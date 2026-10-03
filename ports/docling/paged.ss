;; Paged-attention decoding for Granite Docling. Not a standalone program: append it to
;; docling.ss (cat docling.ss paged.ss > docling-paged.ss) and run the `caption-paged`
;; pipeline on the native CUDA device.
;;
;; The K/V of every layer lives in one pool of 16-token blocks (pk.i / pv.i) instead of the
;; dense [hd, decode-length, kv-heads] caches; decode attends over exactly the sequence's own
;; length through a block table (vLLM's paged attention kernels). The host owns the table: a
;; single sequence takes blocks 0.. in order, so token j sits in pool slot j.
;;
;; Entries (numpy order):
;;   prefill-image-paged  pixels, ids, gather, pos as prefill-image, slots [1, 2L]
;;                        (little-endian i64 pool slot, 0), at [1, 1] -> logits
;;   decode-step-paged    token [1, n], pos [1, n], tables [n, 128] i32 (block ids),
;;                        lens [1, n] (keys including this token), slots [1, 2n] -> next [n]
;;                        (n sequences per step; n = 1 is plain decoding)
;; Pipelines:
;;   caption-paged  image + prompt -> text, tokens
;;   caption-batch  image + prompt + count (a string, 1..16): prefills once, then decodes
;;                  `count` copies of the sequence together. They share the prompt's full
;;                  K/V blocks; the partial last block is rebuilt per sequence by replaying
;;                  its tokens as a batched decode. -> text, tokens, identical (1 if every
;;                  sequence generated the same tokens)

(define block-size 16)
(define blocks-per-seq (quotient decode-length block-size))
(define max-seqs 16)
(define pool-blocks (* max-seqs blocks-per-seq))

(for-each (lambda (i)
            (define-state (format "pk.~a" i) f16 head-dim block-size (* kv-heads pool-blocks))
            (define-state (format "pv.~a" i) f16 head-dim block-size (* kv-heads pool-blocks)))
          (iota layers))

(define (to-f16 t) (ggml-cast t GGML_TYPE_F16))
(define (to-f32 t) (ggml-cast t GGML_TYPE_F32))

;; k, v [hd, kv-heads, n] -> the pool, at the slots the host chose.
(define (paged-write! i k v slots)
  (let ([n (dim k 2)] [width (* head-dim kv-heads)])
    (effect (paged-cache-write (ggml-reshape-2d (to-f16 k) width n) (ggml-reshape-2d (to-f16 v) width n)
                               (state (format "pk.~a" i)) (state (format "pv.~a" i))
                               slots block-size kv-heads))))

(define (project-v h p n) (ggml-reshape-3d (nn:linear h (p "self_attn.v_proj")) head-dim kv-heads n))

;; Prefill attends over the prompt's own K/V and fills the pool for decoding.
(define (paged-prefill-attention slots)
  (lambda (x p pos mask i)
    (let ([h (rms-norm x (p "input_layernorm"))])
      (let-values ([(q k) (query+key h p pos)])
        (let* ([n (dim h 1)] [v (project-v h p n)])
          (paged-write! i k v slots)
          (nn:linear (attn:sdpa (as-heads q heads n) (as-heads k kv-heads n) (as-heads v kv-heads n)
                                'mask mask 'flash flash-attention)
                     (p "self_attn.o_proj")))))))

(define (paged-decode-attention tables lens slots)
  (lambda (x p pos mask i)
    (let ([h (rms-norm x (p "input_layernorm"))])
      (let-values ([(q k) (query+key h p pos)])
        (let ([v (project-v h p (dim h 1))])
          (paged-write! i k v slots)
          (nn:linear (ggml-reshape-2d (to-f32 (paged-attention (to-f16 q) (state (format "pk.~a" i)) (state (format "pv.~a" i))
                                                               tables lens block-size decode-length
                                                               (/ 1.0 (sqrt (inexact head-dim))) kv-heads))
                                      (* head-dim heads) (dim h 1))
                     (p "self_attn.o_proj")))))))

(model prefill-image-paged (inputs [pixels f32 (512 512 3 _)] [ids i32 (_ 1)] [gather i32 (_ 1)] [pos i32 (_ 1)] [slots i32 (_ 1)] [at i32 (1 1)])
  (define feats (image-features pixels))
  (define x (embed (flat ids) (flat gather) feats))
  (define n (dim x 1))
  (define attention (paged-prefill-attention (flat slots)))
  (define hidden-states
    (rms-norm (let loop ([i 0] [x x]) (if (= i layers) x (loop (+ i 1) (decoder-layer x i (flat pos) (attn:causal-mask n) attention))))
              "decoder.norm"))
  (outputs [logits (ggml-mul-mat (weight "decoder.embed_tokens.weight") (ggml-get-rows hidden-states (flat at))) 2]))

(model decode-step-paged (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [tables i32 (128 _)] [lens i32 (_ 1)] [slots i32 (_ 1)])
  (define x (ggml-reshape-2d (nn:embedding "decoder.embed_tokens.weight" (flat token)) hidden (dim (flat token) 0)))
  (define attention (paged-decode-attention tables (flat lens) (flat slots)))
  (define hidden-states
    (rms-norm (let loop ([i 0] [x x]) (if (= i layers) x (loop (+ i 1) (decoder-layer x i (flat pos) #f attention))))
              "decoder.norm"))
  (outputs [next (ggml-argmax (ggml-mul-mat (weight "decoder.embed_tokens.weight") hidden-states)) 1]))

(define (slot-array start count)
  (list->array (apply append (map (lambda (j) (list (+ start j) 0)) (iota count)))))

(define block-table (list->array (iota pool-blocks)))

(define (greedy-paged first first-pos max-new)
  (let ([limit (min max-new (- decode-length first-pos))])
    (let loop ([tokens (list first)] [pos first-pos])
      (if (done? tokens limit)
          tokens
          (let ([r (run decode-step-paged [token (list->array (list (util:last tokens)))] [pos (list->array (list pos))]
                                          [tables block-table] [lens (list->array (list (+ pos 1)))] [slots (slot-array pos 1)])])
            (loop (append tokens (util:->integers (array->list (output r 'next)))) (+ pos 1)))))))

(pipeline caption-paged ([image image] [prompt string])
  (let-values ([(tiles rows cols) (image-tile image 2048 512 'mean '(0.5 0.5 0.5) 'std '(0.5 0.5 0.5))])
    (let* ([rows (car (util:->integers (array->list rows)))]
           [cols (car (util:->integers (array->list cols)))]
           [n-tiles (car (array-shape tiles))]
           [n-image-tokens (* tile-tokens n-tiles)]
           [expanded (expand-image (apply-chat-template prompt) rows cols)])
      (let-values ([(ids gather pos n) (prompt-inputs expanded n-image-tokens)])
        (let* ([at (list->array (list (- n 1)))]
               [r (run prefill-image-paged [pixels tiles] [ids ids] [gather gather] [pos pos] [slots (slot-array 0 n)] [at at])]
               [first (car (array->list (array-argmax (output r 'logits))))])
          (generation-results (greedy-paged first n 512)))))))

;; --- several sequences per decode step

(define (all? pred xs) (or (null? xs) (and (pred (car xs)) (all? pred (cdr xs)))))

(define (seq-table s k) (map (lambda (j) (if (or (= s 0) (< j k)) j (+ (* s blocks-per-seq) j))) (iota blocks-per-seq)))
(define (seq-slot s k p) (+ (* (list-ref (seq-table s k) (quotient p block-size)) block-size) (remainder p block-size)))

(define (batch-step seqs k toks p)
  (let ([r (run decode-step-paged [token (list->array toks)] [pos (list->array (map (lambda (s) p) seqs))]
                                  [tables (array-reshape (list->array (apply append (map (lambda (s) (seq-table s k)) seqs))) (length seqs) blocks-per-seq)]
                                  [lens (list->array (map (lambda (s) (+ p 1)) seqs))]
                                  [slots (list->array (apply append (map (lambda (s) (list (seq-slot s k p) 0)) seqs)))])])
    (util:->integers (array->list (output r 'next)))))

(pipeline caption-batch ([image image] [prompt string] [count string])
  (let-values ([(tiles rows cols) (image-tile image 2048 512 'mean '(0.5 0.5 0.5) 'std '(0.5 0.5 0.5))])
    (let* ([rows (car (util:->integers (array->list rows)))]
           [cols (car (util:->integers (array->list cols)))]
           [b (string->number count)]
           [n-tiles (car (array-shape tiles))]
           [n-image-tokens (* tile-tokens n-tiles)]
           [expanded (expand-image (apply-chat-template prompt) rows cols)])
      (unless (and (fixnum? b) (<= 1 b max-seqs)) (error 'caption-batch "count must be 1..16" count))
      (let-values ([(ids gather pos n) (prompt-inputs expanded n-image-tokens)])
        (let* ([at (list->array (list (- n 1)))]
               [r (run prefill-image-paged [pixels tiles] [ids ids] [gather gather] [pos pos] [slots (slot-array 0 n)] [at at])]
               [first (car (array->list (array-argmax (output r 'logits))))]
               [k (quotient n block-size)]
               [id-list (util:->integers (array->list ids))]
               [others (cdr (iota b))]
               [limit (min 512 (- decode-length n))]
               [all (iota b)])
          (for-each (lambda (p) (unless (null? others) (batch-step others k (map (lambda (s) (list-ref id-list p)) others) p)))
                    (let loop ([p (* k block-size)]) (if (>= p n) '() (cons p (loop (+ p 1))))))
          (let* ([outs (let loop ([outs (map (lambda (s) (list first)) all)] [p n])
                         (if (all? (lambda (o) (done? o limit)) outs)
                             outs
                             (let ([nexts (batch-step all k (map util:last outs) p)])
                               (loop (map (lambda (o nx) (if (done? o limit) o (append o (list nx)))) outs nexts) (+ p 1)))))])
            (results [text (detokenize tok (car outs))] [tokens (car outs)]
                     [identical (list->array (list (if (all? (lambda (o) (equal? o (car outs))) outs) 1 0)))])))))))
