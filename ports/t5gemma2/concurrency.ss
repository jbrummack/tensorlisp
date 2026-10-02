;; Concurrent decoding for T5Gemma 2: B sequences per decode step instead of one.
;; Not a standalone program: crates/tensorlisp/tests/native_concurrency.rs appends
;; it to t5gemma2.ss (after the replacement of paged-max-context) to measure how
;; throughput grows with B on the native CUDA device.
;;
;; Two ways to keep the sequences' K/V:
;;   dense  one [hd, 1088] cache per sequence slot (dk.i / dv.i, 64 slots), every
;;          sequence attends over all 1088 rows under a mask
;;   paged  one pool of 16-token blocks (pk.i / pv.i), a sequence takes only the
;;          blocks it needs and attends over exactly its own length (vLLM's
;;          paged attention kernels); the host owns the block tables
;;
;; Entries (numpy order, n = sequences, m = prompt length):
;;   cross-dense   memory [1, m, 640], rows [1, m] i32 -> (none): the memory's K/V into cache rows
;;   cross-paged   memory [1, m, 640], slots [1, 2m] i32 -> (none): the same into pool slots
;;                 (slots hold one little-endian i64 per token: slot, 0)
;;   decode-dense  token [1, n], pos [1, n], rows [1, n] (cache row of this token),
;;                 mask [n, 1088] f32 -> next [n]
;;   decode-paged  token [1, n], pos [1, n], tables [n, 68] i32 (block ids), lens [1, n]
;;                 (keys including this token), slots [1, 2n] (this token's slot) -> next [n]
;; Pipelines fill-dense / fill-paged: prompt + [slot-or-layout] array -> length

(define max-seqs 64)
(define block-size 16)
(define paged-max-context 1088)
(define max-blocks (quotient (+ paged-max-context block-size -1) block-size))
(define pool-blocks (* max-seqs max-blocks))

(for-each (lambda (i)
            (define-state (format "dk.~a" i) f16 head-dim cache-rows max-seqs)
            (define-state (format "dv.~a" i) f16 head-dim cache-rows max-seqs)
            (define-state (format "pk.~a" i) f16 head-dim block-size pool-blocks)
            (define-state (format "pv.~a" i) f16 head-dim block-size pool-blocks))
          (iota layers))

(define (to-f16 t) (ggml-cast t GGML_TYPE_F16))
(define (to-f32 t) (ggml-cast t GGML_TYPE_F32))

;; The dense caches as [hd, rows * slots], or the first n slots as [hd, rows, 1, n].
(define (dense-rows kind i) (ggml-reshape-2d (state (format "~a.~a" kind i)) head-dim (* cache-rows max-seqs)))
(define (dense-slots kind i n)
  (ggml-reshape-4d (tensor:slice (state (format "~a.~a" kind i)) 2 0 n) head-dim cache-rows 1 n))

(define (k-memory p memory2) (rms-norm (nn:linear memory2 (p "self_attn.k_proj")) (p "self_attn.k_norm")))
(define (v-memory p memory2) (nn:linear memory2 (p "self_attn.v_proj")))

(model cross-dense (inputs [memory f32 (640 _ 1)] [rows i32 (_ 1)])
  (define memory2 (ggml-reshape-2d memory hidden (dim memory 1)))
  (for-each
    (lambda (i)
      (let ([p (prefix-fn "decoder" i)])
        (effect (ggml-set-rows (dense-rows "dk" i) (k-memory p memory2) (flat rows)))
        (effect (ggml-set-rows (dense-rows "dv" i) (v-memory p memory2) (flat rows)))))
    (iota layers))
  (outputs))

(model cross-paged (inputs [memory f32 (640 _ 1)] [slots i32 (_ 1)])
  (define memory2 (ggml-reshape-2d memory hidden (dim memory 1)))
  (for-each
    (lambda (i)
      (let ([p (prefix-fn "decoder" i)])
        (effect (paged-cache-write (to-f16 (k-memory p memory2)) (to-f16 (v-memory p memory2))
                                   (state (format "pk.~a" i)) (state (format "pv.~a" i))
                                   (flat slots) block-size))))
    (iota layers))
  (outputs))

(define (dense-layer x i pos rows mask n)
  (let ([p (prefix-fn "decoder" i)])
    (let ([x (sandwich x p "pre_self_attn_layernorm" "post_self_attn_layernorm"
               (lambda (h)
                 (let-values ([(q k) (query+key h p pos i)])
                   (effect (ggml-set-rows (dense-rows "dk" i) (ggml-reshape-2d k head-dim n) rows))
                   (effect (ggml-set-rows (dense-rows "dv" i) (v-memory p h) rows))
                   (nn:linear (ggml-reshape-2d
                                (attn:sdpa (ggml-reshape-4d q head-dim 1 heads n)
                                           (dense-slots "dk" i n) (dense-slots "dv" i n)
                                           'mask mask 'flash #t)
                                (* head-dim heads) n)
                              (p "self_attn.o_proj")))))])
      (sandwich x p "pre_feedforward_layernorm" "post_feedforward_layernorm" (lambda (h) (mlp h p))))))

(define (paged-layer x i pos tables lens slots n)
  (let ([p (prefix-fn "decoder" i)])
    (let ([x (sandwich x p "pre_self_attn_layernorm" "post_self_attn_layernorm"
               (lambda (h)
                 (let-values ([(q k) (query+key h p pos i)])
                   (let ([kc (state (format "pk.~a" i))] [vc (state (format "pv.~a" i))])
                     (effect (paged-cache-write (ggml-reshape-2d (to-f16 k) head-dim n) (to-f16 (v-memory p h)) kc vc slots block-size))
                     (nn:linear (ggml-reshape-2d
                                  (to-f32 (paged-attention (to-f16 q) kc vc tables lens block-size paged-max-context
                                                        (/ 1.0 (sqrt (inexact head-dim)))))
                                  (* head-dim heads) n)
                                (p "self_attn.o_proj"))))))])
      (sandwich x p "pre_feedforward_layernorm" "post_feedforward_layernorm" (lambda (h) (mlp h p))))))

(define (next-tokens hidden-states)
  (ggml-argmax (ggml-mul-mat (weight "encoder.embed_tokens.weight") (rms-norm hidden-states "decoder.norm"))))

(model decode-dense (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [rows i32 (_ 1)] [mask f32 (1088 _)])
  (define n (dim token 0))
  (define valid (attn:padding-mask mask 1))
  (define x (decoder-embed (flat token)))
  (outputs [next (next-tokens (let loop ([i 0] [x x])
                                (if (= i layers) x (loop (+ i 1) (dense-layer x i (flat pos) (flat rows) valid n)))))
                 1]))

(model decode-paged (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [tables i32 (68 _)] [lens i32 (_ 1)] [slots i32 (_ 1)])
  (define n (dim token 0))
  (define x (decoder-embed (flat token)))
  (outputs [next (next-tokens (let loop ([i 0] [x x])
                                (if (= i layers)
                                    x
                                    (loop (+ i 1) (paged-layer x i (flat pos) tables (flat lens) (flat slots) n)))))
                 1]))

;; The prompt's encoder output and length.
(define (prompt-memory prompt)
  (let-values ([(ids gather pos mask) (encoder-inputs prompt 0)])
    (values (output (run encode [ids ids] [gather gather] [pos pos] [mask mask]) 'memory)
            (car (array-shape mask)))))

(define (first-integer a) (car (util:->integers (array->list a))))

;; slot: the cache slot of this sequence. Its memory goes to rows slot * 1088 + 64 ...
(pipeline fill-dense ([prompt string] [slot array])
  (let-values ([(memory m) (prompt-memory prompt)])
    (when (> m memory-capacity) (error 'fill-dense "prompt too long" m))
    (let ([base (+ (* (first-integer slot) cache-rows) decode-length)])
      (run cross-dense [memory memory] [rows (list->array (map (lambda (j) (+ base j)) (iota m)))]))
    (results [length (list->array (list m))])))

;; layout: [sequence, blocks per sequence]. The sequence owns blocks
;; sequence * blocks ..., so its token j is in pool slot sequence * blocks * 16 + j.
(pipeline fill-paged ([prompt string] [layout array])
  (let-values ([(memory m) (prompt-memory prompt)])
    (let* ([l (util:->integers (array->list layout))]
           [base (* (car l) (cadr l) block-size)])
      (run cross-paged [memory memory]
           [slots (list->array (apply append (map (lambda (j) (list (+ base j) 0)) (iota m))))]))
    (results [length (list->array (list m))])))
