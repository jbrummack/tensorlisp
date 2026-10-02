;; Concurrent decoding for Gemma 4: B sequences per decode step instead of one.
;; Not a standalone program: crates/tensorlisp/tests/native_gemma4.rs appends it to
;; gemma4.ss to measure how throughput grows with B on the native CUDA device.
;;
;; Two ways to keep the sequences' K/V (only the 15 layers with their own K/V store any):
;;   dense  one [hd, 640] cache per sequence slot (dk.i / dv.i, 64 slots), every
;;          sequence attends over all 640 rows under a mask (exact sliding window)
;;   paged  one pool of 16-token blocks (pk.i / pv.i), a sequence takes only the blocks
;;          it needs and attends over exactly its own length (vLLM's paged attention
;;          kernels); the host owns the block tables. The kernel has no sliding window,
;;          so this is exact only for contexts up to 512 tokens.
;;
;; Entries (numpy order, n = sequences, m = prompt length):
;;   prefill-dense  token [1, m], pos [1, m], rows [1, m] (cache row of each token) -> next [1]
;;   prefill-paged  token [1, m], pos [1, m], slots [1, 2m] (little-endian i64 pool slot, 0) -> next [1]
;;                  both attend over the prompt's own K/V, not the cache
;;   decode-dense   token [1, n], pos [1, n], rows [1, n] -> next [n]
;;   decode-paged   token [1, n], pos [1, n], tables [n, 40] i32 (block ids), lens [1, n]
;;                  (keys including this token), slots [1, 2n] -> next [n]
;; Pipelines fill-dense / fill-paged: prompt + [slot-or-layout] array -> length, first (the first generated token)

(define max-seqs 64)
(define block-size 16)
(define batch-rows 640)
(define paged-max-context 640)
(define pool-blocks (* max-seqs 40))

(for-each (lambda (i)
            (define-state (format "dk.~a" i) f16 (head-dim i) batch-rows max-seqs)
            (define-state (format "dv.~a" i) f16 (head-dim i) batch-rows max-seqs)
            (define-state (format "pk.~a" i) f16 (head-dim i) block-size pool-blocks)
            (define-state (format "pv.~a" i) f16 (head-dim i) block-size pool-blocks))
          (iota first-shared))

(define (to-f16 t) (ggml-cast t GGML_TYPE_F16))
(define (to-f32 t) (ggml-cast t GGML_TYPE_F32))

;; Scaled embeddings [1536, n] and the per-layer inputs of token ids.
(define (embed-inputs ids)
  (let ([embeds (ggml-reshape-2d (ggml-scale (nn:embedding "embed_tokens.weight" ids) embed-scale) hidden (dim ids 0))])
    (values embeds (per-layer-inputs ids embeds))))

(define (next-tokens x)
  (ggml-argmax (ggml-mul-mat (weight "embed_tokens.weight") (rms-norm x "norm"))))

(define (own-layer-input x i pos)
  (let* ([h (rms-norm x ((prefix-fn i) "input_layernorm"))])
    (values h (project-q h i pos))))

;; --- prefill: the prompt attends over its own K/V; write(i, k, v) stores them

(define (prefill-pass ids positions write)
  (let-values ([(embeds ple) (embed-inputs ids)])
    (let* ([n (dim ids 0)]
           [masks (cons (attn:causal-mask n) (attn:causal-mask n 'window sliding-window))])
      (let loop ([i 0] [x embeds] [kvs '()])
        (if (= i layers)
            (next-tokens (tensor:slice x 1 (- n 1) 1))
            (let-values ([(h q) (own-layer-input x i positions)])
              (let* ([own (if (< i first-shared)
                              (let-values ([(k v) (project-kv h i positions)])
                                (write i k v)
                                (list (cons i (cons k v))))
                              '())]
                     [kvs (append own kvs)]
                     [kv (cdr (assv (kv-source i) kvs))]
                     [hd (head-dim i)])
                (loop (+ i 1)
                      (finish-layer x
                                    (attn:sdpa (ggml-permute q 0 2 1 3)
                                               (ggml-reshape-4d (car kv) hd n 1 1) (ggml-reshape-4d (cdr kv) hd n 1 1)
                                               'mask (if (full? i) (car masks) (cdr masks)) 'scale 1.0 'flash #t)
                                    i ple)
                      kvs))))))))

(define (dense-rows kind i) (ggml-reshape-2d (state (format "~a.~a" kind i)) (head-dim i) (* batch-rows max-seqs)))
(define (dense-slots kind i n)
  (ggml-reshape-4d (tensor:slice (state (format "~a.~a" kind i)) 2 0 n) (head-dim i) batch-rows 1 n))

(model prefill-dense (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [rows i32 (_ 1)])
  (define rows* (flat rows))
  (outputs [next (prefill-pass (flat token) (flat pos)
                               (lambda (i k v)
                                 (let ([hd (head-dim i)] [n (dim k 2)])
                                   (effect (ggml-set-rows (dense-rows "dk" i) (ggml-reshape-2d k hd n) rows*))
                                   (effect (ggml-set-rows (dense-rows "dv" i) (ggml-reshape-2d v hd n) rows*)))))
                 1]))

(model prefill-paged (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [slots i32 (_ 1)])
  (define slots* (flat slots))
  (outputs [next (prefill-pass (flat token) (flat pos)
                               (lambda (i k v)
                                 (let ([hd (head-dim i)] [n (dim k 2)])
                                   (effect (paged-cache-write (ggml-reshape-2d (to-f16 k) hd n) (ggml-reshape-2d (to-f16 v) hd n)
                                                              (state (format "pk.~a" i)) (state (format "pv.~a" i))
                                                              slots* block-size)))))
                 1]))

;; --- decode: one token per sequence

;; Additive masks [640, 1, 1, n] from each sequence's position.
(define (batch-mask pos window)
  (let* ([n (dim pos 0)]
         [q (ggml-repeat-4d (ggml-reshape-4d (ggml-cast pos GGML_TYPE_F32) 1 1 1 n) batch-rows 1 1 n)]
         [dist (ggml-sub q (ggml-reshape-4d (ggml-arange 0.0 (inexact batch-rows) 1.0) batch-rows 1 1 1))]
         [flags (let ([causal (ggml-step (ggml-scale-bias dist 1.0 1.0))])
                  (if window
                      (ggml-mul causal (ggml-step (ggml-scale-bias dist -1.0 (inexact window))))
                      causal))])
    (ggml-scale-bias flags (- attn:blocked) attn:blocked)))

(define (decode-pass ids positions attend)
  (let-values ([(embeds ple) (embed-inputs ids)])
    (let loop ([i 0] [x embeds])
      (if (= i layers)
          (next-tokens x)
          (let-values ([(h q) (own-layer-input x i positions)])
            (loop (+ i 1) (finish-layer x (attend i h q) i ple)))))))

(model decode-dense (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [rows i32 (_ 1)])
  (define positions (flat pos))
  (define rows* (flat rows))
  (define n (dim positions 0))
  (define masks (cons (batch-mask positions #f) (batch-mask positions sliding-window)))
  (outputs [next (decode-pass (flat token) positions
                              (lambda (i h q)
                                (let ([hd (head-dim i)])
                                  (when (< i first-shared)
                                    (let-values ([(k v) (project-kv h i positions)])
                                      (effect (ggml-set-rows (dense-rows "dk" i) (ggml-reshape-2d k hd n) rows*))
                                      (effect (ggml-set-rows (dense-rows "dv" i) (ggml-reshape-2d v hd n) rows*))))
                                  (let ([src (kv-source i)])
                                    (ggml-reshape-2d
                                      (attn:sdpa (ggml-reshape-4d q hd 1 heads n)
                                                 (dense-slots "dk" src n) (dense-slots "dv" src n)
                                                 'mask (if (full? i) (car masks) (cdr masks)) 'scale 1.0 'flash #t)
                                      (* hd heads) n)))))
                 1]))

(model decode-paged (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [tables i32 (40 _)] [lens i32 (_ 1)] [slots i32 (_ 1)])
  (define positions (flat pos))
  (define n (dim positions 0))
  (define lens* (flat lens))
  (define slots* (flat slots))
  (outputs [next (decode-pass (flat token) positions
                              (lambda (i h q)
                                (let ([hd (head-dim i)])
                                  (when (< i first-shared)
                                    (let-values ([(k v) (project-kv h i positions)])
                                      (effect (paged-cache-write (ggml-reshape-2d (to-f16 k) hd n) (ggml-reshape-2d (to-f16 v) hd n)
                                                                 (state (format "pk.~a" i)) (state (format "pv.~a" i))
                                                                 slots* block-size))))
                                  (let ([src (kv-source i)])
                                    (ggml-reshape-2d
                                      (to-f32 (paged-attention (to-f16 q) (state (format "pk.~a" src)) (state (format "pv.~a" src))
                                                               tables lens* block-size paged-max-context 1.0))
                                      (* hd heads) n)))))
                 1]))

;; --- pipelines

(define (first-integer a) (car (util:->integers (array->list a))))

(define (prompt-ids prompt)
  (let-values ([(ids mask) (tokenize tok prompt)])
    (when (> (+ (car (array-shape ids)) 32) batch-rows) (error 'fill "prompt too long" (car (array-shape ids))))
    ids))

(define (fill-results r m) (results [length (list->array (list m))] [first (output r 'next)]))

;; slot: the cache slot of this sequence; its token j goes to row slot * 640 + j.
(pipeline fill-dense ([prompt string] [slot array])
  (let* ([ids (prompt-ids prompt)] [m (car (array-shape ids))] [base (* (first-integer slot) batch-rows)])
    (fill-results (run prefill-dense [token ids] [pos (list->array (iota m))]
                       [rows (list->array (map (lambda (j) (+ base j)) (iota m)))])
                  m)))

;; layout: [sequence, blocks per sequence]. The sequence owns blocks sequence * blocks ...,
;; so its token j is in pool slot sequence * blocks * 16 + j.
(pipeline fill-paged ([prompt string] [layout array])
  (let* ([ids (prompt-ids prompt)] [m (car (array-shape ids))]
         [l (util:->integers (array->list layout))]
         [base (* (car l) (cadr l) block-size)])
    (fill-results (run prefill-paged [token ids] [pos (list->array (iota m))]
                       [slots (list->array (apply append (map (lambda (j) (list (+ base j) 0)) (iota m))))])
                  m)))
