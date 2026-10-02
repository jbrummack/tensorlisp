;; Gemma 4 E2B (google/gemma-4-E2B), text decoder only, ported from
;; transformers' modeling_gemma4.py.
;;
;; Entries (shapes in numpy order; the batch is 1):
;;   step   token [1, n] i32, pos [1, n] i32 -> logits [1, 262144] (last token), next [1] (argmax)
;;          n tokens at positions pos (a prefill chunk or one decode token): writes their
;;          K/V into the cache at pos, attends over the cache, returns the last token's logits
;; Pipelines (raw inputs):
;;   generate  prompt -> text, tokens        greedy, at most 32 new tokens
;;
;; What differs from Gemma 3: head size 256 on sliding layers and 512 on every 5th
;; (global) layer, no 1/sqrt(hd) attention scale, a scale-free norm on V, RMSNorm
;; without the +1, "proportional" RoPE on global layers (a quarter of the dims rotate),
;; per-layer embeddings (PLE) feeding every layer, a layer scalar, final logit
;; softcapping, and K/V sharing: layers 15..34 have no K/V projections and attend over
;; the cache of layer 13 (sliding) or 14 (global).
;;
;; Shapes in comments are ggml order (innermost first).

(import (tl tensor) (tl nn) (tl attn) (tl util))

(define hidden 1536)
(define heads 8)
(define layers 35)
(define first-shared 15)
(define ple-dim 256)
(define eps 1e-6)
(define sliding-window 512)
(define softcap 30.0)
(define (full? i) (= (remainder (+ i 1) 5) 0))
(define (head-dim i) (if (full? i) 512 256))
(define (kv-source i) (cond [(< i first-shared) i] [(full? i) 14] [else 13]))

;; Tokens the KV cache holds (the longest prompt + generation).
(define capacity 2048)
;; ggml's GELU uses an f16 lookup table on the CPU (~1e-3); #t computes the
;; tanh approximation exactly from ops.
(define exact-gelu #f)

(define dim tensor:dim)
(define flat tensor:flatten)
(define (rms-norm x name) (nn:rms-norm x name 'eps eps))
(define (gelu x) (if exact-gelu (nn:gelu-tanh-exact x) (nn:gelu-tanh x)))
(define (prefix-fn i) (lambda (name) (format "layers.~a.~a" i name)))

;; transformers keeps the embedding scale in the checkpoint's bf16:
;; sqrt(1536) = 39.19 becomes 39.25. The PLE scale sqrt(256) = 16 is exact.
(define embed-scale 39.25)
(define ple-scale 16.0)

;; Proportional RoPE: of a global head's 256 rotation pairs (i, i + 256) only the
;; first 64 rotate, the others have frequency 0. ggml divides the angle by the
;; frequency factor, so those get a huge one.
(define (freq-factors)
  (ggml-scale-bias (ggml-step (ggml-scale-bias (ggml-arange 0.0 256.0 1.0) 1.0 -63.5)) 1e30 1.0))

(define (rope x pos i)
  (if (full? i)
      (ggml-rope-ext x pos (freq-factors) 512 2 0 1000000.0 1.0 0.0 1.0 32.0 1.0)
      (attn:rope x pos 'base 10000.0)))

(define (mlp x p)
  (nn:linear (ggml-mul (gelu (nn:linear x (p "mlp.gate_proj"))) (nn:linear x (p "mlp.up_proj")))
             (p "mlp.down_proj")))

;; --- KV cache: one f16 key and one value cache [hd, capacity] per layer that has its own K/V

(define (cache kind i) (state (format "~a.~a" kind i)))
(for-each (lambda (i)
            (define-state (format "k.~a" i) f16 (head-dim i) capacity)
            (define-state (format "v.~a" i) f16 (head-dim i) capacity))
          (iota first-shared))

;; Additive masks [capacity, n] from the query positions: key j is allowed where
;; 0 <= pos - j (< window for sliding layers). Unwritten rows have j > pos.
(define (position-mask pos window)
  (let* ([n (dim pos 0)]
         [q (ggml-repeat-4d (ggml-reshape-2d (ggml-cast pos GGML_TYPE_F32) 1 n) capacity n 1 1)]
         [dist (ggml-sub q (ggml-reshape-2d (ggml-arange 0.0 (inexact capacity) 1.0) capacity 1))]
         [flags (let ([causal (ggml-step (ggml-scale-bias dist 1.0 1.0))])
                  (if window
                      (ggml-mul causal (ggml-step (ggml-scale-bias dist -1.0 (inexact window))))
                      causal))])
    (ggml-scale-bias flags (- attn:blocked) attn:blocked)))

;; --- per-layer embeddings

;; [256, n, 35]: token identity (a 262144 x 8960 table) plus a projection of the
;; scaled embedding, normed, times 1/sqrt(2).
(define (per-layer-inputs ids embeds)
  (let* ([n (dim embeds 1)]
         [token (ggml-scale (nn:embedding "embed_tokens_per_layer.weight" ids) ple-scale)]
         [proj (ggml-scale (ggml-mul-mat (weight "per_layer_model_projection.weight") embeds)
                           (/ 1.0 (sqrt (inexact hidden))))]
         [proj (rms-norm (ggml-reshape-3d proj ple-dim layers n) "per_layer_projection_norm")]
         [sum (ggml-scale (ggml-add proj (ggml-reshape-3d token ple-dim layers n)) (sqrt 0.5))])
    (ggml-cont (ggml-permute sum 0 2 1 3))))

;; --- decoder

(define (project-q h i pos)
  (let ([p (prefix-fn i)] [n (dim h 1)])
    (rope (rms-norm (ggml-reshape-3d (nn:linear h (p "self_attn.q_proj")) (head-dim i) heads n) (p "self_attn.q_norm")) pos i)))

;; This layer's own key (normed, rotated) and value (scale-free norm), each [hd, 1, n].
(define (project-kv h i pos)
  (let ([p (prefix-fn i)] [n (dim h 1)] [hd (head-dim i)])
    (values (rope (rms-norm (ggml-reshape-3d (nn:linear h (p "self_attn.k_proj")) hd 1 n) (p "self_attn.k_norm")) pos i)
            (ggml-rms-norm (ggml-reshape-3d (nn:linear h (p "self_attn.v_proj")) hd 1 n) eps))))

;; Everything after attention: o_proj, MLP, the per-layer embedding and the layer scalar.
;; attended [heads * hd, n], ple [256, n, 35].
(define (finish-layer x attended i ple)
  (let* ([p (prefix-fn i)]
         [n (dim x 1)]
         [x (ggml-add x (rms-norm (nn:linear attended (p "self_attn.o_proj")) (p "post_attention_layernorm")))]
         [x (ggml-add x (rms-norm (mlp (rms-norm x (p "pre_feedforward_layernorm")) p) (p "post_feedforward_layernorm")))]
         [gate (ggml-mul (gelu (nn:linear x (p "per_layer_input_gate")))
                         (ggml-reshape-2d (tensor:slice ple 2 i 1) ple-dim n))]
         [x (ggml-add x (rms-norm (nn:linear gate (p "per_layer_projection")) (p "post_per_layer_input_norm")))])
    (ggml-mul x (weight (p "layer_scalar")))))

(define (layer x i pos masks ple)
  (let* ([hd (head-dim i)]
         [n (dim x 1)]
         [h (rms-norm x ((prefix-fn i) "input_layernorm"))]
         [q (project-q h i pos)])
    (when (< i first-shared)
      (let-values ([(k v) (project-kv h i pos)])
        (effect (ggml-set-rows (cache "k" i) (ggml-reshape-2d k hd n) pos))
        (effect (ggml-set-rows (cache "v" i) (ggml-reshape-2d v hd n) pos))))
    (let ([src (kv-source i)])
      (finish-layer x
                    (attn:sdpa (ggml-permute q 0 2 1 3)
                               (ggml-reshape-4d (cache "k" src) hd capacity 1 1)
                               (ggml-reshape-4d (cache "v" src) hd capacity 1 1)
                               'mask (if (full? i) (car masks) (cdr masks)) 'scale 1.0 'flash #t)
                    i ple))))

(define (softcapped logits)
  (ggml-scale (ggml-tanh (ggml-scale logits (/ 1.0 softcap))) softcap))

(model step (inputs [token i32 (_ 1)] [pos i32 (_ 1)])
  (define ids (flat token))
  (define positions (flat pos))
  (define n (dim ids 0))
  (define embeds (ggml-reshape-2d (ggml-scale (nn:embedding "embed_tokens.weight" ids) embed-scale) hidden n))
  (define ple (per-layer-inputs ids embeds))
  (define masks (cons (position-mask positions #f) (position-mask positions sliding-window)))
  (define last-hidden
    (let ([x (let loop ([i 0] [x embeds])
               (if (= i layers) x (loop (+ i 1) (layer x i positions masks ple))))])
      (rms-norm (tensor:slice x 1 (- n 1) 1) "norm")))
  (define raw (ggml-mul-mat (weight "embed_tokens.weight") last-hidden))
  (outputs [logits (softcapped raw) 2] [next (ggml-argmax raw) 1]))

;; --- pipeline

(define tok (tokenizer (asset "tokenizer.json")))
(define eos (token-id tok "<eos>"))

(define (generation-results tokens)
  (results [text (detokenize tok tokens)] [tokens tokens]))

(pipeline generate ([prompt string])
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let* ([prompt-ids (util:->integers (array->list ids))]
           [m (length prompt-ids)])
      (when (> (+ m 32) capacity) (error 'generate "prompt too long for the cache" m))
      (let loop ([r (run step [token ids] [pos (list->array (iota m))])] [tokens '()] [pos m])
        (let* ([next (car (util:->integers (array->list (output r 'next))))]
               [tokens (append tokens (list next))])
          (if (or (= next eos) (= (length tokens) 32))
              (generation-results tokens)
              (loop (run step [token (list->array (list next))] [pos (list->array (list pos))])
                    tokens (+ pos 1))))))))
