;; (tl attn), imported as attn:  -- attention.
;;
;; Layouts (ggml order): tokens [D, L, B]; per-head tensors [hd, L, heads, B]
;; (split-heads); additive masks [L_k, L_q, 1, B] with 0 where attention is
;; allowed and a large negative number (`blocked`) elsewhere, combined with
;; ggml-add. Masks built here broadcast over heads (and batch unless given).
(library (tl attn)
  (export blocked padding-mask causal-mask window-mask relative-mask
          split-heads merge-heads sdpa rope multi-head)
  (import (rnrs) (only (chezscheme) format quotient remainder) (tensorlisp runtime) (tl tensor) (tl nn))

  ;; Additive value for blocked positions. Not -inf: 0/1 flags times -inf give NaN.
  (define blocked -1e30)

  ;; 0/1 flags (1 = allowed) as additive mask values.
  (define (flags->mask allowed) (ggml-scale-bias allowed (- blocked) blocked))

  ;; Keys that may be attended: valid [L_k] or [L_k, B] with 1 for real tokens,
  ;; 0 for padding -> [L_k, n-q, 1, B] (soft-max-ext needs a full row per query).
  (define (padding-mask valid n-q)
    (let* ([lk (dim valid 0)] [batch (dim valid 1)]
           [m (ggml-reshape-4d (flags->mask (if (contiguous? valid) valid (ggml-cont valid))) lk 1 1 batch)])
      (ggml-repeat-4d m lk n-q 1 batch)))

  ;; Mask [n-k, n-q] for positions 0.. on both sides, from the distance
  ;; dist = q - k: allowed where scale * dist + bias > 0 for every
  ;; (scale . bias) in conditions. Built in the graph (arange, step).
  (define (relative-mask n-k n-q conditions)
    (when (null? conditions) (error 'attn:relative-mask "no conditions"))
    (let* ([q (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.0 (inexact n-q) 1.0) 1 n-q) n-k n-q 1 1)]
           [dist (ggml-sub q (ggml-reshape-2d (ggml-arange 0.0 (inexact n-k) 1.0) n-k 1))]
           [step (lambda (c) (ggml-step (ggml-scale-bias dist (inexact (car c)) (inexact (cdr c)))))])
      (flags->mask (fold-left (lambda (acc c) (ggml-mul acc (step c))) (step (car conditions)) (cdr conditions)))))

  ;; Causal mask [n, n]: a query sees itself and earlier keys; with 'window w
  ;; only the last w (HF sliding_window with causal masks: 0 <= q - k < w).
  (define (causal-mask n . opts)
    (let* ([o (%options 'attn:causal-mask opts '((window . #f)))]
           [w (%opt o 'window)])
      (relative-mask n n (if w
                             (list '(1 . 1) (cons -1 (%int 'attn:causal-mask 'window w)))
                             (list '(1 . 1))))))

  ;; Bidirectional window [n, n]: keys up to 'left - 1 before and 'right - 1
  ;; after the query (HF: 0 <= q - k < left, or 0 < k - q < right).
  (define (window-mask n . opts)
    (let* ([o (%options 'attn:window-mask opts '((left . #f) (right . #f)))]
           [left (%opt o 'left)] [right (%opt o 'right)])
      (unless (and left right) (error 'attn:window-mask "'left and 'right are required"))
      (relative-mask n n (list (cons -1 (%int 'attn:window-mask 'left left))
                               (cons 1 (%int 'attn:window-mask 'right right))))))

  (define (f16 t) (if (eq? (dtype t) 'f16) t (ggml-cast t GGML_TYPE_F16)))

  ;; [heads * hd, L, B] -> [hd, L, heads, B] (a view; head-major channels like torch).
  (define (split-heads x heads)
    (let ([d (dim x 0)])
      (unless (= 0 (remainder d heads)) (error 'attn:split-heads (format "~a channels don't split into ~a heads" d heads)))
      (ggml-permute (ggml-reshape-4d (if (contiguous? x) x (ggml-cont x)) (quotient d heads) heads (dim x 1) (dim x 2))
                    0 2 1 3)))

  ;; [hd, L, heads, B] -> [heads * hd, L, B].
  (define (merge-heads x)
    (ggml-reshape-3d (ggml-cont (ggml-permute x 0 2 1 3)) (* (dim x 0) (dim x 2)) (dim x 1) (dim x 3)))

  ;; Scaled dot-product attention. q [hd, L_q, heads, B]; k and v
  ;; [hd, L_k, kv-heads, B] where kv-heads divides heads (grouped/multi-query
  ;; attention: consecutive query heads share a K/V head, like HF repeat_kv;
  ;; no copies). -> [heads * hd, L_q, B].
  ;; 'mask #f or additive [L_k, L_q(, 1, B)]; 'scale (1/sqrt(hd));
  ;; 'flash #t uses ggml-flash-attn-ext (K/V and mask in f16, cast unless they
  ;; already are, e.g. an f16 KV cache: faster on GPUs for long sequences and
  ;; single-token decoding, slightly less exact).
  (define (sdpa q k v . opts)
    (let* ([o (%options 'attn:sdpa opts '((mask . #f) (scale . #f) (flash . #f)))]
           [hd (dim q 0)] [lq (dim q 1)] [heads (dim q 2)] [batch (dim q 3)]
           [scale (let ([s (%opt o 'scale)]) (if s (%real 'attn:sdpa 'scale s) (/ 1.0 (sqrt (inexact hd)))))]
           [mask (%opt o 'mask)])
      (unless (= 0 (remainder heads (dim k 2)))
        (error 'attn:sdpa (format "~a K/V heads don't divide ~a query heads" (dim k 2) heads)))
      (if (%opt o 'flash)
          ;; Result [hd, heads, L_q, B] is [heads * hd, L_q, B] in memory.
          (ggml-reshape-3d
            (ggml-flash-attn-ext (if (contiguous? q) q (ggml-cont q)) (f16 k) (f16 v) (and mask (f16 mask)) scale 0.0 0.0)
            (* hd heads) lq batch)
          (let* ([probs (ggml-soft-max-ext (ggml-mul-mat (ggml-cont k) (ggml-cont q)) mask scale 0.0)]  ; [L_k, L_q, heads, B]
                 [vt (ggml-cont (ggml-permute v 1 0 2 3))]                                          ; [L_k, hd, kv-heads, B]
                 [out (ggml-mul-mat vt probs)])                                                      ; [hd, L_q, heads, B]
            (merge-heads out)))))

  ;; Rotary position embedding on x [hd, heads, L] (before split-heads), pos
  ;; i32 [L]. 'base (10000), 'scale: position multiplier (HF "linear" scaling
  ;; with factor f is 1/f), 'mode: neox (rotate halves, HF rotate_half; default)
  ;; or normal (rotate adjacent pairs, original GPT-J/LLaMA-Meta), 'dims rotated (hd).
  (define (rope x pos . opts)
    (let* ([o (%options 'attn:rope opts '((base . 10000.0) (scale . 1.0) (mode . neox) (dims . #f)))]
           [mode (case (%opt o 'mode) [(neox) 2] [(normal) 0]
                   [else (error 'attn:rope "mode must be neox or normal" (%opt o 'mode))])]
           [dims (let ([d (%opt o 'dims)]) (if d (%int 'attn:rope 'dims d) (dim x 0)))])
      (ggml-rope-ext x pos #f dims mode 0 (%real 'attn:rope 'base (%opt o 'base))
                     (%real 'attn:rope 'scale (%opt o 'scale)) 0.0 1.0 32.0 1.0)))

  ;; Self-attention with a fused q/k/v projection laid out [q; k; v] on x
  ;; [D, L, B] -> [D, L, B]. 'names: torch (nn.MultiheadAttention:
  ;; prefix.in_proj_weight/_bias, prefix.out_proj; default) or timm
  ;; (prefix.qkv, prefix.proj). 'mask, 'flash as in sdpa.
  (define (multi-head x prefix heads . opts)
    (let* ([o (%options 'attn:multi-head opts '((names . torch) (mask . #f) (flash . #f)))]
           [d (dim x 0)] [len (dim x 1)] [batch (dim x 2)] [hd (quotient d heads)]
           [qkv (case (%opt o 'names)
                  [(torch) (ggml-add (ggml-mul-mat (weight (string-append prefix ".in_proj_weight")) x)
                                     (weight (string-append prefix ".in_proj_bias")))]
                  [(timm) (linear x (string-append prefix ".qkv"))]
                  [else (error 'attn:multi-head "names must be torch or timm" (%opt o 'names))])]   ; [3D, L, B]
           [es (stride qkv 0)] [row (* 3 d es)]
           [part (lambda (i)                                                                    ; [hd, L, heads, B]
                   (ggml-permute (ggml-view-4d qkv hd heads len batch (* hd es) row (* row len) (* i d es))
                                 0 2 1 3))]
           [out (sdpa (part 0) (part 1) (part 2) 'mask (%opt o 'mask) 'flash (%opt o 'flash))])
      (linear out (string-append prefix (if (eq? (%opt o 'names) 'torch) ".out_proj" ".proj"))))))
