;; TIPSv2 vision encoder (the vision tower of google/tipsv2-b14): a DINOv2-style
;; ViT-B/14 with one register token and LayerScale, ported from the checkpoint's
;; image_encoder.py.
;;
;; Input (numpy order):  image [batch, 3, H, W], RGB in [0, 1] (no mean/std
;;                       normalization); H and W multiples of 14, 448 native.
;;                       Other sizes interpolate the position embeddings
;;                       (bilinear, antialiased), like the reference.
;; Outputs:              cls      [batch, 768]           first token after the final norm
;;                       register [batch, 768]           register token after the final norm
;;                       patches  [batch, H/14 * W/14, 768]  patch tokens, row-major
;; Taps:                 embed, block.00 .. block.11, norm   [batch, tokens, 768]
;;
;; Shapes in comments are ggml order (innermost first): D = 768, N = tokens, B = batch.

(define width 768)
(define heads 12)
(define patch 14)
(define layers 12)
(define ln-eps 1e-6)
;; Fused attention kernel (K/V in f16): faster on Metal, slightly less exact.
(define flash-attention #t)

;; --- generic helpers (candidates for the tl stdlib, see docs/stdlib-candidates.md)

(define (dim t i)
  (let ([s (shape t)])
    (if (< i (length s)) (list-ref s i) 1)))

;; Byte stride of dimension i; past the rank, the size of everything before.
(define (nb t i)
  (let ([s (strides t)])
    (if (< i (length s)) (list-ref s i) (* (nb t (- i 1)) (dim t (- i 1))))))

(define (linear x prefix)
  (ggml-add (ggml-mul-mat (weight (string-append prefix "weight")) x)
            (weight (string-append prefix "bias"))))

(define (layer-norm x prefix eps)
  (ggml-add (ggml-mul (ggml-norm x eps) (weight (string-append prefix "weight")))
            (weight (string-append prefix "bias"))))

;; Self-attention with a fused qkv projection laid out [q; k; v] (DINOv2/timm
;; `qkv`, torch MultiheadAttention `in_proj`). x: [D, N, B]; mask #f or additive
;; [N, N, 1, B]. qkv-prefix and out-prefix name the two linear layers.
(define (fused-attention x qkv-prefix out-prefix mask n-heads)
  (let* ([d (dim x 0)] [len (dim x 1)] [batch (dim x 2)] [hd (quotient d n-heads)]
         [qkv (linear x qkv-prefix)]                            ; [3D, N, B]
         [es (nb qkv 0)]
         [row (* 3 d es)]
         [part (lambda (i)                                      ; [hd, heads, N, B]
                 (ggml-view-4d qkv hd n-heads len batch (* hd es) row (* row len) (* i d es)))]
         [q (ggml-permute (part 0) 0 2 1 3)]                    ; [hd, N, heads, B]
         [k (ggml-permute (part 1) 0 2 1 3)]
         [v (ggml-permute (part 2) 0 2 1 3)]
         [scale (/ 1.0 (sqrt (inexact hd)))]
         [merged
          (if flash-attention
              ;; Result is [hd, heads, N, B], i.e. already [D, N, B] in memory.
              (ggml-reshape-3d (ggml-flash-attn-ext (ggml-cont q)
                                                    (ggml-cast k GGML_TYPE_F16)
                                                    (ggml-cast v GGML_TYPE_F16)
                                                    mask scale 0.0 0.0)
                               d len batch)
              (let* ([scores (ggml-soft-max-ext (ggml-mul-mat (ggml-cont k) (ggml-cont q)) mask scale 0.0)]  ; [N_k, N_q, heads, B]
                     [vt (ggml-cont (ggml-permute (part 2) 1 2 0 3))]                                     ; [N, hd, heads, B]
                     [ctx (ggml-mul-mat vt scores)])                                                      ; [hd, N_q, heads, B]
                (ggml-reshape-3d (ggml-cont (ggml-permute ctx 0 2 1 3)) d len batch)))])
    (linear merged out-prefix)))

;; torch Conv2d with kernel = stride = p (ViT patch embedding) on image
;; [W, H, C, B]: tokens [D, W/p * H/p, B], row-major over the patch grid.
;; im2col in f32 then one matmul; ggml-conv-2d would round the unfolded image to f16.
(define (patch-embed image prefix p)
  (let* ([kernel (weight (string-append prefix "weight"))]                  ; [p, p, C, D]
         [cols (ggml-im2col kernel image p p 0 0 1 1 #t GGML_TYPE_F32)]    ; [p*p*C, gw, gh, B]
         [k (dim cols 0)] [n (* (dim cols 1) (dim cols 2))] [batch (dim cols 3)]
         [out (ggml-mul-mat (ggml-reshape-2d kernel k (dim kernel 3))       ; [D, n*B]
                            (ggml-reshape-2d cols k (* n batch)))])
    (ggml-add (ggml-reshape-3d out (dim out 0) n batch) (weight (string-append prefix "bias")))))

;; Channels-first image-like tensor [W, H, C] from tokens [C, W*H] (row-major
;; grid of w x h), and back.
(define (tokens->grid tokens w h)
  (ggml-cont (ggml-permute (ggml-reshape-3d (ggml-cont tokens) (dim tokens 0) w h) 2 0 1 3)))
(define (grid->tokens grid)
  (let ([w (dim grid 0)] [h (dim grid 1)] [c (dim grid 2)])
    (ggml-reshape-2d (ggml-cont (ggml-permute grid 1 2 0 3)) c (* w h))))

;; --- TIPSv2 vision encoder

;; Position embeddings of the patch tokens for a gw x gh grid: [D, gw*gh].
;; The checkpoint's are for a square grid (32 x 32 at 448 px); other grids are
;; resized with torch's bilinear + antialias interpolation.
(define (patch-positions gw gh)
  (let* ([pos (weight "pos_embed")]                             ; [D, 1 + grid^2]
         [n (- (dim pos 1) 1)]
         [grid (exact (round (sqrt n)))]
         [patches (ggml-view-2d pos width n (nb pos 1) (nb pos 1))])  ; skip the cls row
    (if (and (= gw grid) (= gh grid))
        patches
        (grid->tokens
          (ggml-interpolate (tokens->grid patches grid grid) gw gh width 1
                            (+ GGML_SCALE_MODE_BILINEAR GGML_SCALE_FLAG_ANTIALIAS))))))

(define (block x i)
  (let* ([prefix (string-append "blocks." (number->string i) ".")]
         [p (lambda (name) (string-append prefix name))]
         [attn (fused-attention (layer-norm x (p "norm1.") ln-eps) (p "attn.qkv.") (p "attn.proj.") #f heads)]
         [x (ggml-add x (ggml-mul attn (weight (p "ls1.gamma"))))]
         [mlp (linear (ggml-gelu-erf (linear (layer-norm x (p "norm2.") ln-eps) (p "mlp.fc1."))) (p "mlp.fc2."))])
    (ggml-add x (ggml-mul mlp (weight (p "ls2.gamma"))))))

(model (inputs [image f32 (_ _ 3 batch)])
  (define w (dim image 0))
  (define h (dim image 1))
  (define batch (dim image 3))
  (define checked
    (unless (and (zero? (remainder w patch)) (zero? (remainder h patch)))
      (error 'tipsv2-vision "image height and width must be multiples of 14" (list h w))))
  (define gw (quotient w patch))
  (define gh (quotient h patch))

  (define patch-tokens
    (ggml-add (patch-embed image "patch_embed.proj." patch) (patch-positions gw gh)))

  ;; [cls + pos_0, register, patches + pos_1..] : [D, 2 + gw*gh, B]
  (define pos (weight "pos_embed"))
  (define cls (ggml-add (ggml-repeat-4d (weight "cls_token") width 1 batch 1)
                        (ggml-view-2d pos width 1 (nb pos 1) 0)))
  (define register (ggml-repeat-4d (weight "register_tokens") width 1 batch 1))
  (define x0 (tap "embed" (ggml-concat (ggml-concat cls register 1) patch-tokens 1) 3))

  (define x
    (let loop ([i 0] [x x0])
      (if (= i layers)
          x
          (loop (+ i 1) (tap (format "block.~2,'0d" i) (block x i) 3)))))
  (define final (tap "norm" (layer-norm x "norm." ln-eps) 3))

  ;; Token k of every image: [D, B] rows of final, nb2 apart.
  (define (token k) (ggml-view-2d final width batch (nb final 2) (* k (nb final 1))))
  (outputs [cls (token 0) 2]
           [register (token 1) 2]
           [patches (ggml-view-3d final width (* gw gh) batch (nb final 1) (nb final 2) (* 2 (nb final 1))) 3]))
