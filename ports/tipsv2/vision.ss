;; TIPSv2 vision encoder (the vision tower of google/tipsv2-b14): a DINOv2-style
;; ViT-B/14 with one register token and LayerScale, ported from the checkpoint's
;; image_encoder.py.
;;
;; Input (numpy order):  image [batch, 3, H, W], RGB in [0, 1] (no mean/std
;;                       normalization); H and W multiples of 14, 448 native.
;;                       Other sizes interpolate the position embeddings
;;                       (bilinear, antialiased), like the reference.
;; Raw input:            image, preprocessed like the checkpoint's image processor
;;                        (448 x 448 bilinear, rescaled to [0, 1])
;; Outputs:              cls      [batch, 768]           first token after the final norm
;;                       register [batch, 768]           register token after the final norm
;;                       patches  [batch, H/14 * W/14, 768]  patch tokens, row-major
;; Taps:                 embed, block.00 .. block.11, norm   [batch, tokens, 768]
;;
;; Shapes in comments are ggml order (innermost first): D = 768, N = tokens, B = batch.

(import (tl tensor) (tl nn) (tl attn) (tl vision))

(define width 768)
(define heads 12)
(define patch 14)
(define layers 12)
(define ln-eps 1e-6)
;; Fused attention kernel (K/V in f16): faster on Metal, slightly less exact.
(define flash-attention #t)

;; --- Preprocessing: processor_config.json (448 x 448, resample 2 = bilinear, rescale 1/255)

(preprocess ([image image])
  (model-inputs [image (image->array (image-resize image 448 448 'bilinear))]))

;; --- TIPSv2 vision encoder

;; Position embeddings of the patch tokens for a gw x gh grid: [D, gw*gh]
;; (the checkpoint's 32 x 32 grid, resized like torch's bilinear + antialias).
(define (patch-positions gw gh)
  (let* ([pos (weight "pos_embed")]                                          ; [D, 1 + grid^2]
         [patches (tensor:slice pos 1 1 (- (tensor:dim pos 1) 1))])           ; skip the cls row
    (vision:resize-positions (ggml-cont patches) gw gh)))

(define (block x i)
  (let* ([p (lambda (name) (format "blocks.~a.~a" i name))]
         [attn (attn:multi-head (nn:layer-norm x (p "norm1") 'eps ln-eps) (p "attn") heads
                                'names 'timm 'flash flash-attention)]
         [x (ggml-add x (ggml-mul attn (weight (p "ls1.gamma"))))]
         [mlp (nn:linear (nn:gelu (nn:linear (nn:layer-norm x (p "norm2") 'eps ln-eps) (p "mlp.fc1"))) (p "mlp.fc2"))])
    (ggml-add x (ggml-mul mlp (weight (p "ls2.gamma"))))))

(model (inputs [image f32 (_ _ 3 batch)])
  (define w (tensor:dim image 0))
  (define h (tensor:dim image 1))
  (define batch (tensor:dim image 3))
  (define checked
    (unless (and (zero? (remainder w patch)) (zero? (remainder h patch)))
      (error 'tipsv2-vision "image height and width must be multiples of 14" (list h w))))
  (define gw (quotient w patch))
  (define gh (quotient h patch))

  (define patch-tokens
    (ggml-add (nn:patch-embed image "patch_embed.proj" patch) (patch-positions gw gh)))

  ;; [cls + pos_0, register, patches + pos_1..] : [D, 2 + gw*gh, B]
  (define pos (weight "pos_embed"))
  (define cls (ggml-add (ggml-repeat-4d (weight "cls_token") width 1 batch 1)
                        (tensor:slice pos 1 0 1)))
  (define register (ggml-repeat-4d (weight "register_tokens") width 1 batch 1))
  (define x0 (tap "embed" (tensor:concat (list cls register patch-tokens) 1) 3))

  (define x
    (let loop ([i 0] [x x0])
      (if (= i layers)
          x
          (loop (+ i 1) (tap (format "block.~2,'0d" i) (block x i) 3)))))
  (define final (tap "norm" (nn:layer-norm x "norm" 'eps ln-eps) 3))

  ;; Token k of every image: [D, B].
  (define (token k) (ggml-reshape-2d (ggml-cont (tensor:slice final 1 k 1)) width batch))
  (outputs [cls (token 0) 2]
           [register (token 1) 2]
           [patches (tensor:slice final 1 2 (* gw gh)) 3]))
