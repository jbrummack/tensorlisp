;; (tl vision), imported as vision:  -- image-token layouts and detection heads.
;; Shapes are ggml order: tokens [C, N], images [W, H, C, B].
(library (tl vision)
  (export tokens->grid grid->tokens resize-positions anchor-grid dfl decode-ltrb)
  (import (rnrs) (only (chezscheme) format quotient remainder) (tensorlisp runtime) (tl tensor))

  ;; Tokens [C, w*h] (row-major grid) -> image layout [w, h, C], and back.
  (define (tokens->grid tokens w h)
    (ggml-cont (ggml-permute (ggml-reshape-3d (ggml-cont tokens) (dim tokens 0) w h) 2 0 1 3)))
  (define (grid->tokens grid)
    (let ([w (dim grid 0)] [h (dim grid 1)] [c (dim grid 2)])
      (ggml-reshape-2d (ggml-cont (ggml-permute grid 1 2 0 3)) c (* w h))))

  ;; Position embeddings [C, g*g] of a square g x g grid resized to w x h:
  ;; torch F.interpolate(mode="bilinear", antialias=True) (DINOv2/ViT
  ;; interpolate_pos_encoding with a target size). Unchanged if the size matches.
  (define (resize-positions pos w h)
    (let* ([n (dim pos 1)] [g (exact (round (sqrt n)))])
      (unless (= (* g g) n) (error 'vision:resize-positions (format "~a positions are not a square grid" n)))
      (if (and (= w g) (= h g))
          pos
          (grid->tokens (ggml-interpolate (tokens->grid pos g g) w h (dim pos 0) 1
                                          (+ GGML_SCALE_MODE_BILINEAR GGML_SCALE_FLAG_ANTIALIAS))))))

  ;; Cell centers of a w x h grid, row-major: two tensors [w*h], x + 0.5 and y + 0.5.
  (define (anchor-grid w h)
    (let ([n (* w h)])
      (values (ggml-reshape-1d (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.5 (inexact w) 1.0) w 1) w h 1 1) n)
              (ggml-reshape-1d (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.5 (inexact h) 1.0) 1 h) w h 1 1) n))))

  ;; Distribution focal loss decoding (YOLOv8+ DFL): raw [A, 4 * bins]
  ;; (per side a distribution over bins) -> expected distances [A, 4] (l t r b).
  ;; 'bins (16).
  (define (dfl raw . opts)
    (let* ([o (%options 'vision:dfl opts '((bins . 16)))]
           [bins (%int 'vision:dfl 'bins (%opt o 'bins))]
           [a (dim raw 0)]
           [logits (ggml-reshape-3d (if (contiguous? raw) raw (ggml-cont raw)) a bins 4)]   ; [A, bins, 4]
           [probs (ggml-soft-max (ggml-cont (ggml-permute logits 1 0 2 3)))]                   ; [bins, A, 4]
           [bin-values (ggml-reshape-2d (ggml-arange 0.0 (inexact bins) 1.0) bins 1)])
      (ggml-reshape-2d (ggml-mul-mat (ggml-reshape-2d probs bins (* a 4)) bin-values) a 4)))

  ;; Distances [A, 4] (l t r b, in cells) from the cell centers of a w x h grid
  ;; -> boxes [A, 4] cx, cy, w, h in pixels (Ultralytics dist2bbox, times the
  ;; level's stride: pixels per cell).
  (define (decode-ltrb dist w h pixels-per-cell)
    (let*-values ([(a) (* w h)]
                  [(ax ay) (anchor-grid w h)])
      (let* ([side (lambda (j) (ggml-view-1d dist a (* j (stride dist 1))))]
             [half (lambda (t) (ggml-scale t 0.5))]
             [s (inexact pixels-per-cell)]
             [column (lambda (t) (ggml-reshape-2d (ggml-scale t s) a 1))])
        (concat (list (column (ggml-add ax (half (ggml-sub (side 2) (side 0)))))
                      (column (ggml-add ay (half (ggml-sub (side 3) (side 1)))))
                      (column (ggml-add (side 0) (side 2)))
                      (column (ggml-add (side 1) (side 3))))
                1)))))
