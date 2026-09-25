;; (tl tensor), imported as tensor:  -- shape and layout helpers.
;; Shapes and axes are ggml order (innermost first); strides are in bytes.
(library (tl tensor)
  (export dim stride as-f32 flatten slice concat)
  (import (rnrs) (only (chezscheme) format quotient remainder) (tensorlisp runtime))

  ;; Size of axis i; 1 past the rank. (`shape` drops trailing size-1 axes,
  ;; so batch 1 would otherwise be missing.)
  (define (dim t i)
    (let ([s (shape t)])
      (if (< i (length s)) (list-ref s i) 1)))

  ;; Byte stride of axis i; past the rank, the size of everything before it.
  (define (stride t i)
    (let ([s (strides t)])
      (if (< i (length s)) (list-ref s i) (* (stride t (- i 1)) (dim t (- i 1))))))

  ;; t, cast to f32 unless it already is (e.g. f16 weights added to activations).
  (define (as-f32 t) (if (eq? (dtype t) 'f32) t (ggml-cast t GGML_TYPE_F32)))

  ;; All elements as one axis (copies if t isn't contiguous).
  (define (flatten t)
    (let ([t (if (contiguous? t) t (ggml-cont t))])
      (ggml-reshape-1d t (fold-left * 1 (shape t)))))

  ;; View of elements [from, from + n) along axis (0-3); other axes unchanged.
  (define (slice t axis from n)
    (unless (and (fixnum? axis) (<= 0 axis 3)) (error 'tensor:slice "axis must be 0-3" axis))
    (unless (<= 0 from (+ from n) (dim t axis))
      (error 'tensor:slice (format "~a..~a is outside axis ~a of size ~a" from (+ from n) axis (dim t axis))))
    (let ([ne (lambda (i) (if (= i axis) n (dim t i)))])
      (ggml-view-4d t (ne 0) (ne 1) (ne 2) (ne 3) (stride t 1) (stride t 2) (stride t 3)
                    (* from (stride t axis)))))

  ;; Concatenation of a list of tensors along axis.
  (define (concat ts axis)
    (when (null? ts) (error 'tensor:concat "no tensors"))
    (fold-left (lambda (acc t) (ggml-concat acc t axis)) (car ts) (cdr ts))))
