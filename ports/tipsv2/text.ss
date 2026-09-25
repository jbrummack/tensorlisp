;; TIPSv2 text encoder (the text tower of google/tipsv2-b14), ported from the
;; checkpoint's text_encoder.py.
;;
;; Raw input:            text, tokenized by (preprocess ...) with the checkpoint's
;;                        SentencePiece model (asset "tokenizer.model")
;; Inputs (numpy order):  ids      [batch, 64]  SentencePiece ids (lowercased text, no BOS/EOS)
;;                        paddings [batch, 64]  1.0 at padding positions, else 0.0
;; Output:                embedding [batch, 768], the masked mean of the final
;;                        layer norm (not L2-normalized, like encode_text).
;; Taps:                  embed, layer.00 .. layer.11, final_ln   [batch, 64, 768]
;;
;; Shapes in comments are ggml order (innermost first): D = 768, L = 64, B = batch.

(import (tl tensor) (tl nn) (tl attn))

(define width 768)
(define heads 12)
(define layers 12)
(define ln-eps 1e-5)

;; --- Preprocessing: the checkpoint's Tokenizer.tokenize(texts, max_len=64)

(define tok (tokenizer (asset "tokenizer.model")))

(preprocess ([text string])
  (let-values ([(ids mask) (tokenize tok text 'lowercase #t 'special-tokens #f 'pad-to 64 'pad-id 0)])
    (model-inputs [ids ids]
                  [paddings (array-affine mask -1 1)])))

;; --- TIPSv2 text encoder

(define (resblock x i mask valid)
  (let* ([p (lambda (name) (format "transformer.resblocks.~a.~a" i name))]
         [x (ggml-add x (attn:multi-head (nn:layer-norm x (p "ln_1") 'eps ln-eps) (p "attn") heads 'mask mask))]
         [h (nn:layer-norm x (p "ln_2") 'eps ln-eps)]
         [h (ggml-mul (ggml-relu (nn:linear h (p "mlp.c_fc"))) valid)]
         [h (ggml-mul (nn:linear h (p "mlp.c_proj")) valid)])
    (ggml-add x h)))

(model (inputs [ids i32 (64 batch)] [paddings f32 (64 batch)])
  (define len (tensor:dim ids 0))
  (define batch (tensor:dim ids 1))
  (define x0
    (tap "embed" (ggml-add (ggml-scale (nn:embedding "token_embedding.weight" ids) (sqrt (inexact width)))
                           (nn:sinusoidal-positions width len))
         3))
  (define tokens (ggml-scale-bias paddings -1.0 1.0))                   ; [L, B], 1 = token
  (define valid (ggml-reshape-3d tokens 1 len batch))                   ; [1, L, B]
  (define mask (attn:padding-mask tokens len))                          ; [L, L, 1, B]
  (define x
    (let loop ([i 0] [x x0])
      (if (= i layers)
          x
          (loop (+ i 1) (tap (format "layer.~2,'0d" i) (resblock x i mask valid) 3)))))
  (define final (tap "final_ln" (nn:layer-norm x "ln_final" 'eps ln-eps) 3))
  (outputs [embedding (nn:masked-mean final valid 'eps 1e-8) 2]))
