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

(define width 768)
(define heads 12)
(define head-dim 64)
(define layers 12)
(define ln-eps 1e-5)

;; --- generic helpers (candidates for the tl stdlib, see docs/stdlib-candidates.md)

;; Size of dimension i (ggml order); dimensions past the rank are 1.
(define (dim t i)
  (let ([s (shape t)])
    (if (< i (length s)) (list-ref s i) 1)))

;; torch.nn.Linear: y = x W^T + b, with W stored as [out, in].
(define (linear x prefix)
  (ggml-add (ggml-mul-mat (weight (string-append prefix "weight")) x)
            (weight (string-append prefix "bias"))))

;; torch.nn.LayerNorm over the innermost dimension.
(define (layer-norm x prefix eps)
  (ggml-add (ggml-mul (ggml-norm x eps) (weight (string-append prefix "weight")))
            (weight (string-append prefix "bias"))))

;; [sin(p f_0) .. sin(p f_n-1), cos(p f_0) .. cos(p f_n-1)] for positions
;; p = 0 .. len-1 and f_i = exp(-i ln(max-timescale) / (n - 1)), n = size / 2.
;; Result: [size, len].
(define (sinusoidal-positions size len max-timescale)
  (let* ([n (quotient size 2)]
         [freqs (ggml-exp (ggml-scale (ggml-arange 0.0 (inexact n) 1.0)
                                      (- (/ (log max-timescale) (max 1 (- n 1))))))]
         [positions (ggml-arange 0.0 (inexact len) 1.0)]
         ;; Outer product as a K=1 matmul: [1, n] x [1, len] -> [n, len].
         [angles (ggml-mul-mat (ggml-reshape-2d freqs 1 n) (ggml-reshape-2d positions 1 len))])
    (ggml-concat (ggml-sin angles) (ggml-cos angles) 0)))

;; torch.nn.MultiheadAttention (fused in_proj, batch_first=False semantics),
;; self-attention over x: [D, L, B]. mask: [L, L, 1, B] additive, 0 or -1e30.
(define (multi-head-attention x prefix mask n-heads)
  (let* ([d (dim x 0)] [len (dim x 1)] [batch (dim x 2)] [hd (quotient d n-heads)]
         [qkv (ggml-add (ggml-mul-mat (weight (string-append prefix "in_proj_weight")) x)
                        (weight (string-append prefix "in_proj_bias")))]   ; [3D, L, B]
         [es (car (strides qkv))]
         [row (* 3 d es)]
         ;; [hd, heads, L, B] views of q, k, v inside the fused projection.
         [part (lambda (i)
                 (ggml-view-4d qkv hd n-heads len batch (* hd es) row (* row len) (* i d es)))]
         [q (ggml-cont (ggml-permute (part 0) 0 2 1 3))]       ; [hd, L, heads, B]
         [k (ggml-cont (ggml-permute (part 1) 0 2 1 3))]       ; [hd, L, heads, B]
         [v (ggml-cont (ggml-permute (part 2) 1 2 0 3))]       ; [L, hd, heads, B]
         [scores (ggml-soft-max-ext (ggml-mul-mat k q) mask    ; [L_k, L_q, heads, B]
                                    (/ 1.0 (sqrt (inexact hd))) 0.0)]
         [ctx (ggml-mul-mat v scores)]                          ; [hd, L_q, heads, B]
         [merged (ggml-reshape-3d (ggml-cont (ggml-permute ctx 0 2 1 3)) d len batch)])
    (linear merged (string-append prefix "out_proj."))))

;; Additive attention mask from key paddings [L, B] (1 = padding): [L, L, 1, B].
(define (key-padding-mask paddings)
  (let ([len (dim paddings 0)] [batch (dim paddings 1)])
    (ggml-repeat-4d (ggml-scale (ggml-reshape-4d paddings len 1 1 batch) -1e30)
                    len len 1 batch)))

;; Mean over the sequence (dim 1) of x: [D, L, B], counting only positions
;; where valid: [1, L, B] is 1. Result: [D, B].
(define (masked-mean x valid eps)
  (let* ([d (dim x 0)] [batch (dim x 2)]
         [sums (ggml-sum-rows (ggml-cont (ggml-transpose (ggml-mul x valid))))]  ; [1, D, B]
         [counts (ggml-scale-bias (ggml-sum-rows (ggml-cont (ggml-permute valid 1 0 2 3))) 1.0 eps)])  ; [1, 1, B]
    (ggml-reshape-2d (ggml-div sums counts) d batch)))

;; --- Preprocessing: the checkpoint's Tokenizer.tokenize(texts, max_len=64)

(define tok (tokenizer (asset "tokenizer.model")))

(preprocess ([text string])
  (let-values ([(ids mask) (tokenize tok text 'lowercase #t 'special-tokens #f 'pad-to 64 'pad-id 0)])
    (model-inputs [ids ids]
                  [paddings (array-affine mask -1 1)])))

;; --- TIPSv2 text encoder

(define (layer-name i)
  (string-append "layer." (if (< i 10) "0" "") (number->string i)))

(define (resblock x i mask valid)
  (let* ([prefix (string-append "transformer.resblocks." (number->string i) ".")]
         [x (ggml-add x (multi-head-attention (layer-norm x (string-append prefix "ln_1.") ln-eps)
                                              (string-append prefix "attn.") mask heads))]
         [h (layer-norm x (string-append prefix "ln_2.") ln-eps)]
         [h (ggml-mul (ggml-relu (linear h (string-append prefix "mlp.c_fc."))) valid)]
         [h (ggml-mul (linear h (string-append prefix "mlp.c_proj.")) valid)])
    (ggml-add x h)))

(model (inputs [ids i32 (64 batch)] [paddings f32 (64 batch)])
  (define len (dim ids 0))
  (define batch (dim ids 1))
  ;; get-rows looks rows up per batch entry of its table, so flatten the ids.
  (define tokens
    (ggml-reshape-3d (ggml-get-rows (weight "token_embedding.weight") (ggml-reshape-1d ids (* len batch)))
                     width len batch))
  (define x0
    (tap "embed" (ggml-add (ggml-scale tokens (sqrt (inexact width)))
                           (sinusoidal-positions width len 10000.0))
         3))
  (define valid (ggml-reshape-3d (ggml-scale-bias paddings -1.0 1.0) 1 len batch))   ; [1, L, B]
  (define mask (key-padding-mask paddings))
  (define x
    (let loop ([i 0] [x x0])
      (if (= i layers)
          x
          (loop (+ i 1) (tap (layer-name i) (resblock x i mask valid) 3)))))
  (define final (tap "final_ln" (layer-norm x "ln_final." ln-eps) 3))
  (outputs [embedding (masked-mean final valid 1e-8) 2]))
