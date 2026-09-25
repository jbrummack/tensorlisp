;; (tl nn), imported as nn:  -- layers with PyTorch semantics.
;;
;; `prefix` is a module's path in the state dict, without the trailing dot:
;; (nn:linear x "blocks.0.mlp.fc1") uses "blocks.0.mlp.fc1.weight" and, if the
;; file has it, "...bias". Shapes are ggml order: tokens [D, L, B], images
;; [W, H, C, B] (torch's NCHW memory layout). Options are 'name value pairs.
(library (tl nn)
  (export linear layer-norm rms-norm embedding gelu gelu-tanh gelu-tanh-exact
          conv2d conv2d-depthwise patch-embed upsample-nearest max-pool avg-pool
          sinusoidal-positions masked-mean)
  (import (rnrs) (only (chezscheme) format quotient remainder) (tensorlisp runtime) (tl tensor))

  (define (param prefix name) (string-append prefix "." name))

  ;; Weight `prefix.name`, or #f if the file doesn't have it.
  (define (optional prefix name)
    (let ([n (param prefix name)]) (and (weight? n) (weight n))))

  (define (add-bias y bias) (if bias (ggml-add y bias) y))

  ;; torch.nn.Linear: x [in, ...] -> [out, ...]; weight [out, in] in torch.
  (define (linear x prefix)
    (add-bias (ggml-mul-mat (weight (param prefix "weight")) x) (optional prefix "bias")))

  ;; torch.nn.LayerNorm over axis 0. 'eps (default 1e-5, torch's; many ViTs use 1e-6).
  (define (layer-norm x prefix . opts)
    (let* ([o (%options 'nn:layer-norm opts '((eps . 1e-5)))]
           [y (ggml-norm x (%real 'nn:layer-norm 'eps (%opt o 'eps)))]
           [w (optional prefix "weight")])
      (add-bias (if w (ggml-mul y w) y) (optional prefix "bias"))))

  ;; RMSNorm over axis 0: x / rms(x) * weight. 'eps (default 1e-6); 'offset adds
  ;; a constant to the weight (Gemma: 1, i.e. (1 + weight)) -- one extra op per
  ;; call; better stored in the file (tl convert --offset).
  (define (rms-norm x prefix . opts)
    (let* ([o (%options 'nn:rms-norm opts '((eps . 1e-6) (offset . 0)))]
           [w (weight (param prefix "weight"))]
           [offset (%real 'nn:rms-norm 'offset (%opt o 'offset))])
      (ggml-mul (ggml-rms-norm x (%real 'nn:rms-norm 'eps (%opt o 'eps)))
                (if (= offset 0.0) w (ggml-scale-bias w 1.0 offset)))))

  ;; torch.nn.Embedding: table (a weight name or tensor, [D, vocab]) and ids
  ;; of any shape up to [L, B] -> [D, L, B]. (get-rows looks rows up per
  ;; batch of a 3-D table, so the ids are flattened first.)
  (define (embedding table ids)
    (let* ([table (if (string? table) (weight table) table)]
           [rows (ggml-get-rows table (flatten ids))])
      (ggml-reshape-3d rows (dim table 0) (dim ids 0) (* (dim ids 1) (dim ids 2)))))

  ;; nn.GELU() (erf).
  (define (gelu x) (ggml-gelu-erf x))
  ;; nn.GELU(approximate="tanh") / gelu_pytorch_tanh with ggml's kernel: on the
  ;; CPU it goes through an f16 table (~1e-3 error), exact on GPUs.
  (define (gelu-tanh x) (ggml-gelu x))
  ;; The tanh approximation computed exactly from ops (for comparisons).
  (define (gelu-tanh-exact x)
    (let ([inner (ggml-scale (ggml-add x (ggml-scale (ggml-mul x (ggml-mul x x)) 0.044715))
                             (sqrt (/ 2.0 3.141592653589793)))])
      (ggml-mul (ggml-scale x 0.5) (ggml-scale-bias (ggml-tanh inner) 1.0 1.0))))

  ;; torch.nn.Conv2d (groups = 1) on x [W, H, C, B] -> [W', H', C_out, B].
  ;; 'stride (1), 'padding (default kernel/2, "same" for odd kernels),
  ;; 'method: auto (im2col on GPUs, direct on the CPU: each ~3x faster on its
  ;; device than the other), im2col (patches + one matmul) or direct.
  (define (conv2d x prefix . opts)
    (let* ([o (%options 'nn:conv2d opts '((stride . 1) (padding . #f) (method . auto)))]
           [kernel (weight (param prefix "weight"))]                  ; [k, k, C_in, C_out]
           [k (dim kernel 0)] [c-in (dim kernel 2)] [c-out (dim kernel 3)]
           [s (%int 'nn:conv2d 'stride (%opt o 'stride))]
           [p (let ([p (%opt o 'padding)]) (if p (%int 'nn:conv2d 'padding p) (quotient k 2)))]
           [method (case (%opt o 'method)
                     [(auto) (if (eq? (device) 'gpu) 'im2col 'direct)]
                     [(im2col direct) (%opt o 'method)]
                     [else (error 'nn:conv2d "method must be auto, im2col or direct" (%opt o 'method))])]
           [y (if (eq? method 'direct)
                  (ggml-conv-2d-direct kernel x s s p p 1 1)
                  (im2col-conv x kernel k c-in c-out s p))])
      (add-bias y (let ([b (optional prefix "bias")]) (and b (ggml-reshape-3d b 1 1 c-out))))))

  (define (im2col-conv x kernel k c-in c-out s p)
    (let* ([batch (dim x 3)]
           [pointwise? (and (= k 1) (= s 1) (= p 0))]
           ;; [k*k*C, W', H', B]; for 1x1 convolutions the pixels with channels innermost.
           [cols (if pointwise?
                     (ggml-cont (ggml-permute (ggml-reshape-3d x (* (dim x 0) (dim x 1)) c-in batch) 1 0 2 3))
                     (ggml-im2col kernel x s s p p 1 1 #t GGML_TYPE_F32))]
           [w-out (if pointwise? (dim x 0) (dim cols 1))]
           [h-out (if pointwise? (dim x 1) (dim cols 2))]
           [n (* w-out h-out batch)]
           ;; [W'H'B, C_out] = [W', H', B, C_out] in memory; the kernel is the
           ;; second matmul operand, which must be f32.
           [y (ggml-mul-mat (ggml-reshape-2d cols (* k k c-in) n)
                            (ggml-reshape-2d (as-f32 kernel) (* k k c-in) c-out))])
      (if (= batch 1)
          (ggml-reshape-4d y w-out h-out c-out 1)
          (ggml-cont (ggml-permute (ggml-reshape-4d y w-out h-out batch c-out) 0 1 3 2)))))

  ;; Depthwise Conv2d (groups = channels, weight [C, 1, k, k] in torch).
  (define (conv2d-depthwise x prefix . opts)
    (let* ([o (%options 'nn:conv2d-depthwise opts '((stride . 1) (padding . #f)))]
           [kernel (weight (param prefix "weight"))]                  ; [k, k, 1, C]
           [k (dim kernel 0)]
           [s (%int 'nn:conv2d-depthwise 'stride (%opt o 'stride))]
           [p (let ([p (%opt o 'padding)]) (if p (%int 'nn:conv2d-depthwise 'padding p) (quotient k 2)))])
      (add-bias (ggml-conv-2d-dw-direct kernel x s s p p 1 1)
                (let ([b (optional prefix "bias")]) (and b (ggml-reshape-3d b 1 1 (dim kernel 3)))))))

  ;; ViT patch embedding (Conv2d with kernel = stride = patch) on an image
  ;; [W, H, C, B] -> tokens [D, W/p * H/p, B], row-major over the patch grid.
  ;; im2col in f32 (ggml-conv-2d would round the pixels to f16).
  (define (patch-embed image prefix patch)
    (let* ([kernel (weight (param prefix "weight"))]                  ; [p, p, C, D]
           [cols (ggml-im2col kernel image patch patch 0 0 1 1 #t GGML_TYPE_F32)]   ; [p*p*C, gw, gh, B]
           [k (dim cols 0)] [n (* (dim cols 1) (dim cols 2))] [batch (dim cols 3)]
           [out (ggml-mul-mat (ggml-reshape-2d kernel k (dim kernel 3)) (ggml-reshape-2d cols k (* n batch)))])
      (add-bias (ggml-reshape-3d out (dim out 0) n batch) (optional prefix "bias"))))

  ;; nn.Upsample(scale_factor=factor, mode="nearest") on [W, H, C, B].
  (define (upsample-nearest x factor) (ggml-upscale x factor GGML_SCALE_MODE_NEAREST))

  (define (pool who op x k opts)
    (let* ([o (%options who opts '((stride . #f) (padding . 0)))]
           [s (let ([s (%opt o 'stride)]) (if s (%int who 'stride s) k))]
           [p (inexact (%int who 'padding (%opt o 'padding)))])
      (ggml-pool-2d x op k k s s p p)))

  ;; nn.MaxPool2d(k, stride = k by default, padding): padding never wins.
  (define (max-pool x k . opts) (pool 'nn:max-pool GGML_OP_POOL_MAX x k opts))
  ;; nn.AvgPool2d(k, ...) with count_include_pad=True (padding counts as zeros).
  (define (avg-pool x k . opts) (pool 'nn:avg-pool GGML_OP_POOL_AVG x k opts))

  ;; Transformer position signal [size, len] for positions 0 .. len-1:
  ;; [sin(p f_0) .. sin(p f_n-1), cos(p f_0) .. cos(p f_n-1)], n = size/2,
  ;; f_i = exp(-i ln(max-timescale) / divisor). 'max-timescale (10000),
  ;; 'divisor: n-1 (T5X, TIPS; default) or n (tensor2tensor, diffusion timesteps),
  ;; 'order: sin-cos (default) or cos-sin.
  (define (sinusoidal-positions size len . opts)
    (let* ([o (%options 'nn:sinusoidal-positions opts '((max-timescale . 10000.0) (divisor . n-1) (order . sin-cos)))]
           [n (quotient size 2)]
           [divisor (case (%opt o 'divisor)
                      [(n-1) (max 1 (- n 1))] [(n) n]
                      [else (error 'nn:sinusoidal-positions "divisor must be n-1 or n" (%opt o 'divisor))])]
           [freqs (ggml-exp (ggml-scale (ggml-arange 0.0 (inexact n) 1.0)
                                        (- (/ (log (%real 'nn:sinusoidal-positions 'max-timescale (%opt o 'max-timescale)))
                                              divisor))))]
           [positions (ggml-arange 0.0 (inexact len) 1.0)]
           ;; Outer product as a K=1 matmul: [1, n] x [1, len] -> [n, len].
           [angles (ggml-mul-mat (ggml-reshape-2d freqs 1 n) (ggml-reshape-2d positions 1 len))])
      (case (%opt o 'order)
        [(sin-cos) (ggml-concat (ggml-sin angles) (ggml-cos angles) 0)]
        [(cos-sin) (ggml-concat (ggml-cos angles) (ggml-sin angles) 0)]
        [else (error 'nn:sinusoidal-positions "order must be sin-cos or cos-sin" (%opt o 'order))])))

  ;; Mean over the sequence of x [D, L, B], counting positions where valid
  ;; ([L, B] or [1, L, B]) is 1. 'eps is added to the count. -> [D, B].
  (define (masked-mean x valid . opts)
    (let* ([o (%options 'nn:masked-mean opts '((eps . 0.0)))]
           [d (dim x 0)] [len (dim x 1)] [batch (dim x 2)]
           [valid (ggml-reshape-3d (if (contiguous? valid) valid (ggml-cont valid)) 1 len batch)]
           ;; sum-rows reduces axis 0 only: move the sequence there.
           [sums (ggml-sum-rows (ggml-cont (ggml-transpose (ggml-mul x valid))))]              ; [1, D, B]
           [counts (ggml-scale-bias (ggml-sum-rows (ggml-cont (ggml-permute valid 1 0 2 3)))
                                    1.0 (%real 'nn:masked-mean 'eps (%opt o 'eps)))])      ; [1, 1, B]
      (ggml-reshape-2d (ggml-div sums counts) d batch))))
