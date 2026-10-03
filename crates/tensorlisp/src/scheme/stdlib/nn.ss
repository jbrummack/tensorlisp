;; (tl nn), imported as nn:  -- layers with PyTorch semantics.
;;
;; `prefix` is a module's path in the state dict, without the trailing dot:
;; (nn:linear x "blocks.0.mlp.fc1") uses "blocks.0.mlp.fc1.weight" and, if the
;; file has it, "...bias". Shapes are ggml order: tokens [D, L, B], images
;; [W, H, C, B] (torch's NCHW memory layout). Options are 'name value pairs.
(library (tl nn)
  (export linear layer-norm rms-norm embedding gelu gelu-tanh gelu-tanh-exact
          conv2d conv2d-depthwise conv-transpose2d patch-embed upsample-nearest max-pool avg-pool
          sinusoidal-positions masked-mean)
  (import (rnrs) (only (chezscheme) format quotient remainder iota) (tensorlisp runtime) (tl tensor))

  (define (param prefix name) (string-append prefix "." name))

  ;; Weight `prefix.name`, or #f if the file doesn't have it.
  (define (optional prefix name)
    (let ([n (param prefix name)]) (and (weight? n) (weight n))))

  (define (add-bias y bias) (if bias (ggml-add y bias) y))

  ;; torch.nn.Linear: x [in, ...] -> [out, ...]; weight [out, in] in torch.
  ;; With a LoRA adapter attached to `prefix` (lora-attach!), its delta is added.
  (define (linear x prefix)
    (let* ([y (ggml-mul-mat (weight (param prefix "weight")) x)]
           [delta (lora-delta x prefix)])
      (add-bias (if delta (ggml-add y delta) y) (optional prefix "bias"))))

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

  ;; A per-axis option in PyTorch's order: an integer n (both axes) or a list
  ;; (height width). Returns the values for x (width) and y (height).
  (define (axes who name v)
    (cond
      [(fixnum? v) (values v v)]
      [(and (list? v) (= (length v) 2) (for-all fixnum? v)) (values (cadr v) (car v))]
      [else (error who (format "~a must be an integer or a list (height width)" name) v)]))

  ;; Default padding: kernel/2 per axis ("same" for odd kernels, stride 1).
  (define (padding who o kw kh)
    (let ([p (%opt o 'padding)])
      (if p (axes who 'padding p) (values (quotient kw 2) (quotient kh 2)))))

  (define (conv-bias prefix c-out)
    (let ([b (optional prefix "bias")]) (and b (ggml-reshape-3d b 1 1 c-out))))

  ;; torch.nn.Conv2d (groups = 1) on x [W, H, C, B] -> [W', H', C_out, B].
  ;; Kernels may be non-square. 'stride (1) and 'padding (kernel/2, "same"
  ;; for odd kernels) are an integer or a list (height width) like PyTorch's
  ;; tuples. 'method: auto (im2col on GPUs, direct on the CPU: each ~3x
  ;; faster on its device than the other), im2col (patches + one matmul;
  ;; in bands of output rows when the patches would exceed 256 MiB) or direct.
  (define (conv2d x prefix . opts)
    (let*-values ([(o) (%options 'nn:conv2d opts '((stride . 1) (padding . #f) (method . auto)))]
                  [(kernel) (weight (param prefix "weight"))]          ; [kw, kh, C_in, C_out]
                  [(kw kh c-in c-out) (values (dim kernel 0) (dim kernel 1) (dim kernel 2) (dim kernel 3))]
                  [(sx sy) (axes 'nn:conv2d 'stride (%opt o 'stride))]
                  [(px py) (padding 'nn:conv2d o kw kh)]
                  [(method) (case (%opt o 'method)
                              [(auto) (if (eq? (device) 'gpu) 'im2col 'direct)]
                              [(im2col direct) (%opt o 'method)]
                              [else (error 'nn:conv2d "method must be auto, im2col or direct" (%opt o 'method))])])
      (add-bias (if (eq? method 'direct)
                    (ggml-conv-2d-direct kernel x sx sy px py 1 1)
                    (im2col-conv x kernel kw kh c-in c-out sx sy px py))
                (conv-bias prefix c-out))))

  ;; im2col's columns take 4 * kw * kh * C_in bytes per output pixel (10 GB
  ;; for a 9x9 conv over 256 channels at 400 x 304 pixels). Beyond this they
  ;; are made and multiplied in bands of output rows, one after the other,
  ;; so the allocator reuses the memory.
  (define im2col-limit (* 256 1024 1024))

  (define (im2col-conv x kernel kw kh c-in c-out sx sy px py)
    (let* ([w-out (+ 1 (quotient (- (+ (dim x 0) (* 2 px)) kw) sx))]
           [h-out (+ 1 (quotient (- (+ (dim x 1) (* 2 py)) kh) sy))]
           [row-bytes (* 4 kw kh c-in w-out (dim x 3))]
           [rows (max 1 (quotient im2col-limit row-bytes))])
      (if (>= rows h-out)
          (im2col-matmul x kernel kw kh c-in c-out sx sy px py)
          (let ([padded (if (= px py 0) x (ggml-pad-ext x px px py py 0 0 0 0))])
            (let loop ([r 0] [bands '()])
              (if (>= r h-out)
                  (concat (reverse bands) 1)
                  (let* ([n (min rows (- h-out r))]
                         [band (ggml-cont (slice padded 1 (* r sy) (+ (* (- n 1) sy) kh)))])
                    (loop (+ r n) (cons (im2col-matmul band kernel kw kh c-in c-out sx sy 0 0) bands)))))))))

  (define (im2col-matmul x kernel kw kh c-in c-out sx sy px py)
    (let* ([batch (dim x 3)]
           [k (* kw kh c-in)]
           [pointwise? (and (= kw kh sx sy 1) (= px py 0))]
           ;; [kw*kh*C, W', H', B]; for 1x1 convolutions the pixels with channels innermost.
           [cols (if pointwise?
                     (ggml-cont (ggml-permute (ggml-reshape-3d x (* (dim x 0) (dim x 1)) c-in batch) 1 0 2 3))
                     (ggml-im2col kernel x sx sy px py 1 1 #t GGML_TYPE_F32))]
           [w-out (if pointwise? (dim x 0) (dim cols 1))]
           [h-out (if pointwise? (dim x 1) (dim cols 2))]
           [n (* w-out h-out batch)]
           ;; [W'H'B, C_out] = [W', H', B, C_out] in memory; the kernel is the
           ;; second matmul operand, which must be f32.
           [y (ggml-mul-mat (ggml-reshape-2d cols k n) (ggml-reshape-2d (as-f32 kernel) k c-out))])
      (if (= batch 1)
          (ggml-reshape-4d y w-out h-out c-out 1)
          (ggml-cont (ggml-permute (ggml-reshape-4d y w-out h-out batch c-out) 0 1 3 2)))))

  ;; Depthwise Conv2d (groups = channels, weight [C, 1, kh, kw] in torch);
  ;; 'stride and 'padding like conv2d.
  (define (conv2d-depthwise x prefix . opts)
    (let*-values ([(o) (%options 'nn:conv2d-depthwise opts '((stride . 1) (padding . #f)))]
                  [(kernel) (weight (param prefix "weight"))]          ; [kw, kh, 1, C]
                  [(sx sy) (axes 'nn:conv2d-depthwise 'stride (%opt o 'stride))]
                  [(px py) (padding 'nn:conv2d-depthwise o (dim kernel 0) (dim kernel 1))])
      (add-bias (ggml-conv-2d-dw-direct kernel x sx sy px py 1 1) (conv-bias prefix (dim kernel 3)))))

  ;; torch.nn.ConvTranspose2d (groups 1, padding 0, weight [C_in, C_out, kh, kw]
  ;; in torch) on x [W, H, C_in, 1] -> [(W-1)*s + kw, (H-1)*s + kh, C_out, 1].
  ;; 'stride (1) is the same on both axes. Batch 1 only (ggml's CPU kernel
  ;; ignores the batch). Kernel = stride (upsampling, no overlap) is a matmul
  ;; and a pixel shuffle; other kernels use ggml's op, which is very slow on
  ;; Metal.
  (define (conv-transpose2d x prefix . opts)
    (let* ([o (%options 'nn:conv-transpose2d opts '((stride . 1)))]
           [kernel (weight (param prefix "weight"))]                  ; [kw, kh, C_out, C_in]
           [s (%int 'nn:conv-transpose2d 'stride (%opt o 'stride))])
      (unless (= (dim x 3) 1) (error 'nn:conv-transpose2d "batch must be 1" (shape x)))
      (add-bias (if (= (dim kernel 0) (dim kernel 1) s)
                    (unpatchify x kernel s)
                    (ggml-conv-transpose-2d-p0 (as-f32 kernel) x s))
                (conv-bias prefix (dim kernel 2)))))

  ;; Each pixel times the kernel [k, k, C_out, C_in] becomes a k x k patch.
  (define (unpatchify x kernel k)
    (let* ([w (dim x 0)] [h (dim x 1)] [c-in (dim x 2)] [c-out (dim kernel 2)] [n (* w h)]
           [pixels (ggml-cont (ggml-transpose (ggml-reshape-2d x n c-in)))]                     ; [C_in, W*H]
           [taps (ggml-cont (ggml-transpose (ggml-reshape-2d (as-f32 kernel) (* k k c-out) c-in)))] ; [C_in, k*k*C_out]
           [y (ggml-mul-mat pixels taps)]                             ; [W*H, k*k*C_out]: [w, h, kx, ky, C_out]
           [es (stride y 0)]
           ;; Kernel row ky: [w, h, kx, C_out] -> [kx + k*w, 1, h, C_out].
           [row (lambda (ky)
                  (let ([v (ggml-view-4d y w h k c-out (* w es) (* n es) (* k k n es) (* ky k n es))])
                    (ggml-reshape-4d (ggml-cont (ggml-permute v 1 2 0 3)) (* k w) 1 h c-out)))])
      ;; Rows interleaved: [k*w, k (ky), h, C_out] is [k*w, k*h, C_out] in memory.
      (ggml-reshape-4d (concat (map row (iota k)) 1) (* k w) (* k h) c-out 1)))

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
