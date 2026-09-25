;; YOLO11 (Ultralytics) object detector, ported from ultralytics/nn/modules
;; (Conv, C3k2, C3k, SPPF, C2PSA, Detect). Weights: the model's state dict
;; after fuse() (batch norms folded into the convolutions), see reference.py.
;; The block structure is read from the weights (C3k vs Bottleneck, repeats),
;; so the n/s/m/l/x sizes all work.
;;
;; Input (numpy order): image [batch, 3, H, W], RGB in [0, 1], H and W
;;                      multiples of 32 (640 x 640 letterboxed natively).
;; Raw input:           image, letterboxed to 640 x 640 (fill 114).
;; Output:              head [batch, 84, anchors]: cx, cy, w, h in input pixels,
;;                      then 80 class probabilities (Ultralytics' inference output;
;;                      anchors = (H/8 * W/8) + (H/16 * W/16) + (H/32 * W/32)).
;; Taps:                layer.00 .. layer.22 (each block's output).
;; Postprocess:         boxes [n, 4] xyxy on the original image, scores [n],
;;                      classes [n], labels (text): NMS like the predictor
;;                      (confidence 0.25, IoU 0.7, max 300).
;;
;; Shapes in comments are ggml order: [W, H, C, B] like torch's NCHW memory.

(define input-size 640)
(define reg-max 16)
(define class-names
  '#("person" "bicycle" "car" "motorcycle" "airplane" "bus" "train" "truck" "boat" "traffic light"
     "fire hydrant" "stop sign" "parking meter" "bench" "bird" "cat" "dog" "horse" "sheep" "cow"
     "elephant" "bear" "zebra" "giraffe" "backpack" "umbrella" "handbag" "tie" "suitcase" "frisbee"
     "skis" "snowboard" "sports ball" "kite" "baseball bat" "baseball glove" "skateboard" "surfboard"
     "tennis racket" "bottle" "wine glass" "cup" "fork" "knife" "spoon" "bowl" "banana" "apple"
     "sandwich" "orange" "broccoli" "carrot" "hot dog" "pizza" "donut" "cake" "chair" "couch"
     "potted plant" "bed" "dining table" "toilet" "tv" "laptop" "mouse" "remote" "keyboard"
     "cell phone" "microwave" "oven" "toaster" "sink" "refrigerator" "book" "clock" "vase" "scissors"
     "teddy bear" "hair drier" "toothbrush"))

;; --- helpers

(define (dim t i)
  (let ([s (shape t)])
    (if (< i (length s)) (list-ref s i) 1)))

(define (join . parts) (apply string-append parts))

(define (as-f32 t) (if (eq? (dtype t) 'f32) t (ggml-cast t GGML_TYPE_F32)))

;; Convolutions as im2col + one matmul (ggml's matmul kernels; 2.6x faster
;; on Metal) or ggml's direct convolution kernel (3.7x faster on the CPU).
(define (im2col-convolutions) (eq? (device) 'gpu))

;; torch Conv2d (bias from the folded batch norm), "same" padding, on [W, H, C, 1].
(define (conv2d x name stride)
  (let* ([kernel (weight (join name ".weight"))]           ; [k, k, C_in, C_out]
         [k (dim kernel 0)] [pad (quotient k 2)] [c-out (dim kernel 3)]
         [y (if (im2col-convolutions)
                (let* ([cols (if (and (= k 1) (= stride 1))
                                 ;; 1x1: the patches are the pixels, channels innermost.
                                 (ggml-cont (ggml-permute (ggml-reshape-3d x (* (dim x 0) (dim x 1)) (dim x 2) 1) 1 0 2 3))
                                 (ggml-im2col kernel x stride stride pad pad 1 1 #t GGML_TYPE_F32))]  ; [k*k*C, W', H']
                       [w-out (if (= k 1) (dim x 0) (dim cols 1))]
                       [h-out (if (= k 1) (dim x 1) (dim cols 2))]
                       [n (* w-out h-out)])
                  ;; [W'H', C_out] is [W', H', C_out] in memory.
                  ;; The kernel is the matmul's second operand, which must be f32.
                  (ggml-reshape-4d (ggml-mul-mat (ggml-reshape-2d cols (dim cols 0) n)
                                                 (ggml-reshape-2d (as-f32 kernel) (* k k (dim kernel 2)) c-out))
                                   w-out h-out c-out 1))
                (ggml-conv-2d-direct kernel x stride stride pad pad 1 1))])
    (ggml-add y (ggml-reshape-3d (weight (join name ".bias")) 1 1 c-out))))

;; Depthwise (groups = channels): kernel [k, k, 1, C].
(define (conv2d-dw x name stride)
  (let* ([kernel (weight (join name ".weight"))]
         [pad (quotient (dim kernel 0) 2)]
         [y (ggml-conv-2d-dw-direct kernel x stride stride pad pad 1 1)])
    (ggml-add y (ggml-reshape-3d (weight (join name ".bias")) 1 1 (dim kernel 3)))))

;; Ultralytics Conv: conv + (folded) batch norm + SiLU.
(define (conv x prefix stride . act?)
  (let ([y (conv2d x (join prefix ".conv") stride)])
    (if (or (null? act?) (car act?)) (ggml-silu y) y)))

(define (dwconv x prefix stride)
  (ggml-silu (conv2d-dw x (join prefix ".conv") stride)))

;; Channels [from, from + n) of [W, H, C, B].
(define (channels x from n)
  (let ([s (strides x)])
    (ggml-view-4d x (dim x 0) (dim x 1) n (dim x 3) (list-ref s 1) (list-ref s 2)
                  (* (dim x 0) (dim x 1) (dim x 2) 4) (* from (list-ref s 2)))))

(define (concat-channels xs) (fold-left (lambda (acc x) (ggml-concat acc x 2)) (car xs) (cdr xs)))

;; Indices k = 0, 1, ... while (weight? (make-name k)).
(define (count-while make-name)
  (let loop ([k 0]) (if (weight? (make-name k)) (loop (+ k 1)) k)))

(define (index s k) (join s "." (number->string k)))

;; --- blocks

(define (bottleneck x p)
  (ggml-add x (conv (conv x (join p ".cv1") 1) (join p ".cv2") 1)))

;; C3k: cv3(cat(m(cv1(x)), cv2(x))), m = Bottlenecks.
(define (c3k x p)
  (let* ([n (count-while (lambda (k) (join (index (join p ".m") k) ".cv1.conv.weight")))]
         [m (let loop ([k 0] [y (conv x (join p ".cv1") 1)])
              (if (= k n) y (loop (+ k 1) (bottleneck y (index (join p ".m") k)))))])
    (conv (concat-channels (list m (conv x (join p ".cv2") 1))) (join p ".cv3") 1)))

;; C3k2 (a C2f): split cv1(x) in two, chain the inner blocks on the second
;; half, concatenate every intermediate, cv2. Inner blocks are C3k (they have
;; a cv3) or Bottlenecks.
(define (c3k2 x p)
  (let* ([y (conv x (join p ".cv1") 1)]
         [c (quotient (dim y 2) 2)]
         [n (count-while (lambda (k) (join (index (join p ".m") k) ".cv1.conv.weight")))]
         [parts (let loop ([k 0] [parts (list (channels y c c) (channels y 0 c))])   ; newest first
                  (if (= k n)
                      (reverse parts)
                      (let* ([mp (index (join p ".m") k)]
                             [block (if (weight? (join mp ".cv3.conv.weight")) c3k bottleneck)])
                        (loop (+ k 1) (cons (block (car parts) mp) parts)))))])
    (conv (concat-channels parts) (join p ".cv2") 1)))

;; SPPF: cv1, three 5x5 max pools (stride 1, "same"), cv2 over all four.
;; The released YOLO11 checkpoints have SiLU after cv1; current Ultralytics
;; source builds SPPF's cv1 without it (activations are pickled with the model).
(define sppf-cv1-activation #t)
(define (sppf x p)
  (let* ([y0 (conv x (join p ".cv1") 1 sppf-cv1-activation)]
         [pool (lambda (t) (ggml-pool-2d t GGML_OP_POOL_MAX 5 5 1 1 2.0 2.0))]
         [y1 (pool y0)] [y2 (pool y1)] [y3 (pool y2)])
    (conv (concat-channels (list y0 y1 y2 y3)) (join p ".cv2") 1)))

;; Ultralytics Attention: heads of 64 channels, keys/queries half as wide,
;; plus a depthwise positional conv of v. x: [W, H, C, B] with B = 1.
(define (attention x p)
  (let* ([w (dim x 0)] [h (dim x 1)] [c (dim x 2)] [n (* w h)]
         [heads (max (quotient c 64) 1)] [hd (quotient c heads)] [kd (quotient hd 2)]
         [qkv (ggml-reshape-3d (conv x (join p ".qkv") 1 #f) n (+ kd kd hd) heads)]   ; [N, 2kd + hd, heads]
         [part (lambda (from rows)
                 (ggml-view-3d qkv n rows heads (list-ref (strides qkv) 1) (list-ref (strides qkv) 2)
                               (* from (list-ref (strides qkv) 1))))]
         [q (ggml-cont (ggml-transpose (part 0 kd)))]                                   ; [kd, N, heads]
         [k (ggml-cont (ggml-transpose (part kd kd)))]
         [v (ggml-cont (part (* 2 kd) hd))]                                             ; [N, hd, heads]
         [attn (ggml-soft-max-ext (ggml-mul-mat k q) #f (/ 1.0 (sqrt (inexact kd))) 0.0)] ; [N_k, N_q, heads]
         [out (ggml-mul-mat attn v)]                                                    ; [N_q, hd, heads]
         [v-image (ggml-reshape-4d v w h c 1)])
    (conv (ggml-add (ggml-reshape-4d out w h c 1) (conv2d-dw v-image (join p ".pe.conv") 1))
          (join p ".proj") 1 #f)))

(define (psa-block x p)
  (let ([x (ggml-add x (attention x (join p ".attn")))])
    (ggml-add x (conv (conv x (join p ".ffn.0") 1) (join p ".ffn.1") 1 #f))))

(define (c2psa x p)
  (let* ([y (conv x (join p ".cv1") 1)]
         [c (quotient (dim y 2) 2)]
         [n (count-while (lambda (k) (join (index (join p ".m") k) ".attn.qkv.conv.weight")))]
         [b (let loop ([k 0] [b (channels y c c)])
              (if (= k n) b (loop (+ k 1) (psa-block b (index (join p ".m") k)))))])
    (conv (concat-channels (list (channels y 0 c) b)) (join p ".cv2") 1)))

(define (upsample x) (ggml-upscale x 2 GGML_SCALE_MODE_NEAREST))

;; --- detection head

;; Distribution focal loss decoding: [A, 64] logits -> [A, 4] expected
;; distances (softmax over 16 bins per side, then the mean bin).
(define (dfl raw)
  (let* ([a (dim raw 0)]
         [bins (ggml-reshape-3d (ggml-cont raw) a reg-max 4)]                          ; [A, 16, 4]
         [probs (ggml-soft-max (ggml-cont (ggml-permute bins 1 0 2 3)))]               ; [16, A, 4]
         [weights (ggml-reshape-2d (ggml-arange 0.0 (inexact reg-max) 1.0) reg-max 1)])
    (ggml-reshape-2d (ggml-mul-mat (ggml-reshape-2d probs reg-max (* a 4)) weights) a 4)))

;; One level: [A, 4 + classes] (cx, cy, w, h in pixels; probabilities), A = W * H.
(define (detect-level x i image-width)
  (let* ([w (dim x 0)] [h (dim x 1)] [a (* w h)]
         [stride (inexact (quotient image-width w))]
         [p (lambda (s) (join "23." s "." (number->string i)))]
         [box (conv2d (conv (conv x (join (p "cv2") ".0") 1) (join (p "cv2") ".1") 1) (join (p "cv2") ".2") 1)]
         [cls (conv2d (conv (dwconv (conv (dwconv x (join (p "cv3") ".0.0") 1) (join (p "cv3") ".0.1") 1)
                                    (join (p "cv3") ".1.0") 1)
                              (join (p "cv3") ".1.1") 1)
                      (join (p "cv3") ".2") 1)]
         [dist (dfl (ggml-reshape-2d box a (* 4 reg-max)))]                              ; [A, 4]: l t r b
         [side (lambda (j) (ggml-view-1d dist a (* j a 4)))]
         ;; Anchor centers: (x + 0.5, y + 0.5) over the grid, row-major.
         [ax (ggml-reshape-1d (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.5 (inexact w) 1.0) w 1) w h 1 1) a)]
         [ay (ggml-reshape-1d (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.5 (inexact h) 1.0) 1 h) w h 1 1) a)]
         [half (lambda (t) (ggml-scale t 0.5))]
         [cx (ggml-add ax (half (ggml-sub (side 2) (side 0))))]
         [cy (ggml-add ay (half (ggml-sub (side 3) (side 1))))]
         [bw (ggml-add (side 0) (side 2))]
         [bh (ggml-add (side 1) (side 3))]
         [col (lambda (t) (ggml-reshape-2d (ggml-scale t stride) a 1))]
         [boxes (concat-channels-1 (list (col cx) (col cy) (col bw) (col bh)))])
    (ggml-concat boxes (ggml-sigmoid (ggml-reshape-2d cls a (dim cls 2))) 1)))

(define (concat-channels-1 xs) (fold-left (lambda (acc x) (ggml-concat acc x 1)) (car xs) (cdr xs)))

;; --- model

(preprocess ([image image])
  (model-inputs [image (image->array (image-letterbox image input-size input-size 'fill '(114 114 114)))]))

(model (inputs [image f32 (_ _ 3 1)])
  (define checked
    (unless (and (= 0 (remainder (dim image 0) 32)) (= 0 (remainder (dim image 1) 32)))
      (error 'yolo11 "image height and width must be multiples of 32" (list (dim image 1) (dim image 0)))))
  (define layers (make-vector 23 #f))
  (define (layer i x)
    (vector-set! layers i (tap (format "layer.~2,'0d" i) x 4))
    x)
  (define (L i) (vector-ref layers i))
  (define (named i) (number->string i))
  (let* ([x (layer 0 (conv image (named 0) 2))]
         [x (layer 1 (conv x (named 1) 2))]
         [x (layer 2 (c3k2 x (named 2)))]
         [x (layer 3 (conv x (named 3) 2))]
         [x (layer 4 (c3k2 x (named 4)))]
         [x (layer 5 (conv x (named 5) 2))]
         [x (layer 6 (c3k2 x (named 6)))]
         [x (layer 7 (conv x (named 7) 2))]
         [x (layer 8 (c3k2 x (named 8)))]
         [x (layer 9 (sppf x (named 9)))]
         [x (layer 10 (c2psa x (named 10)))]
         [x (layer 11 (upsample x))]
         [x (layer 12 (concat-channels (list x (L 6))))]
         [x (layer 13 (c3k2 x (named 13)))]
         [x (layer 14 (upsample x))]
         [x (layer 15 (concat-channels (list x (L 4))))]
         [x (layer 16 (c3k2 x (named 16)))]
         [x (layer 17 (conv x (named 17) 2))]
         [x (layer 18 (concat-channels (list x (L 13))))]
         [x (layer 19 (c3k2 x (named 19)))]
         [x (layer 20 (conv x (named 20) 2))]
         [x (layer 21 (concat-channels (list x (L 10))))]
         [x (layer 22 (c3k2 x (named 22)))])
    (let ([levels (map (lambda (i f) (detect-level (L f) i (dim image 0))) '(0 1 2) '(16 19 22))])
      (outputs [head (fold-left (lambda (acc l) (ggml-concat acc l 0)) (car levels) (cdr levels)) 3]))))

;; --- postprocess: NMS on the head output, boxes back on the original image

(define (format-detection name score)
  (format "~a ~a" name (/ (round (* 100 score)) 100.0)))

(postprocess (head image)
  (let* ([rows (array-transpose head)]                                   ; [anchors, 84]
         [size (image-size image)])
    (let-values ([(boxes scores classes indices)
                  (detect (array-slice rows 0 4 'axis 1) (array-slice rows 4 #f 'axis 1)
                          'format 'cxcywh 'score-threshold 0.25 'iou 0.7 'max 300)])
      (let ([labels (map (lambda (c s) (format-detection (vector-ref class-names (exact (round c))) s))
                         (array->list classes) (array->list scores))])
        (results [boxes (boxes-unletterbox boxes input-size input-size (car size) (cadr size))]
                 [scores scores]
                 [classes classes]
                 [labels (fold-left (lambda (acc l) (if (string=? acc "") l (string-append acc ", " l))) "" labels)])))))
