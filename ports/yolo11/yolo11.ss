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

(import (tl tensor) (tl nn) (tl attn) (tl vision) (tl util))

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

(define dim tensor:dim)

;; Ultralytics Conv: conv (with the folded batch norm) + SiLU. Convolutions
;; run as im2col + matmul on GPUs, ggml's direct kernel on the CPU.
(define (conv x prefix stride . act?)
  (let ([y (nn:conv2d x (format "~a.conv" prefix) 'stride stride)])
    (if (or (null? act?) (car act?)) (ggml-silu y) y)))

(define (dwconv x prefix stride)
  (ggml-silu (nn:conv2d-depthwise x (format "~a.conv" prefix) 'stride stride)))

(define (channels x from n) (tensor:slice x 2 from n))
(define (concat-channels xs) (tensor:concat xs 2))

;; Number of blocks prefix.m.0, prefix.m.1, ... (the model size's depth).
(define (block-count prefix)
  (let loop ([k 0]) (if (weight? (format "~a.m.~a.cv1.conv.weight" prefix k)) (loop (+ k 1)) k)))

(define (block prefix k) (format "~a.m.~a" prefix k))

;; --- blocks

(define (bottleneck x p)
  (ggml-add x (conv (conv x (format "~a.cv1" p) 1) (format "~a.cv2" p) 1)))

;; C3k: cv3(cat(m(cv1(x)), cv2(x))), m = Bottlenecks.
(define (c3k x p)
  (let ([m (let loop ([k 0] [y (conv x (format "~a.cv1" p) 1)])
             (if (= k (block-count p)) y (loop (+ k 1) (bottleneck y (block p k)))))])
    (conv (concat-channels (list m (conv x (format "~a.cv2" p) 1))) (format "~a.cv3" p) 1)))

;; C3k2 (a C2f): split cv1(x) in two, chain the inner blocks on the second
;; half, concatenate every intermediate, cv2. Inner blocks are C3k (they have
;; a cv3) or Bottlenecks.
(define (c3k2 x p)
  (let* ([y (conv x (format "~a.cv1" p) 1)]
         [c (quotient (dim y 2) 2)]
         [parts (let loop ([k 0] [parts (list (channels y c c) (channels y 0 c))])   ; newest first
                  (if (= k (block-count p))
                      (reverse parts)
                      (let* ([b (block p k)]
                             [inner (if (weight? (format "~a.cv3.conv.weight" b)) c3k bottleneck)])
                        (loop (+ k 1) (cons (inner (car parts) b) parts)))))])
    (conv (concat-channels parts) (format "~a.cv2" p) 1)))

;; SPPF: cv1, three 5x5 max pools (stride 1, "same"), cv2 over all four.
;; The released YOLO11 checkpoints have SiLU after cv1; current Ultralytics
;; source builds SPPF's cv1 without it (activations are pickled with the model).
(define sppf-cv1-activation #t)
(define (sppf x p)
  (let* ([y0 (conv x (format "~a.cv1" p) 1 sppf-cv1-activation)]
         [pool (lambda (t) (nn:max-pool t 5 'stride 1 'padding 2))]
         [y1 (pool y0)] [y2 (pool y1)] [y3 (pool y2)])
    (conv (concat-channels (list y0 y1 y2 y3)) (format "~a.cv2" p) 1)))

;; Ultralytics Attention: heads of 64 channels, keys/queries half as wide,
;; plus a depthwise positional conv of v. x: [W, H, C, 1].
(define (attention x p)
  (let* ([w (dim x 0)] [h (dim x 1)] [c (dim x 2)] [n (* w h)]
         [heads (max (quotient c 64) 1)] [hd (quotient c heads)] [kd (quotient hd 2)]
         ;; Per head [q (kd); k (kd); v (hd)] channels, tokens innermost: [N, 2kd + hd, heads].
         [qkv (ggml-reshape-3d (conv x (format "~a.qkv" p) 1 #f) n (+ kd kd hd) heads)]
         [part (lambda (from rows) (tensor:slice qkv 1 from rows))]
         [per-head (lambda (t) (ggml-reshape-4d (ggml-cont (ggml-transpose t)) (dim t 1) n heads 1))]   ; [rows, N, heads, 1]
         [v (part (* 2 kd) hd)]
         [out (attn:sdpa (per-head (part 0 kd)) (per-head (part kd kd)) (per-head v))]                ; [C, N]
         [image (lambda (t) (ggml-reshape-4d t w h c 1))])
    (conv (ggml-add (image (ggml-cont (ggml-transpose (ggml-reshape-2d out c n))))
                    (nn:conv2d-depthwise (image (ggml-cont v)) (format "~a.pe.conv" p)))
          (format "~a.proj" p) 1 #f)))

(define (psa-block x p)
  (let ([x (ggml-add x (attention x (format "~a.attn" p)))])
    (ggml-add x (conv (conv x (format "~a.ffn.0" p) 1) (format "~a.ffn.1" p) 1 #f))))

(define (c2psa x p)
  (let* ([y (conv x (format "~a.cv1" p) 1)]
         [c (quotient (dim y 2) 2)]
         [n (let loop ([k 0]) (if (weight? (format "~a.m.~a.attn.qkv.conv.weight" p k)) (loop (+ k 1)) k))]
         [b (let loop ([k 0] [b (channels y c c)])
              (if (= k n) b (loop (+ k 1) (psa-block b (block p k)))))])
    (conv (concat-channels (list (channels y 0 c) b)) (format "~a.cv2" p) 1)))

;; --- detection head

;; One level: [A, 4 + classes] (cx, cy, w, h in pixels; probabilities), A = W * H.
(define (detect-level x i image-width)
  (let* ([w (dim x 0)] [h (dim x 1)] [a (* w h)]
         [p (lambda (branch k) (format "23.~a.~a.~a" branch i k))]
         [box (nn:conv2d (conv (conv x (p "cv2" 0) 1) (p "cv2" 1) 1) (p "cv2" 2))]
         [cls (nn:conv2d (conv (dwconv (conv (dwconv x (p "cv3" "0.0") 1) (p "cv3" "0.1") 1) (p "cv3" "1.0") 1)
                               (p "cv3" "1.1") 1)
                         (p "cv3" 2))]
         [boxes (vision:decode-ltrb (vision:dfl (ggml-reshape-2d box a (* 4 reg-max)) 'bins reg-max)
                                    w h (quotient image-width w))])
    (ggml-concat boxes (ggml-sigmoid (ggml-reshape-2d cls a (dim cls 2))) 1)))

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
         [x (layer 11 (nn:upsample-nearest x 2))]
         [x (layer 12 (concat-channels (list x (L 6))))]
         [x (layer 13 (c3k2 x (named 13)))]
         [x (layer 14 (nn:upsample-nearest x 2))]
         [x (layer 15 (concat-channels (list x (L 4))))]
         [x (layer 16 (c3k2 x (named 16)))]
         [x (layer 17 (conv x (named 17) 2))]
         [x (layer 18 (concat-channels (list x (L 13))))]
         [x (layer 19 (c3k2 x (named 19)))]
         [x (layer 20 (conv x (named 20) 2))]
         [x (layer 21 (concat-channels (list x (L 10))))]
         [x (layer 22 (c3k2 x (named 22)))])
    (let ([levels (map (lambda (i f) (detect-level (L f) i (dim image 0))) '(0 1 2) '(16 19 22))])
      (outputs [head (tensor:concat levels 0) 3]))))

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
                 [labels (util:string-join labels ", ")])))))
