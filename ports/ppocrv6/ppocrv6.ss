;; PP-OCRv6 (medium): text detection (PP-OCRv6_medium_det: LCNetV4 backbone,
;; RepLKFPN neck, DB head) and recognition (PP-OCRv6_medium_rec: LCNetV4,
;; SVTR encoder, CTC head), ported from transformers' pp_lcnet_v4,
;; pp_ocrv6_medium_det and pp_ocrv6_small_rec. Weights: both models with
;; their batch norms folded into the convolutions, see reference.py.
;;
;; Entries (numpy order):
;;   det   image [1, 3, H, W] (BGR, ImageNet-normalized, H and W multiples of 32)
;;         -> prob [1, 1, H, W], the text probability map.
;;         Taps det.stage1 .. det.stage4 (backbone), det.neck.
;;         Raw input: image. Postprocess: boxes [n, 4, 2] (corners on the
;;         original image), scores [n].
;;   rec   image [1, 3, 48, W] (BGR, normalized to [-1, 1]; padded to 320 wide
;;         in the graph) -> probs [1, T, 18710] per time step (T = W/8),
;;         column 0 is CTC's blank. Taps rec.backbone, rec.pooled, rec.encoder.
;; Pipelines:
;;   ocr        image -> boxes [n, 4, 2] and scores [n] in reading order,
;;              text-scores (list), text (one line per box).
;;   recognize  image of one text line -> text, score.
;; Asset: characters (the recognizer's vocabulary, one per line, blank first).
;;
;; Shapes in comments are ggml order: [W, H, C, B] like torch's NCHW memory.

(import (tl tensor) (tl nn) (tl attn) (tl util))

(define dim tensor:dim)

;; --- LCNetV4 backbone

;; Folded Conv + BN (+ ReLU): every backbone conv is prefix.weight + prefix.bias.
(define (conv x p . opts) (apply nn:conv2d x p opts))
(define (conv-relu x p . opts) (ggml-relu (apply conv x p opts)))

;; Large stem: 3x3/2, a 2x2 branch next to a 2x2 max pool, 3x3/2, 1x1 (all ReLU).
(define (stem x p)
  (let* ([s (lambda (name) (format "~a.stem.~a" p name))]
         [pad (lambda (t) (ggml-pad t 1 1 0 0))]                     ; F.pad (0, 1, 0, 1)
         [x (pad (conv-relu x (s "stem1") 'stride 2))]
         [branch (conv-relu (pad (conv-relu x (s "stem2a") 'padding 0)) (s "stem2b") 'padding 0)]
         [pooled (nn:max-pool x 2 'stride 1)])
    (conv-relu (conv-relu (tensor:concat (list pooled branch) 2) (s "stem3") 'stride 2) (s "stem4"))))

;; Squeeze-and-excitation: x * hardsigmoid(fc2(relu(fc1(mean over pixels)))).
(define (squeeze-excite x p)
  (let* ([c (dim x 2)]
         [fc (lambda (v name)
               (let ([w (weight (format "~a.~a.weight" p name))])      ; [1, 1, in, out]
                 (ggml-add (ggml-mul-mat (ggml-reshape-2d w (dim w 2) (dim w 3)) v)
                           (weight (format "~a.~a.bias" p name)))))]
         [pooled (ggml-reshape-2d (ggml-mean (ggml-reshape-2d x (* (dim x 0) (dim x 1)) c)) c 1)]
         [gate (ggml-hardsigmoid (fc (ggml-relu (fc pooled "fc1")) "fc2"))])
    (ggml-mul x (ggml-reshape-4d gate 1 1 c 1))))

;; Depthwise separable block: depthwise conv (+ SE), then a 1x1 - GELU - 1x1
;; MLP over channels, residual when the shape is kept.
(define (block x p stride)
  (let* ([t (nn:conv2d-depthwise x (format "~a.token_conv" p) 'stride stride)]
         [t (if (weight? (format "~a.se.fc1.weight" p)) (squeeze-excite t (format "~a.se" p)) t)]
         [y (conv (nn:gelu (conv t (format "~a.channel_conv1" p))) (format "~a.channel_conv2" p))])
    (if (and (equal? stride 1) (= (dim t 2) (dim y 2))) (ggml-add t y) y)))

(define (block-count p stage)
  (let loop ([k 0]) (if (weight? (format "~a.~a.~a.token_conv.weight" p stage k)) (loop (+ k 1)) k)))

;; The four stages' outputs. strides: the stride of each stage's first block
;; (the others have stride 1), an integer or (height width).
(define (backbone image p strides)
  (let loop ([stage 0] [x (stem image p)] [outs '()])
    (if (= stage 4)
        (reverse outs)
        (let ([y (let blocks ([k 0] [x x])
                   (if (= k (block-count p stage))
                       x
                       (blocks (+ k 1) (block x (format "~a.~a.~a" p stage k) (if (= k 0) (list-ref strides stage) 1)))))])
          (loop (+ stage 1) y (cons y outs))))))

;; --- detection: RepLKFPN neck and DB head

;; Multi-scale convolutions k x k + k x 1 + 1 x k (k = 7, 5, 3), 1x1 back
;; up (ReLU), residual.
(define (intraclass x p)
  (let* ([n (lambda (name) (format "~a.~a" p name))]
         [mix (lambda (h size)
                (ggml-add (ggml-add (conv h (n (format "square.~a" size))) (conv h (n (format "vertical.~a" size))))
                          (conv h (n (format "horizontal.~a" size)))))]
         [h (fold-left mix (conv x (n "reduce")) '(long mid short))])
    (ggml-add x (conv-relu h (n "final")))))

(define (upsample x factor) (if (= factor 1) x (nn:upsample-nearest x factor)))

;; Top-down sums, 9x9 projections, bottom-up sums (3x3/2), 9x9 laterals,
;; intraclass blocks, all scales up to stride 4 and concatenated (coarsest
;; first): [W/4, H/4, 256].
(define (neck stages)
  (let* ([n (lambda (name i) (format "det.neck.~a.~a" name i))]
         [adjusted (map (lambda (x i) (conv x (n "adjust" i))) stages '(0 1 2 3))]
         [top-down (fold-right (lambda (x acc) (if (null? acc) (list x) (cons (ggml-add x (upsample (car acc) 2)) acc)))
                               '() adjusted)]
         [projected (map (lambda (x i) (conv x (n "project" i))) top-down '(0 1 2 3))]
         [bottom-up (let loop ([i 1] [acc (list (car projected))])
                      (if (= i 4)
                          (reverse acc)
                          (loop (+ i 1) (cons (ggml-add (list-ref projected i) (conv (car acc) (n "down" (- i 1)) 'stride 2)) acc))))]
         [refined (map (lambda (x i) (intraclass (conv x (n "lateral" i)) (n "intra" i))) bottom-up '(0 1 2 3))])
    (tensor:concat (reverse (map upsample refined '(1 2 4 8))) 2)))

;; 3x3 conv (ReLU), two 2x2/2 transposed convs (ReLU, sigmoid): [W, H, 1, 1].
(define (db-head x)
  (let* ([x (conv-relu x "det.head.down")]
         [x (ggml-relu (nn:conv-transpose2d x "det.head.up" 'stride 2))])
    (ggml-sigmoid (nn:conv-transpose2d x "det.head.final" 'stride 2))))

(model det (inputs [image f32 (_ _ 3 1)])
  (define checked
    (unless (and (= 0 (remainder (dim image 0) 32)) (= 0 (remainder (dim image 1) 32)))
      (error 'ppocrv6 "the detector's input height and width must be multiples of 32" (list (dim image 1) (dim image 0)))))
  (define stages
    (map (lambda (x i) (tap (format "det.stage~a" i) x 4))
         (backbone image "det.backbone" '(1 2 2 2)) '(1 2 3 4)))
  (define features (tap "det.neck" (neck stages) 4))
  (outputs [prob (db-head features) 4]))

;; --- recognition: SVTR encoder and CTC head

(define rec-height 48)
(define rec-min-width 320)
(define rec-heads 8)

;; Pre-LN transformer block (eps 1e-6), SiLU MLP, on tokens [C, T].
(define (svtr-block x p)
  (let* ([n (lambda (name) (format "~a.~a" p name))]
         [x (ggml-add x (attn:multi-head (nn:layer-norm x (n "norm1") 'eps 1e-6) (n "attn") rec-heads 'names 'timm))])
    (ggml-add x (nn:linear (ggml-silu (nn:linear (nn:layer-norm x (n "norm2") 'eps 1e-6) (n "mlp.fc1"))) (n "mlp.fc2")))))

;; x [T, 1, C, 1] -> tokens [192, T]: skip = conv1x1(x); h = conv1x1(x);
;; h + dwconv1x7(h) (all SiLU); transformer blocks; LayerNorm; + skip.
(define (svtr x)
  (let* ([silu-conv (lambda (t name . opts) (ggml-silu (apply conv t (string-append "rec.svtr." name) opts)))]
         [tokens (lambda (t) (ggml-cont (ggml-transpose (ggml-reshape-2d t (dim t 0) (dim t 2)))))]   ; [C, T]
         [skip (silu-conv x "skip")]
         [h (silu-conv x "reduce")]
         [h (ggml-add h (ggml-silu (nn:conv2d-depthwise h "rec.svtr.local")))]
         [h (let loop ([i 0] [t (tokens h)])
              (if (weight? (format "rec.svtr.blocks.~a.norm1.weight" i))
                  (loop (+ i 1) (svtr-block t (format "rec.svtr.blocks.~a" i)))
                  t))])
    (ggml-add (nn:layer-norm h "rec.svtr.norm" 'eps 1e-6) (tokens skip))))

(model rec (inputs [image f32 (_ 48 3 1)])
  (define padded
    (let ([w (dim image 0)])
      (if (< w rec-min-width) (ggml-pad image (- rec-min-width w) 0 0 0) image)))
  (define features (tap "rec.backbone" (list-ref (backbone padded "rec.backbone" '(1 1 (2 1) (2 1))) 3) 4))
  ;; F.avg_pool2d(x, (3, 2)): [W/4, 3, 768] -> [W/8, 1, 768].
  (define pooled (tap "rec.pooled" (ggml-pool-2d features GGML_OP_POOL_AVG 2 3 2 3 0.0 0.0) 4))
  (define encoded (tap "rec.encoder" (svtr pooled) 3))
  (outputs [probs (ggml-soft-max (nn:linear encoded "rec.ctc")) 3]))

;; --- host side: resizing, DB boxes, crops, CTC (transformers' processors +
;; PaddleOCR's pipeline)

;; From the models' preprocessor_config.json / inference.yml.
(define det-limit-side 736)      ; limit_type "min": the shorter side is at least this
(define det-max-side 4000)
(define db-options '(threshold 0.2 box-threshold 0.45 max-candidates 3000 unclip-ratio 1.4 min-size 3))
(define rec-max-width 3200)
(define characters (vocabulary (asset "characters")))

;; PPOCRV5ServerDetImageProcessor.get_image_size (float math like Python's).
(define (det-size width height)
  (let* ([short (min width height)]
         [ratio (if (< short det-limit-side) (/ (inexact det-limit-side) short) 1.0)]
         [w (exact (truncate (* width ratio)))]
         [h (exact (truncate (* height ratio)))]
         [longest (max w h)]
         [scale (if (> longest det-max-side) (/ (inexact det-max-side) longest) 1.0)]
         [w (if (> longest det-max-side) (exact (truncate (* w scale))) w)]
         [h (if (> longest det-max-side) (exact (truncate (* h scale))) h)]
         [multiple (lambda (v) (max 32 (* 32 (exact (round (/ v 32.0))))))])
    (list (multiple w) (multiple h))))

;; Resized (bilinear), BGR, normalized with the config's (BGR-ordered) mean
;; and std applied before the channels are swapped, as transformers does.
(define (det-input image)
  (let ([size (apply det-size (image-size image))])
    (image->array (image-resize image (car size) (cadr size) 'bilinear)
                  'channels 'bgr 'mean '(0.406 0.456 0.485) 'std '(0.225 0.224 0.229))))

(define (text-boxes-of prob image)
  (let ([size (image-size image)])
    (apply text-boxes prob (car size) (cadr size) db-options)))

;; PPOCRV6SmallRecImageProcessor: height 48, width from the aspect ratio
;; (at most 3200), bilinear without antialiasing, BGR, [-1, 1]. Narrower
;; lines are padded to 320 in the rec graph.
(define (rec-width width height)
  (let* ([aspect (/ (inexact width) height)]
         [target (exact (truncate (* rec-height (max aspect (/ (inexact rec-min-width) rec-height)))))])
    (if (> target rec-max-width)
        rec-max-width
        (min target (exact (ceiling (* rec-height aspect)))))))

(define (rec-input line)
  (let ([size (image-size line)])
    (image->array (image-resize line (rec-width (car size) (cadr size)) rec-height 'bilinear-no-antialias)
                  'channels 'bgr 'mean '(0.5 0.5 0.5) 'std '(0.5 0.5 0.5))))

;; Text and score of one line image.
(define (recognize-line line)
  (let-values ([(ids score) (ctc-greedy (output (run rec [image (rec-input line)]) 'probs))])
    (values (vocabulary-text characters ids) score)))

(preprocess ([image image])
  (model-inputs [image (det-input image)]))

(postprocess (prob image)
  (let-values ([(boxes scores) (text-boxes-of prob image)])
    (results [boxes boxes] [scores scores])))

;; Detection, reading order, a crop per box, recognition (PaddleOCR's OCR
;; pipeline without document orientation, unwarping and text-line
;; orientation).
(pipeline ocr ([image image])
  (let*-values ([(boxes scores) (text-boxes-of (output (run det [image (det-input image)]) 'prob) image)]
                [(order) (text-boxes-order boxes)]
                [(lines) (map (lambda (i)
                                (let-values ([(text score) (recognize-line (image-crop-text image boxes i))])
                                  (cons text score)))
                              (util:->integers (array->list order)))])
    (results [boxes (array-take boxes order)]
             [scores (array-take scores order)]
             [text-scores (map cdr lines)]
             [text (util:string-join (map car lines) "\n")])))

(pipeline recognize ([image image])
  (let-values ([(text score) (recognize-line image)])
    (results [text text] [score score])))
