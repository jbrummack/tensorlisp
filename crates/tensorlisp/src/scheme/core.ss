;; Core of the (tensorlisp runtime) library. Spliced into the library body
;; after the generated raw bindings and constants, before the generated ops,
;; which use %current-ctx, %unwrap and %wrap.
;;
;; Shapes inside Scheme are in ggml order: (ne0 ne1 ...), innermost first.

(define tl_tensor_ne (foreign-procedure "tl_tensor_ne" (uptr int) integer-64))
(define tl_tensor_type (foreign-procedure "tl_tensor_type" (uptr) int))
(define tl_tensor_nb (foreign-procedure "tl_tensor_nb" (uptr int) size_t))

;; Tensors remember the ggml context they belong to, so a tensor kept from an
;; earlier graph build (whose context is freed) can't be used again.
(define-record-type tensor
  (fields ptr ctx)
  (nongenerative tensorlisp-tensor)
  (sealed #t))

;; Graph context, weight context and state context of the graph currently
;; being built, and the graph itself (for effects).
(define %ctx (make-parameter 0))
(define %weights (make-parameter 0))
(define %states (make-parameter 0))
;; Kind of the device the graph is built for: cpu or gpu.
(define %device (make-parameter #f))

(define (device)
  (or (%device) (error 'device "the device is only known inside a model while its graph is built")))
(define %graph (make-parameter 0))

;; The op being called and its (param . tensor) arguments, for reporting ggml
;; assertions. Set while the op's arguments are evaluated, cleared by %wrap
;; once ggml returned.
(define %op #f)
(define %pending '())

(define (%current-ctx who)
  (let ([ctx (%ctx)])
    (when (eqv? ctx 0)
      (error who "graph ops can only be used inside a model while its graph is built"))
    (set! %op who)
    ctx))

(define (%tensor-ptr who t)
  (unless (tensor? t) (error who "expected a tensor" t))
  (let ([ctx (tensor-ctx t)])
    (unless (or (eqv? ctx (%ctx)) (eqv? ctx (%weights)) (eqv? ctx (%states)))
      (error who "tensor belongs to a graph that no longer exists" t))
    (tensor-ptr t)))

;; Tensor argument named param of a generated op.
(define (%unwrap who param t)
  (cond
    [(tensor? t)
     (let ([p (%tensor-ptr who t)])
       (set! %pending (cons (cons param t) %pending))
       p)]
    [(not t) 0] ; optional tensor arguments (e.g. a mask) take #f
    [else (error who "expected a tensor or #f" t)]))

(define (%wrap who ptr)
  (set! %pending '())
  (when (eqv? ptr 0) (error who "ggml returned NULL"))
  (make-tensor ptr (%current-ctx who)))

;; ggml's abort callback on the Scheme thread (see ggml-sys csrc/guard.h).
;; Raising here unwinds through ggml's C frames back to the host call's guard.
(define %abort-callable
  (let ([code (foreign-callable
                (lambda (message)
                  (let ([who (or %op 'ggml)]
                        [args (map (lambda (a) (format "~%  ~a = ~s" (car a) (cdr a)))
                                   (reverse %pending))])
                    (set! %pending '())
                    (raise
                      (condition
                        (make-error)
                        (make-who-condition who)
                        (make-message-condition
                          (apply string-append "ggml assertion failed: " message args))))))
                (string)
                void)])
    (lock-object code)
    code))

(define ($tl-abort-handler)
  (foreign-callable-entry-point %abort-callable))

(define (shape t)
  (let ([p (%tensor-ptr 'shape t)])
    (let loop ([i (- (ggml_n_dims p) 1)] [dims '()])
      (if (< i 0) dims (loop (- i 1) (cons (tl_tensor_ne p i) dims))))))

;; Byte strides (nb0 nb1 ...) in ggml order, one per dimension of (shape t);
;; what ggml-view-* take as nb arguments.
(define (strides t)
  (let ([p (%tensor-ptr 'strides t)])
    (let loop ([i (- (ggml_n_dims p) 1)] [nbs '()])
      (if (< i 0) nbs (loop (- i 1) (cons (tl_tensor_nb p i) nbs))))))

(define (contiguous? t) (ggml_is_contiguous (%tensor-ptr 'contiguous? t)))

(define (dtype t)
  (string->symbol (ggml_type_name (tl_tensor_type (%tensor-ptr 'dtype t)))))

;; A definition, since a library body can't have expressions before definitions.
(define %tensor-writer
  (record-writer (record-type-descriptor tensor)
    (lambda (t port wr)
      (if (or (eqv? (tensor-ctx t) (%ctx)) (eqv? (tensor-ctx t) (%weights)) (eqv? (tensor-ctx t) (%states)))
          (fprintf port "#<tensor ~a ~s>" (dtype t) (shape t))
          (fprintf port "#<tensor (stale)>")))))

;; Weights

(define (%weights-ctx who)
  (let ([w (%weights)])
    (when (eqv? w 0)
      (error who "weights can only be used inside a model while its graph is built"))
    w))

(define (weight? name)
  (unless (string? name) (error 'weight? "expected a string" name))
  (not (eqv? 0 (ggml_get_tensor (%weights-ctx 'weight?) name))))

(define (weight name)
  (unless (string? name) (error 'weight "expected a string" name))
  (let* ([w (%weights-ctx 'weight)]
         [p (ggml_get_tensor w name)])
    (when (eqv? p 0) (error 'weight "no tensor with this name in the model file" name))
    (make-tensor p w)))

;; State: tensors that keep their contents between runs (e.g. a KV cache),
;; declared at the top level and allocated next to the weights, zeroed at load.
;;
;; (define-state "cache.k.0" f32 256 64)    ; name, f32 or f16, ggml dims (expressions)
;;
;; Inside a model, (state name) is the tensor; write it with an op that
;; returns a view of it (ggml-set-rows, ggml-cpy into a view) wrapped in
;; (effect t), which adds the write to the graph right away, so ops built
;; afterwards read the new contents.

(define-syntax define-state
  (syntax-rules ()
    [(_ name type dim ...) (%define-state! name 'type (list dim ...))]))

(define (%define-state! name type dims)
  (let ([slot (%loading)])
    (unless slot (error 'define-state "state can only be declared while a program is loaded"))
    (unless (string? name) (error 'define-state "the name must be a string" name))
    (unless (memq type '(f32 f16)) (error 'define-state "type must be f32 or f16" type))
    (unless (and (list? dims) (<= 1 (length dims) 4) (for-all (lambda (d) (and (fixnum? d) (> d 0))) dims))
      (error 'define-state "dims must be a list of 1-4 positive integers" dims))
    (when (assoc name (vector-ref slot 4)) (error 'define-state "state declared twice" name))
    (vector-set! slot 4 (cons (list name type dims) (vector-ref slot 4)))))

(define (state name)
  (unless (string? name) (error 'state "expected a string" name))
  (let ([ctx (%states)])
    (when (eqv? ctx 0) (error 'state "state can only be used inside a model while its graph is built"))
    (let ([p (ggml_get_tensor ctx name)])
      (when (eqv? p 0) (error 'state "no state with this name; declare it with define-state" name))
      (make-tensor p ctx))))

;; Adds t (e.g. a write into state) to the graph now; returns t.
(define (effect t)
  (let ([g (%graph)])
    (when (eqv? g 0) (error 'effect "effects can only be used inside a model while its graph is built"))
    (ggml_build_forward_expand g (%tensor-ptr 'effect t))
    t))

;; Taps: named intermediate tensors that can be read back on request, e.g. to
;; compare them with a reference implementation. (tap name t [rank]) returns t.

;; While a graph is built: box of ((name tensor rank) ...), newest first.
(define %taps (make-parameter #f))

(define tap
  (case-lambda
    [(name t) (tap name t #f)]
    [(name t rank)
     (unless (string? name) (error 'tap "expected a string name" name))
     (let ([p (%tensor-ptr 'tap t)]
           [slot (%taps)])
       (unless slot (error 'tap "tap can only be used while a graph is built"))
       (when (assoc name (unbox slot)) (error 'tap "duplicate tap name" name))
       ;; Name graph tensors after their tap; weights keep their names.
       (when (eqv? (tensor-ctx t) (%ctx)) (ggml_set_name p name))
       (set-box! slot (cons (list name t rank) (unbox slot)))
       t)]))

;; --- Pre- and postprocessing: host values (autopro, see host.rs). Host
;; arrays are plain data on the Rust side, not graph tensors: preprocess
;; returns them as the model's inputs, postprocess receives the model's
;; outputs as host arrays.

(define tl_host_error (foreign-procedure "tl_host_error" () string))
(define tl_host_tokenizer (foreign-procedure "tl_host_tokenizer" (integer-64) integer-64))
(define tl_host_tokenize
  (foreign-procedure "tl_host_tokenize" (integer-64 string int int integer-64 integer-64 integer-64) integer-64))
(define tl_host_image_width (foreign-procedure "tl_host_image_width" (integer-64) integer-64))
(define tl_host_image_height (foreign-procedure "tl_host_image_height" (integer-64) integer-64))
(define tl_host_image_resize
  (foreign-procedure "tl_host_image_resize" (integer-64 int integer-64 integer-64 integer-64 int) integer-64))
(define tl_host_image_center_crop
  (foreign-procedure "tl_host_image_center_crop" (integer-64 integer-64 integer-64) integer-64))
(define tl_host_image_to_tensor
  (foreign-procedure "tl_host_image_to_tensor"
    (integer-64 double int double double double double double double int int) integer-64))
(define tl_host_audio_rate (foreign-procedure "tl_host_audio_rate" (integer-64) integer-64))
(define tl_host_audio_length (foreign-procedure "tl_host_audio_length" (integer-64) integer-64))
(define tl_host_audio_resample (foreign-procedure "tl_host_audio_resample" (integer-64 integer-64) integer-64))
(define tl_host_audio_pad (foreign-procedure "tl_host_audio_pad" (integer-64 integer-64) integer-64))
(define tl_host_audio_to_tensor (foreign-procedure "tl_host_audio_to_tensor" (integer-64 int integer-64) integer-64))
(define tl_host_log_mel
  (foreign-procedure "tl_host_log_mel"
    (integer-64 integer-64 integer-64 integer-64 integer-64 integer-64 double double double double) integer-64))
(define tl_host_whisper_features (foreign-procedure "tl_host_whisper_features" (integer-64 integer-64) integer-64))
(define tl_host_tensor_rank (foreign-procedure "tl_host_tensor_rank" (integer-64) integer-64))
(define tl_host_tensor_dim (foreign-procedure "tl_host_tensor_dim" (integer-64 integer-64) integer-64))
(define tl_host_tensor_affine (foreign-procedure "tl_host_tensor_affine" (integer-64 double double) integer-64))
(define tl_host_tensor_reshape
  (foreign-procedure "tl_host_tensor_reshape" (integer-64 integer-64 integer-64 integer-64 integer-64) integer-64))
(define tl_host_tensor_new (foreign-procedure "tl_host_tensor_new" (integer-64) integer-64))
(define tl_host_tensor_set (foreign-procedure "tl_host_tensor_set" (integer-64 integer-64 double) integer-64))
(define tl_host_tensor_len (foreign-procedure "tl_host_tensor_len" (integer-64) integer-64))
(define tl_host_tensor_get (foreign-procedure "tl_host_tensor_get" (integer-64 integer-64) double))
(define tl_host_tensor_slice
  (foreign-procedure "tl_host_tensor_slice" (integer-64 integer-64 integer-64 integer-64) integer-64))
(define tl_host_tensor_transpose (foreign-procedure "tl_host_tensor_transpose" (integer-64) integer-64))
(define tl_host_tensor_take (foreign-procedure "tl_host_tensor_take" (integer-64 integer-64) integer-64))
(define tl_host_tensor_argmax (foreign-procedure "tl_host_tensor_argmax" (integer-64) integer-64))
(define tl_host_boxes_convert (foreign-procedure "tl_host_boxes_convert" (integer-64 int int) integer-64))
(define tl_host_nms (foreign-procedure "tl_host_nms" (integer-64 integer-64 integer-64 double) integer-64))
(define tl_host_detect
  (foreign-procedure "tl_host_detect"
    (integer-64 integer-64 int double double int int integer-64 integer-64) integer-64))
(define tl_host_boxes_scale (foreign-procedure "tl_host_boxes_scale" (integer-64 double double) integer-64))
(define tl_host_boxes_clip (foreign-procedure "tl_host_boxes_clip" (integer-64 double double) integer-64))
(define tl_host_boxes_unletterbox
  (foreign-procedure "tl_host_boxes_unletterbox" (integer-64 integer-64 integer-64 integer-64 integer-64) integer-64))
(define tl_host_image_letterbox
  (foreign-procedure "tl_host_image_letterbox"
    (integer-64 integer-64 integer-64 integer-64 integer-64 integer-64 int) integer-64))
(define tl_host_dbscan (foreign-procedure "tl_host_dbscan" (integer-64 double integer-64 int) integer-64))
(define tl_host_run_begin (foreign-procedure "tl_host_run_begin" (integer-64 string) integer-64))
(define tl_host_run_input (foreign-procedure "tl_host_run_input" (integer-64 string integer-64) integer-64))
(define tl_host_run_exec (foreign-procedure "tl_host_run_exec" (integer-64) integer-64))
(define tl_host_run_output_name (foreign-procedure "tl_host_run_output_name" (integer-64 integer-64) string))
(define tl_host_run_output (foreign-procedure "tl_host_run_output" (integer-64 integer-64) integer-64))
(define tl_host_detokenize (foreign-procedure "tl_host_detokenize" (integer-64 integer-64 int) string))
(define tl_host_token_id (foreign-procedure "tl_host_token_id" (integer-64 string) integer-64))
(define tl_host_cluster_centroids (foreign-procedure "tl_host_cluster_centroids" (integer-64 integer-64) integer-64))
(define tl_host_text_boxes
  (foreign-procedure "tl_host_text_boxes"
    (integer-64 integer-64 integer-64 double double integer-64 double double) integer-64))
(define tl_host_text_boxes_order (foreign-procedure "tl_host_text_boxes_order" (integer-64) integer-64))
(define tl_host_image_crop_text (foreign-procedure "tl_host_image_crop_text" (integer-64 integer-64 integer-64) integer-64))
(define tl_host_ctc_greedy (foreign-procedure "tl_host_ctc_greedy" (integer-64) integer-64))
(define tl_host_vocabulary (foreign-procedure "tl_host_vocabulary" (integer-64) integer-64))
(define tl_host_vocabulary_size (foreign-procedure "tl_host_vocabulary_size" (integer-64) integer-64))
(define tl_host_vocabulary_text (foreign-procedure "tl_host_vocabulary_text" (integer-64 integer-64) string))

;; kind: bytes, tokenizer, image, audio or array.
(define-record-type host
  (fields id kind)
  (nongenerative tensorlisp-host)
  (sealed #t))

(define %host-writer
  (record-writer (record-type-descriptor host)
    (lambda (h port wr) (fprintf port "#<~a>" (host-kind h)))))

;; Wraps the id a host function returned; 0 means it failed.
(define (%host who kind id)
  (when (eqv? id 0) (error who (tl_host_error)))
  (make-host id kind))

(define (%host-id who kind h)
  (unless (and (host? h) (eq? (host-kind h) kind))
    (error who (format "expected ~a ~a" (if (memq kind '(array image audio)) "an" "a") kind) h))
  (host-id h))

(define (%int who name v)
  (unless (and (fixnum? v) (>= v 0)) (error who (format "~a must be a non-negative integer" name) v))
  v)

(define (%real who name v)
  (unless (real? v) (error who (format "~a must be a number" name) v))
  (inexact v))

;; Options as 'name value pairs, checked against defaults: ((name . value) ...).
(define (%options who opts defaults)
  (let loop ([opts opts] [acc defaults])
    (cond
      [(null? opts) acc]
      [(or (null? (cdr opts)) (not (symbol? (car opts))))
       (error who "options must be 'name value pairs" opts)]
      [(not (assq (car opts) defaults))
       (error who (format "unknown option ~s; options are ~s" (car opts) (map car defaults)))]
      [else (loop (cddr opts) (cons (cons (car opts) (cadr opts)) acc))])))

(define (%opt options name) (cdr (assq name options)))

;; Assets: files stored in the model (TL_ASSET.<name>), e.g. a tokenizer.
(define %assets (make-parameter '()))

(define (asset name)
  (let ([entry (assoc name (%assets))])
    (unless entry
      (error 'asset "no asset with this name in the model file" name (map car (%assets))))
    (make-host (cdr entry) 'bytes)))

;; Text

(define (tokenizer source)
  (%host 'tokenizer 'tokenizer (tl_host_tokenizer (%host-id 'tokenizer 'bytes source))))

;; Returns two arrays [length]: token ids and attention mask (1 = token).
(define (tokenize tok text . opts)
  (unless (string? text) (error 'tokenize "expected a string" text))
  (let* ([o (%options 'tokenize opts
              '((lowercase . #f) (special-tokens . #t) (max-length . #f) (pad-to . #f) (pad-id . #f)))]
         [int-or (lambda (name) (let ([v (%opt o name)]) (if v (%int 'tokenize name v) -1)))]
         [ids (%host 'tokenize 'array
                (tl_host_tokenize (%host-id 'tokenize 'tokenizer tok) text
                                  (if (%opt o 'lowercase) 1 0) (if (%opt o 'special-tokens) 1 0)
                                  (int-or 'max-length) (int-or 'pad-to) (int-or 'pad-id)))])
    (values ids (make-host (+ (host-id ids) 1) 'array))))

;; Text from token ids (an array or a list), e.g. generated tokens.
(define (detokenize tok ids . opts)
  (let* ([o (%options 'detokenize opts '((skip-special . #t)))]
         [text (tl_host_detokenize (%host-id 'detokenize 'tokenizer tok)
                                   (%host-id 'detokenize 'array (if (list? ids) (list->array ids) ids))
                                   (if (%opt o 'skip-special) 1 0))])
    (unless text (error 'detokenize (tl_host_error)))
    text))

;; Id of a token (e.g. a special token such as "<eos>"), or #f.
(define (token-id tok token)
  (unless (string? token) (error 'token-id "expected a string" token))
  (let ([id (tl_host_token_id (%host-id 'token-id 'tokenizer tok) token)])
    (and (>= id 0) id)))

;; Images (8-bit RGB; resizing reproduces Pillow, except bilinear-no-antialias:
;; torch's bilinear interpolation without antialiasing, close to OpenCV's
;; INTER_LINEAR, which skips pixels when downscaling)

(define (%filter who f)
  (case f
    [(nearest) 0] [(lanczos) 1] [(bilinear) 2] [(bicubic) 3] [(box) 4] [(hamming) 5] [(bilinear-no-antialias) 6]
    [else (error who "filter must be nearest, bilinear, bicubic, lanczos, box, hamming or bilinear-no-antialias" f)]))

(define (image-size img)
  (let ([id (%host-id 'image-size 'image img)])
    (list (tl_host_image_width id) (tl_host_image_height id))))

(define (%resize who img kind a b c filter)
  (%host who 'image (tl_host_image_resize (%host-id who 'image img) kind a b c (%filter who filter))))

(define (image-resize img width height filter)
  (%resize 'image-resize img 0 (%int 'image-resize 'width width) (%int 'image-resize 'height height) 0 filter))

;; Shorter edge to `edge`, keeping the aspect ratio (HF shortest_edge).
(define image-resize-shortest
  (case-lambda
    [(img edge filter) (image-resize-shortest img edge filter #f)]
    [(img edge filter max-longest)
     (%resize 'image-resize-shortest img 1 (%int 'image-resize-shortest 'edge edge)
              (if max-longest (%int 'image-resize-shortest 'max-longest max-longest) -1) 0 filter)]))

(define (image-resize-longest img edge filter)
  (%resize 'image-resize-longest img 2 (%int 'image-resize-longest 'edge edge) 0 0 filter))

;; Sides rounded to multiples of `multiple`, pixel count kept within bounds.
(define (image-resize-multiple img multiple min-pixels max-pixels filter)
  (%resize 'image-resize-multiple img 3 (%int 'image-resize-multiple 'multiple multiple)
           (%int 'image-resize-multiple 'min-pixels min-pixels)
           (%int 'image-resize-multiple 'max-pixels max-pixels) filter))

(define (image-center-crop img width height)
  (%host 'image-center-crop 'image
    (tl_host_image_center_crop (%host-id 'image-center-crop 'image img)
                               (%int 'image-center-crop 'width width) (%int 'image-center-crop 'height height))))

;; Fits the image into width x height keeping its aspect ratio, centered on
;; 'fill (YOLO input, Ultralytics' LetterBox geometry).
(define (image-letterbox img width height . opts)
  (let* ([o (%options 'image-letterbox opts '((fill . (114 114 114)) (filter . bilinear)))]
         [fill (%opt o 'fill)])
    (unless (and (list? fill) (= (length fill) 3) (for-all (lambda (c) (and (fixnum? c) (<= 0 c 255))) fill))
      (error 'image-letterbox "fill must be a list of 3 integers 0-255" fill))
    (%host 'image-letterbox 'image
      (tl_host_image_letterbox (%host-id 'image-letterbox 'image img)
                               (%int 'image-letterbox 'width width) (%int 'image-letterbox 'height height)
                               (car fill) (cadr fill) (caddr fill) (%filter 'image-letterbox (%opt o 'filter))))))

;; [3, H, W] (or [H, W, 3]) array: x * scale, then (x - mean) / std per channel.
(define (image->array img . opts)
  (let* ([o (%options 'image->array opts '((scale . 1/255) (mean . #f) (std . #f) (channels . rgb) (layout . chw)))]
         [triple (lambda (name)
                   (let ([v (%opt o name)])
                     (unless (and (list? v) (= (length v) 3) (for-all real? v))
                       (error 'image->array (format "~a must be a list of 3 numbers" name) v))
                     (map inexact v)))]
         [normalize? (or (%opt o 'mean) (%opt o 'std))]
         [mean (if normalize? (triple 'mean) '(0.0 0.0 0.0))]
         [std (if normalize? (triple 'std) '(1.0 1.0 1.0))]
         [scale (%opt o 'scale)])
    (unless (memq (%opt o 'channels) '(rgb bgr)) (error 'image->array "channels must be rgb or bgr" (%opt o 'channels)))
    (unless (memq (%opt o 'layout) '(chw hwc)) (error 'image->array "layout must be chw or hwc" (%opt o 'layout)))
    (%host 'image->array 'array
      (tl_host_image_to_tensor (%host-id 'image->array 'image img)
                               (if scale (%real 'image->array 'scale scale) -1.0)
                               (if normalize? 1 0)
                               (car mean) (cadr mean) (caddr mean) (car std) (cadr std) (caddr std)
                               (if (eq? (%opt o 'channels) 'bgr) 1 0)
                               (if (eq? (%opt o 'layout) 'hwc) 1 0)))))

;; Audio (mono)

(define (audio-rate a) (tl_host_audio_rate (%host-id 'audio-rate 'audio a)))
(define (audio-length a) (tl_host_audio_length (%host-id 'audio-length 'audio a)))

(define (audio-resample a rate)
  (%host 'audio-resample 'audio
    (tl_host_audio_resample (%host-id 'audio-resample 'audio a) (%int 'audio-resample 'rate rate))))

;; Zero-pads or truncates to `length` samples.
(define (audio-pad a length)
  (%host 'audio-pad 'audio (tl_host_audio_pad (%host-id 'audio-pad 'audio a) (%int 'audio-pad 'length length))))

;; Returns two arrays [length]: samples and attention mask (1 = sample).
(define (audio->array a . opts)
  (let* ([o (%options 'audio->array opts '((normalize . #f) (length . #f)))]
         [len (%opt o 'length)]
         [values-array (%host 'audio->array 'array
                         (tl_host_audio_to_tensor (%host-id 'audio->array 'audio a) (if (%opt o 'normalize) 1 0)
                                                  (if len (%int 'audio->array 'length len) -1)))])
    (values values-array (make-host (+ (host-id values-array) 1) 'array))))

;; Mel spectrogram [mels, frames], transformers' spectrogram() semantics.
(define (log-mel a . opts)
  (let* ([o (%options 'log-mel opts
              '((n-fft . 400) (hop . 160) (win . #f) (mels . 80) (f-min . 0) (f-max . #f) (power . 2)
                (center . #t) (scale . slaney) (norm . slaney) (log . log10) (floor . 1e-10)))]
         [scale (case (%opt o 'scale) [(htk) 0] [(slaney) 1] [(kaldi) 2]
                  [else (error 'log-mel "scale must be htk, slaney or kaldi" (%opt o 'scale))])]
         [norm (case (%opt o 'norm) [(slaney) 1] [(#f none) 0]
                 [else (error 'log-mel "norm must be slaney or #f" (%opt o 'norm))])]
         [log (case (%opt o 'log) [(#f none) 0] [(ln) 1] [(log10) 2]
                [else (error 'log-mel "log must be log10, ln or #f" (%opt o 'log))])]
         [flags (+ (if (%opt o 'center) 1 0) (* 2 scale) (* 8 norm) (* 16 log))])
    (%host 'log-mel 'array
      (tl_host_log_mel (%host-id 'log-mel 'audio a) (%int 'log-mel 'n-fft (%opt o 'n-fft)) (%int 'log-mel 'hop (%opt o 'hop))
                       (if (%opt o 'win) (%int 'log-mel 'win (%opt o 'win)) 0) (%int 'log-mel 'mels (%opt o 'mels)) flags
                       (%real 'log-mel 'f-min (%opt o 'f-min)) (if (%opt o 'f-max) (%real 'log-mel 'f-max (%opt o 'f-max)) -1.0)
                       (%real 'log-mel 'power (%opt o 'power)) (%real 'log-mel 'floor (%opt o 'floor))))))

;; WhisperFeatureExtractor: [mels, 3000] from 16 kHz audio (padded to 30 s).
(define (whisper-features a mels)
  (%host 'whisper-features 'array
    (tl_host_whisper_features (%host-id 'whisper-features 'audio a) (%int 'whisper-features 'mels mels))))

;; Arrays

(define (array-shape x)
  (let ([id (%host-id 'array-shape 'array x)])
    (map (lambda (i) (tl_host_tensor_dim id i)) (iota (tl_host_tensor_rank id)))))

;; x * scale + bias, elementwise.
(define (array-affine x scale bias)
  (%host 'array-affine 'array
    (tl_host_tensor_affine (%host-id 'array-affine 'array x) (%real 'array-affine 'scale scale) (%real 'array-affine 'bias bias))))

;; Same data, new shape (1-4 dims, numpy order).
(define (array-reshape x . dims)
  (unless (<= 1 (length dims) 4) (error 'array-reshape "1 to 4 dimensions" dims))
  (let ([d (append (map (lambda (v) (%int 'array-reshape 'dimension v)) dims) '(0 0 0))])
    (%host 'array-reshape 'array
      (tl_host_tensor_reshape (%host-id 'array-reshape 'array x) (car d) (cadr d) (caddr d) (cadddr d)))))

;; Array access and manipulation

(define (array-length x) (tl_host_tensor_len (%host-id 'array-length 'array x)))

;; All elements in row-major order.
(define (array->list x)
  (let ([id (%host-id 'array->list 'array x)])
    (map (lambda (i) (tl_host_tensor_get id i)) (iota (tl_host_tensor_len id)))))

;; A 1-d array from a list of numbers.
(define (list->array xs)
  (unless (and (list? xs) (for-all real? xs)) (error 'list->array "expected a list of numbers" xs))
  (let ([a (%host 'list->array 'array (tl_host_tensor_new (length xs)))])
    (let loop ([xs xs] [i 0])
      (unless (null? xs)
        (tl_host_tensor_set (host-id a) i (inexact (car xs)))
        (loop (cdr xs) (+ i 1))))
    a))

(define (%integer who name v)
  (unless (fixnum? v) (error who (format "~a must be an integer" name) v))
  v)

;; x[start:end] along 'axis (default 0); negative bounds count from the end,
;; end #f is the end.
(define (array-slice x start end . opts)
  (let ([o (%options 'array-slice opts '((axis . 0)))])
    (%host 'array-slice 'array
      (tl_host_tensor_slice (%host-id 'array-slice 'array x) (%int 'array-slice 'axis (%opt o 'axis))
                            (%integer 'array-slice 'start start)
                            (if end (%integer 'array-slice 'end end) (greatest-fixnum))))))

;; Axes reversed ([a, b] -> [b, a]).
(define (array-transpose x)
  (%host 'array-transpose 'array (tl_host_tensor_transpose (%host-id 'array-transpose 'array x))))

(define (%index-array who xs)
  (if (list? xs) (list->array xs) xs))

;; Entries along axis 0 at indices (an array or a list), e.g. kept detections.
(define (array-take x indices)
  (%host 'array-take 'array
    (tl_host_tensor_take (%host-id 'array-take 'array x) (%host-id 'array-take 'array (%index-array 'array-take indices)))))

;; Index of the largest value along the last axis.
(define (array-argmax x)
  (%host 'array-argmax 'array (tl_host_tensor_argmax (%host-id 'array-argmax 'array x))))

;; Detection (boxes are [n, 4] arrays; torchvision / Ultralytics semantics)

(define (%box-format who f)
  (case f [(xyxy) 0] [(xywh) 1] [(cxcywh) 2]
    [else (error who "box format must be xyxy, xywh or cxcywh" f)]))

(define (boxes-convert boxes from to)
  (%host 'boxes-convert 'array
    (tl_host_boxes_convert (%host-id 'boxes-convert 'array boxes)
                           (%box-format 'boxes-convert from) (%box-format 'boxes-convert to))))

;; Greedy NMS on xyxy boxes: kept indices by descending score. With 'classes
;; (array or list), boxes of different classes don't suppress each other.
(define (nms boxes scores . opts)
  (let ([o (%options 'nms opts '((iou . 0.5) (classes . #f)))])
    (%host 'nms 'array
      (tl_host_nms (%host-id 'nms 'array boxes) (%host-id 'nms 'array scores)
                   (let ([c (%opt o 'classes)]) (if c (%host-id 'nms 'array (%index-array 'nms c)) 0))
                   (%real 'nms 'iou (%opt o 'iou))))))

;; Detector head to detections, like Ultralytics' non_max_suppression: boxes
;; [n, 4] (format 'format), class-scores [n, classes] (probabilities).
;; Returns four arrays: boxes [k, 4] xyxy, scores [k], classes [k] and the
;; row of each detection in the input [k], best first.
(define (detect boxes class-scores . opts)
  (let* ([o (%options 'detect opts
              '((format . cxcywh) (score-threshold . 0.25) (iou . 0.45) (class-agnostic . #f)
                (multi-label . #f) (max-candidates . 30000) (max . 300)))]
         [first (%host 'detect 'array
                  (tl_host_detect (%host-id 'detect 'array boxes) (%host-id 'detect 'array class-scores)
                                  (%box-format 'detect (%opt o 'format))
                                  (%real 'detect 'score-threshold (%opt o 'score-threshold))
                                  (%real 'detect 'iou (%opt o 'iou))
                                  (if (%opt o 'class-agnostic) 1 0) (if (%opt o 'multi-label) 1 0)
                                  (%int 'detect 'max-candidates (%opt o 'max-candidates))
                                  (%int 'detect 'max (%opt o 'max))))]
         [id (host-id first)])
    (values first (make-host (+ id 1) 'array) (make-host (+ id 2) 'array) (make-host (+ id 3) 'array))))

;; x coordinates times sx, y times sy (e.g. normalized boxes to pixels).
(define (boxes-scale boxes sx sy)
  (%host 'boxes-scale 'array
    (tl_host_boxes_scale (%host-id 'boxes-scale 'array boxes) (%real 'boxes-scale 'sx sx) (%real 'boxes-scale 'sy sy))))

(define (boxes-clip boxes width height)
  (%host 'boxes-clip 'array
    (tl_host_boxes_clip (%host-id 'boxes-clip 'array boxes) (%real 'boxes-clip 'width width) (%real 'boxes-clip 'height height))))

;; xyxy boxes on a letterboxed (model-width x model-height) input back onto
;; the original image, clipped (the inverse of image-letterbox).
(define (boxes-unletterbox boxes model-width model-height width height)
  (let ([i (lambda (name v) (%int 'boxes-unletterbox name v))])
    (%host 'boxes-unletterbox 'array
      (tl_host_boxes_unletterbox (%host-id 'boxes-unletterbox 'array boxes)
                                 (i 'model-width model-width) (i 'model-height model-height)
                                 (i 'width width) (i 'height height)))))

;; Clustering

;; DBSCAN (scikit-learn semantics) on the rows of x [n, d]: a label per row,
;; -1 for noise. 'metric: euclidean or cosine.
(define (dbscan x eps min-samples . opts)
  (let ([o (%options 'dbscan opts '((metric . euclidean)))])
    (%host 'dbscan 'array
      (tl_host_dbscan (%host-id 'dbscan 'array x) (%real 'dbscan 'eps eps) (%int 'dbscan 'min-samples min-samples)
                      (case (%opt o 'metric) [(euclidean) 0] [(cosine) 1]
                        [else (error 'dbscan "metric must be euclidean or cosine" (%opt o 'metric))])))))

;; Mean row per cluster: [clusters, d] (noise ignored).
(define (cluster-centroids x labels)
  (%host 'cluster-centroids 'array
    (tl_host_cluster_centroids (%host-id 'cluster-centroids 'array x) (%host-id 'cluster-centroids 'array labels))))

;; OCR (PaddleOCR semantics)

;; Text boxes from a DB probability map prob [H, W] (or [1, 1, H, W]) of an
;; image resized from width x height: two arrays, boxes [n, 4, 2] (corners
;; top-left, top-right, bottom-right, bottom-left as x, y on the original
;; image) and scores [n]. Options as transformers' / PaddleOCR's DB
;; postprocessing: 'threshold (pixels that are text), 'box-threshold (mean
;; probability of a box), 'max-candidates, 'unclip-ratio (how far boxes
;; grow), 'min-size (shorter side, map pixels).
(define (text-boxes prob width height . opts)
  (let* ([o (%options 'text-boxes opts
              '((threshold . 0.3) (box-threshold . 0.6) (max-candidates . 1000) (unclip-ratio . 1.5) (min-size . 3)))]
         [r (lambda (name) (%real 'text-boxes name (%opt o name)))]
         [first (%host 'text-boxes 'array
                  (tl_host_text_boxes (%host-id 'text-boxes 'array prob)
                                      (%int 'text-boxes 'width width) (%int 'text-boxes 'height height)
                                      (r 'threshold) (r 'box-threshold)
                                      (%int 'text-boxes 'max-candidates (%opt o 'max-candidates))
                                      (r 'unclip-ratio) (r 'min-size)))])
    (values first (make-host (+ (host-id first) 1) 'array))))

;; Reading order of text boxes [n, 4, 2]: indices [n], top to bottom, left
;; to right within a line (PaddleOCR's sort_quad_boxes).
(define (text-boxes-order boxes)
  (%host 'text-boxes-order 'array (tl_host_text_boxes_order (%host-id 'text-boxes-order 'array boxes))))

;; The text line in box i of boxes [n, 4, 2], straightened (PaddleOCR's
;; min-area-rectangle crop: perspective warp, bicubic; rotated 90 degrees
;; counterclockwise when 1.5 times taller than wide).
(define (image-crop-text img boxes i)
  (%host 'image-crop-text 'image
    (tl_host_image_crop_text (%host-id 'image-crop-text 'image img) (%host-id 'image-crop-text 'array boxes)
                             (%integer 'image-crop-text 'i i))))

;; Greedy CTC decoding of per-step probabilities [T, classes] (or [1, T,
;; classes]; class 0 is the blank): two values, the class ids (an array) and
;; their mean probability (0 without any).
(define (ctc-greedy probs)
  (let ([ids (%host 'ctc-greedy 'array (tl_host_ctc_greedy (%host-id 'ctc-greedy 'array probs)))])
    (values ids (tl_host_tensor_get (+ (host-id ids) 1) 0))))

;; Strings by index from UTF-8 text with one entry per line (bytes, e.g. an
;; asset), such as a recognizer's characters.
(define (vocabulary source)
  (%host 'vocabulary 'vocabulary (tl_host_vocabulary (%host-id 'vocabulary 'bytes source))))

(define (vocabulary-size v)
  (tl_host_vocabulary_size (%host-id 'vocabulary-size 'vocabulary v)))

;; The entries at ids (an array or a list), concatenated.
(define (vocabulary-text v ids)
  (let ([text (tl_host_vocabulary_text (%host-id 'vocabulary-text 'vocabulary v)
                                       (%host-id 'vocabulary-text 'array (if (list? ids) (list->array ids) ids)))])
    (unless text (error 'vocabulary-text (tl_host_error)))
    text))

;; (preprocess ([text string] [photo image]) body ... (model-inputs [ids array] ...))
;; Raw input kinds: string, image, audio, array. Runs per example on the host;
;; each result array is one example of the model input (the batch dimension
;; is added by stacking examples).
(define-syntax model-inputs
  (syntax-rules ()
    [(_ [name expr] ...) (list (cons 'name expr) ...)]))

(define-syntax preprocess
  (syntax-rules ()
    [(_ ([name kind] ...) body ...)
     (%register-preprocess! '((name kind) ...) (lambda (name ...) body ...))]))

(define (%register-preprocess! spec proc)
  (let ([slot (%loading)])
    (unless slot (error 'preprocess "preprocess can only be used while a program is loaded"))
    (when (vector-ref slot 1) (error 'preprocess "a program can define only one preprocess"))
    (%check-raw-spec 'preprocess spec)
    (vector-set! slot 1 (cons spec proc))))

;; (postprocess (name ...) body ... (results [name value] ...))
;; Each name is a model output (one example of it: the batch dimension is
;; split off) or a raw input of preprocess. Values are host arrays, numbers
;; or lists of numbers. Runs per example on the host.
(define-syntax results
  (syntax-rules ()
    [(_ [name expr] ...) (list (cons 'name expr) ...)]))

(define-syntax postprocess
  (syntax-rules ()
    [(_ (name ...) body ...)
     (%register-postprocess! '(name ...) (lambda (name ...) body ...))]))

(define (%register-postprocess! names proc)
  (let ([slot (%loading)])
    (unless slot (error 'postprocess "postprocess can only be used while a program is loaded"))
    (when (vector-ref slot 2) (error 'postprocess "a program can define only one postprocess"))
    (unless (for-all symbol? names) (error 'postprocess "arguments must be names of outputs or raw inputs" names))
    (vector-set! slot 2 (cons names proc))))

;; Model definition
;;
;; (model (inputs [x f32 (784 1)] [y f32])
;;   (define h ...)
;;   (outputs [logits (ggml-add ...)] [probs (ggml-soft-max ...) 2]))
;;
;; Input shapes are optional; a dimension may be #f or a symbol to accept any
;; size. The optional integer after an output is its rank (number of
;; dimensions); by default trailing dimensions of size 1 are dropped.
;;
;; A program may define several named entries, (model encode (inputs ...) ...),
;; e.g. an encoder run once and a decoder run per step; an unnamed model is
;; the entry `main`. The default entry (for preprocess, postprocess and runs
;; that name none) is `main`, or else the first one defined.

(define-syntax inputs (lambda (x) (syntax-violation 'inputs "misplaced auxiliary keyword" x)))

(define-syntax outputs
  (syntax-rules ()
    [(_ [name expr opt ...] ...) (list (list 'name expr opt ...) ...)]))

(define-syntax model
  (syntax-rules (inputs)
    [(_ (inputs [name type dims ...] ...) body ...)
     (%register-model! 'main '((name type dims ...) ...) (lambda (name ...) body ...))]
    [(_ entry (inputs [name type dims ...] ...) body ...)
     (%register-model! 'entry '((name type dims ...) ...) (lambda (name ...) body ...))]))

(define %dtypes '(f32 i32))

(define (%check-input-spec spec)
  (unless (and (list? spec) (<= 2 (length spec) 3) (symbol? (car spec)))
    (error 'model "input must look like [name dtype] or [name dtype (dims ...)]" spec))
  (unless (memq (cadr spec) %dtypes)
    (error 'model "unsupported input dtype, expected one of" (cadr spec) %dtypes))
  (when (= (length spec) 3)
    (let ([dims (caddr spec)])
      (unless (and (list? dims)
                   (<= 1 (length dims) 4)
                   (for-all (lambda (d) (or (not d) (symbol? d) (and (fixnum? d) (> d 0)))) dims))
        (error 'model "input dims must be a list of 1-4 positive integers, symbols or #f" dims)))))

;; Set by the host while a program is evaluated:
;; #(entries preprocess postprocess pipelines states), entries, pipelines and
;; states as reversed lists of (name ...).
(define %loading (make-parameter #f))

(define (%register-model! entry spec builder)
  (let ([slot (%loading)])
    (cond
      [(not slot) (error 'model "model can only be used while a program is loaded")]
      [(not (symbol? entry)) (error 'model "an entry name must be a symbol" entry)]
      [(assq entry (vector-ref slot 0))
       (error 'model (if (eq? entry 'main) "a program can define only one unnamed model" "duplicate entry name") entry)]
      [(assq entry (vector-ref slot 3)) (error 'model "a pipeline already has this name" entry)]
      [else
       (for-each %check-input-spec spec)
       (let ([names (map car spec)])
         (unless (= (length names) (length (%dedupe names)))
           (error 'model "duplicate input names" names)))
       (vector-set! slot 0 (cons (list entry spec builder) (vector-ref slot 0)))])))

(define (%dedupe xs)
  (let loop ([xs xs] [seen '()])
    (cond
      [(null? xs) (reverse seen)]
      [(memq (car xs) seen) (loop (cdr xs) seen)]
      [else (loop (cdr xs) (cons (car xs) seen))])))

;; Pipelines: host code that runs entries, e.g. a generation loop.
;;
;; (pipeline caption ([photo image] [prompt string])
;;   (let* ([enc (run encode [pixels ...] [ids ...])]
;;          [dec (run decode [memory (output enc 'memory)] ...)])
;;     (results [text (detokenize tok ...)])))
;;
;; (run entry [input array] ...) runs a model entry on host arrays (a missing
;; leading batch dimension of 1 is added) and returns its outputs; (output r
;; name) picks one. Results may also be strings.
(define-syntax pipeline
  (syntax-rules ()
    [(_ name ([arg kind] ...) body ...)
     (%register-pipeline! 'name '((arg kind) ...) (lambda (arg ...) body ...))]))

(define (%register-pipeline! name spec proc)
  (let ([slot (%loading)])
    (unless slot (error 'pipeline "pipeline can only be used while a program is loaded"))
    (when (or (assq name (vector-ref slot 3)) (assq name (vector-ref slot 0)))
      (error 'pipeline "an entry or pipeline already has this name" name))
    (%check-raw-spec 'pipeline spec)
    (vector-set! slot 3 (cons (list name spec proc) (vector-ref slot 3)))))

(define (%check-raw-spec who spec)
  (for-each (lambda (s)
              (unless (and (symbol? (car s)) (memq (cadr s) '(string image audio array)))
                (error who "raw inputs look like [name kind], kind: string, image, audio or array" s)))
            spec))

;; The program a pipeline runs in (its id), while it runs.
(define %current-program (make-parameter #f))

(define-syntax run
  (syntax-rules ()
    [(_ entry [name expr] ...) (%run 'entry (list (cons 'name expr) ...))]))

;; Returns ((output-name . array) ...).
(define (%run entry inputs)
  (let ([id (%current-program)])
    (unless id (error 'run "run can only be used in a pipeline"))
    (let ([r (tl_host_run_begin id (symbol->string entry))])
      (when (eqv? r 0) (error 'run (tl_host_error)))
      (for-each (lambda (i)
                  (unless (symbol? (car i)) (error 'run "inputs look like [name array]" i))
                  (when (eqv? (tl_host_run_input r (symbol->string (car i)) (%host-id 'run 'array (cdr i))) 0)
                    (error 'run (tl_host_error))))
                inputs)
      (let ([n (- (tl_host_run_exec r) 1)])
        (when (< n 0) (error (string->symbol (format "run ~a" entry)) (tl_host_error)))
        (map (lambda (i)
               (cons (string->symbol (tl_host_run_output_name r i))
                     (%host 'run 'array (tl_host_run_output r i))))
             (iota n))))))

(define (output results name)
  (let ([entry (assq name results)])
    (unless entry (error 'output (format "no output ~s; outputs are ~s" name (map car results))))
    (cdr entry)))

;; Environment programs are evaluated in: pure R6RS (plus the R5RS names like
;; quotient), a few pure Chez utilities, (tensorlisp), and the stdlib
;; libraries the program imports. No eval, ports, files or foreign-procedure.
(define (%program-environment imports)
  (copy-environment
    (apply environment
      '(rnrs base) '(rnrs lists) '(rnrs control)
      '(rnrs arithmetic fixnums) '(rnrs arithmetic flonums)
      '(only (rnrs r5rs) quotient remainder modulo exact->inexact inexact->exact)
      '(only (chezscheme) iota list-head format fold-left fold-right)
      '(tensorlisp)
      imports)
    #t))

;; The stdlib: (tl name) binds its exports as name:export unless the import
;; spec says otherwise (prefix, only, rename, except).
(define %stdlib '(tensor nn attn vision util))

;; Import specs of the program's leading (import ...) forms, checked and with
;; the canonical prefixes applied.
(define (%program-imports forms)
  (define (library-name spec)
    (cond
      [(and (pair? spec) (memq (car spec) '(prefix only except rename)) (pair? (cdr spec)))
       (library-name (cadr spec))]
      [(and (pair? spec) (eq? (car spec) 'library) (pair? (cdr spec))) (cadr spec)]
      [else spec]))
  (define (check spec)
    (let ([name (library-name spec)])
      (unless (and (list? name) (= (length name) 2) (eq? (car name) 'tl) (memq (cadr name) %stdlib))
        (error 'import (format "only the stdlib can be imported: ~a" (map (lambda (n) (list 'tl n)) %stdlib)) name))
      (if (equal? spec name)
          `(prefix ,name ,(string->symbol (format "~a:" (cadr name))))
          spec)))
  (apply append
    (map (lambda (form)
           (unless (list? form) (error 'import "expected (import library ...)" form))
           (map check (cdr form)))
         forms)))

(define (%import-form? form) (and (pair? form) (eq? (car form) 'import)))

;; Host entry points, called from Rust.

(define %models (make-eqv-hashtable))

;; %models: id -> #(entries preprocess assets postprocess pipelines), entries
;; ((name spec builder) ...) and pipelines ((name spec proc) ...) in
;; definition order.
(define (%model-entries m) (vector-ref m 0))
(define (%model-preprocess m) (vector-ref m 1))
(define (%model-assets m) (vector-ref m 2))
(define (%model-postprocess m) (vector-ref m 3))
(define (%model-pipelines m) (vector-ref m 4))

(define (%default-entry m)
  (or (assq 'main (%model-entries m)) (car (%model-entries m))))

(define (%raw-spec->strings spec)
  (map (lambda (s) (list (symbol->string (car s)) (symbol->string (cadr s)))) spec))

;; Evaluates a program with its assets ((name . bytes-id) ...). Returns
;; (entries raw-inputs post-args pipelines):
;; entries ((entry-name ((name-string dtype-string dims-or-#f) ...)) ...), default first;
;; raw-inputs ((name-string kind-string) ...) or #f without preprocess;
;; post-args (name-string ...) or #f without postprocess;
;; pipelines ((name-string raw-inputs) ...);
;; states ((name-string type-string (dim ...)) ...).
(define ($tl-load-program id text assets)
  (let* ([port (open-input-string text)]
         [forms (let loop ([acc '()])
                  (let ([form (read port)])
                    (if (eof-object? form) (reverse acc) (loop (cons form acc)))))]
         [imports (let loop ([forms forms] [acc '()])
                    (if (and (pair? forms) (%import-form? (car forms)))
                        (loop (cdr forms) (cons (car forms) acc))
                        (reverse acc)))]
         [body (list-tail forms (length imports))]
         [env (%program-environment (%program-imports imports))]
         [slot (vector '() #f #f '() '())])
    (for-each (lambda (form)
                (when (%import-form? form) (error 'import "imports must come before everything else" form)))
              body)
    (parameterize ([%loading slot] [%assets assets])
      (for-each (lambda (form) (eval form env)) body))
    (when (null? (vector-ref slot 0)) (error 'load "the program does not define a model"))
    (let* ([pre (vector-ref slot 1)]
           [post (vector-ref slot 2)]
           [m (vector (reverse (vector-ref slot 0)) pre assets post (reverse (vector-ref slot 3)))]
           [default (%default-entry m)])
      (hashtable-set! %models id m)
      (list
        (map (lambda (e)
               (list (symbol->string (car e))
                     (map (lambda (spec)
                            (list (symbol->string (car spec))
                                  (symbol->string (cadr spec))
                                  (and (= (length spec) 3)
                                       (map (lambda (d) (and (fixnum? d) d)) (caddr spec)))))
                          (cadr e))))
             (cons default (remq default (%model-entries m))))
        (and pre (%raw-spec->strings (car pre)))
        (and post (map symbol->string (car post)))
        (map (lambda (p) (list (symbol->string (car p)) (%raw-spec->strings (cadr p))))
             (%model-pipelines m))
        (map (lambda (st) (list (car st) (symbol->string (cadr st)) (caddr st)))
             (reverse (vector-ref slot 4)))))))

(define (%host-arg a)
  (if (string? a) a (make-host (cdr a) (if (eq? (car a) 'tensor) 'array (car a)))))

;; Runs preprocess on one example. raws: per raw input, a string or
;; (kind . host-id). Returns ((input-name . array-id) ...) in the default
;; entry's input order.
(define ($tl-preprocess id raws)
  (let* ([m (hashtable-ref %models id #f)]
         [pre (and m (%model-preprocess m))])
    (unless pre (error 'preprocess "the program has no preprocess form"))
    (let* ([args (map %host-arg raws)]
           [result (parameterize ([%assets (%model-assets m)]) (apply (cdr pre) args))]
           [names (map car (cadr (%default-entry m)))])
      (unless (and (list? result) (for-all pair? result))
        (error 'preprocess "the body must end in (model-inputs [name array] ...)" result))
      (map (lambda (name)
             (let ([entry (assq name result)])
               (unless entry (error 'preprocess "no array for model input" name))
               (list (symbol->string name) (%host-id 'model-inputs 'array (cdr entry)))))
           names))))

;; (results ...) of postprocess or a pipeline for the host:
;; ((name-string kind payload) ...), kind "array" with a host id, "scalar"
;; with a number, "vector" with a list of numbers or "text" with a string.
(define (%export-results who result)
  (unless (and (list? result) (for-all (lambda (r) (and (pair? r) (symbol? (car r)))) result))
    (error who "the body must end in (results [name value] ...)" result))
  (map (lambda (r)
         (let ([name (symbol->string (car r))] [v (cdr r)])
           (cond
             [(host? v) (list name "array" (%host-id 'results 'array v))]
             [(real? v) (list name "scalar" (inexact v))]
             [(string? v) (list name "text" v)]
             [(and (list? v) (for-all real? v)) (list name "vector" (map inexact v))]
             [else (error 'results (format "~a must be an array, a number, a string or a list of numbers" name) v)])))
       result))

;; Runs postprocess on one example. args: per postprocess argument, a string
;; or (kind . host-id).
(define ($tl-postprocess id args)
  (let* ([m (hashtable-ref %models id #f)]
         [post (and m (%model-postprocess m))])
    (unless post (error 'postprocess "the program has no postprocess form"))
    (%export-results 'postprocess
      (parameterize ([%assets (%model-assets m)])
        (apply (cdr post) (map %host-arg args))))))

;; Runs pipeline name on raw inputs (in its declared order, like preprocess).
(define ($tl-pipeline id name raws)
  (let* ([m (hashtable-ref %models id #f)]
         [p (and m (assq (string->symbol name) (%model-pipelines m)))])
    (unless p (error 'pipeline "the program has no pipeline with this name" name))
    (%export-results 'pipeline
      (parameterize ([%assets (%model-assets m)] [%current-program id])
        (apply (caddr p) (map %host-arg raws))))))

;; Creates an input tensor for spec (name dtype ...) with ggml dims ne (1-4 sizes).
(define (%new-input ctx spec ne)
  (let* ([type (case (cadr spec) [(f32) GGML_TYPE_F32] [(i32) GGML_TYPE_I32])]
         [name (symbol->string (car spec))]
         [t (case (length ne)
              [(1) (apply ggml_new_tensor_1d ctx type ne)]
              [(2) (apply ggml_new_tensor_2d ctx type ne)]
              [(3) (apply ggml_new_tensor_3d ctx type ne)]
              [(4) (apply ggml_new_tensor_4d ctx type ne)])])
    (ggml_set_name t name)
    (ggml_set_input t)
    t))

;; Outputs are read back as contiguous f32.
(define (%finish-output ctx t)
  (let* ([t (if (ggml_is_contiguous t) t (ggml_cont ctx t))]
         [t (if (eqv? (tl_tensor_type t) GGML_TYPE_F32) t (ggml_cast ctx t GGML_TYPE_F32))])
    (ggml_set_output t)
    t))

;; Marks tensor t (named name) as a graph result: (name-string address rank).
(define (%result ctx who name t rank)
  (unless (tensor? t) (error who "not a tensor" name t))
  (let* ([p (%finish-output ctx (%tensor-ptr who t))]
         [n-dims (ggml_n_dims p)]
         [rank (or rank n-dims)])
    (unless (and (fixnum? rank) (<= n-dims rank 4))
      (error who
        (format "~a has ~a dimensions, its rank must be an integer from ~a to 4" name n-dims n-dims)
        rank))
    (list name p rank)))

;; Builds the whole graph of entry (a string) of model id in ctx. input-dims
;; holds the ggml dims of each input, in input order. taps is #t for all taps or a list of tap
;; names to read back. Returns
;; (graph (input-address ...) (output ...) (tap ...) (tap-name ...)), where
;; outputs and taps are (name-string address rank).
(define ($tl-build id entry ctx weights states device input-dims graph-size taps)
  (let ([m (hashtable-ref %models id #f)]
        [tap-slot (box '())])
    (unless m (error 'build "unknown model" id))
    (let ([m (let ([e (assq (string->symbol entry) (%model-entries m))])
               (unless e (error 'build "unknown entry" entry))
               (cons (cadr e) (caddr e)))])
    (set! %op 'inputs)
    (set! %pending '())
    (parameterize ([%ctx ctx] [%weights weights] [%states states] [%taps tap-slot] [%device (string->symbol device)]
                   [%graph (ggml_new_graph_custom ctx graph-size #f)])
      (let* ([inputs (map (lambda (spec ne) (%new-input ctx spec ne))
                          (car m) input-dims)]
             [outs (apply (cdr m) (map (lambda (p) (make-tensor p ctx)) inputs))])
        (unless (list? outs)
          (error 'model "the model body must end in (outputs [name tensor] ...)" outs))
        (set! %op 'outputs)
        (let* ([outputs
                (map (lambda (o)
                       (%result ctx 'outputs (symbol->string (car o)) (cadr o)
                                (and (pair? (cddr o)) (caddr o))))
                     outs)]
               [tapped (reverse (unbox tap-slot))]
               [selected
                (if (eq? taps #t)
                    tapped
                    (map (lambda (name)
                           (or (assoc name tapped)
                               (error 'tap (format "the program has no tap named ~s" name)
                                      (map car tapped))))
                         taps))]
               [tap-results
                (map (lambda (tp) (%result ctx 'tap (car tp) (cadr tp) (caddr tp))) selected)]
               [graph (%graph)])
          (for-each (lambda (r) (ggml_build_forward_expand graph (cadr r)))
                    (append outputs tap-results))
          ;; Inputs the body doesn't use still get memory, so they can be set.
          (for-each (lambda (t) (ggml_build_forward_expand graph t)) inputs)
          (list graph inputs outputs tap-results (map car tapped))))))))

(define ($tl-unload id)
  (hashtable-delete! %models id))
