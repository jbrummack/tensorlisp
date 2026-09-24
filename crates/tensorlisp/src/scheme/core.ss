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

;; Graph context and weight context of the graph currently being built.
(define %ctx (make-parameter 0))
(define %weights (make-parameter 0))

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
    (unless (or (eqv? ctx (%ctx)) (eqv? ctx (%weights)))
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

(define (dtype t)
  (string->symbol (ggml_type_name (tl_tensor_type (%tensor-ptr 'dtype t)))))

;; A definition, since a library body can't have expressions before definitions.
(define %tensor-writer
  (record-writer (record-type-descriptor tensor)
    (lambda (t port wr)
      (if (or (eqv? (tensor-ctx t) (%ctx)) (eqv? (tensor-ctx t) (%weights)))
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

;; --- Preprocessing: host values (autopro, see host.rs). Host arrays are
;; plain data on the Rust side, not graph tensors; preprocess returns them as
;; the model's inputs.

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

;; Images (8-bit RGB; resizing reproduces Pillow)

(define (%filter who f)
  (case f
    [(nearest) 0] [(lanczos) 1] [(bilinear) 2] [(bicubic) 3] [(box) 4] [(hamming) 5]
    [else (error who "filter must be nearest, bilinear, bicubic, lanczos, box or hamming" f)]))

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
    (for-each (lambda (s)
                (unless (and (symbol? (car s)) (memq (cadr s) '(string image audio array)))
                  (error 'preprocess "raw inputs look like [name kind], kind: string, image, audio or array" s)))
              spec)
    (vector-set! slot 1 (cons spec proc))))

;; Model definition
;;
;; (model (inputs [x f32 (784 1)] [y f32])
;;   (define h ...)
;;   (outputs [logits (ggml-add ...)] [probs (ggml-soft-max ...) 2]))
;;
;; Input shapes are optional; a dimension may be #f or a symbol to accept any
;; size. The optional integer after an output is its rank (number of
;; dimensions); by default trailing dimensions of size 1 are dropped.

(define-syntax inputs (lambda (x) (syntax-violation 'inputs "misplaced auxiliary keyword" x)))

(define-syntax outputs
  (syntax-rules ()
    [(_ [name expr opt ...] ...) (list (list 'name expr opt ...) ...)]))

(define-syntax model
  (syntax-rules (inputs)
    [(_ (inputs [name type dims ...] ...) body ...)
     (%register-model! '((name type dims ...) ...) (lambda (name ...) body ...))]))

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

;; Set by the host while a program is evaluated: #((spec . builder) preprocess).
(define %loading (make-parameter #f))

(define (%register-model! spec builder)
  (let ([slot (%loading)])
    (cond
      [(not slot) (error 'model "model can only be used while a program is loaded")]
      [(vector-ref slot 0) (error 'model "a program can define only one model")]
      [else
       (for-each %check-input-spec spec)
       (let ([names (map car spec)])
         (unless (= (length names) (length (%dedupe names)))
           (error 'model "duplicate input names" names)))
       (vector-set! slot 0 (cons spec builder))])))

(define (%dedupe xs)
  (let loop ([xs xs] [seen '()])
    (cond
      [(null? xs) (reverse seen)]
      [(memq (car xs) seen) (loop (cdr xs) seen)]
      [else (loop (cdr xs) (cons (car xs) seen))])))

;; Environment programs are evaluated in: pure R6RS (plus the R5RS names like
;; quotient), a few pure Chez utilities, and (tensorlisp). No eval, ports,
;; files or foreign-procedure.
(define (%program-environment)
  (copy-environment
    (environment '(rnrs base) '(rnrs lists) '(rnrs control)
                 '(rnrs arithmetic fixnums) '(rnrs arithmetic flonums)
                 '(only (rnrs r5rs) quotient remainder modulo exact->inexact inexact->exact)
                 '(only (chezscheme) iota list-head format fold-left fold-right)
                 '(tensorlisp))
    #t))

;; Host entry points, called from Rust.

(define %models (make-eqv-hashtable))

;; %models: id -> #(input-spec builder preprocess assets)
(define (%model-spec m) (vector-ref m 0))
(define (%model-builder m) (vector-ref m 1))
(define (%model-preprocess m) (vector-ref m 2))
(define (%model-assets m) (vector-ref m 3))

;; Evaluates a program with its assets ((name . bytes-id) ...). Returns
;; (inputs raw-inputs): inputs ((name-string dtype-string dims-or-#f) ...),
;; raw-inputs ((name-string kind-string) ...) or #f without preprocess.
(define ($tl-load-program id text assets)
  (let ([env (%program-environment)]
        [slot (vector #f #f)]
        [port (open-input-string text)])
    (parameterize ([%loading slot] [%assets assets])
      (let loop ()
        (let ([form (read port)])
          (unless (eof-object? form)
            (eval form env)
            (loop)))))
    (unless (vector-ref slot 0) (error 'load "the program does not define a model"))
    (let ([model (vector-ref slot 0)] [pre (vector-ref slot 1)])
      (hashtable-set! %models id (vector (car model) (cdr model) pre assets))
      (list
        (map (lambda (spec)
               (list (symbol->string (car spec))
                     (symbol->string (cadr spec))
                     (and (= (length spec) 3)
                          (map (lambda (d) (and (fixnum? d) d)) (caddr spec)))))
             (car model))
        (and pre (map (lambda (s) (list (symbol->string (car s)) (symbol->string (cadr s)))) (car pre)))))))

;; Runs preprocess on one example. raws: per raw input, a string or
;; (kind . host-id). Returns ((input-name . array-id) ...) in input order.
(define ($tl-preprocess id raws)
  (let* ([m (hashtable-ref %models id #f)]
         [pre (and m (%model-preprocess m))])
    (unless pre (error 'preprocess "the program has no preprocess form"))
    (let* ([args (map (lambda (spec raw)
                        (if (string? raw) raw (make-host (cdr raw) (if (eq? (car raw) 'tensor) 'array (car raw)))))
                      (car pre) raws)]
           [result (parameterize ([%assets (%model-assets m)]) (apply (cdr pre) args))]
           [names (map car (%model-spec m))])
      (unless (and (list? result) (for-all pair? result))
        (error 'preprocess "the body must end in (model-inputs [name array] ...)" result))
      (map (lambda (name)
             (let ([entry (assq name result)])
               (unless entry (error 'preprocess "no array for model input" name))
               (list (symbol->string name) (%host-id 'model-inputs 'array (cdr entry)))))
           names))))

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

;; Builds the whole graph for model id in ctx. input-dims holds the ggml dims
;; of each input, in input order. taps is #t for all taps or a list of tap
;; names to read back. Returns
;; (graph (input-address ...) (output ...) (tap ...) (tap-name ...)), where
;; outputs and taps are (name-string address rank).
(define ($tl-build id ctx weights input-dims graph-size taps)
  (let ([m (hashtable-ref %models id #f)]
        [tap-slot (box '())])
    (unless m (error 'build "unknown model" id))
    (let ([m (cons (%model-spec m) (%model-builder m))])
    (set! %op 'inputs)
    (set! %pending '())
    (parameterize ([%ctx ctx] [%weights weights] [%taps tap-slot])
      (let* ([inputs (map (lambda (spec ne) (%new-input ctx spec ne))
                          (car m) input-dims)]
             [outs (apply (cdr m) (map (lambda (p) (make-tensor p ctx)) inputs))])
        (unless (and (list? outs) (pair? outs))
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
               [graph (ggml_new_graph_custom ctx graph-size #f)])
          (for-each (lambda (r) (ggml_build_forward_expand graph (cadr r)))
                    (append outputs tap-results))
          (list graph inputs outputs tap-results (map car tapped))))))))

(define ($tl-unload id)
  (hashtable-delete! %models id))
