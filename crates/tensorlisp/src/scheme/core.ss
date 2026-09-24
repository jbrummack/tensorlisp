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

;; Set by the host while a program is evaluated; receives (spec . builder).
(define %loading (make-parameter #f))

(define (%register-model! spec builder)
  (let ([slot (%loading)])
    (cond
      [(not slot) (error 'model "model can only be used while a program is loaded")]
      [(unbox slot) (error 'model "a program can define only one model")]
      [else
       (for-each %check-input-spec spec)
       (let ([names (map car spec)])
         (unless (= (length names) (length (%dedupe names)))
           (error 'model "duplicate input names" names)))
       (set-box! slot (cons spec builder))])))

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

;; Evaluates a program and returns its input specs as
;; ((name-string dtype-string dims-or-#f) ...).
(define ($tl-load-program id text)
  (let ([env (%program-environment)]
        [slot (box #f)]
        [port (open-input-string text)])
    (parameterize ([%loading slot])
      (let loop ()
        (let ([form (read port)])
          (unless (eof-object? form)
            (eval form env)
            (loop)))))
    (unless (unbox slot) (error 'load "the program does not define a model"))
    (hashtable-set! %models id (unbox slot))
    (map (lambda (spec)
           (list (symbol->string (car spec))
                 (symbol->string (cadr spec))
                 (and (= (length spec) 3)
                      (map (lambda (d) (and (fixnum? d) d)) (caddr spec)))))
         (car (unbox slot)))))

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
          (list graph inputs outputs tap-results (map car tapped)))))))

(define ($tl-unload id)
  (hashtable-delete! %models id))
