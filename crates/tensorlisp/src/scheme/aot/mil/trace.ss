;;; scheme/aot/mil/trace.ss
;;;
;;; MIL trace state: the runtime side of the highlevel -> MIL nanopass
;;; (shadow.ss, compile.ss, next to this file). A "trace" is built by
;;; running a compiled program for real -- each shadow op in shadow.ss both
;;; computes its real value (via the ggml op it wraps, so a model's own
;;; shape-dependent control flow stays correct) and records, here, which
;;; %name the resulting tensor got in the MIL program being accumulated.
;;;
;;; This is a from-scratch, self-contained port of the same mechanism
;;; core.ss's own (hand-written, `(tl generic)`-targeted) MIL trace pass
;;; already uses internally (`%mil-names`/`%mil-emit!`/`%mil-ref`/etc.,
;;; private to core.ss) -- not a reuse of it, since those are core.ss
;;; internals no other library can see (core.ss is what every stdlib
;;; library imports, not the other way around). Keeping this state and its
;;; accessors in their own library, rather than folded into shadow.ss,
;;; mirrors that same separation one level up: "what a trace is" apart from
;;; "what each op does to it".
(library (tl aot mil trace)
  (export mil-trace-reset! mil-fresh! mil-sanitize
          mil-emit! mil-ref mil-declare-input! mil-declare-weight!
          mil-render mil-render! mil-last-render)
  (import (rnrs) (rnrs hashtables) (only (chezscheme) format with-output-to-string))

  ;; real tensor (eq?) -> %name symbol.
  (define %names (make-eq-hashtable))
  ;; gguf weight name (string) -> %name symbol, so the same weight
  ;; referenced twice is only declared once.
  (define %weight-names (make-hashtable string-hash string=?))
  (define %stmts '())    ; reversed list of (set %name (op ...)) forms
  (define %weights '())  ; reversed list of (gguf-name %name dtype dims)
  (define %counter 0)

  (define (mil-trace-reset!)
    (set! %names (make-eq-hashtable))
    (set! %weight-names (make-hashtable string-hash string=?))
    (set! %stmts '())
    (set! %weights '())
    (set! %counter 0))

  (define (mil-fresh! prefix)
    (let ([n %counter])
      (set! %counter (+ n 1))
      (string->symbol (format "%~a_~a" prefix n))))

  ;; MIL identifiers can't contain '.'; the gguf name (the real lookup key)
  ;; stays untouched, only the %name gets sanitized.
  (define (mil-sanitize s)
    (list->string (map (lambda (c) (if (char=? c #\.) #\_ c)) (string->list s))))

  ;; Records that `t` (a real tensor) was just produced by `node` (an
  ;; already-built MIL op-node sexpr, e.g. (relu (x %x_0)), typically from
  ;; one of language.ss's `mil-*` constructors); returns `t` unchanged so
  ;; the real (reference) computation this shadow wraps is unaffected.
  (define (mil-emit! t node prefix)
    (let ([nm (mil-fresh! prefix)])
      (set! %stmts (cons (list 'set nm node) %stmts))
      (hashtable-set! %names t nm)
      t))

  ;; The %name a previously-emitted (or declared-weight/input) real tensor
  ;; was given.
  (define (mil-ref t)
    (or (hashtable-ref %names t #f)
        (error 'mil-ref
               "tensor was not produced by a traced (mil-shadow:*) op -- likely reached through an uncovered primitive upstream"
               t)))

  ;; Declares `t` as a MIL graph input named `%name` -- compile.ss's model
  ;; injection calls this once per (model (inputs ...) ...) parameter, so a
  ;; model's own declared inputs are mil-ref-able from its very first op,
  ;; the same way a weight becomes ref-able the moment it's first declared.
  (define (mil-declare-input! t name)
    (hashtable-set! %names t (string->symbol (format "%~a" name)))
    t)

  ;; Declares `t` (a real weight tensor, from `weight`) as a MIL graph input
  ;; named after its gguf name (sanitized), unless already declared, and
  ;; returns its %name.
  (define (mil-declare-weight! t gguf-name dtype dims)
    (or (hashtable-ref %weight-names gguf-name #f)
        (let ([nm (string->symbol (format "%~a" (mil-sanitize gguf-name)))])
          (hashtable-set! %weight-names gguf-name nm)
          (set! %weights (cons (list gguf-name nm dtype dims) %weights))
          (hashtable-set! %names t nm)
          nm)))

  ;; Renders the trace accumulated so far as one program sexpr, textually --
  ;; same shape as core.ss's own %mil-render:
  ;; (program <name>
  ;;   (inputs (%x (tensor f32 (dims ...))) ...)
  ;;   (weights (%w (tensor f32 (dims ...)) (source "gguf.name")) ...)
  ;;   (block (set %name (op (arg val) ...)) ...)
  ;;   (output %name))
  ;; `inputs` is `((name dtype dims) ...)`; `output-tensor` is the real
  ;; tensor whose %name becomes the program's output.
  (define (mil-render program-name inputs output-tensor)
    (define (render-input i)
      (list (string->symbol (format "%~a" (car i)))
            (list 'tensor (cadr i) (caddr i))))
    (define (render-weight w)
      (list (cadr w) (list 'tensor (caddr w) (cadddr w)) (list 'source (car w))))
    (with-output-to-string
      (lambda ()
        (write (list 'program (string->symbol program-name)
                      (cons 'inputs (map render-input inputs))
                      (cons 'weights (map render-weight (reverse %weights)))
                      (cons 'block (reverse %stmts))
                      (list 'output (mil-ref output-tensor)))))))

  ;; A model builder's own tensors (and %weights/%ctx) stop being valid the
  ;; moment the graph build returns, so rendering has to happen from
  ;; *inside* the model body (where `output-tensor` is still live), not
  ;; from `postprocess`. `mil-render!` is that: same args, stores the text
  ;; (surviving on the Scheme thread after the call returns, same as
  ;; core.ss's own $tl-mil-last-render) instead of returning it, and returns
  ;; `output-tensor` unchanged so it can be spliced in wherever the model
  ;; would otherwise just use that tensor directly.
  (define %last-render #f)
  (define (mil-render! program-name inputs output-tensor)
    (set! %last-render (mil-render program-name inputs output-tensor))
    output-tensor)

  (define (mil-last-render) %last-render))
