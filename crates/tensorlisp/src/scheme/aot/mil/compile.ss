;;; scheme/aot/mil/compile.ss
;;;
;;; The highlevel -> MIL nanopass itself: a source-to-source rewrite (same
;;; shape as core.ss's own `$tl-compile-generic-to-mil`/`reference_compiler
;;; .ss`'s ggml pass next door) that renames each covered call's head to its
;;; `mil-shadow:*` shadow (shadow.ss) and injects a trace reset plus one
;;; input declaration per `(model (inputs ...) ...)` parameter. The rewrite
;;; alone doesn't produce a MIL program -- the *rewritten* source still has
;;; to be loaded and run for real (real weights, real inputs), the same way
;;; `$tl-compile-generic-to-mil`'s output does: running it both computes the
;;; normal numeric result and, as a side effect, populates the trace;
;;; calling `(mil-render! ...)` somewhere in the (rewritten) body then makes
;;; the accumulated MIL text available via `mil-last-render` once the run
;;; returns.
;;;
;;; Renaming only ever touches the head position of a call form -- never an
;;; argument, a quoted datum, or a `let`-bound name -- so it can't misfire
;;; on option keywords or literal data; it also can't detect a covered name
;;; locally shadowed by `let`, same known, accepted simplification every
;;; other nanopass in this codebase makes.
(library (tl aot mil compile)
  (export hl-mil-compile-form hl-mil-compile-program)
  (import (chezscheme) (tl aot mil trace))

  ;; Recognizes both a highlevel.ss `-t` name and the literal `ggml-*` name
  ;; it aliases (a model may write either), plus `(tl tensor)`'s own
  ;; `slice`/`concat` (bare or `tensor:`-prefixed) and plain `weight` --
  ;; shadow.ss's actual vocabulary, see its own doc comment for why these
  ;; particular ops and nothing past them.
  (define (%rename op)
    (case op
      [(add-t ggml-add) 'mil-shadow:add]
      [(sub-t ggml-sub) 'mil-shadow:sub]
      [(mul-t ggml-mul) 'mil-shadow:mul]
      [(relu-t ggml-relu) 'mil-shadow:relu]
      [(silu-t ggml-silu) 'mil-shadow:silu]
      [(sigmoid-t ggml-sigmoid) 'mil-shadow:sigmoid]
      [(mul-mat-t ggml-mul-mat) 'mil-shadow:mul-mat]
      [(reshape-1d-t ggml-reshape-1d) 'mil-shadow:reshape-1d]
      [(reshape-2d-t ggml-reshape-2d) 'mil-shadow:reshape-2d]
      [(reshape-3d-t ggml-reshape-3d) 'mil-shadow:reshape-3d]
      [(reshape-4d-t ggml-reshape-4d) 'mil-shadow:reshape-4d]
      [(conv-2d-direct-t ggml-conv-2d-direct) 'mil-shadow:conv-2d-direct]
      [(conv-2d-dw-direct-t ggml-conv-2d-dw-direct) 'mil-shadow:conv-2d-dw-direct]
      [(concat-t ggml-concat) 'mil-shadow:concat]
      [(slice tensor:slice) 'mil-shadow:tensor-slice]
      [(concat tensor:concat) 'mil-shadow:tensor-concat]
      [(pool-2d-t ggml-pool-2d) 'mil-shadow:pool-2d]
      [(upscale-t ggml-upscale) 'mil-shadow:upscale]
      [(weight) 'mil-shadow:weight]
      [else #f]))

  ;; Like `map`, but also handles an improper (dotted) list -- needed since
  ;; this walks *every* pair in a form, including a `. rest`-style variadic
  ;; parameter list, not just call expressions (same reason core.ss's own
  ;; %mil-map-form exists).
  (define (%map-form lst)
    (if (pair? lst)
        (cons (%compile-form (car lst)) (%map-form (cdr lst)))
        (%compile-form lst)))

  (define (%compile-form form)
    (cond
      [(and (pair? form) (eq? (car form) 'model)) (%inject-model form)]
      [(and (pair? form) (symbol? (car form)))
       (cons (or (%rename (car form)) (car form)) (%map-form (cdr form)))]
      [(pair? form) (cons (%compile-form (car form)) (%compile-form (cdr form)))]
      [else form]))

  ;; (model (inputs [name type dims ...] ...) body ...) or
  ;; (model entry (inputs [name type dims ...] ...) body ...): after
  ;; renaming the body as usual, injects a trace reset and one
  ;; (mil-declare-input! name "name") per declared input, right before the
  ;; (renamed) body -- a model's own parameters are otherwise never
  ;; mil-ref-able, since nothing else ever produces them via mil-emit!/
  ;; mil-declare-weight!. Same structure as core.ss's own %mil-inject-model.
  (define (%inject-model form)
    (let*-values ([(entry+spec rest) (values (cadr form) (cddr form))]
                  [(entry spec body)
                   (if (and (pair? entry+spec) (eq? (car entry+spec) 'inputs))
                       (values #f entry+spec rest)
                       (values entry+spec (car rest) (cdr rest)))]
                  [(names) (map car (cdr spec))]
                  ;; A `define`, not a bare expression: internal-body syntax
                  ;; requires every define before any expression, and each
                  ;; needs its own throwaway bound name (internal defines
                  ;; use letrec* semantics -- reusing `n` would shadow the
                  ;; outer parameter with an as-yet-unassigned binding of
                  ;; the same name).
                  [(decls) (map (lambda (n)
                                  (list 'define (string->symbol (format "%hl-mil-input-decl-~a" n))
                                        (list 'mil-declare-input! n (symbol->string n))))
                                names)]
                  [(reset) (list 'define '%hl-mil-reset-decl (list 'mil-trace-reset!))])
      (append (list 'model) (if entry (list entry) '()) (list spec) (list reset) decls (map %compile-form body))))

  (define (hl-mil-compile-form form) (%compile-form form))

  (define (%import-form? form) (and (pair? form) (eq? (car form) 'import)))

  ;; Reads every top-level form from `src`, drops its leading (import ...)
  ;; forms (replaced with a fixed import of every library a rename might
  ;; target), and renames every covered call to its mil-shadow: shadow. The
  ;; result is ordinary tensorlisp source: load and run it exactly as-is to
  ;; both get the normal numeric output AND, as a side effect, populate the
  ;; trace (see the module doc comment). NOTE: `(tl aot mil shadow)` isn't
  ;; on core.ss's %stdlib allowlist yet, so this output can't actually be
  ;; loaded through the normal Model::load path until that's wired in --
  ;; this pass is complete and independently testable (see its own tests),
  ;; but that one remaining integration step is deliberately not done here,
  ;; same reason highlevel.ss/mil/language.ss aren't wired in either.
  (define (hl-mil-compile-program src)
    (let* ([port (open-input-string src)]
           [forms (let loop ([acc '()])
                    (let ([form (read port)])
                      (if (eof-object? form) (reverse acc) (loop (cons form acc)))))]
           [body (remp %import-form? forms)]
           [compiled (map %compile-form body)])
      (with-output-to-string
        (lambda ()
          (write '(import (tl aot mil shadow) (tl nn) (tl attn) (tl vision) (tl util) (tl tensor)))
          (newline)
          (for-each (lambda (f) (write f) (newline)) compiled))))))
