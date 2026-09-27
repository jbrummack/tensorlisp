;;; scheme/aot/reference_compiler.ss
;;;
;;; The nanopass that lowers `(tl aot highlevel)` source into gguf ops: the
;;; real `ggml-*` calls a program's GGUF `TL_TXT` text is actually made of
;;; (see docs/scheme-functions.md's "Generated ggml ops" list). Same idea as
;;; core.ss's `$tl-compile-generic-to-ggml`/`%generic-rename` (a pure
;;; syntax-to-syntax rewrite of call heads, read in and read back out as
;;; source text, no semantic transformation, no shape reasoning -- the
;;; target's own execution computes shapes exactly as it always has), except
;;; systematic rather than hand-curated: every highlevel name is *by
;;; construction* `ggml-` plus itself minus its trailing `-t` (see
;;; highlevel.ss), so the rename is derived from `hl-op-names` instead of
;;; needing a matching case clause per op here.
;;;
;;; Renaming only ever touches the head position of a call form -- never an
;;; argument, a quoted datum, or a `let`-bound name -- so it can't misfire on
;;; option keywords or literal data; it also can't detect a highlevel op name
;;; locally shadowed by `let`, the same known, accepted simplification
;;; `$tl-compile-generic-to-ggml` makes.
(library (tl aot reference-compiler)
  (export hl-compile-form hl-compile-program)
  ;; Only `hl-op-names` -- the ops themselves aren't needed to derive a
  ;; rename, just the name list to check membership against.
  (import (chezscheme) (only (tl aot highlevel) hl-op-names))

  ;; `op-t` -> `ggml-op`, only for names `(tl aot highlevel)` actually
  ;; exports; anything else (a helper the program itself defined, a stdlib
  ;; call, a special form) is left untouched.
  (define (%hl-rename op)
    (and (memq op hl-op-names)
         (let ([s (symbol->string op)])
           (string->symbol (string-append "ggml-" (substring s 0 (- (string-length s) 2)))))))

  (define (hl-compile-form form)
    (cond
      [(and (pair? form) (symbol? (car form)))
       (cons (or (%hl-rename (car form)) (car form))
             (map hl-compile-form (cdr form)))]
      [(pair? form) (cons (hl-compile-form (car form)) (hl-compile-form (cdr form)))]
      [else form]))

  ;; Reads every top-level form from `src` and returns gguf-op-targeted
  ;; source text, forms in the same order, each rewritten independently.
  ;; Unlike `$tl-compile-generic-to-ggml`, this doesn't touch or replace the
  ;; program's own `(import ...)` forms -- a highlevel-vocabulary program is
  ;; expected to `(import (tl aot highlevel))` itself, same as any other
  ;; stdlib import, and the compiled output keeps whatever imports it wrote.
  (define (hl-compile-program src)
    (let* ([port (open-input-string src)]
           [forms (let loop ([acc '()])
                    (let ([form (read port)])
                      (if (eof-object? form) (reverse acc) (loop (cons form acc)))))])
      (with-output-to-string
        (lambda ()
          (for-each (lambda (f) (write (hl-compile-form f)) (newline)) forms))))))
