;; (tl generic), imported bare (no prefix -- see core.ss's %stdlib-unprefixed):
;; the AOT-portable op vocabulary.
;;
;; This is exactly, and only, the subset of ops tensorlisp-aot
;; (../../../../tensorlisp-aot) currently knows how to lower to a backend
;; leaf (CoreML MIL so far; see its src/lib.rs). Each name here is a plain
;; alias or a thin wrapper around an existing, already-validated ggml/(tl
;; nn)/(tl tensor) op -- never new graph-building logic of its own -- so a
;; model written against these names runs exactly like it always has (ggml
;; is still the only backend that actually executes a graph); the only
;; thing "generic" about them is that their names don't commit to ggml,
;; which is what lets a nanopass lowering table key off op name alone
;; instead of ggml's own (C-derived) naming.
;;
;; Growing this list and tensorlisp-aot's lowering table together, one op at
;; a time, each checked against a real tap (see tensorlisp-aot's tests), is
;; the intended way to extend AOT coverage -- instead of hand-porting a
;; whole model's ops in one large, hard-to-verify Rust change.
(library (tl generic)
  (export add sub mul relu silu sigmoid mul-mat reshape
          conv2d conv2d-depthwise max-pool upsample-nearest
          slice concat)
  (import (rnrs) (tensorlisp runtime) (prefix (tl tensor) tensor:) (prefix (tl nn) nn:))

  (define add ggml-add)
  (define sub ggml-sub)
  (define mul ggml-mul)
  (define relu ggml-relu)
  (define silu ggml-silu)
  (define sigmoid ggml-sigmoid)
  (define mul-mat ggml-mul-mat)

  ;; (reshape t d0 ...): 1 to 4 target dims, ggml order (innermost first).
  (define reshape
    (case-lambda
      [(t d0) (ggml-reshape-1d t d0)]
      [(t d0 d1) (ggml-reshape-2d t d0 d1)]
      [(t d0 d1 d2) (ggml-reshape-3d t d0 d1 d2)]
      [(t d0 d1 d2 d3) (ggml-reshape-4d t d0 d1 d2 d3)]))

  ;; nn:conv2d/nn:conv2d-depthwise pick 'direct on CPU by default (im2col on
  ;; GPU) -- tensorlisp-aot's CONV_2D/CONV_2D_DW lowering only understands
  ;; the direct kernel's single ggml op node, so an AOT-targeted model
  ;; should build (and its taps be checked) on Device::Cpu, or pass
  ;; 'method 'direct explicitly.
  (define conv2d nn:conv2d)
  (define conv2d-depthwise nn:conv2d-depthwise)
  (define max-pool nn:max-pool)
  (define upsample-nearest nn:upsample-nearest)

  ;; tensor:slice/tensor:concat, not the raw ggml-view-4d/ggml-concat ops:
  ;; tensorlisp-aot's VIEW lowering specifically relies on tensor:slice's
  ;; single-axis, stride-preserving offset pattern (see its slice_params()
  ;; doc comment), and its CONCAT lowering assumes tensor:concat's
  ;; fold-left-over-pairs binary chaining.
  (define slice tensor:slice)
  (define concat tensor:concat))
