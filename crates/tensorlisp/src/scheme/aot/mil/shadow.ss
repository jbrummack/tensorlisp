;;; scheme/aot/mil/shadow.ss
;;;
;;; Shadow ops: the semantic core of the highlevel -> MIL nanopass. Each
;;; `mil-shadow:*` procedure below takes the exact same arguments as the
;;; real ggml/(tl tensor) op it wraps, calls that real op to get the actual
;;; (reference) result -- so a model's own shape-dependent control flow
;;; keeps working exactly as it always has -- and also emits, via
;;; trace.ss's `mil-emit!`, the equivalent node from mil/language.ss's
;;; generated MIL vocabulary. Returns the real tensor, unchanged.
;;;
;;; Coverage is intentionally *not* every op highlevel.ss/language.ss
;;; expose: it's exactly the set tensorlisp-aot's Rust `lower_*` functions
;;; (crates/tensorlisp-aot/src/lib.rs) already lower and have proven end to
;;; end on real ANE hardware (its own module doc comment: "proven ... on
;;; tensorlisp's mlp test fixture ... and, since, on real yolo11n taps for
;;; every op below") -- ADD, SUB, MUL, RELU, SILU, SIGMOID, MUL_MAT,
;;; RESHAPE, CONV_2D, CONV_2D_DW, CONCAT, POOL_2D, UPSCALE, VIEW. Each
;;; shadow below is a direct Scheme port of that Rust function's semantics
;;; (parameter order/meaning, axis conversion, the fixed assumptions each
;;; makes), not an independent re-derivation -- see each one's own comment
;;; for exactly which Rust function it mirrors and where it necessarily
;;; differs (mainly: this pass sees a call's own arguments directly, before
;;; the underlying ggml call, where the Rust lowering has to decode raw
;;; ggml node fields (`op_params`, `view_offs`) after the fact from an
;;; already-built graph -- strictly easier here, not a different result).
;;;
;;; An op with no shadow here just isn't traced: the real tensor still
;;; flows through (still numerically correct for the reference run), but a
;;; later covered op that tries to use one as a MIL argument fails loudly
;;; via `mil-ref`, rather than silently emitting a wrong or partial graph --
;;; same failure mode core.ss's own `(tl generic)` MIL pass already accepts.
;;;
;;; One addition beyond tensorlisp-aot's own op set: `mil-shadow:weight`,
;;; shadowing plain `weight` (not a ggml op at all). Without it, a weight
;;; tensor CONV_2D/CONV_2D_DW read straight off the gguf file would never
;;; be `mil-ref`-able -- nothing else declares it -- so conv would fail
;;; immediately on any real model. core.ss's own mil-trace:conv2d avoids
;;; needing this by declaring its weight inline (it already takes a gguf
;;; name prefix, not a bare tensor); the raw-op-level shadows here take an
;;; already-resolved tensor instead, so declaring happens at the `weight`
;;; call site itself, one step earlier.
(library (tl aot mil shadow)
  (export mil-shadow:add mil-shadow:sub mil-shadow:mul
          mil-shadow:relu mil-shadow:silu mil-shadow:sigmoid
          mil-shadow:mul-mat
          mil-shadow:reshape-1d mil-shadow:reshape-2d
          mil-shadow:reshape-3d mil-shadow:reshape-4d
          mil-shadow:conv-2d-direct mil-shadow:conv-2d-dw-direct
          mil-shadow:concat mil-shadow:tensor-concat
          mil-shadow:pool-2d
          mil-shadow:upscale
          mil-shadow:tensor-slice
          mil-shadow:weight)
  (import (rnrs) (tensorlisp runtime) (prefix (tl tensor) tensor:)
          (tl aot mil trace) (tl aot mil language))

  ;; ---------------------------------------------------------- elementwise / matmul

  (define (mil-shadow:add a b) (mil-emit! (ggml-add a b) (mil-add (mil-ref a) (mil-ref b)) "add"))
  (define (mil-shadow:sub a b) (mil-emit! (ggml-sub a b) (mil-sub (mil-ref a) (mil-ref b)) "sub"))
  (define (mil-shadow:mul a b) (mil-emit! (ggml-mul a b) (mil-mul (mil-ref a) (mil-ref b)) "mul"))
  (define (mil-shadow:relu x) (mil-emit! (ggml-relu x) (mil-relu (mil-ref x)) "relu"))
  (define (mil-shadow:silu x) (mil-emit! (ggml-silu x) (mil-silu (mil-ref x)) "silu"))
  (define (mil-shadow:sigmoid x) (mil-emit! (ggml-sigmoid x) (mil-sigmoid (mil-ref x)) "sigmoid"))

  ;; Mirrors lower_mul_mat: ggml_mul_mat(a, b) = b @ a^T (tensorlisp's
  ;; `linear` calls it as (ggml-mul-mat weight x), i.e. a = weight, b =
  ;; activations); MIL matmul(x=b, y=a, transpose_x=#f, transpose_y=#t)
  ;; keeps b's batch rank, same as the real ggml call does.
  (define (mil-shadow:mul-mat a b)
    (mil-emit! (ggml-mul-mat a b) (mil-matmul (mil-ref b) (mil-ref a) #f #t) "matmul"))

  ;; ---------------------------------------------------------------------- reshape

  ;; Mirrors lower_reshape: ggml's own dims are innermost-first, MIL's
  ;; declared shape is outermost-first -- reverse. Unlike lower_reshape
  ;; (which reads the already-built node's `ne`, since it works from
  ;; GraphInfo), the target dims are simply this call's own arguments.
  (define (mil-shadow:reshape-1d x d0)
    (mil-emit! (ggml-reshape-1d x d0) (mil-reshape (mil-ref x) (list d0)) "reshape"))
  (define (mil-shadow:reshape-2d x d0 d1)
    (mil-emit! (ggml-reshape-2d x d0 d1) (mil-reshape (mil-ref x) (list d1 d0)) "reshape"))
  (define (mil-shadow:reshape-3d x d0 d1 d2)
    (mil-emit! (ggml-reshape-3d x d0 d1 d2) (mil-reshape (mil-ref x) (list d2 d1 d0)) "reshape"))
  (define (mil-shadow:reshape-4d x d0 d1 d2 d3)
    (mil-emit! (ggml-reshape-4d x d0 d1 d2 d3) (mil-reshape (mil-ref x) (list d3 d2 d1 d0)) "reshape"))

  ;; Declares a gguf weight as a MIL graph input the moment source code
  ;; reads it, named after its gguf name (sanitized) -- see the module doc
  ;; comment for why this is needed at all.
  (define (mil-shadow:weight name)
    (let ([t (weight name)])
      (mil-declare-weight! t name (dtype t) (shape t))
      t))

  ;; ------------------------------------------------------------------------- conv

  ;; Mirrors lower_conv_2d/conv2d: a = kernel [kw,kh,C_in,C_out] (ggml
  ;; order), b = input [W,H,C,N]; s0/p0/d0 are ggml's own W-axis (dim 0),
  ;; s1/p1/d1 its H-axis (dim 1); MIL wants [h, w] (its own axis order is
  ;; reversed from ggml's). No bias input at this raw level --
  ;; ggml-conv-2d-direct itself takes none (a caller adding one does so with
  ;; a separate ggml-add afterward, which mil-shadow:add already covers on
  ;; its own, same as lower_conv_2d never sets a "bias" input either).
  (define (mil-shadow:conv-2d-direct a b s0 s1 p0 p1 d0 d1)
    (mil-emit! (ggml-conv-2d-direct a b s0 s1 p0 p1 d0 d1)
               (mil-conv (mil-ref b) (mil-ref a) #f (list s1 s0) "custom" (list p1 p1 p0 p0) (list d1 d0) 1)
               "conv"))

  ;; Mirrors lower_conv_2d_dw: depthwise, groups = the real output's channel
  ;; count. lower_conv_2d_dw reads `ctx.node.ne[2]` off the already-built
  ;; node; the equivalent value here is the real output's own ggml-order
  ;; dim 2, read back via tensor:dim once it's computed.
  (define (mil-shadow:conv-2d-dw-direct a b s0 s1 p0 p1 d0 d1)
    (let ([y (ggml-conv-2d-dw-direct a b s0 s1 p0 p1 d0 d1)])
      (mil-emit! y
                 (mil-conv (mil-ref b) (mil-ref a) #f (list s1 s0) "custom" (list p1 p1 p0 p0) (list d1 d0) (tensor:dim y 2))
                 "conv")))

  ;; ---------------------------------------------------------------------- concat

  ;; Mirrors lower_concat: `dim` is ggml axis order (0 = innermost); MIL
  ;; wants ndarray axis order (0 = outermost) over the *real* (broadcast)
  ;; rank of the two operands, not a hardcoded 4 the way core.ss's own
  ;; %mil-axis4 (a rank-4-only shortcut) does.
  (define (mil-shadow:concat a b dim)
    (let* ([rank (max (length (shape a)) (length (shape b)))]
           [axis (- rank 1 dim)])
      (mil-emit! (ggml-concat a b dim) (mil-concat (list (mil-ref a) (mil-ref b)) axis #f) "concat")))

  ;; tensor:concat folds ggml-concat pairwise, left to right (see
  ;; tensor.ss); mirror that so the trace gets one MIL concat node per real
  ;; ggml-concat node, exactly as many as the real fold performs -- same
  ;; structure as core.ss's own mil-trace:tensor-concat.
  (define (mil-shadow:tensor-concat ts axis)
    (when (null? ts) (error 'mil-shadow:tensor-concat "no tensors"))
    (fold-left (lambda (acc t) (mil-shadow:concat acc t axis)) (car ts) (cdr ts)))

  ;; ---------------------------------------------------------------------- pool

  ;; Mirrors lower_pool_2d: `op` selects GGML_OP_POOL_MAX/GGML_OP_POOL_AVG;
  ;; k0/s0/p0 are ggml's own W axis, k1/s1/p1 its H axis; MIL wants [h, w].
  ;; ggml's own p0/p1 are floats (see ggml_pool_2d's C signature); MIL's
  ;; `pad` wants integers, same truncation lower_pool_2d's own comment
  ;; notes is exact for whole-pixel padding. mil-avg-pool's
  ;; `exclude-padding-from-average` has no ggml-side equivalent to derive
  ;; from -- #f is a guess, unlike MAX_POOL this isn't validated against a
  ;; real tap (tensorlisp-aot's own PassTable only actually exercises
  ;; MAX_POOL so far).
  (define (mil-shadow:pool-2d x op k0 k1 s0 s1 p0 p1)
    (let* ([y (ggml-pool-2d x op k0 k1 s0 s1 p0 p1)]
           [pad (list (%round p1) (%round p1) (%round p0) (%round p0))]
           [node (cond
                   [(= op GGML_OP_POOL_MAX) (mil-max-pool (mil-ref x) (list k1 k0) (list s1 s0) "custom" pad #f)]
                   [(= op GGML_OP_POOL_AVG) (mil-avg-pool #f (mil-ref x) (list k1 k0) (list s1 s0) "custom" pad #f)]
                   [else (error 'mil-shadow:pool-2d "unsupported ggml pool op" op)])])
      (mil-emit! y node "pool")))

  (define (%round x) (exact (round x)))

  ;; ------------------------------------------------------------------- upscale

  ;; Mirrors lower_upscale: only GGML_SCALE_MODE_NEAREST is covered (the
  ;; only mode tensorlisp-aot's Rust lowering handles); target size is
  ;; computed the same way core.ss's own mil-trace:upsample-nearest does
  ;; (ggml order [W,H,C,N], so dim 0 = W, dim 1 = H).
  (define (mil-shadow:upscale x factor mode)
    (unless (= mode GGML_SCALE_MODE_NEAREST)
      (error 'mil-shadow:upscale "only GGML_SCALE_MODE_NEAREST is covered" mode))
    (let ([target-h (* factor (tensor:dim x 1))]
          [target-w (* factor (tensor:dim x 0))])
      (mil-emit! (ggml-upscale x factor mode)
                 (mil-resize-nearest-neighbor (mil-ref x) target-h target-w)
                 "resize")))

  ;; --------------------------------------------------------------------- slice

  ;; Mirrors lower_view: tensor:slice is the only view-producer tensorlisp
  ;; source ever calls (see tensor.ss and lower_view's own doc comment) -- a
  ;; single-axis, stride-preserving slice. Unlike lower_view, which has to
  ;; decode a raw view_offs/nb pair after the fact from an already-built
  ;; node, this shadow sees the slice's own (axis from n) arguments
  ;; directly, before the underlying ggml-view-4d call -- begin/size are
  ;; known outright, no decoding needed. Calls tensor:slice directly for the
  ;; real value (unlike core.ss's own mil-trace:tensor-slice, which
  ;; couldn't -- core.ss is what (tl tensor) imports, not the other way
  ;; around; this library has no such restriction).
  (define (mil-shadow:tensor-slice x axis from n)
    (unless (and (fixnum? axis) (<= 0 axis 3)) (error 'mil-shadow:tensor-slice "axis must be 0-3" axis))
    (let* ([full (list (tensor:dim x 0) (tensor:dim x 1) (tensor:dim x 2) (tensor:dim x 3))]
           [begin4 (map (lambda (i) (if (= i axis) from 0)) '(0 1 2 3))]
           [size4 (map (lambda (i sz) (if (= i axis) n sz)) '(0 1 2 3) full)])
      (mil-emit! (tensor:slice x axis from n)
                 (mil-slice-by-size (mil-ref x) (reverse begin4) (reverse size4))
                 "slice"))))
