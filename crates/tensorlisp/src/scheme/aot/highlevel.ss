;;; scheme/aot/highlevel.ss
;;;
;;; The high-level tensor-op vocabulary for tensorlisp's AOT compiler: one
;;; name per generated ggml op (see docs/scheme-functions.md's "Generated
;;; ggml ops" list), `-t` appended in place of the "ggml-" prefix (`add` ->
;;; `add-t`, `conv-2d-dw` -> `conv-2d-dw-t`) so this vocabulary's names don't
;;; collide with R6RS's own numeric procedures (`abs`, `div`, `exp`, `log`,
;;; `sin`, `cos`, `sqrt`, `floor`, `round` are all real R6RS bindings) or
;;; shadow anything a program importing this alongside other stdlib
;;; libraries already has in scope -- unlike stripping the prefix outright,
;;; which is what `(tl generic)` does for its much smaller, hand-picked
;;; subset (see stdlib/generic.ss) and gets away with only because none of
;;; its 13 names happen to collide.
;;;
;;; Each name is a plain alias -- exactly the same idea as
;;; stdlib/generic.ss's AOT-portable subset, just extended to the *entire*
;;; ggml op surface instead of only the ops tensorlisp-aot currently knows
;;; how to lower -- so building a graph against these names already runs it
;;; for real (ggml is still the only backend that actually executes a
;;; graph); nothing here is a new IR or new graph-building logic.
;;;
;;; The point of naming these separately from `ggml-*` is the same reason
;;; (tl generic) exists: a name that doesn't commit to ggml lets a nanopass
;;; lowering table key off the op name alone (see reference_compiler.ss,
;;; next to this file, and core.ss's `$tl-compile-generic-to-ggml` for the
;;; pattern this generalizes). `hl-op-names` is the list every such pass
;;; should check membership against -- adding a new op here is the only
;;; thing extending this vocabulary needs; reference_compiler.ss derives its
;;; rename automatically instead of needing a matching hand-written entry.
(library (tl aot highlevel)
  (export hl-op-names
          abs-t abs-inplace-t acc-t acc-inplace-t add-t add-cast-t add-id-t
          add-inplace-t add-rel-pos-t add-rel-pos-inplace-t add1-t
          add1-inplace-t arange-t argmax-t argsort-t argsort-top-k-t cast-t
          ceil-t ceil-inplace-t clamp-t col2im-1d-t concat-t cont-t cont-1d-t
          cont-2d-t cont-3d-t cont-4d-t conv-1d-t conv-1d-dw-t
          conv-1d-dw-ph-t conv-1d-ph-t conv-2d-t conv-2d-direct-t
          conv-2d-dw-t conv-2d-dw-direct-t conv-2d-s1-ph-t conv-2d-sk-p0-t
          conv-3d-t conv-3d-direct-t conv-transpose-1d-t
          conv-transpose-2d-p0-t cos-t cos-inplace-t count-equal-t cpy-t
          cross-entropy-loss-t cross-entropy-loss-back-t cumsum-t diag-t
          diag-mask-inf-t diag-mask-inf-inplace-t diag-mask-zero-t
          diag-mask-zero-inplace-t div-t div-inplace-t dsv4-hc-comb-t
          dsv4-hc-post-t dsv4-hc-pre-t dup-t dup-inplace-t dup-tensor-t elu-t
          elu-inplace-t exp-t exp-inplace-t expm1-t expm1-inplace-t fill-t
          fill-inplace-t flash-attn-back-t flash-attn-ext-t floor-t
          floor-inplace-t gated-delta-net-t gated-linear-attn-t geglu-t
          geglu-erf-t geglu-erf-split-t geglu-erf-swapped-t geglu-quick-t
          geglu-quick-split-t geglu-quick-swapped-t geglu-split-t
          geglu-swapped-t gelu-t gelu-erf-t gelu-erf-inplace-t gelu-inplace-t
          gelu-quick-t gelu-quick-inplace-t get-first-tensor-t
          get-next-tensor-t get-rel-pos-t get-rows-t get-rows-back-t glu-t
          glu-split-t group-norm-t group-norm-inplace-t hardsigmoid-t
          hardswish-t im2col-t im2col-3d-t interpolate-t l2-norm-t
          l2-norm-inplace-t leaky-relu-t lightning-indexer-t log-t
          log-inplace-t mean-t mul-t mul-inplace-t mul-mat-t mul-mat-id-t
          neg-t neg-inplace-t new-tensor-1d-t new-tensor-2d-t new-tensor-3d-t
          new-tensor-4d-t norm-t norm-inplace-t opt-step-adamw-t
          opt-step-sgd-t out-prod-t pad-t pad-circular-t pad-ext-t
          pad-ext-circular-t pad-reflect-1d-t permute-t pool-1d-t pool-2d-t
          pool-2d-back-t reglu-t reglu-split-t reglu-swapped-t relu-t
          relu-inplace-t repeat-t repeat-4d-t repeat-back-t reshape-t
          reshape-1d-t reshape-2d-t reshape-3d-t reshape-4d-t rms-norm-t
          rms-norm-back-t rms-norm-inplace-t roll-t rope-t rope-custom-t
          rope-custom-inplace-t rope-ext-t rope-ext-back-t rope-ext-inplace-t
          rope-inplace-t round-t round-inplace-t rwkv-wkv6-t rwkv-wkv7-t
          scale-t scale-bias-t scale-bias-inplace-t scale-inplace-t set-t
          set-1d-t set-1d-inplace-t set-2d-t set-2d-inplace-t set-inplace-t
          set-rows-t sgn-t sgn-inplace-t sigmoid-t sigmoid-inplace-t silu-t
          silu-back-t silu-inplace-t sin-t sin-inplace-t soft-max-t
          soft-max-ext-t soft-max-ext-back-t soft-max-ext-back-inplace-t
          soft-max-ext-inplace-t soft-max-inplace-t softplus-t
          softplus-inplace-t solve-tri-t sqr-t sqr-inplace-t sqrt-t
          sqrt-inplace-t ssm-conv-t ssm-scan-t step-t step-inplace-t sub-t
          sub-inplace-t sum-t sum-rows-t swiglu-t swiglu-oai-t swiglu-split-t
          swiglu-swapped-t tanh-t tanh-inplace-t timestep-embedding-t top-k-t
          transpose-t tri-t trunc-t trunc-inplace-t unary-t unary-inplace-t
          upscale-t upscale-ext-t view-1d-t view-2d-t view-3d-t view-4d-t
          view-tensor-t win-part-t win-unpart-t xielu-t)
  (import (rnrs) (tensorlisp runtime))

  (define hl-op-names
    '(     abs-t abs-inplace-t acc-t acc-inplace-t add-t add-cast-t add-id-t
     add-inplace-t add-rel-pos-t add-rel-pos-inplace-t add1-t
     add1-inplace-t arange-t argmax-t argsort-t argsort-top-k-t cast-t
     ceil-t ceil-inplace-t clamp-t col2im-1d-t concat-t cont-t cont-1d-t
     cont-2d-t cont-3d-t cont-4d-t conv-1d-t conv-1d-dw-t conv-1d-dw-ph-t
     conv-1d-ph-t conv-2d-t conv-2d-direct-t conv-2d-dw-t
     conv-2d-dw-direct-t conv-2d-s1-ph-t conv-2d-sk-p0-t conv-3d-t
     conv-3d-direct-t conv-transpose-1d-t conv-transpose-2d-p0-t cos-t
     cos-inplace-t count-equal-t cpy-t cross-entropy-loss-t
     cross-entropy-loss-back-t cumsum-t diag-t diag-mask-inf-t
     diag-mask-inf-inplace-t diag-mask-zero-t diag-mask-zero-inplace-t
     div-t div-inplace-t dsv4-hc-comb-t dsv4-hc-post-t dsv4-hc-pre-t dup-t
     dup-inplace-t dup-tensor-t elu-t elu-inplace-t exp-t exp-inplace-t
     expm1-t expm1-inplace-t fill-t fill-inplace-t flash-attn-back-t
     flash-attn-ext-t floor-t floor-inplace-t gated-delta-net-t
     gated-linear-attn-t geglu-t geglu-erf-t geglu-erf-split-t
     geglu-erf-swapped-t geglu-quick-t geglu-quick-split-t
     geglu-quick-swapped-t geglu-split-t geglu-swapped-t gelu-t gelu-erf-t
     gelu-erf-inplace-t gelu-inplace-t gelu-quick-t gelu-quick-inplace-t
     get-first-tensor-t get-next-tensor-t get-rel-pos-t get-rows-t
     get-rows-back-t glu-t glu-split-t group-norm-t group-norm-inplace-t
     hardsigmoid-t hardswish-t im2col-t im2col-3d-t interpolate-t l2-norm-t
     l2-norm-inplace-t leaky-relu-t lightning-indexer-t log-t log-inplace-t
     mean-t mul-t mul-inplace-t mul-mat-t mul-mat-id-t neg-t neg-inplace-t
     new-tensor-1d-t new-tensor-2d-t new-tensor-3d-t new-tensor-4d-t norm-t
     norm-inplace-t opt-step-adamw-t opt-step-sgd-t out-prod-t pad-t
     pad-circular-t pad-ext-t pad-ext-circular-t pad-reflect-1d-t permute-t
     pool-1d-t pool-2d-t pool-2d-back-t reglu-t reglu-split-t
     reglu-swapped-t relu-t relu-inplace-t repeat-t repeat-4d-t
     repeat-back-t reshape-t reshape-1d-t reshape-2d-t reshape-3d-t
     reshape-4d-t rms-norm-t rms-norm-back-t rms-norm-inplace-t roll-t
     rope-t rope-custom-t rope-custom-inplace-t rope-ext-t rope-ext-back-t
     rope-ext-inplace-t rope-inplace-t round-t round-inplace-t rwkv-wkv6-t
     rwkv-wkv7-t scale-t scale-bias-t scale-bias-inplace-t scale-inplace-t
     set-t set-1d-t set-1d-inplace-t set-2d-t set-2d-inplace-t
     set-inplace-t set-rows-t sgn-t sgn-inplace-t sigmoid-t
     sigmoid-inplace-t silu-t silu-back-t silu-inplace-t sin-t
     sin-inplace-t soft-max-t soft-max-ext-t soft-max-ext-back-t
     soft-max-ext-back-inplace-t soft-max-ext-inplace-t soft-max-inplace-t
     softplus-t softplus-inplace-t solve-tri-t sqr-t sqr-inplace-t sqrt-t
     sqrt-inplace-t ssm-conv-t ssm-scan-t step-t step-inplace-t sub-t
     sub-inplace-t sum-t sum-rows-t swiglu-t swiglu-oai-t swiglu-split-t
     swiglu-swapped-t tanh-t tanh-inplace-t timestep-embedding-t top-k-t
     transpose-t tri-t trunc-t trunc-inplace-t unary-t unary-inplace-t
     upscale-t upscale-ext-t view-1d-t view-2d-t view-3d-t view-4d-t
     view-tensor-t win-part-t win-unpart-t xielu-t))

  (define abs-t ggml-abs)
  (define abs-inplace-t ggml-abs-inplace)
  (define acc-t ggml-acc)
  (define acc-inplace-t ggml-acc-inplace)
  (define add-t ggml-add)
  (define add-cast-t ggml-add-cast)
  (define add-id-t ggml-add-id)
  (define add-inplace-t ggml-add-inplace)
  (define add-rel-pos-t ggml-add-rel-pos)
  (define add-rel-pos-inplace-t ggml-add-rel-pos-inplace)
  (define add1-t ggml-add1)
  (define add1-inplace-t ggml-add1-inplace)
  (define arange-t ggml-arange)
  (define argmax-t ggml-argmax)
  (define argsort-t ggml-argsort)
  (define argsort-top-k-t ggml-argsort-top-k)
  (define cast-t ggml-cast)
  (define ceil-t ggml-ceil)
  (define ceil-inplace-t ggml-ceil-inplace)
  (define clamp-t ggml-clamp)
  (define col2im-1d-t ggml-col2im-1d)
  (define concat-t ggml-concat)
  (define cont-t ggml-cont)
  (define cont-1d-t ggml-cont-1d)
  (define cont-2d-t ggml-cont-2d)
  (define cont-3d-t ggml-cont-3d)
  (define cont-4d-t ggml-cont-4d)
  (define conv-1d-t ggml-conv-1d)
  (define conv-1d-dw-t ggml-conv-1d-dw)
  (define conv-1d-dw-ph-t ggml-conv-1d-dw-ph)
  (define conv-1d-ph-t ggml-conv-1d-ph)
  (define conv-2d-t ggml-conv-2d)
  (define conv-2d-direct-t ggml-conv-2d-direct)
  (define conv-2d-dw-t ggml-conv-2d-dw)
  (define conv-2d-dw-direct-t ggml-conv-2d-dw-direct)
  (define conv-2d-s1-ph-t ggml-conv-2d-s1-ph)
  (define conv-2d-sk-p0-t ggml-conv-2d-sk-p0)
  (define conv-3d-t ggml-conv-3d)
  (define conv-3d-direct-t ggml-conv-3d-direct)
  (define conv-transpose-1d-t ggml-conv-transpose-1d)
  (define conv-transpose-2d-p0-t ggml-conv-transpose-2d-p0)
  (define cos-t ggml-cos)
  (define cos-inplace-t ggml-cos-inplace)
  (define count-equal-t ggml-count-equal)
  (define cpy-t ggml-cpy)
  (define cross-entropy-loss-t ggml-cross-entropy-loss)
  (define cross-entropy-loss-back-t ggml-cross-entropy-loss-back)
  (define cumsum-t ggml-cumsum)
  (define diag-t ggml-diag)
  (define diag-mask-inf-t ggml-diag-mask-inf)
  (define diag-mask-inf-inplace-t ggml-diag-mask-inf-inplace)
  (define diag-mask-zero-t ggml-diag-mask-zero)
  (define diag-mask-zero-inplace-t ggml-diag-mask-zero-inplace)
  (define div-t ggml-div)
  (define div-inplace-t ggml-div-inplace)
  (define dsv4-hc-comb-t ggml-dsv4-hc-comb)
  (define dsv4-hc-post-t ggml-dsv4-hc-post)
  (define dsv4-hc-pre-t ggml-dsv4-hc-pre)
  (define dup-t ggml-dup)
  (define dup-inplace-t ggml-dup-inplace)
  (define dup-tensor-t ggml-dup-tensor)
  (define elu-t ggml-elu)
  (define elu-inplace-t ggml-elu-inplace)
  (define exp-t ggml-exp)
  (define exp-inplace-t ggml-exp-inplace)
  (define expm1-t ggml-expm1)
  (define expm1-inplace-t ggml-expm1-inplace)
  (define fill-t ggml-fill)
  (define fill-inplace-t ggml-fill-inplace)
  (define flash-attn-back-t ggml-flash-attn-back)
  (define flash-attn-ext-t ggml-flash-attn-ext)
  (define floor-t ggml-floor)
  (define floor-inplace-t ggml-floor-inplace)
  (define gated-delta-net-t ggml-gated-delta-net)
  (define gated-linear-attn-t ggml-gated-linear-attn)
  (define geglu-t ggml-geglu)
  (define geglu-erf-t ggml-geglu-erf)
  (define geglu-erf-split-t ggml-geglu-erf-split)
  (define geglu-erf-swapped-t ggml-geglu-erf-swapped)
  (define geglu-quick-t ggml-geglu-quick)
  (define geglu-quick-split-t ggml-geglu-quick-split)
  (define geglu-quick-swapped-t ggml-geglu-quick-swapped)
  (define geglu-split-t ggml-geglu-split)
  (define geglu-swapped-t ggml-geglu-swapped)
  (define gelu-t ggml-gelu)
  (define gelu-erf-t ggml-gelu-erf)
  (define gelu-erf-inplace-t ggml-gelu-erf-inplace)
  (define gelu-inplace-t ggml-gelu-inplace)
  (define gelu-quick-t ggml-gelu-quick)
  (define gelu-quick-inplace-t ggml-gelu-quick-inplace)
  (define get-first-tensor-t ggml-get-first-tensor)
  (define get-next-tensor-t ggml-get-next-tensor)
  (define get-rel-pos-t ggml-get-rel-pos)
  (define get-rows-t ggml-get-rows)
  (define get-rows-back-t ggml-get-rows-back)
  (define glu-t ggml-glu)
  (define glu-split-t ggml-glu-split)
  (define group-norm-t ggml-group-norm)
  (define group-norm-inplace-t ggml-group-norm-inplace)
  (define hardsigmoid-t ggml-hardsigmoid)
  (define hardswish-t ggml-hardswish)
  (define im2col-t ggml-im2col)
  (define im2col-3d-t ggml-im2col-3d)
  (define interpolate-t ggml-interpolate)
  (define l2-norm-t ggml-l2-norm)
  (define l2-norm-inplace-t ggml-l2-norm-inplace)
  (define leaky-relu-t ggml-leaky-relu)
  (define lightning-indexer-t ggml-lightning-indexer)
  (define log-t ggml-log)
  (define log-inplace-t ggml-log-inplace)
  (define mean-t ggml-mean)
  (define mul-t ggml-mul)
  (define mul-inplace-t ggml-mul-inplace)
  (define mul-mat-t ggml-mul-mat)
  (define mul-mat-id-t ggml-mul-mat-id)
  (define neg-t ggml-neg)
  (define neg-inplace-t ggml-neg-inplace)
  (define new-tensor-1d-t ggml-new-tensor-1d)
  (define new-tensor-2d-t ggml-new-tensor-2d)
  (define new-tensor-3d-t ggml-new-tensor-3d)
  (define new-tensor-4d-t ggml-new-tensor-4d)
  (define norm-t ggml-norm)
  (define norm-inplace-t ggml-norm-inplace)
  (define opt-step-adamw-t ggml-opt-step-adamw)
  (define opt-step-sgd-t ggml-opt-step-sgd)
  (define out-prod-t ggml-out-prod)
  (define pad-t ggml-pad)
  (define pad-circular-t ggml-pad-circular)
  (define pad-ext-t ggml-pad-ext)
  (define pad-ext-circular-t ggml-pad-ext-circular)
  (define pad-reflect-1d-t ggml-pad-reflect-1d)
  (define permute-t ggml-permute)
  (define pool-1d-t ggml-pool-1d)
  (define pool-2d-t ggml-pool-2d)
  (define pool-2d-back-t ggml-pool-2d-back)
  (define reglu-t ggml-reglu)
  (define reglu-split-t ggml-reglu-split)
  (define reglu-swapped-t ggml-reglu-swapped)
  (define relu-t ggml-relu)
  (define relu-inplace-t ggml-relu-inplace)
  (define repeat-t ggml-repeat)
  (define repeat-4d-t ggml-repeat-4d)
  (define repeat-back-t ggml-repeat-back)
  (define reshape-t ggml-reshape)
  (define reshape-1d-t ggml-reshape-1d)
  (define reshape-2d-t ggml-reshape-2d)
  (define reshape-3d-t ggml-reshape-3d)
  (define reshape-4d-t ggml-reshape-4d)
  (define rms-norm-t ggml-rms-norm)
  (define rms-norm-back-t ggml-rms-norm-back)
  (define rms-norm-inplace-t ggml-rms-norm-inplace)
  (define roll-t ggml-roll)
  (define rope-t ggml-rope)
  (define rope-custom-t ggml-rope-custom)
  (define rope-custom-inplace-t ggml-rope-custom-inplace)
  (define rope-ext-t ggml-rope-ext)
  (define rope-ext-back-t ggml-rope-ext-back)
  (define rope-ext-inplace-t ggml-rope-ext-inplace)
  (define rope-inplace-t ggml-rope-inplace)
  (define round-t ggml-round)
  (define round-inplace-t ggml-round-inplace)
  (define rwkv-wkv6-t ggml-rwkv-wkv6)
  (define rwkv-wkv7-t ggml-rwkv-wkv7)
  (define scale-t ggml-scale)
  (define scale-bias-t ggml-scale-bias)
  (define scale-bias-inplace-t ggml-scale-bias-inplace)
  (define scale-inplace-t ggml-scale-inplace)
  (define set-t ggml-set)
  (define set-1d-t ggml-set-1d)
  (define set-1d-inplace-t ggml-set-1d-inplace)
  (define set-2d-t ggml-set-2d)
  (define set-2d-inplace-t ggml-set-2d-inplace)
  (define set-inplace-t ggml-set-inplace)
  (define set-rows-t ggml-set-rows)
  (define sgn-t ggml-sgn)
  (define sgn-inplace-t ggml-sgn-inplace)
  (define sigmoid-t ggml-sigmoid)
  (define sigmoid-inplace-t ggml-sigmoid-inplace)
  (define silu-t ggml-silu)
  (define silu-back-t ggml-silu-back)
  (define silu-inplace-t ggml-silu-inplace)
  (define sin-t ggml-sin)
  (define sin-inplace-t ggml-sin-inplace)
  (define soft-max-t ggml-soft-max)
  (define soft-max-ext-t ggml-soft-max-ext)
  (define soft-max-ext-back-t ggml-soft-max-ext-back)
  (define soft-max-ext-back-inplace-t ggml-soft-max-ext-back-inplace)
  (define soft-max-ext-inplace-t ggml-soft-max-ext-inplace)
  (define soft-max-inplace-t ggml-soft-max-inplace)
  (define softplus-t ggml-softplus)
  (define softplus-inplace-t ggml-softplus-inplace)
  (define solve-tri-t ggml-solve-tri)
  (define sqr-t ggml-sqr)
  (define sqr-inplace-t ggml-sqr-inplace)
  (define sqrt-t ggml-sqrt)
  (define sqrt-inplace-t ggml-sqrt-inplace)
  (define ssm-conv-t ggml-ssm-conv)
  (define ssm-scan-t ggml-ssm-scan)
  (define step-t ggml-step)
  (define step-inplace-t ggml-step-inplace)
  (define sub-t ggml-sub)
  (define sub-inplace-t ggml-sub-inplace)
  (define sum-t ggml-sum)
  (define sum-rows-t ggml-sum-rows)
  (define swiglu-t ggml-swiglu)
  (define swiglu-oai-t ggml-swiglu-oai)
  (define swiglu-split-t ggml-swiglu-split)
  (define swiglu-swapped-t ggml-swiglu-swapped)
  (define tanh-t ggml-tanh)
  (define tanh-inplace-t ggml-tanh-inplace)
  (define timestep-embedding-t ggml-timestep-embedding)
  (define top-k-t ggml-top-k)
  (define transpose-t ggml-transpose)
  (define tri-t ggml-tri)
  (define trunc-t ggml-trunc)
  (define trunc-inplace-t ggml-trunc-inplace)
  (define unary-t ggml-unary)
  (define unary-inplace-t ggml-unary-inplace)
  (define upscale-t ggml-upscale)
  (define upscale-ext-t ggml-upscale-ext)
  (define view-1d-t ggml-view-1d)
  (define view-2d-t ggml-view-2d)
  (define view-3d-t ggml-view-3d)
  (define view-4d-t ggml-view-4d)
  (define view-tensor-t ggml-view-tensor)
  (define win-part-t ggml-win-part)
  (define win-unpart-t ggml-win-unpart)
  (define xielu-t ggml-xielu))
