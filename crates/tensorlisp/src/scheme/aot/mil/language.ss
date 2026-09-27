;;; scheme/aot/mil/language.ss
;;;
;;; The CoreML MIL graph-builder vocabulary -- the first of the passes
;;; under scheme/aot/mil/ (see trace.ss, shadow.ss, compile.ss next to this
;;; file for the highlevel -> MIL nanopass built on top of it). Generated
;;; (see `crates/tensorlisp-aot/examples/gen_mil_language.rs`) from coreml-rs's own
;;; `coreml_rs::mil::graph::MIL_GRAPH_OPS` -- that crate already ports its MIL
;;; builder pattern to Scheme itself (its `build.rs`, from the same
;;; `spec/mil_ops.json` that generates its Rust `MilOp`/`OpSpec` catalog and its
;;; `cg_leafnode::builder::Builder`-based Rust graph functions), so this file is
;;; a thin, regeneratable wrapper, not a hand-written port.
;;;
;;; Each `(mil-<op> <arg> ...)` below takes the op's own declared arguments
;;; positionally (Internal-kind ones, e.g. a `cond`/`while_loop` block body,
;;; are Python-side plumbing and don't appear here) and returns the same
;;; `(<op> (<arg> <val>) ...)` S-expression coreml-rs's Rust
;;; `cg_leafnode::builder::Builder::op` would build for an identical call --
;;; plain graph data, no execution, no validation (real op semantics/type
;;; checking stay `coreml_rs::mil::builder`'s job once a pass hands it a
;;; concrete `Function`). This is `(tl aot highlevel)`'s MIL-side counterpart:
;;; a second nanopass *target* language, next to `reference_compiler.ss`'s
;;; ggml target -- see `core.ss`'s existing (hand-written, symbolic-trace) MIL
;;; pass for how tensorlisp already renders a MIL program from Scheme; a
;;; highlevel -> MIL pass over *this* vocabulary would build the same kind of
;;; IR node this library returns, one node per op, instead of a text trace.
;;;
;;; To regenerate after a coreml-rs op-catalogue change:
;;;   cargo run -p tensorlisp-aot --example gen_mil_language \
;;;     > crates/tensorlisp/src/scheme/aot/mil/language.ss
(library (tl aot mil language)
  (export mil-op-names
          mil-abs mil-acos mil-add mil-affine mil-argsort mil-asin mil-atan
          mil-atanh mil-avg-pool mil-band-part mil-batch-norm
          mil-batch-to-space mil-cast mil-ceil mil-clamped-relu mil-classify
          mil-clip mil-concat mil-cond mil-const
          mil-constexpr-affine-dequantize mil-constexpr-blockwise-shift-scale
          mil-constexpr-cast mil-constexpr-lut-to-dense
          mil-constexpr-lut-to-sparse
          mil-constexpr-sparse-blockwise-shift-scale
          mil-constexpr-sparse-to-dense mil-conv mil-conv-quantized
          mil-conv-transpose mil-cos mil-cosh mil-crop mil-crop-resize
          mil-cumsum mil-depth-to-space mil-dequantize mil-einsum mil-elu
          mil-equal mil-erf mil-exp mil-exp2 mil-expand-dims mil-fill
          mil-fill-like mil-flatten2d mil-floor mil-floor-div mil-gather
          mil-gather-along-axis mil-gather-nd mil-gelu mil-greater
          mil-greater-equal mil-gru mil-identity mil-instance-norm mil-inverse
          mil-l2-norm mil-l2-pool mil-layer-norm mil-leaky-relu mil-less
          mil-less-equal mil-linear mil-linear-activation mil-list-gather
          mil-list-length mil-list-read mil-list-scatter mil-list-write
          mil-local-response-norm mil-log mil-logical-and mil-logical-not
          mil-logical-or mil-logical-xor mil-lstm mil-make-list mil-matmul
          mil-max-pool mil-maximum mil-minimum mil-mod mil-mul
          mil-non-maximum-suppression mil-non-zero mil-not-equal mil-one-hot
          mil-pad mil-pixel-shuffle mil-pixel-unshuffle mil-pow mil-prelu
          mil-quantize mil-random-bernoulli mil-random-categorical
          mil-random-normal mil-random-uniform mil-range-1d mil-read-state
          mil-real-div mil-reduce-argmax mil-reduce-argmin mil-reduce-l1-norm
          mil-reduce-l2-norm mil-reduce-log-sum mil-reduce-log-sum-exp
          mil-reduce-max mil-reduce-mean mil-reduce-min mil-reduce-prod
          mil-reduce-sum mil-reduce-sum-square mil-relu mil-relu6 mil-resample
          mil-reshape mil-reshape-like mil-resize mil-resize-bilinear
          mil-resize-nearest-neighbor mil-reverse mil-reverse-sequence mil-rnn
          mil-round mil-rsqrt mil-scaled-dot-product-attention mil-scaled-tanh
          mil-scatter mil-scatter-along-axis mil-scatter-nd mil-select
          mil-shape mil-sigmoid mil-sigmoid-hard mil-sign mil-silu mil-sin
          mil-sinh mil-slice-by-index mil-slice-by-size mil-slice-update
          mil-sliding-windows mil-softmax mil-softplus mil-softplus-parametric
          mil-softsign mil-space-to-batch mil-space-to-depth mil-split
          mil-sqrt mil-square mil-squeeze mil-stack mil-sub mil-tan mil-tanh
          mil-threshold mil-thresholded-relu mil-tile mil-topk mil-transpose
          mil-upsample-bilinear mil-upsample-nearest-neighbor mil-while-loop
          mil-write-state)
  (import (rnrs))

  (define mil-op-names
    '(     mil-abs mil-acos mil-add mil-affine mil-argsort mil-asin mil-atan
     mil-atanh mil-avg-pool mil-band-part mil-batch-norm mil-batch-to-space
     mil-cast mil-ceil mil-clamped-relu mil-classify mil-clip mil-concat
     mil-cond mil-const mil-constexpr-affine-dequantize
     mil-constexpr-blockwise-shift-scale mil-constexpr-cast
     mil-constexpr-lut-to-dense mil-constexpr-lut-to-sparse
     mil-constexpr-sparse-blockwise-shift-scale
     mil-constexpr-sparse-to-dense mil-conv mil-conv-quantized
     mil-conv-transpose mil-cos mil-cosh mil-crop mil-crop-resize mil-cumsum
     mil-depth-to-space mil-dequantize mil-einsum mil-elu mil-equal mil-erf
     mil-exp mil-exp2 mil-expand-dims mil-fill mil-fill-like mil-flatten2d
     mil-floor mil-floor-div mil-gather mil-gather-along-axis mil-gather-nd
     mil-gelu mil-greater mil-greater-equal mil-gru mil-identity
     mil-instance-norm mil-inverse mil-l2-norm mil-l2-pool mil-layer-norm
     mil-leaky-relu mil-less mil-less-equal mil-linear mil-linear-activation
     mil-list-gather mil-list-length mil-list-read mil-list-scatter
     mil-list-write mil-local-response-norm mil-log mil-logical-and
     mil-logical-not mil-logical-or mil-logical-xor mil-lstm mil-make-list
     mil-matmul mil-max-pool mil-maximum mil-minimum mil-mod mil-mul
     mil-non-maximum-suppression mil-non-zero mil-not-equal mil-one-hot
     mil-pad mil-pixel-shuffle mil-pixel-unshuffle mil-pow mil-prelu
     mil-quantize mil-random-bernoulli mil-random-categorical
     mil-random-normal mil-random-uniform mil-range-1d mil-read-state
     mil-real-div mil-reduce-argmax mil-reduce-argmin mil-reduce-l1-norm
     mil-reduce-l2-norm mil-reduce-log-sum mil-reduce-log-sum-exp
     mil-reduce-max mil-reduce-mean mil-reduce-min mil-reduce-prod
     mil-reduce-sum mil-reduce-sum-square mil-relu mil-relu6 mil-resample
     mil-reshape mil-reshape-like mil-resize mil-resize-bilinear
     mil-resize-nearest-neighbor mil-reverse mil-reverse-sequence mil-rnn
     mil-round mil-rsqrt mil-scaled-dot-product-attention mil-scaled-tanh
     mil-scatter mil-scatter-along-axis mil-scatter-nd mil-select mil-shape
     mil-sigmoid mil-sigmoid-hard mil-sign mil-silu mil-sin mil-sinh
     mil-slice-by-index mil-slice-by-size mil-slice-update
     mil-sliding-windows mil-softmax mil-softplus mil-softplus-parametric
     mil-softsign mil-space-to-batch mil-space-to-depth mil-split mil-sqrt
     mil-square mil-squeeze mil-stack mil-sub mil-tan mil-tanh mil-threshold
     mil-thresholded-relu mil-tile mil-topk mil-transpose
     mil-upsample-bilinear mil-upsample-nearest-neighbor mil-while-loop
     mil-write-state))

(define (mil-abs x) (list 'abs (list 'x x)))
(define (mil-acos x) (list 'acos (list 'x x)))
(define (mil-add x y) (list 'add (list 'x x) (list 'y y)))
(define (mil-affine x transform-matrix output-height output-width sampling-mode padding-mode padding-value coordinates-mode align-corners) (list 'affine (list 'x x) (list 'transform_matrix transform-matrix) (list 'output_height output-height) (list 'output_width output-width) (list 'sampling_mode sampling-mode) (list 'padding_mode padding-mode) (list 'padding_value padding-value) (list 'coordinates_mode coordinates-mode) (list 'align_corners align-corners)))
(define (mil-argsort x axis ascending) (list 'argsort (list 'x x) (list 'axis axis) (list 'ascending ascending)))
(define (mil-asin x) (list 'asin (list 'x x)))
(define (mil-atan x) (list 'atan (list 'x x)))
(define (mil-atanh x) (list 'atanh (list 'x x)))
(define (mil-avg-pool exclude-padding-from-average x kernel-sizes strides pad-type pad ceil-mode) (list 'avg_pool (list 'exclude_padding_from_average exclude-padding-from-average) (list 'x x) (list 'kernel_sizes kernel-sizes) (list 'strides strides) (list 'pad_type pad-type) (list 'pad pad) (list 'ceil_mode ceil-mode)))
(define (mil-band-part x lower upper) (list 'band_part (list 'x x) (list 'lower lower) (list 'upper upper)))
(define (mil-batch-norm x mean variance gamma beta epsilon) (list 'batch_norm (list 'x x) (list 'mean mean) (list 'variance variance) (list 'gamma gamma) (list 'beta beta) (list 'epsilon epsilon)))
(define (mil-batch-to-space x block-shape crops) (list 'batch_to_space (list 'x x) (list 'block_shape block-shape) (list 'crops crops)))
(define (mil-cast x dtype) (list 'cast (list 'x x) (list 'dtype dtype)))
(define (mil-ceil x) (list 'ceil (list 'x x)))
(define (mil-clamped-relu x alpha beta) (list 'clamped_relu (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-classify probabilities classes) (list 'classify (list 'probabilities probabilities) (list 'classes classes)))
(define (mil-clip x alpha beta) (list 'clip (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-concat values axis interleave) (list 'concat (list 'values values) (list 'axis axis) (list 'interleave interleave)))
(define (mil-cond pred) (list 'cond (list 'pred pred)))
(define (mil-const ) (list 'const))
(define (mil-constexpr-affine-dequantize quantized-data zero-point scale axis) (list 'constexpr_affine_dequantize (list 'quantized_data quantized-data) (list 'zero_point zero-point) (list 'scale scale) (list 'axis axis)))
(define (mil-constexpr-blockwise-shift-scale data scale offset) (list 'constexpr_blockwise_shift_scale (list 'data data) (list 'scale scale) (list 'offset offset)))
(define (mil-constexpr-cast source-val output-dtype) (list 'constexpr_cast (list 'source_val source-val) (list 'output_dtype output-dtype)))
(define (mil-constexpr-lut-to-dense indices lut vector-axis) (list 'constexpr_lut_to_dense (list 'indices indices) (list 'lut lut) (list 'vector_axis vector-axis)))
(define (mil-constexpr-lut-to-sparse indices-mask indices-nonzero-data lut vector-axis) (list 'constexpr_lut_to_sparse (list 'indices_mask indices-mask) (list 'indices_nonzero_data indices-nonzero-data) (list 'lut lut) (list 'vector_axis vector-axis)))
(define (mil-constexpr-sparse-blockwise-shift-scale data-mask nonzero-data scale offset) (list 'constexpr_sparse_blockwise_shift_scale (list 'data_mask data-mask) (list 'nonzero_data nonzero-data) (list 'scale scale) (list 'offset offset)))
(define (mil-constexpr-sparse-to-dense nonzero-data mask) (list 'constexpr_sparse_to_dense (list 'nonzero_data nonzero-data) (list 'mask mask)))
(define (mil-conv x weight bias strides pad-type pad dilations groups) (list 'conv (list 'x x) (list 'weight weight) (list 'bias bias) (list 'strides strides) (list 'pad_type pad-type) (list 'pad pad) (list 'dilations dilations) (list 'groups groups)))
(define (mil-conv-quantized x weight bias quantization-type nbits quant-scale quant-bias strides pad-type pad dilations groups) (list 'conv_quantized (list 'x x) (list 'weight weight) (list 'bias bias) (list 'quantization_type quantization-type) (list 'nbits nbits) (list 'quant_scale quant-scale) (list 'quant_bias quant-bias) (list 'strides strides) (list 'pad_type pad-type) (list 'pad pad) (list 'dilations dilations) (list 'groups groups)))
(define (mil-conv-transpose x weight bias pad output-shape pad-type strides dilations groups) (list 'conv_transpose (list 'x x) (list 'weight weight) (list 'bias bias) (list 'pad pad) (list 'output_shape output-shape) (list 'pad_type pad-type) (list 'strides strides) (list 'dilations dilations) (list 'groups groups)))
(define (mil-cos x) (list 'cos (list 'x x)))
(define (mil-cosh x) (list 'cosh (list 'x x)))
(define (mil-crop x crop-height crop-width) (list 'crop (list 'x x) (list 'crop_height crop-height) (list 'crop_width crop-width)))
(define (mil-crop-resize x boxes box-indices target-height target-width normalized-coordinates spatial-scale box-coordinate-mode sampling-mode pad-value) (list 'crop_resize (list 'x x) (list 'boxes boxes) (list 'box_indices box-indices) (list 'target_height target-height) (list 'target_width target-width) (list 'normalized_coordinates normalized-coordinates) (list 'spatial_scale spatial-scale) (list 'box_coordinate_mode box-coordinate-mode) (list 'sampling_mode sampling-mode) (list 'pad_value pad-value)))
(define (mil-cumsum x axis exclusive reverse) (list 'cumsum (list 'x x) (list 'axis axis) (list 'exclusive exclusive) (list 'reverse reverse)))
(define (mil-depth-to-space x block-size) (list 'depth_to_space (list 'x x) (list 'block_size block-size)))
(define (mil-dequantize input zero-point scale axis) (list 'dequantize (list 'input input) (list 'zero_point zero-point) (list 'scale scale) (list 'axis axis)))
(define (mil-einsum values equation) (list 'einsum (list 'values values) (list 'equation equation)))
(define (mil-elu x alpha) (list 'elu (list 'x x) (list 'alpha alpha)))
(define (mil-equal x y) (list 'equal (list 'x x) (list 'y y)))
(define (mil-erf x) (list 'erf (list 'x x)))
(define (mil-exp x) (list 'exp (list 'x x)))
(define (mil-exp2 x) (list 'exp2 (list 'x x)))
(define (mil-expand-dims x axes) (list 'expand_dims (list 'x x) (list 'axes axes)))
(define (mil-fill shape value) (list 'fill (list 'shape shape) (list 'value value)))
(define (mil-fill-like ref-tensor value) (list 'fill_like (list 'ref_tensor ref-tensor) (list 'value value)))
(define (mil-flatten2d x axis) (list 'flatten2d (list 'x x) (list 'axis axis)))
(define (mil-floor x) (list 'floor (list 'x x)))
(define (mil-floor-div x y) (list 'floor_div (list 'x x) (list 'y y)))
(define (mil-gather x indices axis batch-dims validate-indices) (list 'gather (list 'x x) (list 'indices indices) (list 'axis axis) (list 'batch_dims batch-dims) (list 'validate_indices validate-indices)))
(define (mil-gather-along-axis x indices axis validate-indices) (list 'gather_along_axis (list 'x x) (list 'indices indices) (list 'axis axis) (list 'validate_indices validate-indices)))
(define (mil-gather-nd x indices batch-dims validate-indices) (list 'gather_nd (list 'x x) (list 'indices indices) (list 'batch_dims batch-dims) (list 'validate_indices validate-indices)))
(define (mil-gelu x mode) (list 'gelu (list 'x x) (list 'mode mode)))
(define (mil-greater x y) (list 'greater (list 'x x) (list 'y y)))
(define (mil-greater-equal x y) (list 'greater_equal (list 'x x) (list 'y y)))
(define (mil-gru x initial-h weight-ih weight-hh bias direction output-sequence recurrent-activation activation reset-after input-bias) (list 'gru (list 'x x) (list 'initial_h initial-h) (list 'weight_ih weight-ih) (list 'weight_hh weight-hh) (list 'bias bias) (list 'direction direction) (list 'output_sequence output-sequence) (list 'recurrent_activation recurrent-activation) (list 'activation activation) (list 'reset_after reset-after) (list 'input_bias input-bias)))
(define (mil-identity x) (list 'identity (list 'x x)))
(define (mil-instance-norm x gamma beta epsilon) (list 'instance_norm (list 'x x) (list 'gamma gamma) (list 'beta beta) (list 'epsilon epsilon)))
(define (mil-inverse x epsilon) (list 'inverse (list 'x x) (list 'epsilon epsilon)))
(define (mil-l2-norm x epsilon) (list 'l2_norm (list 'x x) (list 'epsilon epsilon)))
(define (mil-l2-pool x kernel-sizes strides pad-type pad ceil-mode) (list 'l2_pool (list 'x x) (list 'kernel_sizes kernel-sizes) (list 'strides strides) (list 'pad_type pad-type) (list 'pad pad) (list 'ceil_mode ceil-mode)))
(define (mil-layer-norm x axes gamma beta epsilon) (list 'layer_norm (list 'x x) (list 'axes axes) (list 'gamma gamma) (list 'beta beta) (list 'epsilon epsilon)))
(define (mil-leaky-relu x alpha) (list 'leaky_relu (list 'x x) (list 'alpha alpha)))
(define (mil-less x y) (list 'less (list 'x x) (list 'y y)))
(define (mil-less-equal x y) (list 'less_equal (list 'x x) (list 'y y)))
(define (mil-linear x weight bias) (list 'linear (list 'x x) (list 'weight weight) (list 'bias bias)))
(define (mil-linear-activation x alpha beta) (list 'linear_activation (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-list-gather ls indices) (list 'list_gather (list 'ls ls) (list 'indices indices)))
(define (mil-list-length ls) (list 'list_length (list 'ls ls)))
(define (mil-list-read ls index) (list 'list_read (list 'ls ls) (list 'index index)))
(define (mil-list-scatter ls indices value) (list 'list_scatter (list 'ls ls) (list 'indices indices) (list 'value value)))
(define (mil-list-write ls index value) (list 'list_write (list 'ls ls) (list 'index index) (list 'value value)))
(define (mil-local-response-norm x size alpha beta k) (list 'local_response_norm (list 'x x) (list 'size size) (list 'alpha alpha) (list 'beta beta) (list 'k k)))
(define (mil-log x epsilon) (list 'log (list 'x x) (list 'epsilon epsilon)))
(define (mil-logical-and x y) (list 'logical_and (list 'x x) (list 'y y)))
(define (mil-logical-not x) (list 'logical_not (list 'x x)))
(define (mil-logical-or x y) (list 'logical_or (list 'x x) (list 'y y)))
(define (mil-logical-xor x y) (list 'logical_xor (list 'x x) (list 'y y)))
(define (mil-lstm x initial-h initial-c weight-ih weight-hh bias peephole weight-ih-back weight-hh-back bias-back peephole-back direction output-sequence recurrent-activation cell-activation activation clip) (list 'lstm (list 'x x) (list 'initial_h initial-h) (list 'initial_c initial-c) (list 'weight_ih weight-ih) (list 'weight_hh weight-hh) (list 'bias bias) (list 'peephole peephole) (list 'weight_ih_back weight-ih-back) (list 'weight_hh_back weight-hh-back) (list 'bias_back bias-back) (list 'peephole_back peephole-back) (list 'direction direction) (list 'output_sequence output-sequence) (list 'recurrent_activation recurrent-activation) (list 'cell_activation cell-activation) (list 'activation activation) (list 'clip clip)))
(define (mil-make-list init-length dynamic-length elem-shape dtype) (list 'make_list (list 'init_length init-length) (list 'dynamic_length dynamic-length) (list 'elem_shape elem-shape) (list 'dtype dtype)))
(define (mil-matmul x y transpose-x transpose-y) (list 'matmul (list 'x x) (list 'y y) (list 'transpose_x transpose-x) (list 'transpose_y transpose-y)))
(define (mil-max-pool x kernel-sizes strides pad-type pad ceil-mode) (list 'max_pool (list 'x x) (list 'kernel_sizes kernel-sizes) (list 'strides strides) (list 'pad_type pad-type) (list 'pad pad) (list 'ceil_mode ceil-mode)))
(define (mil-maximum x y) (list 'maximum (list 'x x) (list 'y y)))
(define (mil-minimum x y) (list 'minimum (list 'x x) (list 'y y)))
(define (mil-mod x y) (list 'mod (list 'x x) (list 'y y)))
(define (mil-mul x y) (list 'mul (list 'x x) (list 'y y)))
(define (mil-non-maximum-suppression boxes scores iou-threshold max-boxes per-class-suppression) (list 'non_maximum_suppression (list 'boxes boxes) (list 'scores scores) (list 'iou_threshold iou-threshold) (list 'max_boxes max-boxes) (list 'per_class_suppression per-class-suppression)))
(define (mil-non-zero x) (list 'non_zero (list 'x x)))
(define (mil-not-equal x y) (list 'not_equal (list 'x x) (list 'y y)))
(define (mil-one-hot indices one-hot-vector-size axis on-value off-value) (list 'one_hot (list 'indices indices) (list 'one_hot_vector_size one-hot-vector-size) (list 'axis axis) (list 'on_value on-value) (list 'off_value off-value)))
(define (mil-pad x pad mode constant-val) (list 'pad (list 'x x) (list 'pad pad) (list 'mode mode) (list 'constant_val constant-val)))
(define (mil-pixel-shuffle x upscale-factor) (list 'pixel_shuffle (list 'x x) (list 'upscale_factor upscale-factor)))
(define (mil-pixel-unshuffle x downscale-factor) (list 'pixel_unshuffle (list 'x x) (list 'downscale_factor downscale-factor)))
(define (mil-pow x y) (list 'pow (list 'x x) (list 'y y)))
(define (mil-prelu x alpha) (list 'prelu (list 'x x) (list 'alpha alpha)))
(define (mil-quantize input zero-point scale axis output-dtype) (list 'quantize (list 'input input) (list 'zero_point zero-point) (list 'scale scale) (list 'axis axis) (list 'output_dtype output-dtype)))
(define (mil-random-bernoulli shape prob seed) (list 'random_bernoulli (list 'shape shape) (list 'prob prob) (list 'seed seed)))
(define (mil-random-categorical x mode size seed) (list 'random_categorical (list 'x x) (list 'mode mode) (list 'size size) (list 'seed seed)))
(define (mil-random-normal shape mean stddev seed) (list 'random_normal (list 'shape shape) (list 'mean mean) (list 'stddev stddev) (list 'seed seed)))
(define (mil-random-uniform shape low high seed) (list 'random_uniform (list 'shape shape) (list 'low low) (list 'high high) (list 'seed seed)))
(define (mil-range-1d end start step) (list 'range_1d (list 'end end) (list 'start start) (list 'step step)))
(define (mil-read-state input) (list 'read_state (list 'input input)))
(define (mil-real-div x y) (list 'real_div (list 'x x) (list 'y y)))
(define (mil-reduce-argmax x axis keep-dims output-dtype) (list 'reduce_argmax (list 'x x) (list 'axis axis) (list 'keep_dims keep-dims) (list 'output_dtype output-dtype)))
(define (mil-reduce-argmin x axis keep-dims output-dtype) (list 'reduce_argmin (list 'x x) (list 'axis axis) (list 'keep_dims keep-dims) (list 'output_dtype output-dtype)))
(define (mil-reduce-l1-norm x axes keep-dims) (list 'reduce_l1_norm (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-l2-norm x axes keep-dims) (list 'reduce_l2_norm (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-log-sum x axes keep-dims) (list 'reduce_log_sum (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-log-sum-exp x axes keep-dims) (list 'reduce_log_sum_exp (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-max x axes keep-dims) (list 'reduce_max (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-mean x axes keep-dims) (list 'reduce_mean (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-min x axes keep-dims) (list 'reduce_min (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-prod x axes keep-dims) (list 'reduce_prod (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-sum x axes keep-dims) (list 'reduce_sum (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-reduce-sum-square x axes keep-dims) (list 'reduce_sum_square (list 'x x) (list 'axes axes) (list 'keep_dims keep-dims)))
(define (mil-relu x) (list 'relu (list 'x x)))
(define (mil-relu6 x) (list 'relu6 (list 'x x)))
(define (mil-resample x coordinates sampling-mode padding-mode padding-value coordinates-mode align-corners) (list 'resample (list 'x x) (list 'coordinates coordinates) (list 'sampling_mode sampling-mode) (list 'padding_mode padding-mode) (list 'padding_value padding-value) (list 'coordinates_mode coordinates-mode) (list 'align_corners align-corners)))
(define (mil-reshape x shape) (list 'reshape (list 'x x) (list 'shape shape)))
(define (mil-reshape-like x ref-tensors begins ends end-masks) (list 'reshape_like (list 'x x) (list 'ref_tensors ref-tensors) (list 'begins begins) (list 'ends ends) (list 'end_masks end-masks)))
(define (mil-resize x shape resized-dims interpolation-mode sampling-mode) (list 'resize (list 'x x) (list 'shape shape) (list 'resized_dims resized-dims) (list 'interpolation_mode interpolation-mode) (list 'sampling_mode sampling-mode)))
(define (mil-resize-bilinear x target-size-height target-size-width sampling-mode) (list 'resize_bilinear (list 'x x) (list 'target_size_height target-size-height) (list 'target_size_width target-size-width) (list 'sampling_mode sampling-mode)))
(define (mil-resize-nearest-neighbor x target-size-height target-size-width) (list 'resize_nearest_neighbor (list 'x x) (list 'target_size_height target-size-height) (list 'target_size_width target-size-width)))
(define (mil-reverse x axes) (list 'reverse (list 'x x) (list 'axes axes)))
(define (mil-reverse-sequence x lengths seq-axis batch-axis) (list 'reverse_sequence (list 'x x) (list 'lengths lengths) (list 'seq_axis seq-axis) (list 'batch_axis batch-axis)))
(define (mil-rnn x initial-h weight-ih weight-hh bias direction output-sequence activation) (list 'rnn (list 'x x) (list 'initial_h initial-h) (list 'weight_ih weight-ih) (list 'weight_hh weight-hh) (list 'bias bias) (list 'direction direction) (list 'output_sequence output-sequence) (list 'activation activation)))
(define (mil-round x) (list 'round (list 'x x)))
(define (mil-rsqrt x epsilon) (list 'rsqrt (list 'x x) (list 'epsilon epsilon)))
(define (mil-scaled-dot-product-attention query key value attn-mask) (list 'scaled_dot_product_attention (list 'query query) (list 'key key) (list 'value value) (list 'attn_mask attn-mask)))
(define (mil-scaled-tanh x alpha beta) (list 'scaled_tanh (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-scatter data indices updates axis mode validate-indices) (list 'scatter (list 'data data) (list 'indices indices) (list 'updates updates) (list 'axis axis) (list 'mode mode) (list 'validate_indices validate-indices)))
(define (mil-scatter-along-axis data indices updates axis mode validate-indices) (list 'scatter_along_axis (list 'data data) (list 'indices indices) (list 'updates updates) (list 'axis axis) (list 'mode mode) (list 'validate_indices validate-indices)))
(define (mil-scatter-nd data indices updates mode validate-indices) (list 'scatter_nd (list 'data data) (list 'indices indices) (list 'updates updates) (list 'mode mode) (list 'validate_indices validate-indices)))
(define (mil-select cond a b) (list 'select (list 'cond cond) (list 'a a) (list 'b b)))
(define (mil-shape x) (list 'shape (list 'x x)))
(define (mil-sigmoid x) (list 'sigmoid (list 'x x)))
(define (mil-sigmoid-hard x alpha beta) (list 'sigmoid_hard (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-sign x) (list 'sign (list 'x x)))
(define (mil-silu x) (list 'silu (list 'x x)))
(define (mil-sin x) (list 'sin (list 'x x)))
(define (mil-sinh x) (list 'sinh (list 'x x)))
(define (mil-slice-by-index x begin end stride begin-mask end-mask squeeze-mask) (list 'slice_by_index (list 'x x) (list 'begin begin) (list 'end end) (list 'stride stride) (list 'begin_mask begin-mask) (list 'end_mask end-mask) (list 'squeeze_mask squeeze-mask)))
(define (mil-slice-by-size x begin size) (list 'slice_by_size (list 'x x) (list 'begin begin) (list 'size size)))
(define (mil-slice-update x update begin end stride begin-mask end-mask squeeze-mask) (list 'slice_update (list 'x x) (list 'update update) (list 'begin begin) (list 'end end) (list 'stride stride) (list 'begin_mask begin-mask) (list 'end_mask end-mask) (list 'squeeze_mask squeeze-mask)))
(define (mil-sliding-windows x axis size stride) (list 'sliding_windows (list 'x x) (list 'axis axis) (list 'size size) (list 'stride stride)))
(define (mil-softmax x axis) (list 'softmax (list 'x x) (list 'axis axis)))
(define (mil-softplus x) (list 'softplus (list 'x x)))
(define (mil-softplus-parametric x alpha beta) (list 'softplus_parametric (list 'x x) (list 'alpha alpha) (list 'beta beta)))
(define (mil-softsign x) (list 'softsign (list 'x x)))
(define (mil-space-to-batch x block-shape paddings) (list 'space_to_batch (list 'x x) (list 'block_shape block-shape) (list 'paddings paddings)))
(define (mil-space-to-depth x block-size) (list 'space_to_depth (list 'x x) (list 'block_size block-size)))
(define (mil-split x num-splits split-sizes axis) (list 'split (list 'x x) (list 'num_splits num-splits) (list 'split_sizes split-sizes) (list 'axis axis)))
(define (mil-sqrt x) (list 'sqrt (list 'x x)))
(define (mil-square x) (list 'square (list 'x x)))
(define (mil-squeeze x axes) (list 'squeeze (list 'x x) (list 'axes axes)))
(define (mil-stack values axis) (list 'stack (list 'values values) (list 'axis axis)))
(define (mil-sub x y) (list 'sub (list 'x x) (list 'y y)))
(define (mil-tan x) (list 'tan (list 'x x)))
(define (mil-tanh x) (list 'tanh (list 'x x)))
(define (mil-threshold x alpha) (list 'threshold (list 'x x) (list 'alpha alpha)))
(define (mil-thresholded-relu x alpha) (list 'thresholded_relu (list 'x x) (list 'alpha alpha)))
(define (mil-tile x reps) (list 'tile (list 'x x) (list 'reps reps)))
(define (mil-topk x k axis ascending sort return-indices output-indices-dtype) (list 'topk (list 'x x) (list 'k k) (list 'axis axis) (list 'ascending ascending) (list 'sort sort) (list 'return_indices return-indices) (list 'output_indices_dtype output-indices-dtype)))
(define (mil-transpose x perm) (list 'transpose (list 'x x) (list 'perm perm)))
(define (mil-upsample-bilinear x scale-factor-height scale-factor-width align-corners half-pixel-centers) (list 'upsample_bilinear (list 'x x) (list 'scale_factor_height scale-factor-height) (list 'scale_factor_width scale-factor-width) (list 'align_corners align-corners) (list 'half_pixel_centers half-pixel-centers)))
(define (mil-upsample-nearest-neighbor x scale-factor-height scale-factor-width) (list 'upsample_nearest_neighbor (list 'x x) (list 'scale_factor_height scale-factor-height) (list 'scale_factor_width scale-factor-width)))
(define (mil-while-loop loop-vars) (list 'while_loop (list 'loop_vars loop-vars)))
(define (mil-write-state input data) (list 'write_state (list 'input input) (list 'data data))))
