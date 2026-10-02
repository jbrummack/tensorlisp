//! Lowers a tensorlisp ggml reference graph into a MIL [`Function`], node by
//! node, using [`tensorlisp::GraphInfo`] (built from the same ggml graph the
//! CPU/Metal/CUDA backends actually run) as ground truth: every op, source
//! wiring and output shape here comes directly from what ggml already
//! computed, not from re-deriving it, so a bad lowering can't quietly
//! disagree with the reference.
//!
//! This is the CoreML MIL pass of tensorlisp's nanopass AOT compiler: model
//! source is written against `(tl generic)` (see
//! `../../crates/tensorlisp/src/scheme/stdlib/generic.ss`), an unprefixed
//! alias/wrapper vocabulary over ggml/`(tl nn)`/`(tl tensor)` ops that
//! always executes exactly like the underlying op it wraps (ggml is still
//! the only backend that ever actually runs a graph -- that's the
//! reference/testing pass, and it's "free": aliasing means a generic-op
//! program already *is* a ggml program). This module is the other pass:
//! [`LOWER_PASS`] maps each ggml op name `(tl generic)` is designed to
//! produce to a MIL lowering function, registered one entry at a time
//! (`register_*` below) instead of matched in a growing statement -- see
//! [`cg_leafnode::nanopass::PassTable`].
//!
//! Proven end to end (`GraphInfo` -> `mil::Function` -> serialized
//! `spec::Model` -> run on ANE via `coreml_rs::runtime`) on tensorlisp's
//! `mlp` test fixture (`MUL_MAT`/`ADD`/`RELU` only) and, since, on real
//! yolo11n taps for every op below. [`lower_until`] lowers only a *prefix*
//! of the graph ending at a named tap, so each newly-added op can be
//! checked against that tap's real CPU value before trusting it in the
//! full model -- the same "small, checkable slice first" approach the
//! `mlp` proof used, applied per op instead of per model.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use cg_leafnode::nanopass::PassTable;
use coreml_rs::mil::{DataType, Function, MilOp, OpBuilder, Opset, TensorType, Value, Var};
use half::f16;
use tensorlisp::gguf::{GgufFile, TensorInfo};
use tensorlisp::{DType, GraphInfo, NodeInfo};

/// The dtype every intermediate/weight tensor in a lowered MIL graph is
/// built in. Declared model inputs/outputs stay `Float32` regardless (see
/// `build`'s own cast-at-the-boundary logic), so nothing calling into a
/// lowered model needs to change -- this only affects what CoreML's
/// compiler sees *inside* the graph, which is what actually decides
/// whether it'll place ops on the ANE (fp32 graphs were observed falling
/// back to CPU across every model in this crate's own test suite; the ANE
/// overwhelmingly prefers fp16).
const DTYPE: DataType = DataType::Float16;

#[derive(Debug, thiserror::Error)]
pub enum AotError {
    #[error("node {index} ({op}): unsupported ggml op")]
    UnsupportedOp { index: usize, op: String },
    #[error("node {index} ({op}): needs source {want}, has {got}")]
    SourceCount { index: usize, op: String, want: usize, got: usize },
    #[error("no weight, input or prior node named {0:?}")]
    UnknownSource(String),
    #[error("weight {0:?} is not f32 (AOT lowering only handles f32 so far)")]
    NonFloatWeight(String),
    #[error("no node found for {0:?}")]
    UnknownOutput(String),
    #[error(transparent)]
    Mil(#[from] coreml_rs::mil::MilError),
    #[error(transparent)]
    Gguf(#[from] tensorlisp::Error),
}

/// Lowers `info` (from [`tensorlisp::Model::graph`] on the same GGUF at
/// `gguf_path`) into a MIL `Function` for `opset`. `inputs` gives each
/// declared input's fixed MIL shape (ndarray order) -- MIL functions need
/// fixed shapes, unlike ggml's per-call graph rebuilding.
///
/// Returns the function plus a declared-name -> actual-MIL-output-name map:
/// MIL identifiers can't contain `.`, so a tap like `layer.00` is exposed
/// under a sanitized name (see [`sanitize_ident`]), which a caller reading
/// the serialized model's outputs needs to look up by.
pub fn lower_graph(
    gguf_path: &Path,
    info: &GraphInfo,
    inputs: &[(&str, Vec<u64>)],
    opset: Opset,
) -> Result<(Function, HashMap<String, String>), AotError> {
    // `GraphInfo` gives an output's name and shape but not which node
    // produces it -- ggml's cgraph output tensors aren't necessarily named
    // (tensorlisp tracks them by pointer internally, not by ggml name), so
    // there is no name to match on in general. ggml still names a node
    // whenever *something else* (an input, weight or tap) already had a
    // name and this node further wraps it (appending " (reshaped)" etc.),
    // so try that first; a single, unnamed output falls back to the last
    // node in the graph. A model with several unnamed outputs would need
    // `tensorlisp::GraphInfo` to expose the producing node index directly
    // (a small upstream change, not attempted here).
    let outputs: Vec<(String, usize)> = info
        .outputs
        .iter()
        .map(|(name, _)| {
            let idx = match find_named_node(&info.nodes, name) {
                Some(i) => i,
                None if info.outputs.len() == 1 => {
                    info.nodes.len().checked_sub(1).ok_or_else(|| AotError::UnknownOutput(name.clone()))?
                }
                None => return Err(AotError::UnknownOutput(name.clone())),
            };
            Ok((name.clone(), idx))
        })
        .collect::<Result<_, AotError>>()?;

    let indices: Vec<usize> = (0..info.nodes.len()).collect();
    build(gguf_path, &info.nodes, &indices, inputs, opset, &outputs)
}

/// Lowers only the prefix of `info`'s graph up to (and including) the node
/// named `tap` -- e.g. one of yolo11n's `layer.NN` taps -- exposing it as
/// the function's sole output. Lets a new op be checked against that tap's
/// real value from a CPU run without needing every op the rest of the
/// graph uses to be implemented yet.
pub fn lower_until(
    gguf_path: &Path,
    info: &GraphInfo,
    inputs: &[(&str, Vec<u64>)],
    opset: Opset,
    tap: &str,
) -> Result<(Function, HashMap<String, String>), AotError> {
    let idx = find_named_node(&info.nodes, tap).ok_or_else(|| AotError::UnknownOutput(tap.to_string()))?;
    let indices = dependency_closure(&info.nodes, idx);
    build(gguf_path, &info.nodes, &indices, inputs, opset, &[(tap.to_string(), idx)])
}

/// `target` plus every node it transitively depends on via `%N` source
/// references, ascending. A raw index prefix (`0..=target`) would also drag
/// in nodes `target` doesn't actually depend on -- ggml's node order is *a*
/// valid topological order for the whole graph, not necessarily one where
/// unrelated subgraphs (e.g. yolo11n's anchor-grid `ARANGE`, independent of
/// the image input) sort after everything a given tap needs.
fn dependency_closure(nodes: &[NodeInfo], target: usize) -> Vec<usize> {
    let mut needed = std::collections::HashSet::new();
    let mut stack = vec![target];
    while let Some(i) = stack.pop() {
        if !needed.insert(i) {
            continue;
        }
        for src in &nodes[i].srcs {
            if let Some(rest) = src.strip_prefix('%')
                && let Ok(j) = rest.parse::<usize>()
            {
                stack.push(j);
            }
        }
    }
    let mut indices: Vec<usize> = needed.into_iter().collect();
    indices.sort_unstable();
    indices
}

/// MIL identifiers are `[A-Za-z_][A-Za-z0-9_@]*`; ggml/tensorlisp names
/// (`fc1.weight`, `layer.00`, `0.conv.weight`) often aren't -- notably,
/// several yolo11n weight/layer names start with a digit.
fn sanitize_ident(s: &str) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    match cleaned.chars().next() {
        Some(c) if c.is_ascii_digit() => format!("_{cleaned}"),
        _ => cleaned,
    }
}

/// The most-wrapped (last) node whose name is `name` or starts with
/// `"{name} ("` (ggml appends " (reshaped)"/" (cont)"/... each time it
/// wraps an already-named tensor further).
fn find_named_node(nodes: &[NodeInfo], name: &str) -> Option<usize> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name == name || n.name.starts_with(&format!("{name} (")))
        .map(|(i, _)| i)
        .next_back()
}

fn build(
    gguf_path: &Path,
    nodes: &[NodeInfo],
    indices: &[usize],
    inputs: &[(&str, Vec<u64>)],
    opset: Opset,
    outputs: &[(String, usize)],
) -> Result<(Function, HashMap<String, String>), AotError> {
    let file = GgufFile::open(gguf_path)?;
    let infos = file.tensor_infos();

    if std::env::var("TENSORLISP_AOT_DEBUG").is_ok() {
        for &i in indices {
            let n = &nodes[i];
            eprintln!("{i}: op={:?} name={:?} ne={:?} op_params={:?} srcs={:?}", n.op, n.name, n.ne, &n.op_params[..8], n.srcs);
        }
    }

    let mut func = Function::new(opset);
    let mut named: HashMap<String, Var> = HashMap::new();
    for (name, shape) in inputs {
        // The declared input stays fp32 (so nothing calling into this
        // model needs to change), cast to fp16 immediately -- every
        // lowering function beyond this point builds `DTYPE` (fp16)
        // tensors, so this is the one and only fp32->fp16 conversion.
        let ty = TensorType::new(DataType::Float32, shape.iter().copied());
        let raw = func.input(name, ty)?;
        let fp16_ty = TensorType::new(DTYPE, shape.iter().copied());
        let cast = func.op(MilOp::Cast).input("x", &raw).input("dtype", "fp16").output(fp16_ty).build()?;
        named.insert((*name).to_string(), cast);
    }

    let mut node_vars: HashMap<usize, Var> = HashMap::with_capacity(indices.len());
    for &index in indices {
        let node = &nodes[index];
        let var = lower_node(index, node, None, &node_vars, &mut named, &file, gguf_path, &infos, &mut func)?;
        node_vars.insert(index, var);
    }

    // Declared outputs are the other fp16->fp32 boundary; each output's
    // *name* (what `find_named_node`/callers address it by) now comes from
    // this cast, not from the underlying (still fp16, unnamed) node the
    // way a single `forced_name` on `lower_node` used to set it directly.
    let mut output_vars: Vec<Var> = Vec::with_capacity(outputs.len());
    let mut output_names: HashMap<String, String> = HashMap::with_capacity(outputs.len());
    for (name, idx) in outputs {
        let sanitized = sanitize_ident(name);
        let fp16_var = &node_vars[idx];
        let dims = fp16_var.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnknownOutput(name.clone()))?;
        let fp32_ty = TensorType::new(DataType::Float32, dims);
        let cast = func.op(MilOp::Cast).input("x", fp16_var).input("dtype", "fp32").name(sanitized.clone()).output(fp32_ty).build()?;
        output_vars.push(cast);
        output_names.insert(name.clone(), sanitized);
    }
    func.set_outputs(&output_vars)?;

    Ok((func, output_names))
}

/// One node's worth of context handed to a registered lowering function:
/// everything [`resolve`] needs to turn a `srcs` entry into a [`Var`], plus
/// the node itself and the [`Function`] being built. Taking this by value
/// (not `&mut`) lets a lowering fn return an [`OpBuilder`] that still
/// borrows `func` with the ctx's own lifetime, rather than one shortened by
/// going through an intermediate `&mut LowerCtx` reference.
struct LowerCtx<'a> {
    index: usize,
    node: &'a NodeInfo,
    node_vars: &'a HashMap<usize, Var>,
    named: &'a mut HashMap<String, Var>,
    file: &'a GgufFile,
    path: &'a Path,
    infos: &'a [TensorInfo],
    func: &'a mut Function,
}

impl<'a> LowerCtx<'a> {
    fn src(&mut self, i: usize) -> Result<Var, AotError> {
        let s = self
            .node
            .srcs
            .get(i)
            .ok_or_else(|| AotError::SourceCount {
                index: self.index,
                op: self.node.op.clone(),
                want: i + 1,
                got: self.node.srcs.len(),
            })?
            .clone();
        resolve(&s, self.node_vars, self.named, self.file, self.path, self.infos, self.func)
    }
}

/// A lowering function: takes one node's context, returns an unfinished
/// [`OpBuilder`] (the caller applies the declared output's forced name, if
/// any, and calls `.build()`). Registered per ggml op name in [`LOWER_PASS`].
type LowerFn = for<'a> fn(LowerCtx<'a>) -> Result<OpBuilder<'a>, AotError>;

fn lower_binary(mut ctx: LowerCtx<'_>, op: MilOp) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let y = ctx.src(1)?;
    // ggml_n_dims (and so `node.ne`) drops trailing size-1 dims (e.g. a
    // batch of 1), but MIL still computes and expects the *true* rank of
    // its actual (broadcast) inputs -- pad the declared output back up to
    // that rank rather than trusting `node.ne`'s length verbatim, or a
    // later op/output feature whose declared shape this feeds into won't
    // match what Core ML actually produces.
    let ty = ty_like(ctx.node, rank_of(&x).max(rank_of(&y)));
    Ok(ctx.func.op(op).input("x", &x).input("y", &y).output(ty))
}
fn lower_add(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_binary(ctx, MilOp::Add)
}
fn lower_sub(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_binary(ctx, MilOp::Sub)
}
fn lower_mul(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_binary(ctx, MilOp::Mul)
}

fn lower_unary(mut ctx: LowerCtx<'_>, op: MilOp) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let ty = ty_like(ctx.node, rank_of(&x));
    Ok(ctx.func.op(op).input("x", &x).output(ty))
}
fn lower_relu(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_unary(ctx, MilOp::Relu)
}
fn lower_silu(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_unary(ctx, MilOp::Silu)
}
fn lower_sigmoid(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    lower_unary(ctx, MilOp::Sigmoid)
}

/// `ggml_gelu_erf` (the exact, erf-based GELU -- the only GELU variant
/// either tipsv2's or ppocrv6's `nn:gelu` calls ever use; plain
/// `GELU`/`GELU_QUICK`, the tanh/sigmoid approximations, don't appear).
/// MIL's `gelu` op takes an explicit `mode`; `"EXACT"` is the erf form.
fn lower_gelu_erf(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let ty = ty_like(ctx.node, rank_of(&x));
    let op: MilOp = "gelu".parse().map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "gelu missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(op).input("x", &x).input("mode", "EXACT").output(ty))
}

/// `ggml_hardsigmoid`: `min(1, max(0, (x + 3) / 6))` (see
/// `vendor/ggml/src/ggml-cpu/vec.h`'s `ggml_vec_hardsigmoid_f32`) -- MIL's
/// `sigmoid_hard` is the same piecewise-linear shape, `min(1, max(0, alpha
/// * x + beta))`, so `alpha = 1/6`, `beta = 0.5` matches it exactly.
fn lower_hardsigmoid(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let ty = ty_like(ctx.node, rank_of(&x));
    let alpha = fp16_scalar(ctx.func, 1.0 / 6.0)?;
    let beta = fp16_scalar(ctx.func, 0.5)?;
    let op: MilOp = "sigmoid_hard"
        .parse()
        .map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "sigmoid_hard missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(op).input("x", &x).input("alpha", &alpha).input("beta", &beta).output(ty))
}

/// `ggml_norm(a, eps)`: plain normalization (subtract mean, divide by std)
/// over ggml's axis 0 (MIL's last axis) -- *not* an affine layer norm
/// (nn.ss's own `layer-norm` applies the learned scale/shift afterward,
/// with separate `ggml-mul`/`ggml-add` calls, both already covered). MIL's
/// `layer_norm` op takes `gamma`/`beta` as declared (non-optional in the
/// generated Scheme wrapper) but they're optional in the real op spec;
/// omitting them here leaves CoreML's own identity defaults (1, 0), so
/// this stays the same plain normalization ggml computes. `op_params[0]`
/// is `eps` as a raw `f32` bit pattern (see `ggml_norm_impl`).
fn lower_norm(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let eps = f32::from_bits(ctx.node.op_params[0] as u32);
    let dims = x.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "NORM needs a fully-known source shape".to_string(),
    })?;
    let channels = *dims.last().ok_or_else(|| AotError::UnsupportedOp { index: ctx.index, op: "NORM source must be at least rank 1".to_string() })?;
    let one = f16::from_f32(1.0).to_bits();
    let zero = f16::from_f32(0.0).to_bits();
    let gamma = ctx.func.constant(Value::tensor_f16_bits(&[channels], &vec![one; channels as usize])?)?;
    let beta = ctx.func.constant(Value::tensor_f16_bits(&[channels], &vec![zero; channels as usize])?)?;
    let eps_c = fp16_scalar(ctx.func, eps)?;
    let op: MilOp = "layer_norm".parse().map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "layer_norm missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(op).input("x", &x).input("axes", vec![-1i32]).input("gamma", &gamma).input("beta", &beta).input("epsilon", &eps_c).output(ty))
}

/// `ggml_mean(a)`: the mean over ggml's axis 0 (MIL's last axis), keeping
/// it as a size-1 axis (`result->ne[0] = 1`, see `ggml_mean`) -- no
/// `op_params` at all.
fn lower_mean(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let op: MilOp = "reduce_mean".parse().map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "reduce_mean missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(op).input("x", &x).input("axes", vec![rank as i32 - 1]).input("keep_dims", true).output(ty))
}

/// `ggml_pad(a, p0, p1, p2, p3)` is `ggml_pad_ext(a, 0,p0, 0,p1, 0,p2,
/// 0,p3)` -- right-side-only zero padding, `op_params` (as `i32`s, not the
/// usual raw-`f32`-bits convention) `[lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3,
/// circular]` (see `ggml_pad_ext`). MIL's `pad` wants a flat `[before,
/// after]` list per axis, outermost-first, matching ggml's own axis order
/// reversed like everywhere else in this file; unlike ggml, MIL's `pad`
/// only accepts pairs for the *trailing* axes (it pads from the end of the
/// axis list backward, same as `F.pad`), so this only covers padding
/// confined to ggml's low axes (0/1), which is the only pattern either
/// port's `stem` or `rec`'s width-padding ever uses -- erroring loudly if
/// axis 2 or 3 ever needs padding too.
fn lower_pad(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let p = ctx.node.op_params;
    let (lp2, rp2, lp3, rp3) = (p[4], p[5], p[6], p[7]);
    if lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: "PAD on ggml axis 2 or 3 (only axis 0/1 padding is covered)".to_string() });
    }
    if p[8] != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: "PAD circular mode".to_string() });
    }
    // ggml axis 0 (W) is MIL's last axis, axis 1 (H) second-to-last; MIL's
    // pad list only covers the trailing `rank`-implied axes it's given, in
    // the same [axis1_before, axis1_after, axis0_before, axis0_after]
    // order (outermost of the covered axes first).
    let (lp0, rp0, lp1, rp1) = (p[0], p[1], p[2], p[3]);
    let pad = vec![lp1, rp1, lp0, rp0];
    let zero = fp16_scalar(ctx.func, 0.0)?;
    let op: MilOp = "pad".parse().map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "pad missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(op).input("x", &x).input("pad", pad).input("mode", "constant").input("constant_val", &zero).output(ty))
}

/// ggml_mul_mat(a, b) = b @ a^T (tensorlisp's `linear` calls it as
/// `(ggml-mul-mat weight x)`, i.e. a = weight, b = activations). Output
/// keeps `b`'s batch rank.
fn lower_mul_mat(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let a = ctx.src(0)?;
    let b = ctx.src(1)?;
    let ty = ty_like(ctx.node, rank_of(&b));
    Ok(ctx.func.op(MilOp::Matmul).input("x", &b).input("y", &a).input("transpose_x", false).input("transpose_y", true).output(ty))
}

/// A pure reshape (element count preserved, no slicing) always has
/// `view_offs == 0`; ggml gives slicing views a distinct "VIEW" op name.
/// Unlike other ops, a reshape's declared shape *is* what MIL will actually
/// produce (that's what `shape` controls), so `node.ne` needs no padding --
/// it's self-consistent by construction.
fn lower_reshape(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    if ctx.node.view_offs != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: "RESHAPE with nonzero view_offs".to_string() });
    }
    let x = ctx.src(0)?;
    let shape: Vec<i32> = ctx.node.ne.iter().rev().map(|&d| d as i32).collect();
    let ty = TensorType::new(DTYPE, shape.iter().map(|&d| d as u64));
    Ok(ctx.func.op(MilOp::Reshape).input("x", &x).input("shape", shape).output(ty))
}

/// ggml_conv_2d_direct(a, b, s0,s1,p0,p1,d0,d1): a = kernel [KW,KH,IC,OC]
/// (ggml order), b = input [W,H,C,N]. op_params (see
/// vendor/ggml/src/ggml.c) = [s0,s1,p0,p1,d0,d1] as i32. Output keeps `b`'s
/// rank (always 4 for an image input).
fn lower_conv_2d(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let a = ctx.src(0)?;
    let b = ctx.src(1)?;
    let ty = ty_like(ctx.node, rank_of(&b));
    let (s0, s1, p0, p1, d0, d1) = conv_params(ctx.node);
    Ok(conv2d(ctx.func, &b, &a, [s1, s0], [p1, p0], [d1, d0], 1, ty))
}

/// Depthwise: groups = input channels (ggml requires a->ne[2] == 1,
/// a->ne[3] == b->ne[2], i.e. one KWxKH filter per channel).
fn lower_conv_2d_dw(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let a = ctx.src(0)?;
    let b = ctx.src(1)?;
    // CoreML's conv op accepts an unbatched (rank-3, [C, H, W]) `x` fine
    // when `groups == 1` (every other conv in this model, already proven
    // against real taps) -- but a *grouped* conv (`groups > 1`, only ever
    // true for a depthwise conv here) requires the full rank-4 [N, C, H,
    // W]: without an explicit batch axis, CoreML's compiler can't tell
    // the grouped channel axis apart from a spatial one ("Variadic
    // dimension ... unexpected length 1; expected 2"). pad_rank adds the
    // batch axis back (as 1) only when it's missing, a pure reshape.
    let b = pad_rank(ctx.func, &b, 4)?;
    let ty = ty_like(ctx.node, rank_of(&b));
    let (s0, s1, p0, p1, d0, d1) = conv_params(ctx.node);
    let groups = ctx.node.ne[2];
    Ok(conv2d(ctx.func, &b, &a, [s1, s0], [p1, p0], [d1, d0], groups, ty))
}

/// `ggml_im2col(kernel, image, s0,s1,p0,p1,d0,d1, is_2D, dst_type)`: the
/// only call site in this codebase (`nn:patch-embed`) always uses `s0=s1=
/// patch`, `p0=p1=0`, `d0=d1=1` -- non-overlapping patches, i.e. a plain
/// strided "patchify", not a general sliding-window unfold -- so this only
/// covers that case (erroring loudly otherwise) instead of a fully general
/// im2col, which would need real overlap/padding handling.
///
/// ggml's own CPU kernel (`ggml_compute_forward_im2col_f32`) writes
/// `dst[iic*(KH*KW) + ikh*KW + ikw]` into the combined output axis -- KW
/// fastest, then KH, then IC slowest -- which is *exactly* how a plain
/// contiguous reshape flattens the kernel's own `[KW, KH, IC]` (ggml order,
/// axes 0-2) axes together (`nn:patch-embed`'s own `ggml-reshape-2d kernel
/// k (dim kernel 3)` relies on this). Since patch == stride and padding =
/// 0, `iiw = iow*patch + ikw` (`iow`/`ikw` a size-`patch` sub-index within
/// `iow`), i.e. this is *just* a reshape splitting each of the image's W/H
/// axes into (outer, inner) pairs, with no data movement of its own:
///
/// 1. reshape image `[N, C, H, W]` (MIL order) -> `[N, C, OH, KH, OW, KW]`
///    (splits H into `OH*KH`, W into `OW*KW` -- ordinary row-major splits,
///    inner sub-index last/fastest, matching `iow`/`ioh` being the slower
///    half of each pair).
/// 2. transpose to `[N, OH, OW, C, KH, KW]` (perm indices onto the above:
///    `[0, 2, 4, 1, 3, 5]`) -- groups `C, KH, KW` adjacent, in the same
///    (slowest to fastest) order ggml's own flattening uses.
/// 3. reshape, merging the last three axes -> `[N, OH, OW, C*KH*KW]`,
///    exactly `ggml_im2col`'s own declared output shape (ggml order
///    `[IC*KH*KW, OW, OH, N]`, reversed).
fn lower_im2col(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let kernel = ctx.src(0)?;
    let image = ctx.src(1)?;
    let p = ctx.node.op_params;
    let (s0, s1, p0, p1, d0, d1, is_2d) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
    if is_2d != 1 || p0 != 0 || p1 != 0 || d0 != 1 || d1 != 1 || s0 != s1 {
        return Err(AotError::UnsupportedOp {
            index: ctx.index,
            op: format!("IM2COL (only the non-overlapping, square-patch case is covered): s0={s0} s1={s1} p0={p0} p1={p1} d0={d0} d1={d1} is_2D={is_2d}"),
        });
    }
    let patch = s0 as u64;
    let kernel_dims = kernel.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "IM2COL needs a fully-known kernel shape".to_string(),
    })?;
    // Kernel MIL shape is `[D, C, KH, KW]` (ggml `[KW, KH, C, D]` reversed);
    // KH/KW must match the stride we're asserting a non-overlapping patch
    // with (ggml's own `ggml_calc_conv_output_size` already guarantees the
    // image divides evenly, or `ggml_im2col` itself would have asserted).
    let rank = kernel_dims.len();
    if rank != 4 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("IM2COL kernel must be rank 4, got {rank}") });
    }
    let (c, kh, kw) = (kernel_dims[1], kernel_dims[2], kernel_dims[3]);
    if kh != patch || kw != patch {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("IM2COL kernel {kh}x{kw} doesn't match stride {patch}") });
    }

    let image = pad_rank(ctx.func, &image, 4)?;
    let image_dims = image.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "IM2COL needs a fully-known image shape".to_string(),
    })?;
    if image_dims.len() != 4 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("IM2COL image must be rank 4, got {image_dims:?}") });
    }
    let (n, ic, h, w) = (image_dims[0], image_dims[1], image_dims[2], image_dims[3]);
    if ic != c {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("IM2COL image channels {ic} != kernel channels {c}") });
    }
    let (oh, ow) = (h / patch, w / patch);
    if oh * patch != h || ow * patch != w {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("IM2COL image {h}x{w} isn't a multiple of patch {patch}") });
    }

    // MIL's reshape/transpose only accept rank <= 5, so the naive single
    // 6D reshape + 6D transpose (splitting H *and* W in one step) doesn't
    // fit -- instead fold in one spatial axis at a time, each stage never
    // exceeding rank 5, accumulating the combined axis in the same order
    // ggml's own kernel writes it in (C slowest, then KH, then KW
    // fastest -- see the module doc comment): merge KH onto C first
    // (giving C*KH, C the slower/outer half, KH the faster/inner half --
    // matching "C then KH"), then merge KW onto *that* (giving C*KH*KW,
    // KW the faster/inner half -- matching "...then KW"). Doing it in the
    // other order would leave no way to insert KH back between C and KW.
    let i32v = |v: &[u64]| -> Vec<i32> { v.iter().map(|&d| d as i32).collect() };

    // Split H: [n, ic, h, w] -> [n, ic, oh, patch, w] -> (swap oh/patch)
    // [n, ic, patch, oh, w] -> merge (ic, patch) -> [n, ic*patch, oh, w].
    let h_split = ctx
        .func
        .op(MilOp::Reshape)
        .input("x", &image)
        .input("shape", i32v(&[n, ic, oh, patch, w]))
        .output(TensorType::new(DTYPE, [n, ic, oh, patch, w]))
        .build()?;
    let h_swapped = ctx
        .func
        .op(MilOp::Transpose)
        .input("x", &h_split)
        .input("perm", vec![0, 1, 3, 2, 4])
        .output(TensorType::new(DTYPE, [n, ic, patch, oh, w]))
        .build()?;
    let c1 = ic * patch;
    let h_merged = ctx
        .func
        .op(MilOp::Reshape)
        .input("x", &h_swapped)
        .input("shape", i32v(&[n, c1, oh, w]))
        .output(TensorType::new(DTYPE, [n, c1, oh, w]))
        .build()?;

    // Split W: [n, c1, oh, w] -> [n, c1, oh, ow, patch] -> (swap ow/patch)
    // [n, c1, patch, oh, ow] -> merge (c1, patch) -> [n, c1*patch, oh, ow]
    // (= [n, ic*patch*patch, oh, ow], C then KH then KW, fastest last).
    let w_split = ctx
        .func
        .op(MilOp::Reshape)
        .input("x", &h_merged)
        .input("shape", i32v(&[n, c1, oh, ow, patch]))
        .output(TensorType::new(DTYPE, [n, c1, oh, ow, patch]))
        .build()?;
    let w_swapped = ctx
        .func
        .op(MilOp::Transpose)
        .input("x", &w_split)
        .input("perm", vec![0, 1, 4, 2, 3])
        .output(TensorType::new(DTYPE, [n, c1, patch, oh, ow]))
        .build()?;
    let cols = c1 * patch;
    let w_merged = ctx
        .func
        .op(MilOp::Reshape)
        .input("x", &w_swapped)
        .input("shape", i32v(&[n, cols, oh, ow]))
        .output(TensorType::new(DTYPE, [n, cols, oh, ow]))
        .build()?;

    // [n, cols, oh, ow] -> [n, oh, ow, cols], ggml_im2col's own declared
    // shape (`[IC*KH*KW, OW, OH, N]` ggml order, reversed).
    let out_ty = ty_like(ctx.node, 4);
    Ok(ctx.func.op(MilOp::Transpose).input("x", &w_merged).input("perm", vec![0, 2, 3, 1]).output(out_ty))
}

/// Reshapes `x` to add leading axes of 1 until it reaches `rank`, if it
/// isn't there already -- a pure, value-preserving reshape, not new data.
/// Some MIL ops (grouped conv, tile) need their *input*'s own declared
/// rank to actually match what the op structurally expects (a fixed
/// number of `reps` entries, an unambiguous batch/group axis, ...); unlike
/// elementwise ops, they don't auto-broadcast a lower-rank input the way
/// `ty_like`-only padding (used everywhere else in this file) assumes.
fn pad_rank(func: &mut Function, x: &Var, rank: usize) -> Result<Var, AotError> {
    if rank_of(x) >= rank {
        return Ok(x.clone());
    }
    let Some(dims) = x.ty().as_tensor().and_then(TensorType::fixed_shape) else {
        return Ok(x.clone());
    };
    let mut shape: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    while shape.len() < rank {
        shape.insert(0, 1);
    }
    let ty = TensorType::new(DTYPE, shape.iter().map(|&d| d as u64));
    Ok(func.op(MilOp::Reshape).input("x", x).input("shape", shape).output(ty).build()?)
}

/// ggml_concat(a, b, dim): binary only (variadic concats in the Scheme
/// source, e.g. `concat-channels`, fold left over pairs, so each CONCAT
/// node has exactly 2 srcs); dim is ggml axis order in op_params[0] (see
/// vendor/ggml/src/ggml.c).
fn lower_concat(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let a = ctx.src(0)?;
    let b = ctx.src(1)?;
    // Ground truth for the target rank is this node's own reported output
    // rank, not just the operands' -- one operand alone can be truncated
    // more aggressively than the concatenated result is (e.g. a `[a, 1]`
    // column reshape reporting rank 1, concatenated with others into a
    // genuinely rank-2 `[a, 4]` result). MIL's `concat` needs every input
    // to already be exactly that rank (unlike an elementwise op, it
    // doesn't auto-broadcast a lower-rank operand against the others).
    let rank = ctx.node.ne.len().max(rank_of(&a)).max(rank_of(&b));
    let ty = ty_like(ctx.node, rank);
    let a = pad_rank(ctx.func, &a, rank)?;
    let b = pad_rank(ctx.func, &b, rank)?;
    let dim = ctx.node.op_params[0] as i64;
    let axis = rank as i64 - 1 - dim;
    Ok(ctx.func.op(MilOp::Concat).inputs("values", vec![&a, &b]).input("axis", axis as i32).input("interleave", false).output(ty))
}

/// ggml_pool_2d(a, op, k0,k1,s0,s1,p0,p1): op_params = [op, k0, k1, s0, s1,
/// p0, p1], all i32 (p0/p1 are declared `float` in the C signature but land
/// in an `int32_t[]` array literal, i.e. a real numeric truncation, not a
/// bit-cast like ARANGE's floats -- fine since yolo11n's SPPF only ever
/// pads by whole pixels).
fn lower_pool_2d(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let ty = ty_like(ctx.node, rank_of(&x));
    let (pool_op, k0, k1, s0, s1, p0, p1) = pool_params(ctx.node);
    let is_avg = match pool_op {
        0 => false, // GGML_OP_POOL_MAX
        1 => true,  // GGML_OP_POOL_AVG
        other => return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("POOL_2D op {other}") }),
    };
    let mut builder = ctx
        .func
        .op(if is_avg { MilOp::AvgPool } else { MilOp::MaxPool })
        .input("x", &x)
        .input("kernel_sizes", vec![k1, k0])
        .input("strides", vec![s1, s0])
        .input("pad_type", "custom")
        .input("pad", vec![p1, p1, p0, p0])
        .input("ceil_mode", false);
    if is_avg {
        // ggml's own average pool always includes padding (zeros) in the
        // average (`count_include_pad=True`, see nn.ss's own `avg-pool`
        // doc comment) -- `exclude_padding_from_average` has no ggml-side
        // equivalent to derive from, but ggml's own semantics pin it to
        // `false` unambiguously (unlike MAX_POOL, which needs no such
        // flag at all -- required only for AvgPool, which is why this
        // wasn't needed until ppocrv6's `rec.pooled` first exercised it).
        builder = builder.input("exclude_padding_from_average", false);
    }
    Ok(builder.output(ty))
}

/// ggml_interpolate_impl: op_params[0] = mode (low byte is the
/// ggml_scale_mode, high bits are flags like align-corners); only NEAREST
/// (yolo11n's FPN upsample, via `upsample-nearest`) is handled so far. The
/// target shape is already ground truth (`node.ne`), so MIL's
/// `resize_nearest_neighbor` (which takes an exact target height/width, not
/// a scale factor) needs no source-shape derivation at all.
fn lower_upscale(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let mode = ctx.node.op_params[0] & 0xFF;
    if mode != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("UPSCALE mode {mode} (only NEAREST=0 handled)") });
    }
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let mut shape: Vec<u64> = ctx.node.ne.iter().rev().map(|&d| d as u64).collect();
    while shape.len() < rank {
        shape.insert(0, 1);
    }
    let ty = TensorType::new(DTYPE, shape.clone());
    let h = shape[rank - 2] as i32;
    let w = shape[rank - 1] as i32;
    Ok(ctx.func.op(MilOp::ResizeNearestNeighbor).input("x", &x).input("target_size_height", h).input("target_size_width", w).output(ty))
}

/// tensorlisp's only VIEW producer is `tensor:slice` (see
/// ports/yolo11/yolo11.ss / crates/tensorlisp/src/scheme/stdlib/tensor.ss):
/// a single-axis, stride-preserving slice -- `ggml-view-4d t ne0 ne1 ne2 ne3
/// (stride t 1) (stride t 2) (stride t 3) (* from (stride t axis)))` -- so
/// the view's `nb` always equals its source's `nb`, and `view_offs` is
/// exactly `from * nb[axis]` for whichever single axis was sliced (all
/// others keep the source's own `ne`). That means `view_offs` decodes
/// losslessly into a per-axis begin index via repeated division by `nb`,
/// largest stride first, without needing to know the source's own ggml
/// node (MIL `slice_by_size` then takes it from there).
fn lower_view(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    if middle_collapse_gap(ctx.node, &x).is_some() {
        return lower_view_middle_collapse(ctx, x);
    }
    // `ggml_view_1d` (the only other VIEW producer besides `tensor:slice`
    // in this codebase) always passes n_dims=1 to `ggml_view_impl`, so its
    // node's own reported rank is always exactly 1 -- `tensor:slice`'s own
    // view_4d never reports rank 1 in this model (channel-split views
    // always keep W/H, both > 1). See `lower_view_column_select`'s own doc
    // comment for why this can't instead be told apart by comparing ranks
    // against the *source*'s rank (every source here is inflated by a
    // leading batch=1 axis from the very first op onward, so a source's
    // "true" ggml rank isn't recoverable from its MIL var's rank alone).
    if ctx.node.ne.len() == 1 {
        return lower_view_column_select(ctx, x);
    }
    if is_group_reshape_view(ctx.node, &x) {
        return lower_view_group_reshape(ctx, x);
    }
    // Same reason as `lower_concat`/`lower_repeat`: `slice_by_size` needs
    // `begin`/`size` to actually match `x`'s own declared rank exactly,
    // and that's this node's own reported rank (`node.ne.len()`), not
    // necessarily `x`'s current one (a source further back in the chain
    // isn't guaranteed to already carry the same padding every other
    // tensor here does).
    let rank = ctx.node.ne.len().max(rank_of(&x));
    let x = pad_rank(ctx.func, &x, rank)?;
    let (begin, size) = slice_params(ctx.node, rank);
    let ty = ty_like(ctx.node, rank);
    Ok(ctx.func.op(MilOp::SliceBySize).input("x", &x).input("begin", begin).input("size", size).output(ty))
}

/// A third VIEW pattern, distinct from both `tensor:slice`'s (same-rank,
/// single-axis narrow) and `decode-ltrb`'s (rank-2 -> rank-1 column
/// select): `attn:multi-head`'s own per-Q/K/V-part split,
/// `(ggml-view-4d qkv hd heads len batch (* hd es) row (* row len) (* i d
/// es))`. Since `es`/`row` are `qkv`'s own natural strides (`es = stride
/// qkv 0`, `row = 3*d*es`), this ne/nb combination isn't a same-rank
/// narrowing at all: it slices a contiguous `hd*heads` (= `d`) run out of
/// `qkv`'s actual last axis (`3d`, the fused Q;K;V channels) at offset
/// `i*d`, then reshapes that run into two axes, `heads` (outer) and `hd`
/// (inner) -- one more axis than `qkv` itself has. Applying
/// `tensor:slice`'s decode here (treating `qkv`'s axes as if they already
/// were `[hd, heads, len, batch]` positionally) is wrong: `qkv`'s real
/// axis 2 is `len`, not `heads`, and slicing wants axis 2's size to be
/// `heads`'s size instead, which is either a bogus out-of-range slice or
/// (worse) a silently wrong one.
///
/// Detected by: `nb[1] == ne[0] * elem_size` (axis 0/1 are a contiguous
/// "reshape split" -- necessary but not sufficient, this alone also holds
/// for `tensor:slice` narrowing anything other than its own axis 0) *and*
/// the source's actual last axis is a whole multiple of `ne[0]*ne[1]`
/// (true here: `qkv`'s last axis is `3d`, a multiple of `d = hd*heads`;
/// false for every other VIEW in this codebase (yolo11's own tensor:slice
/// -narrowed attention channel axis, and its channel-split c3k2 views) --
/// verified against real per-node dumps of both, not just derived).
fn is_group_reshape_view(node: &NodeInfo, x: &Var) -> bool {
    if node.ne.len() < 2 || node.nb.len() < 2 {
        return false;
    }
    if node.nb[1] != node.ne[0] as usize * 4 {
        return false;
    }
    let combined = node.ne[0] as u64 * node.ne[1] as u64;
    let Some(dims) = x.ty().as_tensor().and_then(TensorType::fixed_shape) else {
        return false;
    };
    match dims.last() {
        Some(&last) => combined > 0 && last >= combined && last % combined == 0,
        None => false,
    }
}

fn lower_view_group_reshape(ctx: LowerCtx<'_>, x: Var) -> Result<OpBuilder<'_>, AotError> {
    let dims = x.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "VIEW (group-reshape) needs a fully-known source shape".to_string(),
    })?;
    let r = dims.len();
    if r == 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: "VIEW (group-reshape) source must be at least rank 1".to_string() });
    }
    let combined = ctx.node.ne[0] as u64 * ctx.node.ne[1] as u64;
    let begin_elems = ctx.node.view_offs as u64 / 4;
    if begin_elems % combined != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("VIEW (group-reshape): offset {begin_elems} elements doesn't align to {combined}") });
    }

    let mut begin = vec![0i32; r];
    begin[r - 1] = (begin_elems) as i32;
    let mut size: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
    size[r - 1] = combined as i32;
    let mut sliced_dims = dims.clone();
    sliced_dims[r - 1] = combined;
    let sliced = ctx
        .func
        .op(MilOp::SliceBySize)
        .input("x", &x)
        .input("begin", begin)
        .input("size", size)
        .output(TensorType::new(DTYPE, sliced_dims.clone()))
        .build()?;

    // Reshape the sliced run's last axis (`combined`) into two axes,
    // `heads` (outer/slower) then `hd` (inner/faster) -- matching `nb[1]
    // == ne[0]*elem_size`'s own "hd fastest, heads next" structure, one
    // more axis than `x` itself.
    let heads = ctx.node.ne[1] as u64;
    let hd = ctx.node.ne[0] as u64;
    let mut new_dims: Vec<u64> = sliced_dims[..r - 1].to_vec();
    new_dims.push(heads);
    new_dims.push(hd);
    let new_shape: Vec<i32> = new_dims.iter().map(|&d| d as i32).collect();
    Ok(ctx.func.op(MilOp::Reshape).input("x", &sliced).input("shape", new_shape).output(TensorType::new(DTYPE, new_dims)))
}

/// A fourth VIEW pattern: nn.ss's `unpatchify` (`nn:conv-transpose2d`'s
/// kernel-equals-stride "pixel shuffle" path, used by ppocrv6's `db-head`),
/// via `row`'s own `(ggml-view-4d y w h k c-out (* w es) (* n es) (* k k n
/// es) (* ky k n es))`. `y`'s own shape is `[n, k*k*c-out]` (`n = w*h`,
/// `es = stride y 0`): axes 0/1 of the view (`w`, `h`) are a plain split of
/// `y`'s own last axis `n` (same shape as `is_group_reshape_view`'s split,
/// just not needing a slice since nothing is sliced out of `n`); axis 2
/// (`kx`) directly walks `y`'s *other* axis (`k*k*c-out`) at
/// unit-of-`n` granularity (`nb[2] == n * elem_size`); axis 3 (`c-out`)
/// continues from there, but skips a whole extra factor of `k` per step
/// (`nb[3] == nb[2] * ne[2] * k`, not just `nb[2] * ne[2]`) -- that extra
/// `k` is `ky`, a real dimension of `y`'s second axis (`k*k*c-out = kx *
/// ky * c-out`, `kx` fastest) that this particular view fixes to one value
/// and drops, sitting *between* the two axes (`kx`, `c-out`) that survive.
/// Neither `tensor:slice`'s decode (same-rank, no dropped axis) nor
/// `is_group_reshape_view`'s (drops nothing, only ever adds a `heads`/`hd`
/// split at the very end) fit a dropped *middle* axis, so this is handled
/// by reshaping `y` all the way out to five real axes (`[c-out, ky, kx, h,
/// w]`, MIL's own rank limit for reshape/transpose -- only valid because
/// `y` itself is exactly rank 2 here, verified against a real dump, not
/// assumed), slicing out the one `ky` this call wants, and reshaping the
/// result back down to four.
fn middle_collapse_gap(node: &NodeInfo, x: &Var) -> Option<bool> {
    let r = node.ne.len();
    // `c_out == 1` (only `det.head.final`, whose kernel has a single
    // output channel) truncates the reported rank to 3 (ggml drops
    // trailing size-1 axes) -- `ky`'s own size isn't recoverable from
    // `nb[3]` when it doesn't even exist, but `unpatchify` always calls
    // `row` with a *square* `k x k` kernel, so `ky`'s size is simply
    // `ne[2]` (`kx`'s), the same `k` -- no need to cross-check via `nb[3]`
    // at all (done anyway, below, whenever it does exist, as a sanity
    // check rather than the primary source of truth).
    if !(3..=4).contains(&r) || node.nb.len() != r {
        return None;
    }
    let dims = x.ty().as_tensor().and_then(TensorType::fixed_shape)?;
    let &x_last = dims.last()?;
    let elem = 4u64;
    let n01 = node.ne[0] as u64 * node.ne[1] as u64;
    if n01 != x_last {
        return None;
    }
    if node.nb[1] as u64 != node.ne[0] as u64 * elem {
        return None;
    }
    if node.nb[2] as u64 != x_last * elem {
        return None;
    }
    if r == 4 {
        let expected3 = node.nb[2] as u64 * node.ne[2] as u64 * node.ne[2] as u64;
        if node.nb[3] as u64 != expected3 {
            return None;
        }
    }
    Some(true)
}

fn lower_view_middle_collapse(ctx: LowerCtx<'_>, x: Var) -> Result<OpBuilder<'_>, AotError> {
    let dims = x.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "VIEW (middle-collapse) needs a fully-known source shape".to_string(),
    })?;
    if dims.len() != 2 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("VIEW (middle-collapse) only covers a rank-2 source, got {dims:?}") });
    }
    let (kkc, n) = (dims[0], dims[1]);
    let (w, h, kx) = (ctx.node.ne[0] as u64, ctx.node.ne[1] as u64, ctx.node.ne[2] as u64);
    let c_out = ctx.node.ne.get(3).copied().unwrap_or(1) as u64;
    let ky = kx; // square kernel, always (see this function's own doc comment).
    if w * h != n || kx * ky * c_out != kkc {
        return Err(AotError::UnsupportedOp {
            index: ctx.index,
            op: format!("VIEW (middle-collapse): shape mismatch (w={w} h={h} n={n}, kx={kx} ky={ky} c_out={c_out} kkc={kkc})"),
        });
    }
    let offset_elems = ctx.node.view_offs as u64 / 4;
    let ky_stride = n * kx;
    if offset_elems % ky_stride != 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("VIEW (middle-collapse): offset {offset_elems} doesn't align to {ky_stride}") });
    }
    let ky_index = offset_elems / ky_stride;

    let split = ctx
        .func
        .op(MilOp::Reshape)
        .input("x", &x)
        .input("shape", vec![c_out as i32, ky as i32, kx as i32, h as i32, w as i32])
        .output(TensorType::new(DTYPE, [c_out, ky, kx, h, w]))
        .build()?;
    let sliced = ctx
        .func
        .op(MilOp::SliceBySize)
        .input("x", &split)
        .input("begin", vec![0, ky_index as i32, 0, 0, 0])
        .input("size", vec![c_out as i32, 1, kx as i32, h as i32, w as i32])
        .output(TensorType::new(DTYPE, [c_out, 1, kx, h, w]))
        .build()?;
    // Match this node's own reported rank exactly (3 when `c_out == 1` got
    // truncated, matching `ty_like`'s ground-truth convention elsewhere in
    // this file), not forced up to 4.
    let out_rank = ctx.node.ne.len();
    let out_ty = ty_like(ctx.node, out_rank);
    let out_shape: Vec<i32> = if out_rank == 3 { vec![kx as i32, h as i32, w as i32] } else { vec![c_out as i32, kx as i32, h as i32, w as i32] };
    Ok(ctx.func.op(MilOp::Reshape).input("x", &sliced).input("shape", out_shape).output(out_ty))
}

/// `vision.ss`'s `decode-ltrb`, via a direct `(ggml-view-1d dist a (* j
/// (stride dist 1)))`: picks one index `j` along `dist`'s higher axis (a
/// rank-2, contiguous `[a, 4]` tensor) and flattens that axis away
/// entirely, keeping the other (innermost) axis in full -- so the view
/// node's own `ne`/`nb` (rank 1) don't carry enough information to recover
/// which source axis was fixed or at what index the way a same-rank
/// `tensor:slice` view's do (see `slice_params`'s doc comment): a rank-1
/// `nb`/`ne` pair is consistent with infinitely many higher-rank offsets.
/// This instead assumes `x` is standard-contiguous (true for every value
/// this call site actually feeds -- it's always a freshly
/// `ggml-reshape-2d`'d matmul result) and derives the fixed axis's index
/// directly from `view_offs` and `x`'s own declared (MIL) shape -- whose
/// *last* axis is always ggml's own axis 0 regardless of how many extra
/// leading (padding) axes came along with it, so indexing from the end
/// rather than assuming a specific rank is what actually makes this robust
/// to that padding.
fn lower_view_column_select(ctx: LowerCtx<'_>, x: Var) -> Result<OpBuilder<'_>, AotError> {
    let dims = x.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "VIEW needs a fully-known source shape".to_string(),
    })?;
    if dims.len() < 2 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("VIEW column-select needs a source of rank >= 2, got {dims:?}") });
    }
    let rank = dims.len();
    let inner = dims[rank - 1]; // ggml's own axis 0, kept in full.
    if inner != ctx.node.ne[0] as u64 {
        return Err(AotError::UnsupportedOp {
            index: ctx.index,
            op: format!("VIEW keeps {} elements but source's inner axis is {inner}", ctx.node.ne[0]),
        });
    }
    let elem_size = 4u64; // f32 throughout this lowering pass.
    let offset_elems = ctx.node.view_offs as u64 / elem_size;
    if offset_elems % inner != 0 {
        return Err(AotError::UnsupportedOp {
            index: ctx.index,
            op: format!("VIEW offset {offset_elems} elements doesn't align to inner axis size {inner}"),
        });
    }
    let outer_index = (offset_elems / inner) as i32;
    let mut begin = vec![0i32; rank];
    begin[rank - 2] = outer_index;
    let mut size = vec![1i32; rank];
    size[rank - 1] = inner as i32;
    let mut sliced_ty_dims = vec![1u64; rank];
    sliced_ty_dims[rank - 1] = inner;
    let sliced = ctx
        .func
        .op(MilOp::SliceBySize)
        .input("x", &x)
        .input("begin", begin)
        .input("size", size)
        .output(TensorType::new(DTYPE, sliced_ty_dims))
        .build()?;
    let out_len = ctx.node.ne[0] as u64;
    Ok(ctx.func.op(MilOp::Reshape).input("x", &sliced).input("shape", vec![out_len as i32]).output(TensorType::new(DTYPE, [out_len])))
}

/// `ggml_arange(ctx, start, stop, step)`: a fresh leaf tensor, no sources;
/// `op_params` are `[start, stop, step]` as raw `f32` bit patterns (see
/// `NodeInfo::op_params`'s own doc comment). MIL's `range_1d` (declared
/// arg names `start`/`end`/`step`, coremltools' own ordering) covers this
/// directly; the output length is already ground truth in `node.ne[0]`, so
/// there's no need to recompute it (and risk an off-by-one from float
/// rounding) from `start`/`stop`/`step` the way ggml itself does.
fn lower_arange(ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let start = f32::from_bits(ctx.node.op_params[0] as u32);
    let stop = f32::from_bits(ctx.node.op_params[1] as u32);
    let step = f32::from_bits(ctx.node.op_params[2] as u32);
    let n = ctx.node.ne[0] as u64;
    let start_c = fp16_scalar(ctx.func, start)?;
    let stop_c = fp16_scalar(ctx.func, stop)?;
    let step_c = fp16_scalar(ctx.func, step)?;
    let range_op: MilOp = "range_1d".parse().map_err(|_| AotError::UnsupportedOp { index: ctx.index, op: "range_1d missing from the MIL catalogue".to_string() })?;
    Ok(ctx.func.op(range_op).input("start", &start_c).input("end", &stop_c).input("step", &step_c).output(TensorType::new(DTYPE, [n])))
}

/// `ggml_repeat_4d(ctx, a, ne0..ne3)`: broadcasts `a` up to a target shape
/// (no `op_params` at all -- the target shape is simply the result
/// tensor's own `ne`, see `vendor/ggml/src/ggml.c`). MIL's `tile` wants a
/// per-axis repeat COUNT, not a target size, so this divides the (rank-
/// padded) target shape by `a`'s own (rank-padded) shape, axis by axis.
fn lower_repeat(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = ctx.node.ne.len().max(rank_of(&x));
    let ty = ty_like(ctx.node, rank);
    let out_dims = ty.fixed_shape().ok_or_else(|| AotError::UnsupportedOp { index: ctx.index, op: "REPEAT needs a fully-known target shape".to_string() })?;
    // MIL's `tile` needs `reps` to have exactly `x`'s own declared rank
    // (unlike an elementwise op, it doesn't auto-broadcast a lower-rank
    // input against a wider `reps`/output) -- pad `x` itself up to `rank`
    // first, the same reason `lower_conv_2d_dw` pads its own input.
    let x = pad_rank(ctx.func, &x, rank)?;
    let src_dims = x.ty().as_tensor().and_then(TensorType::fixed_shape).ok_or_else(|| AotError::UnsupportedOp {
        index: ctx.index,
        op: "REPEAT needs a fully-known source shape".to_string(),
    })?;
    let reps: Vec<i32> = out_dims.iter().zip(src_dims.iter()).map(|(&o, &s)| (o / s) as i32).collect();
    Ok(ctx.func.op(MilOp::Tile).input("x", &x).input("reps", reps).output(ty))
}

/// `ggml_scale`/`ggml_scale_bias`: both tag `GGML_OP_SCALE`, `op_params =
/// [scale, bias]` as raw `f32` bit patterns (`ggml_scale` itself always
/// passes `bias = 0.0`). MIL has no fused scale-and-bias op, so this is
/// `x * scale + bias` as two real ops -- always both, even when `bias` is
/// exactly 0 (harmless, and simpler than branching on it).
fn lower_scale(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let scale = f32::from_bits(ctx.node.op_params[0] as u32);
    let bias = f32::from_bits(ctx.node.op_params[1] as u32);
    let scale_c = fp16_scalar(ctx.func, scale)?;
    let bias_c = fp16_scalar(ctx.func, bias)?;
    let scaled = ctx.func.op(MilOp::Mul).input("x", &x).input("y", &scale_c).output(ty.clone()).build()?;
    Ok(ctx.func.op(MilOp::Add).input("x", &scaled).input("y", &bias_c).output(ty))
}

/// ggml_cont just re-lays-out `a`'s own values into a fresh contiguous
/// buffer, same logical shape -- a ggml-backend memory-layout requirement
/// with no MIL equivalent (MIL tensors have no "non-contiguous view"
/// concept for a leaf builder to route around). Lowers to a reshape to its
/// own (already-known) shape: value-preserving by construction, and reuses
/// an op every backend already has instead of inventing an identity one.
fn lower_cont(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let shape: Vec<i32> = ctx.node.ne.iter().rev().map(|&d| d as i32).collect();
    Ok(ctx.func.op(MilOp::Reshape).input("x", &x).input("shape", shape).output(ty))
}

/// ggml_transpose swaps ggml axes 0 and 1 (`ne`/`nb` swapped, everything
/// else fixed) -- MIL's own last two axes (its axis order is reversed from
/// ggml's, so ggml's innermost two axes are MIL's last two).
fn lower_transpose(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    let ty = ty_like(ctx.node, rank);
    let mut perm: Vec<i32> = (0..rank as i32).collect();
    if rank >= 2 {
        perm.swap(rank - 1, rank - 2);
    }
    Ok(ctx.func.op(MilOp::Transpose).input("x", &x).input("perm", perm).output(ty))
}

/// `ggml_permute(a, axis0..axis3)`: op_params are exactly `[axis0, axis1,
/// axis2, axis3]` (see `ggml_permute`'s own `ggml_set_op_params` call),
/// where `axisI` is the ggml axis input axis `I` moves *to*. Every call
/// tensorlisp's own stdlib actually makes (attn.ss's split-heads/
/// merge-heads: `(0 2 1 3)`, swapping the middle two axes; sdpa's own
/// value-transpose: `(1 0 2 3)`, swapping the first two) is a single pair
/// of ggml axes swapping with everything else fixed -- so, like
/// `lower_permute`'s sibling ops, this only needs to recognize *which*
/// pair swaps, not implement a fully general permutation (one that moved
/// data into/out of a padding axis beyond the reported rank would need
/// more care than a same-rank swap does, and doesn't occur here).
fn lower_permute(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let x = ctx.src(0)?;
    let rank = rank_of(&x);
    if rank == 0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: "PERMUTE needs a source of rank >= 1".to_string() });
    }
    let ty = ty_like(ctx.node, rank);
    let p = ctx.node.op_params;
    // axis[s] = the ggml axis input axis `s` moves *to* (see ggml_permute's
    // own `ggml_set_op_params` call). Invert it: ggml_src_for_dst[d] = the
    // input axis that ends up at output axis `d`.
    let axis = [p[0], p[1], p[2], p[3]];
    let mut ggml_src_for_dst = [0i32; 4];
    for (s, &d) in axis.iter().enumerate() {
        ggml_src_for_dst[d as usize] = s as i32;
    }
    // MIL output axis `m` (0 = outermost) <-> ggml output axis `rank-1-m`
    // (0 = innermost); same conversion for the matching *input* axis.
    // Arbitrary permutations are fine (not just a single-pair swap) as
    // long as no axis beyond `rank` (ggml's own padding slots, always
    // identity-mapped) is actually touched -- verified false only by
    // `unpatchify`'s own 3-cycle `(1 2 0 3)`, not yet by anything that
    // crosses into padding.
    let mut perm = vec![0i32; rank];
    for m in 0..rank {
        let g_out = rank - 1 - m;
        let g_in = ggml_src_for_dst[g_out] as usize;
        if g_in >= rank {
            return Err(AotError::UnsupportedOp {
                index: ctx.index,
                op: format!("PERMUTE {axis:?} moves data from ggml axis {g_in} beyond rank {rank}"),
            });
        }
        perm[m] = (rank - 1 - g_in) as i32;
    }
    Ok(ctx.func.op(MilOp::Transpose).input("x", &x).input("perm", perm).output(ty))
}

/// `ggml_soft_max_ext(a, mask, scale, max_bias)`: op_params are `[scale,
/// max_bias]` as raw `f32` bit patterns (same convention as `ARANGE`'s,
/// see `NodeInfo::op_params`'s own doc comment). Only `max_bias == 0.0` is
/// covered -- the only value tensorlisp's own `attn:sdpa` ever passes
/// (ALiBi-style linear bias slopes aren't used anywhere in this model).
/// softmax(a * scale + mask) over ggml's axis 0 (innermost) = MIL's last
/// axis; MIL has no fused scale/mask/softmax op, so this decomposes into
/// the same three real ops ggml's own reference computation performs.
fn lower_soft_max(mut ctx: LowerCtx<'_>) -> Result<OpBuilder<'_>, AotError> {
    let a = ctx.src(0)?;
    let scale = f32::from_bits(ctx.node.op_params[0] as u32);
    let max_bias = f32::from_bits(ctx.node.op_params[1] as u32);
    if max_bias != 0.0 {
        return Err(AotError::UnsupportedOp { index: ctx.index, op: format!("SOFT_MAX with max_bias {max_bias} (only 0.0 is covered)") });
    }
    let rank = rank_of(&a);
    let ty = ty_like(ctx.node, rank);
    let scale_c = fp16_scalar(ctx.func, scale)?;
    let scaled = ctx.func.op(MilOp::Mul).input("x", &a).input("y", &scale_c).output(ty.clone()).build()?;
    let biased = if ctx.node.srcs.len() > 1 {
        let mask = ctx.src(1)?;
        ctx.func.op(MilOp::Add).input("x", &scaled).input("y", &mask).output(ty.clone()).build()?
    } else {
        scaled
    };
    Ok(ctx.func.op(MilOp::Softmax).input("x", &biased).input("axis", (rank - 1) as i32).output(ty))
}

/// The CoreML MIL lowering pass: ggml op name -> lowering function. Each
/// entry here, plus its corresponding `(tl generic)` alias, is the whole
/// surface for extending AOT coverage -- see the module doc comment.
fn leafnode_pass() -> &'static PassTable<LowerFn> {
    static PASS: OnceLock<PassTable<LowerFn>> = OnceLock::new();
    PASS.get_or_init(|| {
        PassTable::<LowerFn>::new()
            .register("ADD", lower_add)
            .register("SUB", lower_sub)
            .register("MUL", lower_mul)
            .register("RELU", lower_relu)
            .register("SILU", lower_silu)
            .register("SIGMOID", lower_sigmoid)
            .register("MUL_MAT", lower_mul_mat)
            .register("RESHAPE", lower_reshape)
            .register("CONV_2D", lower_conv_2d)
            .register("CONV_2D_DW", lower_conv_2d_dw)
            .register("CONCAT", lower_concat)
            .register("POOL_2D", lower_pool_2d)
            .register("UPSCALE", lower_upscale)
            .register("VIEW", lower_view)
            .register("CONT", lower_cont)
            .register("TRANSPOSE", lower_transpose)
            .register("PERMUTE", lower_permute)
            .register("SOFT_MAX", lower_soft_max)
            .register("ARANGE", lower_arange)
            .register("REPEAT", lower_repeat)
            .register("SCALE", lower_scale)
            .register("NORM", lower_norm)
            .register("GELU_ERF", lower_gelu_erf)
            .register("HARDSIGMOID", lower_hardsigmoid)
            .register("MEAN", lower_mean)
            .register("PAD", lower_pad)
            .register("IM2COL", lower_im2col)
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_node(
    index: usize,
    node: &NodeInfo,
    forced_name: Option<String>,
    node_vars: &HashMap<usize, Var>,
    named: &mut HashMap<String, Var>,
    file: &GgufFile,
    path: &Path,
    infos: &[TensorInfo],
    func: &mut Function,
) -> Result<Var, AotError> {
    let lower = leafnode_pass()
        .get(node.op.as_str())
        .ok_or_else(|| AotError::UnsupportedOp { index, op: node.op.clone() })?;
    let ctx = LowerCtx { index, node, node_vars, named, file, path, infos, func };
    let builder = lower(ctx)?;
    let builder = match forced_name {
        Some(name) => builder.name(name),
        None => builder,
    };
    Ok(builder.build()?)
}

/// `CONV_2D`/`CONV_2D_DW` op_params (see `vendor/ggml/src/ggml.c`'s
/// `ggml_conv_2d_direct`/`ggml_conv_2d_dw_direct`): `[s0, s1, p0, p1, d0, d1]`.
fn conv_params(node: &NodeInfo) -> (i32, i32, i32, i32, i32, i32) {
    let p = node.op_params;
    (p[0], p[1], p[2], p[3], p[4], p[5])
}

/// `POOL_2D` op_params (see `vendor/ggml/src/ggml.c`'s `ggml_pool_2d`):
/// `[op, k0, k1, s0, s1, p0, p1]`.
fn pool_params(node: &NodeInfo) -> (i32, i32, i32, i32, i32, i32, i32) {
    let p = node.op_params;
    (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
}

/// Shared `conv`/`conv_2d_dw` lowering. `stride`/`pad`/`dilation` are
/// `[h, w]` (MIL's order); ggml gives `s0/p0/d0` for its W axis (dim 0) and
/// `s1/p1/d1` for H (dim 1), so callers pass `[s1, s0]` etc.
#[allow(clippy::too_many_arguments)]
fn conv2d<'f>(
    func: &'f mut Function,
    input: &Var,
    weight: &Var,
    stride: [i32; 2],
    pad: [i32; 2],
    dilation: [i32; 2],
    groups: i64,
    out_ty: TensorType,
) -> coreml_rs::mil::OpBuilder<'f> {
    func.op(MilOp::Conv)
        .input("x", input)
        .input("weight", weight)
        .input("strides", vec![stride[0], stride[1]])
        .input("pad_type", "custom")
        .input("pad", vec![pad[0], pad[0], pad[1], pad[1]])
        .input("dilations", vec![dilation[0], dilation[1]])
        .input("groups", groups as i32)
        .output(out_ty)
}

/// A scalar constant, built at `DTYPE` (fp16) instead of the fp32 every
/// plain Rust `f32` auto-wraps as via `Arg`'s own `From<f32>` -- needed
/// wherever a scalar gets combined with an already-`DTYPE` tensor in an op
/// (`mul`/`add`) that requires matching operand dtypes; MIL's own builder
/// rejects the mismatch outright (`"inputs of type T disagree (Float16 vs
/// Float32)"`), it isn't just a CoreML-compiler-time nuisance.
fn fp16_scalar(func: &mut Function, v: f32) -> Result<Var, AotError> {
    let bits = f16::from_f32(v).to_bits();
    Ok(func.constant(Value::tensor_f16_bits(&[], &[bits])?)?)
}

fn resolve(
    src: &str,
    node_vars: &HashMap<usize, Var>,
    named: &mut HashMap<String, Var>,
    file: &GgufFile,
    path: &Path,
    infos: &[TensorInfo],
    func: &mut Function,
) -> Result<Var, AotError> {
    if let Some(rest) = src.strip_prefix('%') {
        let idx: usize = rest.parse().map_err(|_| AotError::UnknownSource(src.to_string()))?;
        return node_vars.get(&idx).cloned().ok_or_else(|| AotError::UnknownSource(src.to_string()));
    }
    if let Some(v) = named.get(src) {
        return Ok(v.clone());
    }
    let idx = infos
        .iter()
        .position(|t| t.name == src)
        .ok_or_else(|| AotError::UnknownSource(src.to_string()))?;
    let info = &infos[idx];
    if info.dtype != DType::F32 {
        return Err(AotError::NonFloatWeight(src.to_string()));
    }
    let bytes = file.read_tensor_bytes(path, idx as i64)?;
    // Weight constants are always fp32 in the GGUF file; pack them down to
    // fp16 bits here to match `DTYPE` (see this crate's own doc comment on
    // it) -- every other tensor in the graph is fp16, and MIL's ops need
    // matching operand dtypes, so a weight left at fp32 wouldn't type-check
    // against the (fp16) activation it's used with.
    let bits: Vec<u16> = bytes
        .chunks_exact(4)
        .map(|c| f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits())
        .collect();
    let shape: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();
    let value = Value::tensor_f16_bits(&shape, &bits)?;
    // Sanitized const name; the lookup cache still keys on ggml's own name.
    let var = func.constant_named(&sanitize_ident(src), value)?;
    named.insert(src.to_string(), var.clone());
    Ok(var)
}

fn rank_of(v: &Var) -> usize {
    v.ty().as_tensor().and_then(|t| t.rank()).unwrap_or(0)
}

/// `NodeInfo::ne` is ggml order (innermost first) and, per `ggml_n_dims`,
/// has trailing size-1 dims (typically an unbatched N=1) already dropped;
/// MIL wants ndarray order (outermost first) and the *true* rank its
/// actual (already rank-correct) inputs carry. Left-pads with 1s to
/// `rank` -- the rank of whichever resolved source var this op's output
/// rank actually follows -- rather than trusting `node.ne.len()` verbatim.
fn ty_like(node: &NodeInfo, rank: usize) -> TensorType {
    let mut shape: Vec<u64> = node.ne.iter().rev().map(|&d| d as u64).collect();
    while shape.len() < rank {
        shape.insert(0, 1);
    }
    TensorType::new(DTYPE, shape)
}

/// Decodes a `VIEW` node's `view_offs` (byte offset from its source's base)
/// into per-axis `begin`/`size` for MIL's `slice_by_size` (`x[begin[i]..begin[i]+size[i]]`
/// per axis). `node.nb`/`node.ne` are ggml order (innermost first) and
/// already sliced to `ggml_n_dims`, i.e. only the axes that survived
/// truncation; higher (truncated, size-1) axes contribute nothing to the
/// offset, so decoding just the present axes is exact. Since `tensor:slice`
/// only ever changes one axis's `ne` and keeps every `nb` equal to the
/// source's own strides, `view_offs` is a multiple of exactly one `nb`
/// entry -- dividing by strides largest-first recovers that axis's begin
/// index (and 0 for every other axis) the same way positional-notation
/// decoding recovers digits from decreasing place values.
fn slice_params(node: &NodeInfo, rank: usize) -> (Vec<i32>, Vec<i32>) {
    let n = node.ne.len();
    let mut begin_ggml = vec![0i64; n];
    let mut offset = node.view_offs as i64;
    for axis in (0..n).rev() {
        let stride = node.nb[axis] as i64;
        if stride == 0 {
            continue;
        }
        begin_ggml[axis] = offset / stride;
        offset -= begin_ggml[axis] * stride;
    }
    let mut begin: Vec<i32> = begin_ggml.iter().rev().map(|&v| v as i32).collect();
    let mut size: Vec<i32> = node.ne.iter().rev().map(|&d| d as i32).collect();
    while begin.len() < rank {
        begin.insert(0, 0);
    }
    while size.len() < rank {
        size.insert(0, 1);
    }
    (begin, size)
}
