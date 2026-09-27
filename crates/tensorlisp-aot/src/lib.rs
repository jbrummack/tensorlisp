//! Lowers a tensorlisp ggml reference graph into a MIL [`Function`], node by
//! node, using [`tensorlisp::GraphInfo`] (built from the same ggml graph the
//! CPU/Metal/CUDA backends actually run) as ground truth: every op, source
//! wiring and output shape here comes directly from what ggml already
//! computed, not from re-deriving it, so a bad lowering can't quietly
//! disagree with the reference.
//!
//! Only the small op set the `mlp` fixture needs (`MUL_MAT`, `ADD`, `RELU`)
//! is wired up so far -- this proves the pipeline mechanism (`GraphInfo` ->
//! `mil::Function` -> serialized `spec::Model` -> run on ANE via
//! `coreml_rs::runtime`) end to end on a small, numerically-checkable model
//! before scaling to a real model's full op set (see `cg-leafnode`'s design
//! notes for the staging rationale). Cross-platform: building the MIL
//! `Function` needs no Apple-only APIs, only actually *running* it does
//! (left to callers, e.g. `coreml_rs::runtime`, which is itself Apple-only).

use std::collections::HashMap;
use std::path::Path;

use coreml_rs::mil::{DataType, Function, MilOp, Opset, TensorType, Value, Var};
use tensorlisp::gguf::{GgufFile, TensorInfo};
use tensorlisp::{DType, GraphInfo, NodeInfo};

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
    #[error("no output node found for {0:?}")]
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
pub fn lower_graph(
    gguf_path: &Path,
    info: &GraphInfo,
    inputs: &[(&str, Vec<u64>)],
    opset: Opset,
) -> Result<Function, AotError> {
    let file = GgufFile::open(gguf_path)?;
    let infos = file.tensor_infos();

    let mut func = Function::new(opset);
    let mut named: HashMap<String, Var> = HashMap::new();
    for (name, shape) in inputs {
        let ty = TensorType::new(DataType::Float32, shape.iter().copied());
        named.insert((*name).to_string(), func.input(name, ty)?);
    }

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
    let mut output_index: HashMap<String, usize> = HashMap::new();
    for (name, _) in &info.outputs {
        let named_idx = info
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.name == *name || n.name.starts_with(&format!("{name} (")))
            .map(|(i, _)| i)
            .next_back();
        let idx = match named_idx {
            Some(i) => i,
            None if info.outputs.len() == 1 => info.nodes.len().checked_sub(1).ok_or_else(|| AotError::UnknownOutput(name.clone()))?,
            None => return Err(AotError::UnknownOutput(name.clone())),
        };
        output_index.insert(name.clone(), idx);
    }
    let name_by_index: HashMap<usize, String> = output_index.iter().map(|(n, &i)| (i, n.clone())).collect();

    let mut node_vars: Vec<Var> = Vec::with_capacity(info.nodes.len());
    for (index, node) in info.nodes.iter().enumerate() {
        // Give the op its declared output name directly (rather than
        // ggml's auto-generated `add_4`-style one) so the serialized
        // model's output feature is actually called e.g. `logits`.
        let forced_name = name_by_index.get(&index).cloned();
        let var = lower_node(index, node, forced_name, &node_vars, &mut named, &file, gguf_path, &infos, &mut func)?;
        node_vars.push(var);
    }

    let outputs: Vec<Var> = info.outputs.iter().map(|(name, _)| node_vars[output_index[name]].clone()).collect();
    func.set_outputs(&outputs)?;

    Ok(func)
}

#[allow(clippy::too_many_arguments)]
fn lower_node(
    index: usize,
    node: &NodeInfo,
    forced_name: Option<String>,
    node_vars: &[Var],
    named: &mut HashMap<String, Var>,
    file: &GgufFile,
    path: &Path,
    infos: &[TensorInfo],
    func: &mut Function,
) -> Result<Var, AotError> {
    let src = |i: usize, named: &mut HashMap<String, Var>, func: &mut Function| -> Result<Var, AotError> {
        let s = node.srcs.get(i).ok_or_else(|| AotError::SourceCount {
            index,
            op: node.op.clone(),
            want: i + 1,
            got: node.srcs.len(),
        })?;
        resolve(s, node_vars, named, file, path, infos, func)
    };
    let ty = mil_type(node);

    let builder = match node.op.as_str() {
        "ADD" => {
            let x = src(0, named, func)?;
            let y = src(1, named, func)?;
            func.op(MilOp::Add).input("x", &x).input("y", &y).output(ty)
        }
        "RELU" => {
            let x = src(0, named, func)?;
            func.op(MilOp::Relu).input("x", &x).output(ty)
        }
        "MUL_MAT" => {
            // ggml_mul_mat(a, b) = b @ a^T (tensorlisp's `linear` calls it as
            // `(ggml-mul-mat weight x)`, i.e. a = weight, b = activations).
            let a = src(0, named, func)?;
            let b = src(1, named, func)?;
            func.op(MilOp::Matmul)
                .input("x", &b)
                .input("y", &a)
                .input("transpose_x", false)
                .input("transpose_y", true)
                .output(ty)
        }
        other => {
            return Err(AotError::UnsupportedOp { index, op: other.to_string() });
        }
    };
    let builder = match forced_name {
        Some(name) => builder.name(name),
        None => builder,
    };
    Ok(builder.build()?)
}

fn resolve(
    src: &str,
    node_vars: &[Var],
    named: &mut HashMap<String, Var>,
    file: &GgufFile,
    path: &Path,
    infos: &[TensorInfo],
    func: &mut Function,
) -> Result<Var, AotError> {
    if let Some(rest) = src.strip_prefix('%') {
        let idx: usize = rest.parse().map_err(|_| AotError::UnknownSource(src.to_string()))?;
        return node_vars.get(idx).cloned().ok_or_else(|| AotError::UnknownSource(src.to_string()));
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
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();
    let value = Value::tensor(&shape, floats)?;
    // MIL identifiers are `[A-Za-z_][A-Za-z0-9_@]*`; ggml weight names
    // (`fc1.weight`) aren't, so give the `const` a sanitized name while
    // still keying the lookup cache on ggml's own name.
    let mil_name: String = src.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    let var = func.constant_named(&mil_name, value)?;
    named.insert(src.to_string(), var.clone());
    Ok(var)
}

/// `NodeInfo::ne` is ggml order (innermost first); MIL (like the rest of
/// this crate's shapes) wants ndarray order (outermost first).
fn mil_type(node: &NodeInfo) -> TensorType {
    let shape: Vec<u64> = node.ne.iter().rev().map(|&d| d as u64).collect();
    TensorType::new(DataType::Float32, shape)
}
