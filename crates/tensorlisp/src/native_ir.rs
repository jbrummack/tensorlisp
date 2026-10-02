//! Backend-independent half of the native devices: converts a built ggml
//! cgraph into `tensorlisp_kernels`' IR and reads results back.

use std::{collections::HashMap, ffi::CStr};

use ggml_sys::ffi::*;
use tensorlisp_kernels::graph::{Graph as IrGraph, Storage, Tensor as IrTensor, TensorId, Ty};

use crate::error::{Error, Result};

pub(crate) fn backend_err(e: tensorlisp_kernels::Error) -> Error {
    Error::Backend(e.to_string())
}

/// Result name, tensor, ndarray-order shape.
pub(crate) type Results = Vec<(String, TensorId, Vec<usize>)>;

pub(crate) struct Lowered {
    pub ir: IrGraph,
    pub outputs: Results,
    pub taps: Results,
}

fn name_of(t: *const ggml_tensor) -> String {
    unsafe { CStr::from_ptr(ggml_get_name(t)) }.to_string_lossy().into_owned()
}

/// Converts a built ggml graph. `weight_index`/`state_index` map ggml tensor
/// pointers (of the file's and the states' contexts) to leaf indices.
pub(crate) fn lower_graph(
    graph: *mut ggml_cgraph,
    inputs: &[*mut ggml_tensor],
    outputs: &[(String, *mut ggml_tensor, usize)],
    taps: &[(String, *mut ggml_tensor, usize)],
    weight_index: &HashMap<usize, usize>,
    state_index: &HashMap<usize, usize>,
) -> Lowered {
    let mut ir = IrGraph::default();
    let mut ids: HashMap<*mut ggml_tensor, TensorId> = HashMap::new();

    // Every tensor the nodes touch, nodes first and in order.
    let n = unsafe { ggml_graph_n_nodes(graph) };
    let mut order: Vec<*mut ggml_tensor> = Vec::new();
    let mut register = |t: *mut ggml_tensor, order: &mut Vec<*mut ggml_tensor>| {
        if !t.is_null() && !ids.contains_key(&t) {
            ids.insert(t, order.len());
            order.push(t);
        }
    };
    for i in 0..n {
        register(unsafe { ggml_graph_node(graph, i) }, &mut order);
    }
    let mut cursor = 0;
    while cursor < order.len() {
        let t = order[cursor];
        cursor += 1;
        let (srcs, view_src) = unsafe { ((*t).src, (*t).view_src) };
        for s in srcs {
            register(s, &mut order);
        }
        register(view_src, &mut order);
    }
    for (_, t, _) in outputs.iter().chain(taps) {
        register(*t, &mut order);
    }

    let input_index: HashMap<*mut ggml_tensor, usize> = inputs.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    for &t in &order {
        let r = unsafe { &*t };
        let storage = if !r.view_src.is_null() {
            Storage::Temp
        } else if let Some(&w) = weight_index.get(&(t as usize)) {
            Storage::Weight(w)
        } else if let Some(&s) = state_index.get(&(t as usize)) {
            Storage::State(s)
        } else if let Some(&i) = input_index.get(&t) {
            Storage::Input(i)
        } else {
            Storage::Temp
        };
        ir.tensors.push(IrTensor {
            ty: Ty(r.type_ as u32),
            ne: r.ne,
            nb: [r.nb[0] as u64, r.nb[1] as u64, r.nb[2] as u64, r.nb[3] as u64],
            op: unsafe { CStr::from_ptr(ggml_op_name(r.op)) }.to_string_lossy().into_owned(),
            op_params: r.op_params,
            src: r.src.iter().map(|&s| ids.get(&s).copied()).collect(),
            view_src: ids.get(&r.view_src).copied(),
            view_offs: r.view_offs as u64,
            storage,
            name: name_of(t),
        });
    }
    ir.nodes = (0..n as usize).collect();
    ir.keep = outputs.iter().chain(taps).map(|(_, t, _)| ids[t]).collect();

    if std::env::var_os("TL_NATIVE_TRACE").is_some() {
        let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
        for &id in &ir.nodes {
            *counts.entry(ir.tensors[id].op.as_str()).or_default() += 1;
        }
        eprintln!("native graph: {} nodes: {counts:?}", ir.nodes.len());
    }

    let shape = |t: *mut ggml_tensor, rank: usize| -> Vec<usize> {
        let ne = unsafe { (*t).ne };
        (0..rank).rev().map(|i| ne[i] as usize).collect()
    };
    let results = |list: &[(String, *mut ggml_tensor, usize)]| -> Results {
        list.iter().map(|(name, t, rank)| (name.clone(), ids[t], shape(*t, *rank))).collect()
    };
    let (outputs, taps) = (results(outputs), results(taps));
    Lowered { ir, outputs, taps }
}

/// Reads f32 results through `read(tensor, bytes)`.
pub(crate) fn read_results(
    results: &Results,
    read: impl Fn(TensorId, usize) -> tensorlisp_kernels::Result<Vec<u8>>,
) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
    results
        .iter()
        .map(|(name, id, shape)| {
            let n: usize = shape.iter().product();
            let bytes = read(*id, n * 4).map_err(backend_err)?;
            let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), data)
                .map(|a| (name.clone(), a))
                .map_err(|e| Error::Backend(format!("{name}: {e}")))
        })
        .collect()
}
