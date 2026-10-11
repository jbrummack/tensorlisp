//! Fusion pass over the IR, before planning: sequences ggml's Vulkan backend
//! fuses (`ggml_vk_can_fuse` and friends) are rewritten into one node with a
//! pseudo-op name that `lower.rs` maps onto the fused shader. The last node of a
//! sequence is rewritten in place (so later readers keep their tensor ids and the
//! planner sees the real reads); the others leave `graph.nodes` and are never
//! allocated.
//!
//! | pseudo-op | replaces |
//! |---|---|
//! | `RMS_NORM_MUL` | `RMS_NORM`, `MUL` by a weight (src: x, w; eps in `op_params[0]`) |
//! | `RMS_NORM_MUL_ROPE` | `RMS_NORM_MUL`, `ROPE` (src: x, w, pos, ff; rope's `op_params`, eps in `[15]`) |
//!
//! ggml also fuses `ROPE`+`VIEW`+`SET_ROWS`, but its shader reads i64 row indices and the
//! programs here index with i32, so that variant is not formed.

use std::collections::HashMap;

use crate::graph::{Graph, TensorId, Ty};

/// How many times each pseudo-op was formed.
pub type Counts = std::collections::BTreeMap<&'static str, usize>;

/// Readers of each root tensor (a node reading through a view counts for the root).
fn readers(graph: &Graph) -> Vec<usize> {
    let mut n = vec![0usize; graph.tensors.len()];
    for &id in &graph.nodes {
        for s in graph.tensors[id].src.iter().flatten() {
            n[graph.root(*s).0] += 1;
        }
    }
    n
}

/// Analysis of the current node list, recomputed before each pass.
struct Info {
    readers: Vec<usize>,
    keep: std::collections::HashSet<TensorId>,
    position: HashMap<TensorId, usize>,
}

impl Info {
    fn new(graph: &Graph) -> Info {
        Info {
            readers: readers(graph),
            keep: graph.keep.iter().map(|&k| graph.root(k).0).collect(),
            position: graph.nodes.iter().enumerate().map(|(p, &id)| (id, p)).collect(),
        }
    }

    /// Nodes strictly between `head` and `tail`, if both are nodes in that order.
    fn between<'g>(&self, graph: &'g Graph, head: TensorId, tail: TensorId) -> Option<&'g [TensorId]> {
        let (&ph, &pt) = (self.position.get(&head)?, self.position.get(&tail)?);
        (pt > ph).then(|| &graph.nodes[ph + 1..pt])
    }

    /// `head` has `readers` readers (the tail, plus views of it), and the tail follows it with
    /// nothing but views in between.
    fn chained(&self, graph: &Graph, head: TensorId, tail: TensorId, readers: usize) -> bool {
        self.readers[head] == readers
            && !self.keep.contains(&head)
            && graph.tensors[head].view_src.is_none()
            && self.between(graph, head, tail).is_some_and(|b| b.iter().all(|&m| graph.tensors[m].is_view_op()))
    }
}

fn rope_mode_ok(t: &crate::graph::Tensor) -> bool {
    matches!(t.op_params[2], 0 | 2)
}

pub fn fuse(graph: &mut Graph, disabled: &str) -> Counts {
    let mut counts = Counts::new();
    let mut dead = vec![false; graph.tensors.len()];

    let on = |name: &str| !disabled.split(',').any(|d| d == name);

    // Pass 1: RMS_NORM + MUL.
    let info = Info::new(graph);
    for p in 0..graph.nodes.len() {
        if !on("RMS_NORM_MUL") {
            break;
        }
        let id = graph.nodes[p];
        if graph.tensors[id].op != "MUL" {
            continue;
        }
        let t = &graph.tensors[id];
        let (Some(a), Some(b)) = (t.src(0), t.src(1)) else { continue };
        if graph.tensors[a].op != "RMS_NORM" || !info.chained(graph, a, id, 1) {
            continue;
        }
        let (rms, w) = (&graph.tensors[a], &graph.tensors[b]);
        let Some(x_id) = rms.src(0) else { continue };
        let x = &graph.tensors[x_id];
        let ok = rms.ty == Ty::F32
            && x.ty == Ty::F32
            && w.ty == Ty::F32
            && t.ty == Ty::F32
            && t.ne == rms.ne
            && t.is_contiguous()
            && w.nb[0] == 4
            && w.ne[0] > 0
            && rms.ne[0] % w.ne[0] == 0;
        if !ok {
            continue;
        }
        let eps = rms.op_params[0];
        let t = &mut graph.tensors[id];
        t.op = "RMS_NORM_MUL".into();
        t.src = vec![Some(x_id), Some(b)];
        t.op_params = [0; 16];
        t.op_params[0] = eps;
        dead[a] = true;
        *counts.entry("RMS_NORM_MUL").or_default() += 1;
    }
    graph.nodes.retain(|&id| !dead[id]);

    // Pass 2: RMS_NORM_MUL + ROPE.
    let info = Info::new(graph);
    for p in 0..graph.nodes.len() {
        if !on("RMS_NORM_MUL_ROPE") {
            break;
        }
        let id = graph.nodes[p];
        let r = &graph.tensors[id];
        if r.op != "ROPE" {
            continue;
        }
        let Some(m) = r.src(0) else { continue };
        let mt = &graph.tensors[m];
        if mt.op != "RMS_NORM_MUL" || !info.chained(graph, m, id, 1) {
            continue;
        }
        let ok = rope_mode_ok(r)
            && r.src(1).is_some()
            && mt.is_contiguous()
            && r.is_contiguous()
            && mt.ne[0] <= 1024
            && mt.ne == r.ne
            && matches!(r.ty, Ty::F32 | Ty::F16);
        if !ok {
            continue;
        }
        let (x, w) = (mt.src(0), mt.src(1));
        let eps = mt.op_params[0];
        let (pos, ff) = (r.src(1), r.src(2));
        let r = &mut graph.tensors[id];
        r.op = "RMS_NORM_MUL_ROPE".into();
        r.src = vec![x, w, pos, ff];
        r.op_params[15] = eps;
        dead[m] = true;
        *counts.entry("RMS_NORM_MUL_ROPE").or_default() += 1;
    }
    graph.nodes.retain(|&id| !dead[id]);

    counts
}

