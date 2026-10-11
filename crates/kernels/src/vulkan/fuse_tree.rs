//! Graph passes for the Tree lowering (`TL_NATIVE_TREE`), run before planning:
//!
//! * [`cse`]: nodes that compute the same thing from the same inputs are computed once (Gemma
//!   converts one attention mask to f16 in every layer; the 35 copies become one).
//! * [`elementwise`]: chains and trees of elementwise nodes over dense f32 tensors (add, mul,
//!   scale, gelu, copies to f16, broadcast of a row / a scalar, arange) become one `FUSED_EW`
//!   node: a single kernel, `k:ew` in `stdlib/tree_kernels.ss`, that computes the whole expression
//!   per element from the region's external inputs, so the intermediates are never stored.
//!
//! A node is absorbed into its consumer when it has exactly that one consumer, is not an output,
//! and has the same number of elements. The consumer keeps its tensor id (readers are unchanged);
//! absorbed nodes leave `graph.nodes`.

use std::collections::{HashMap, HashSet};

use crate::graph::{Fused, Graph, Storage, TensorId, Ty};

/// Readers of each root tensor among the real (non-view) nodes, in node order, once each.
fn consumers(graph: &Graph) -> Vec<Vec<TensorId>> {
    let mut cons: Vec<Vec<TensorId>> = vec![vec![]; graph.tensors.len()];
    for &id in &graph.nodes {
        let t = &graph.tensors[id];
        if t.is_view_op() {
            continue;
        }
        for s in t.src.iter().flatten() {
            let r = graph.root(*s).0;
            if cons[r].last() != Some(&id) {
                cons[r].push(id);
            }
        }
    }
    cons
}

fn state_rooted(graph: &Graph, id: TensorId) -> bool {
    matches!(graph.tensors[graph.root(id).0].storage, Storage::State(_))
}

/// Replaces nodes that repeat an earlier node (same op, parameters, shapes and inputs) by that
/// node. Returns how many were removed.
pub fn cse(graph: &mut Graph) -> usize {
    const PURE: &[&str] = &[
        "ADD", "SUB", "MUL", "DIV", "SCALE", "SQR", "SQRT", "SIN", "COS", "LOG", "UNARY", "CPY", "CONT", "DUP", "REPEAT", "ARANGE", "GET_ROWS", "CONCAT", "RMS_NORM", "RMS_NORM_MUL", "NORM", "SOFT_MAX",
        "MUL_MAT", "ROPE", "SUM_ROWS", "MEAN", "ARGMAX",
    ];
    let mut repl: HashMap<TensorId, TensorId> = HashMap::new();
    let mut seen: HashMap<String, TensorId> = HashMap::new();
    let mut dead: HashSet<TensorId> = HashSet::new();
    let map = |repl: &HashMap<TensorId, TensorId>, id: TensorId| repl.get(&id).copied().unwrap_or(id);
    let nodes = graph.nodes.clone();
    for id in nodes {
        // readers see the canonical tensor: rewrite this node's sources (and the roots of its views)
        for s in graph.tensors[id].src.iter_mut().flatten() {
            *s = map(&repl, *s);
        }
        if let Some(v) = graph.tensors[id].view_src {
            graph.tensors[id].view_src = Some(map(&repl, v));
        }
        let t = &graph.tensors[id];
        if !PURE.contains(&t.op.as_str()) || t.view_src.is_some() || t.storage != Storage::Temp || graph.keep.contains(&id) {
            continue;
        }
        if t.src.iter().flatten().any(|&s| state_rooted(graph, s)) {
            continue;
        }
        // CPY / CONT list their destination as the second source; it is the node itself.
        let skip_dst = matches!(t.op.as_str(), "CPY" | "CONT" | "DUP");
        let srcs: Vec<String> = t
            .src
            .iter()
            .enumerate()
            .filter(|(i, _)| !(skip_dst && *i == 1))
            .map(|(_, s)| match s {
                Some(s) => {
                    let x = &graph.tensors[*s];
                    let (r, offs) = graph.root(*s);
                    format!("{r}+{offs}:{:?}:{:?}:{:?}", x.ty, x.ne, x.nb)
                }
                None => "-".into(),
            })
            .collect();
        let key = format!("{}|{:?}|{:?}|{:?}|{:?}|{}", t.op, t.ty, t.ne, t.nb, t.op_params, srcs.join(","));
        match seen.get(&key) {
            Some(&first) => {
                repl.insert(id, first);
                dead.insert(id);
            }
            None => {
                seen.insert(key, id);
            }
        }
    }
    if dead.is_empty() {
        return 0;
    }
    // views of removed nodes read the canonical one; sources of later nodes were rewritten as met
    for t in graph.tensors.iter_mut() {
        if let Some(v) = t.view_src {
            t.view_src = Some(map(&repl, v));
        }
        for s in t.src.iter_mut().flatten() {
            *s = map(&repl, *s);
        }
    }
    for k in graph.keep.iter_mut() {
        *k = map(&repl, *k);
    }
    graph.nodes.retain(|id| !dead.contains(id));
    dead.len()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The same elements as the output, flat.
    Full,
    /// A row of `n` elements repeated: index `e % n`.
    Row(u32),
    /// One element.
    Scalar,
}

enum Body {
    Bin(&'static str),
    Scale(f32, f32),
    Unary(&'static str),
    Copy,
    Bcast,
    Arange(f32, f32),
}

struct Ew {
    body: Body,
    out: Ty,
    n: u32,
    srcs: Vec<(TensorId, Kind)>,
}

fn unary_name(code: i32) -> Option<&'static str> {
    Some(match code {
        0 => "abs",
        1 => "sgn",
        2 => "neg",
        3 => "step",
        4 => "tanh",
        6 => "relu",
        7 => "sigmoid",
        8 => "gelu",
        9 => "gelu_quick",
        10 => "silu",
        13 => "exp",
        18 => "floor",
        19 => "ceil",
        _ => return None,
    })
}

fn classify(graph: &Graph, id: TensorId) -> Option<Ew> {
    let t = &graph.tensors[id];
    if t.view_src.is_some() || t.storage != Storage::Temp || !t.is_contiguous() || t.nelements() == 0 || t.nelements() > u32::MAX as i64 {
        return None;
    }
    let n = t.nelements() as u32;
    // an operand: dense, f32 (or i32 where allowed), not state (a later node may overwrite it)
    let operand = |s: Option<TensorId>, bcast: bool, ints: bool| -> Option<(TensorId, Kind)> {
        let s = s?;
        let x = &graph.tensors[s];
        if !(x.ty == Ty::F32 || (ints && x.ty == Ty::I32)) || !x.is_contiguous() || x.nelements() == 0 || state_rooted(graph, s) {
            return None;
        }
        if x.ne == t.ne {
            Some((s, Kind::Full))
        } else if bcast && x.nelements() == 1 {
            Some((s, Kind::Scalar))
        } else if bcast && x.ne[1..] == [1, 1, 1] && x.ne[0] > 1 && t.ne[0] % x.ne[0] == 0 {
            Some((s, Kind::Row(x.ne[0] as u32)))
        } else {
            None
        }
    };
    let f32_out = t.ty == Ty::F32;
    let (body, srcs, out) = match t.op.as_str() {
        "ADD" | "SUB" | "MUL" | "DIV" if f32_out => {
            let a = operand(t.src(0), false, false)?;
            let b = operand(t.src(1), true, false)?;
            let f = match t.op.as_str() {
                "ADD" => "+",
                "SUB" => "-",
                "MUL" => "*",
                _ => "/",
            };
            (Body::Bin(f), vec![a, b], Ty::F32)
        }
        "SCALE" if f32_out => (Body::Scale(t.op_param_f32(0), t.op_param_f32(1)), vec![operand(t.src(0), false, false)?], Ty::F32),
        "SQR" if f32_out => (Body::Bin("*"), vec![operand(t.src(0), false, false)?, operand(t.src(0), false, false)?], Ty::F32),
        "SQRT" if f32_out => (Body::Unary("sqrt"), vec![operand(t.src(0), false, false)?], Ty::F32),
        "SIN" if f32_out => (Body::Unary("sin"), vec![operand(t.src(0), false, false)?], Ty::F32),
        "COS" if f32_out => (Body::Unary("cos"), vec![operand(t.src(0), false, false)?], Ty::F32),
        "LOG" if f32_out => (Body::Unary("log"), vec![operand(t.src(0), false, false)?], Ty::F32),
        "UNARY" if f32_out => (Body::Unary(unary_name(t.op_params[0])?), vec![operand(t.src(0), false, false)?], Ty::F32),
        "CPY" | "CONT" | "DUP" if matches!(t.ty, Ty::F32 | Ty::F16) => {
            let s = t.src(0)?;
            let x = &graph.tensors[s];
            if !matches!(x.ty, Ty::F32 | Ty::I32) || (x.ty == Ty::I32 && t.ty != Ty::F32) || !x.is_contiguous() || x.nelements() != t.nelements() || state_rooted(graph, s) {
                return None;
            }
            (Body::Copy, vec![(s, Kind::Full)], t.ty)
        }
        "REPEAT" if f32_out => {
            let s = t.src(0)?;
            let x = &graph.tensors[s];
            if x.ty != Ty::F32 || x.nelements() != 1 || state_rooted(graph, s) {
                return None;
            }
            (Body::Bcast, vec![(s, Kind::Scalar)], Ty::F32)
        }
        "ARANGE" if f32_out => (Body::Arange(f32::from_bits(t.op_params[0] as u32), f32::from_bits(t.op_params[2] as u32)), vec![], Ty::F32),
        _ => return None,
    };
    Some(Ew { body, out, n, srcs })
}

fn num(x: f32) -> String {
    if x.is_finite() {
        format!("{:?}", x)
    } else {
        unreachable!("constants go through push words")
    }
}

/// Forms elementwise regions. Returns how many fused nodes were made.
pub fn elementwise(graph: &mut Graph) -> usize {
    let _ = num;
    let cons = consumers(graph);
    let keep: HashSet<TensorId> = graph.keep.iter().map(|&k| graph.root(k).0).collect();
    let mut info: HashMap<TensorId, Ew> = HashMap::new();
    for &id in &graph.nodes {
        if !graph.tensors[id].is_view_op() {
            if let Some(e) = classify(graph, id) {
                info.insert(id, e);
            }
        }
    }
    // child -> the node that absorbs it
    let mut parent: HashMap<TensorId, TensorId> = HashMap::new();
    for (&n, node) in &info {
        for &(s, kind) in &node.srcs {
            if kind != Kind::Full {
                continue;
            }
            let (r, offs) = graph.root(s);
            let Some(p) = info.get(&r) else { continue };
            if offs != 0 || p.out != Ty::F32 || p.n != node.n || graph.tensors[r].nelements() != graph.tensors[s].nelements() || keep.contains(&r) || cons[r].len() != 1 || cons[r][0] != n {
                continue;
            }
            parent.insert(r, n);
        }
    }
    let pos: HashMap<TensorId, usize> = graph.nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut children: HashMap<TensorId, Vec<TensorId>> = HashMap::new();
    for (&c, &p) in &parent {
        children.entry(p).or_default().push(c);
    }

    let mut formed = 0;
    let mut dead: HashSet<TensorId> = HashSet::new();
    let roots: Vec<TensorId> = graph.nodes.iter().copied().filter(|id| info.contains_key(id) && !parent.contains_key(id)).collect();
    for root in roots {
        // members in node order (producers first)
        let mut members = vec![root];
        let mut i = 0;
        while i < members.len() {
            if let Some(c) = children.get(&members[i]) {
                members.extend(c.iter().copied());
            }
            i += 1;
        }
        members.sort_by_key(|m| pos[m]);

        // names are local to the region, so equal regions (every layer's) give the same kernel
        let temp = |m: TensorId| format!("t{}", members.iter().position(|&x| x == m).unwrap());
        let mut inputs: Vec<(TensorId, Kind, Ty)> = Vec::new();
        let mut lets: Vec<String> = Vec::new();
        let mut consts: Vec<u32> = Vec::new();
        let mut name_of_input = |inputs: &mut Vec<(TensorId, Kind, Ty)>, s: TensorId, k: Kind, ty: Ty| -> String {
            let idx = inputs.iter().position(|&(i, kk, _)| i == s && kk == k).unwrap_or_else(|| {
                inputs.push((s, k, ty));
                inputs.len() - 1
            });
            format!("x{idx}")
        };
        for &m in &members {
            let e = &info[&m];
            let mut ops: Vec<String> = Vec::new();
            for &(s, kind) in &e.srcs {
                let (r, offs) = graph.root(s);
                if kind == Kind::Full && offs == 0 && parent.get(&r) == Some(&m) {
                    ops.push(temp(r));
                } else {
                    ops.push(name_of_input(&mut inputs, s, kind, graph.tensors[s].ty));
                }
            }
            let mut c = |v: f32| {
                consts.push(v.to_bits());
                format!("c{}", consts.len() - 1)
            };
            let expr = match &e.body {
                Body::Bin(f) => format!("({f} {} {})", ops[0], ops[1]),
                Body::Scale(a, b) => {
                    let (ca, cb) = (c(*a), c(*b));
                    format!("(+ (* {} {ca}) {cb})", ops[0])
                }
                Body::Unary(f) => format!("(unary {f} {})", ops[0]),
                Body::Copy | Body::Bcast => ops[0].clone(),
                Body::Arange(a, b) => {
                    let (ca, cb) = (c(*a), c(*b));
                    format!("(+ {ca} (* (f32 idx) {cb}))")
                }
            };
            lets.push(format!("({} {expr})", temp(m)));
        }
        let root_e = &info[&root];
        // Row inputs need their length as a push word, in input order; constants follow.
        let mut push: Vec<u32> = vec![root_e.n];
        let mut spec = Vec::new();
        for &(_, k, ty) in &inputs {
            let tn = match ty {
                Ty::F32 => "f32",
                Ty::I32 => "i32",
                _ => unreachable!(),
            };
            match k {
                Kind::Full => spec.push(format!("({tn} full)")),
                Kind::Scalar => spec.push(format!("({tn} scalar)")),
                Kind::Row(r) => {
                    push.push(r);
                    spec.push(format!("({tn} row)"));
                }
            }
        }
        push.extend(&consts);
        let out_ty = if root_e.out == Ty::F16 { "f16" } else { "f32" };
        let args = format!("'({}) '{out_ty} '({}) '{} {}", spec.join(" "), lets.join(" "), temp(root), consts.len());

        for &m in &members {
            if m != root {
                dead.insert(m);
            }
        }
        let t = &mut graph.tensors[root];
        t.op = "FUSED_EW".into();
        t.src = inputs.iter().map(|&(s, _, _)| Some(s)).collect();
        t.op_params = [0; 16];
        graph.fused.insert(root, Fused { args, push, n_inputs: 0, taps: vec![], last_stage: 0 });
        formed += 1;
    }

    // view nodes of removed tensors go too
    loop {
        let before = dead.len();
        for &id in &graph.nodes {
            let t = &graph.tensors[id];
            if t.is_view_op() && t.op != "NONE" && t.src.iter().flatten().any(|&s| dead.contains(&graph.root(s).0)) {
                dead.insert(id);
            }
        }
        if dead.len() == before {
            break;
        }
    }
    graph.nodes.retain(|id| !dead.contains(id));
    formed
}

/// The activation a dp4a mat-vec (or a fused gate/up pair) reads.
fn dot_input(t: &crate::graph::Tensor) -> Option<TensorId> {
    match t.op.as_str() {
        "MUL_MAT" => t.src(1),
        "FUSED_GU" => t.src(2),
        _ => None,
    }
}

/// Producers whose every reader is a dp4a mat-vec write the activations quantized (see
/// `Graph::q8_out`): no f32 copy, no separate quantize launch, and one quantization shared by
/// all the mat-vecs that read it. `dot_ok(id)` says whether mat-vec `id` takes the dp4a path.
pub fn mark_q8_outputs(graph: &mut Graph, dot_ok: &dyn Fn(&Graph, TensorId) -> bool) -> usize {
    let cons = consumers(graph);
    let keep: HashSet<TensorId> = graph.keep.iter().map(|&k| graph.root(k).0).collect();
    let mut marked = Vec::new();
    let mut sidecars: Vec<(TensorId, Vec<TensorId>)> = Vec::new();
    for &id in &graph.nodes {
        let t = &graph.tensors[id];
        let producer = match t.op.as_str() {
            "FUSED_EW" => graph.fused.get(&id).is_some_and(|f| f.args.contains("'f32 '(")),
            "FUSED_ROW" => t.ne[0] % 32 == 0,
            "RMS_NORM" | "RMS_NORM_MUL" => t.ne[0] % 32 == 0,
            "FLASH_ATTN_EXT" => super::lower::fa_split(graph, id).is_some() && t.ne[0] % 32 == 0,
            _ => false,
        };
        if !producer || t.ty != Ty::F32 || t.view_src.is_some() || !t.is_contiguous() || t.nelements() % 32 != 0 || keep.contains(&id) || cons[id].is_empty() {
            continue;
        }
        let all_dot = cons[id].iter().all(|&m| {
            let mm = &graph.tensors[m];
            dot_input(mm).is_some_and(|b| {
                let (r, offs) = graph.root(b);
                let tb = &graph.tensors[b];
                r == id && offs == 0 && tb.nelements() == t.nelements() && tb.is_contiguous()
            }) && mm.src(0).is_some_and(|a| graph.root(a).0 != id)
                && (mm.op == "FUSED_GU" || dot_ok(graph, m))
        });
        if all_dot {
            marked.push(id);
        } else if t.op == "FUSED_ROW" {
            // f32 values some node needs, and mat-vecs that want int8: the chain writes both
            let dots: Vec<TensorId> = cons[id]
                .iter()
                .copied()
                .filter(|&m| {
                    let mm = &graph.tensors[m];
                    dot_input(mm).is_some_and(|b| {
                        let (r, offs) = graph.root(b);
                        let tb = &graph.tensors[b];
                        r == id && offs == 0 && tb.nelements() == t.nelements() && tb.is_contiguous()
                    }) && (mm.op == "FUSED_GU" || dot_ok(graph, m))
                })
                .collect();
            if !dots.is_empty() {
                sidecars.push((id, dots));
            }
        }
    }
    let n = marked.len() + sidecars.len();
    graph.q8_out.extend(marked);
    for (id, dots) in sidecars {
        let elems = graph.tensors[id].nelements();
        let ne0 = (elems as u64 * 9 / 8).div_ceil(4) as i64;
        let mut side = graph.tensors[id].clone();
        side.op = "SIDECAR".into();
        side.src = vec![];
        side.op_params = [0; 16];
        side.op_params[0] = elems as i32;
        side.ne = [ne0, 1, 1, 1];
        side.nb = [4, 4 * ne0 as u64, 4 * ne0 as u64, 4 * ne0 as u64];
        side.view_src = None;
        side.view_offs = 0;
        let sid = graph.tensors.len();
        graph.tensors.push(side);
        // the producer writes it, the mat-vecs read it: both keep it alive (an extra source each)
        graph.tensors[id].src.push(Some(sid));
        for m in dots {
            let t = &mut graph.tensors[m];
            let want = if t.op == "FUSED_GU" { 4 } else { 3 };
            while t.src.len() < want - 1 {
                t.src.push(None);
            }
            if t.src.len() < want {
                t.src.push(Some(sid));
            }
        }
        graph.q8_side.insert(id, sid);
    }
    n
}

// ---- row chains -------------------------------------------------------------------------------

struct NormNode {
    x: TensorId,
    w: Option<(TensorId, Kind)>,
    eps: f32,
}

/// An `RMS_NORM` / `RMS_NORM_MUL` over dense f32 rows of a multiple of four elements.
fn classify_norm(graph: &Graph, id: TensorId) -> Option<NormNode> {
    let t = &graph.tensors[id];
    if !matches!(t.op.as_str(), "RMS_NORM" | "RMS_NORM_MUL") || t.ty != Ty::F32 || t.view_src.is_some() || t.storage != Storage::Temp || !t.is_contiguous() || t.ne[0] % 4 != 0 || t.nelements() == 0 {
        return None;
    }
    let x = t.src(0)?;
    let tx = &graph.tensors[x];
    if tx.ty != Ty::F32 || !tx.is_contiguous() || tx.ne != t.ne || state_rooted(graph, x) {
        return None;
    }
    let w = if t.op == "RMS_NORM_MUL" {
        let w = t.src(1)?;
        let tw = &graph.tensors[w];
        if tw.ty != Ty::F32 || !tw.is_contiguous() || state_rooted(graph, w) {
            return None;
        }
        let kind = if tw.ne == t.ne {
            Kind::Full
        } else if tw.ne[1..] == [1, 1, 1] && tw.ne[0] > 1 && t.ne[0] % tw.ne[0] == 0 {
            Kind::Row(tw.ne[0] as u32)
        } else {
            return None;
        };
        Some((w, kind))
    } else {
        None
    };
    Some(NormNode { x, w, eps: t.op_param_f32(0) })
}

/// `x` is `t` read flat from the start: the same elements in the same order.
fn flat_of(graph: &Graph, x: TensorId, t: TensorId) -> bool {
    let (r, offs) = graph.root(x);
    r == t && offs == 0 && graph.tensors[x].nelements() == graph.tensors[t].nelements() && graph.tensors[x].is_contiguous()
}

/// Chains of row operations become one `FUSED_ROW` node: an `RMS_NORM` followed by any run of
/// elementwise nodes (residual add, scale, ...) and further norms over the same rows, each stage
/// feeding the next through registers. A stage's result that something outside the chain still
/// reads is written out as well (a tap). Returns how many chains were formed.
pub fn row_chains(graph: &mut Graph) -> usize {
    let cons = consumers(graph);
    let keep: HashSet<TensorId> = graph.keep.iter().map(|&k| graph.root(k).0).collect();
    let pos: HashMap<TensorId, usize> = graph.nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut taken: HashSet<TensorId> = HashSet::new();
    let mut formed = 0;
    let mut dead: HashSet<TensorId> = HashSet::new();
    let nodes = graph.nodes.clone();
    for &start in &nodes {
        if taken.contains(&start) {
            continue;
        }
        let Some(n0) = classify_norm(graph, start) else { continue };
        let first_pos = pos[&start];
        let t0 = &graph.tensors[start];
        let (n, nel) = (t0.ne[0] as u32, t0.nelements());
        // an input produced by a node must come before the chain starts (it cannot depend on the chain)
        let before = |s: TensorId| pos.get(&graph.root(s).0).map_or(true, |&p| p < first_pos);
        if !before(n0.x) {
            continue;
        }
        let mut inputs: Vec<(TensorId, Kind, Ty)> = vec![(n0.x, Kind::Full, Ty::F32)];
        let mut consts: Vec<u32> = Vec::new();
        let mut stages: Vec<String> = Vec::new();
        let mut members: Vec<TensorId> = vec![start];
        let mut taps: Vec<(usize, TensorId)> = Vec::new();
        let add_input = |inputs: &mut Vec<(TensorId, Kind, Ty)>, s: TensorId, k: Kind, ty: Ty| -> usize {
            inputs.iter().position(|&(i, kk, _)| i == s && kk == k).unwrap_or_else(|| {
                inputs.push((s, k, ty));
                inputs.len() - 1
            })
        };
        // stage 0: the norm
        let w_idx = match n0.w {
            Some((w, k)) => {
                if !before(w) {
                    continue;
                }
                Some(add_input(&mut inputs, w, k, Ty::F32))
            }
            None => None,
        };
        consts.push(n0.eps.to_bits());
        stages.push(format!("(norm {} {})", w_idx.map_or("#f".to_string(), |i| i.to_string()), consts.len() - 1));
        let mut tail = start;
        let mut min_ext = usize::MAX; // earliest outside reader of a tapped value
        loop {
            let mut next: Option<(TensorId, String, Vec<(TensorId, Kind, Ty)>, Vec<u32>)> = None;
            for &c in &cons[tail] {
                if taken.contains(&c) || members.contains(&c) {
                    continue;
                }
                let tc = &graph.tensors[c];
                if tc.nelements() != nel || tc.ne[0] as u32 != n {
                    continue;
                }
                let mut ins = inputs.clone();
                let mut cs = consts.clone();
                let stage: Option<String> = if let Some(nn) = classify_norm(graph, c) {
                    if !flat_of(graph, nn.x, tail) {
                        None
                    } else {
                        match nn.w {
                            Some((w, k)) if !before(w) => None,
                            w => {
                                let wi = w.map(|(w, k)| add_input(&mut ins, w, k, Ty::F32));
                                cs.push(nn.eps.to_bits());
                                Some(format!("(norm {} {})", wi.map_or("#f".to_string(), |i| i.to_string()), cs.len() - 1))
                            }
                        }
                    }
                } else if let Some(e) = classify(graph, c) {
                    // elementwise: the tail is an operand, the others come from before the chain
                    if e.out != Ty::F32 || !e.srcs.iter().any(|&(s, k)| k == Kind::Full && flat_of(graph, s, tail)) {
                        None
                    } else {
                        let mut ops: Vec<String> = Vec::new();
                        let mut ok = true;
                        for &(s, k) in &e.srcs {
                            if k == Kind::Full && flat_of(graph, s, tail) {
                                ops.push("x".into());
                            } else if before(s) {
                                ops.push(format!("x{}", add_input(&mut ins, s, k, graph.tensors[s].ty)));
                            } else {
                                ok = false;
                            }
                        }
                        if !ok || ins.iter().any(|&(_, _, ty)| ty != Ty::F32) {
                            None
                        } else {
                            let mut cc = |v: f32| {
                                cs.push(v.to_bits());
                                format!("c{}", cs.len() - 1)
                            };
                            let expr = match &e.body {
                                Body::Bin(f) => format!("({f} {} {})", ops[0], ops[1]),
                                Body::Scale(a, b) => {
                                    let (ca, cb) = (cc(*a), cc(*b));
                                    format!("(+ (* {} {ca}) {cb})", ops[0])
                                }
                                Body::Unary(f) => format!("(unary {f} {})", ops[0]),
                                _ => String::new(),
                            };
                            if expr.is_empty() {
                                None
                            } else {
                                Some(format!("(ew ((t0 {expr})) t0)"))
                            }
                        }
                    }
                } else {
                    None
                };
                let Some(stage) = stage else { continue };
                // the tail's other readers must run after this stage (and after any later one)
                let others = cons[tail].iter().filter(|&&o| o != c).map(|o| pos[o]);
                let new_min = others.chain(std::iter::once(min_ext)).min().unwrap_or(usize::MAX);
                if pos[&c] >= new_min {
                    continue;
                }
                min_ext = new_min;
                next = Some((c, stage, ins, cs));
                break;
            }
            let Some((c, stage, ins, cs)) = next else { break };
            // the value we are leaving is written out if anything else reads it
            let others = cons[tail].iter().any(|&o| o != c) || keep.contains(&tail);
            if others {
                taps.push((stages.len() - 1, tail));
            }
            stages.push(stage);
            inputs = ins;
            consts = cs;
            members.push(c);
            tail = c;
        }
        if members.len() < 2 {
            continue;
        }
        // emit
        let root = tail;
        let mut push: Vec<u32> = vec![(nel / n as i64) as u32];
        let mut spec = Vec::new();
        for &(_, k, _) in &inputs {
            match k {
                Kind::Full => spec.push("(f32 full)".to_string()),
                Kind::Scalar => spec.push("(f32 scalar)".to_string()),
                Kind::Row(r) => {
                    push.push(r);
                    spec.push("(f32 row)".to_string());
                }
            }
        }
        push.extend(&consts);
        let args = format!("{n} '({}) '({}) {{OUTS}} {}", spec.join(" "), stages.join(" "), consts.len());
        for &m in &members {
            taken.insert(m);
            if m != root {
                dead.insert(m);
            }
        }
        let mut src: Vec<Option<TensorId>> = inputs.iter().map(|&(s, _, _)| Some(s)).collect();
        let n_inputs = src.len();
        for &(_, t) in &taps {
            src.push(Some(t));
            // the tapped tensor is now written by the fused node, not by a node of its own
            dead.remove(&t);
            dead.insert(t);
        }
        let last_stage = stages.len() - 1;
        let t = &mut graph.tensors[root];
        t.op = "FUSED_ROW".into();
        t.src = src;
        t.op_params = [0; 16];
        graph.fused.insert(root, Fused { args, push, n_inputs, taps: taps.iter().map(|&(s, _)| s).collect(), last_stage });
        formed += 1;
    }
    // view nodes of removed tensors go too (a tapped tensor is still allocated: it is a source of the chain)
    loop {
        let before = dead.len();
        for &id in &graph.nodes {
            let t = &graph.tensors[id];
            if t.is_view_op() && t.op != "NONE" && t.src.iter().flatten().any(|&s| dead.contains(&graph.root(s).0) && !is_tap(graph, graph.root(s).0)) {
                dead.insert(id);
            }
        }
        if dead.len() == before {
            break;
        }
    }
    graph.nodes.retain(|id| !dead.contains(id));
    formed
}

/// A tensor some fused row chain writes besides its own result.
fn is_tap(graph: &Graph, id: TensorId) -> bool {
    graph.fused.iter().any(|(&root, f)| f.n_inputs > 0 && graph.tensors[root].src.iter().skip(f.n_inputs).flatten().any(|&t| t == id))
}

// ---- norm + rope + set_rows ---------------------------------------------------------------------

/// `RMS_NORM[_MUL]`, then optionally `ROPE`, then optionally `SET_ROWS` into a cache, each stage
/// read only by the next, become one `FUSED_NR` node (`k:norm-rope`): a workgroup per row, the
/// normalized row in shared memory for the rotation, the result stored straight into the cache.
/// The node takes the id of the last stage. Sources: x, weight, positions, frequency factors,
/// row index, cache (absent ones are `None`); `op_params` are the rope's with the norm's epsilon in
/// word 0; `Fused.args` is `"<ncols> <weighted> <mode> <dty> <set-rows>"`.
pub fn norm_rope(graph: &mut Graph) -> usize {
    let cons = consumers(graph);
    let keep: HashSet<TensorId> = graph.keep.iter().map(|&k| graph.root(k).0).collect();
    let mut dead: HashSet<TensorId> = HashSet::new();
    let mut formed = 0;
    let nodes = graph.nodes.clone();
    let sole = |id: TensorId, c: TensorId| cons[id].as_slice() == [c] && !keep.contains(&id) && !is_tap(graph, id);
    let mut plans = Vec::new();
    for &start in &nodes {
        let Some(n0) = classify_norm(graph, start) else { continue };
        let tn = &graph.tensors[start];
        if tn.ne[0] > 4096 || tn.ne[0] % 2 != 0 || n0.w.is_some_and(|(_, k)| k == Kind::Full && tn.ne[1..] != [1, 1, 1]) {
            continue;
        }
        let ncols = tn.ne[0] as u32;
        let mut last = start;
        let mut rope: Option<TensorId> = None;
        // ROPE
        if let [c] = cons[start].as_slice() {
            let tr = &graph.tensors[*c];
            if tr.op == "ROPE" && sole(start, *c) && tr.ty == Ty::F32 && tr.is_contiguous() && tr.ne == tn.ne && tr.view_src.is_none() && tr.storage == Storage::Temp
                && flat_of(graph, tr.src(0).unwrap_or(usize::MAX), start)
                && matches!(tr.op_params[2], 0 | 2)
                && tr.op_params[1] > 0 && tr.op_params[1] % 2 == 0 && tr.op_params[1] as i64 <= tn.ne[0]
                && tr.src(1).is_some_and(|p| graph.tensors[p].ty == Ty::I32 && graph.tensors[p].is_contiguous())
                && tr.src(2).map_or(true, |f| graph.tensors[f].ty == Ty::F32 && graph.tensors[f].is_contiguous())
                && tr.src(1).is_some_and(|p| graph.tensors[p].nelements() >= tn.ne[2] * tn.ne[3] && tn.ne[3] == 1)
            {
                rope = Some(*c);
                last = *c;
            }
        }
        // SET_ROWS (one row)
        let mut set: Option<TensorId> = None;
        if let [c] = cons[last].as_slice() {
            let ts = &graph.tensors[*c];
            if ts.op == "SET_ROWS" && sole(last, *c) && matches!(ts.ty, Ty::F32 | Ty::F16) && tn.ne[1..] == [1, 1, 1] && ts.nb[0] == ts.ty.size()
                && ts.src(0).is_some_and(|a| flat_of(graph, a, last))
                && ts.src(1).is_some_and(|i| graph.tensors[i].ty == Ty::I32)
            {
                set = Some(*c);
                last = *c;
            }
        }
        if last == start {
            continue;
        }
        plans.push((start, rope, set, last, n0, ncols));
    }
    for (start, rope, set, last, n0, ncols) in plans {
        let tl = &graph.tensors[last];
        let dty = if set.is_some() { tl.ty } else { Ty::F32 };
        let (mode, mut params) = match rope {
            Some(r) => (if graph.tensors[r].op_params[2] == 2 { "neox" } else { "norm" }, graph.tensors[r].op_params),
            None => ("#f", [0i32; 16]),
        };
        params[0] = n0.eps.to_bits() as i32;
        let (pos, ff) = match rope {
            Some(r) => (graph.tensors[r].src(1), graph.tensors[r].src(2)),
            None => (None, None),
        };
        let ix = set.and_then(|s| graph.tensors[s].src(1));
        let cache = set.and_then(|s| graph.tensors[s].src(2));
        let src = vec![Some(n0.x), n0.w.map(|(w, _)| w), pos, ff, ix, cache];
        let args = format!("{ncols} {} {mode} {} {}", n0.w.is_some() as u8, if dty == Ty::F16 { "f16" } else { "f32" }, set.is_some() as u8);
        dead.insert(start);
        if let (Some(r), Some(_)) = (rope, set) {
            dead.insert(r);
        }
        let t = &mut graph.tensors[last];
        t.op = "FUSED_NR".into();
        t.src = src;
        t.op_params = params;
        graph.fused.insert(last, Fused { args, push: vec![], n_inputs: 0, taps: vec![], last_stage: 0 });
        formed += 1;
    }
    loop {
        let before = dead.len();
        for &id in &graph.nodes {
            let t = &graph.tensors[id];
            if t.is_view_op() && t.op != "NONE" && t.src.iter().flatten().any(|&s| dead.contains(&graph.root(s).0)) {
                dead.insert(id);
            }
        }
        if dead.len() == before {
            break;
        }
    }
    graph.nodes.retain(|id| !dead.contains(id));
    formed
}

// ---- gate / up / gelu-mul -----------------------------------------------------------------------

/// `gelu(W_gate x) * (W_up x)` over one q8_0 activation vector (an FFN's gate and up projections and
/// the gated product) becomes one `FUSED_GU` node (`k:gateup-q8`) that writes the product as int8
/// blocks for the down projection. Needs subgroups of 32 (a workgroup owns 32 rows = one block) and
/// every reader of the product to be a dp4a mat-vec.
pub fn gate_up(graph: &mut Graph, dot_ok: &dyn Fn(&Graph, TensorId) -> bool, subgroup32: bool) -> usize {
    if !subgroup32 {
        return 0;
    }
    let cons = consumers(graph);
    let keep: HashSet<TensorId> = graph.keep.iter().map(|&k| graph.root(k).0).collect();
    let mut dead: HashSet<TensorId> = HashSet::new();
    let mut found = Vec::new();
    for &id in &graph.nodes {
        let t = &graph.tensors[id];
        if t.op != "MUL" || t.ty != Ty::F32 || t.view_src.is_some() || !t.is_contiguous() || t.ne[1..] != [1, 1, 1] || t.ne[0] % 32 != 0 || keep.contains(&id) || cons[id].is_empty() {
            continue;
        }
        let (Some(s0), Some(s1)) = (t.src(0), t.src(1)) else { continue };
        let single = |x: TensorId, c: TensorId| {
            let (r, offs) = graph.root(x);
            offs == 0 && cons[r].as_slice() == [c] && !keep.contains(&r) && graph.tensors[x].nelements() == graph.tensors[r].nelements()
        };
        // one operand is gelu(gate), the other the up projection
        let pick = |gelu_side: TensorId, up_side: TensorId| -> Option<(TensorId, TensorId, TensorId)> {
            let (gr, ur) = (graph.root(gelu_side).0, graph.root(up_side).0);
            let un = &graph.tensors[gr];
            if un.op != "UNARY" || un.op_params[0] != 8 || !single(gelu_side, id) || !single(up_side, id) {
                return None;
            }
            let g = un.src(0)?;
            if !single(g, gr) {
                return None;
            }
            Some((gr, graph.root(g).0, ur))
        };
        let Some((un, g, u)) = pick(s0, s1).or_else(|| pick(s1, s0)) else { continue };
        let (tg, tu) = (&graph.tensors[g], &graph.tensors[u]);
        if tg.op != "MUL_MAT" || tu.op != "MUL_MAT" || tg.ne != tu.ne || tg.ne[0] != t.ne[0] || tg.ty != Ty::F32 || tu.ty != Ty::F32 {
            continue;
        }
        let (Some(wg), Some(wu), Some(xg), Some(xu)) = (tg.src(0), tu.src(0), tg.src(1), tu.src(1)) else { continue };
        let (twg, twu) = (&graph.tensors[wg], &graph.tensors[wu]);
        if twg.ty != Ty::Q8_0 || twu.ty != Ty::Q8_0 || twg.ne != twu.ne || twg.nb != twu.nb || graph.root(xg) != graph.root(xu) || graph.root(xg).1 != 0 {
            continue;
        }
        let tx = &graph.tensors[xg];
        if tx.ty != Ty::F32 || !tx.is_contiguous() || tx.ne != [twg.ne[0], 1, 1, 1] || state_rooted(graph, xg) || !dot_ok(graph, g) || !dot_ok(graph, u) {
            continue;
        }
        // every reader of the product is a dp4a mat-vec (the result is only written as int8)
        let readers_ok = cons[id].iter().all(|&m| {
            let mm = &graph.tensors[m];
            mm.op == "MUL_MAT"
                && mm.src(1).is_some_and(|b| {
                    let (r, offs) = graph.root(b);
                    r == id && offs == 0 && graph.tensors[b].nelements() == t.nelements() && graph.tensors[b].is_contiguous()
                })
                && dot_ok(graph, m)
        });
        if readers_ok {
            found.push((id, wg, wu, xg, [un, g, u]));
        }
    }
    let n = found.len();
    for (id, wg, wu, x, gone) in found {
        let t = &mut graph.tensors[id];
        t.op = "FUSED_GU".into();
        t.src = vec![Some(wg), Some(wu), Some(x)];
        t.op_params = [0; 16];
        graph.q8_out.insert(id);
        dead.extend(gone);
    }
    loop {
        let before = dead.len();
        for &id in &graph.nodes {
            let t = &graph.tensors[id];
            if t.is_view_op() && t.op != "NONE" && t.src.iter().flatten().any(|&s| dead.contains(&graph.root(s).0)) {
                dead.insert(id);
            }
        }
        if dead.len() == before {
            break;
        }
    }
    graph.nodes.retain(|id| !dead.contains(id));
    n
}
