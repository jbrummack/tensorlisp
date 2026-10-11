//! Lowering of graph nodes to Tree kernels (stdlib/tree_kernels.ss). Each function packs the
//! push constants in the order its generator declares them; `Ok(None)` means "no Tree kernel
//! for this node" and the caller falls back to ggml's shader.

use super::*;
use crate::vulkan::tree::TreeKernels;

const WG: u32 = 256;

/// The element type name the generators use, for the types they handle.
fn tname(ty: Ty) -> Option<&'static str> {
    match ty {
        Ty::F32 => Some("f32"),
        Ty::F16 => Some("f16"),
        Ty::I32 => Some("i32"),
        Ty::Q8_0 => Some("q8_0"),
        _ => None,
    }
}

/// Weight types the matmul / mat-vec / gather generators dequantize.
fn weight_name(ty: Ty) -> Option<&'static str> {
    matches!(ty, Ty::F32 | Ty::F16 | Ty::Q8_0).then(|| tname(ty).unwrap())
}

/// A q8_0 mat-vec that runs as "quantize the activations to int8, then dp4a" (what ggml-cuda's
/// `mul_mat_vec_q` does, see docs/design/tree.md): the shape and the scratch it needs. Used by the
/// planner (scratch bytes) and by the lowering, so both agree.
pub(super) struct DotPlan {
    pub n: u32,
    pub xq_bytes: u64,
    pub xd_bytes: u64,
}

pub(super) fn q8_dot_plan(graph: &Graph, caps: &Caps, id: TensorId) -> Option<DotPlan> {
    // Quantizing the activations to int8 changes results by ~0.5% (what ggml's CPU and CUDA backends
    // do too); TL_TREE_DOT=0 keeps the f32 path.
    if !caps.int_dot || std::env::var("TL_TREE_DOT").is_ok_and(|v| v == "0") || caps.subgroup_size == 0 || 128 % caps.subgroup_size != 0 {
        return None;
    }
    let t = &graph.tensors[id];
    if t.op != "MUL_MAT" {
        return None;
    }
    let (ta, tb) = (&graph.tensors[t.src(0)?], &graph.tensors[t.src(1)?]);
    let k = ta.ne[0];
    if ta.ty != Ty::Q8_0 || tb.ty != Ty::F32 || t.ty != Ty::F32 || k % 32 != 0 || tb.ne[0] != k || ta.nb[0] != 34 || tb.nb[0] != 4 || t.nb[0] != 4 {
        return None;
    }
    if ta.ne[2] != 1 || ta.ne[3] != 1 {
        return None;
    }
    let dense = |x: &crate::graph::Tensor| x.nb[2] == x.nb[1] * x.ne[1] as u64 && x.nb[3] == x.nb[2] * x.ne[2] as u64;
    let batch = tb.ne[2] * tb.ne[3];
    let n = if batch == 1 { tb.ne[1] } else if dense(tb) && dense(t) { tb.ne[1] * batch } else { return None };
    // TL_TREE_DOT_MAXN: the most columns the dp4a mat-vec takes (default 64; it holds a
    // sum per column per lane and beats dequantizing q8_0 inside the tensor-core matmul up to there).
    let maxn: i64 = std::env::var("TL_TREE_DOT_MAXN").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    if !(1..=maxn).contains(&n) {
        return None;
    }
    let words = n as u64 * k as u64 / 4;
    Some(DotPlan { n: n as u32, xq_bytes: 4 * words, xd_bytes: 4 * (n as u64 * k as u64 / 32) })
}

/// Flash attention with few queries (decode) runs as a split over the keys: one workgroup per chunk
/// of `FA_CHUNK` keys, then a merge. The scratch holds [m, l, o[hsv]] per (chunk, head, query).
pub(super) const FA_CHUNK: u32 = 128;

pub(super) struct FaSplit {
    pub nchunks: u32,
    pub bytes: u64,
}

pub(super) fn fa_split_plan(graph: &Graph, id: TensorId) -> Option<FaSplit> {
    let t = &graph.tensors[id];
    if t.op != "FLASH_ATTN_EXT" || std::env::var_os("TL_TREE_NOSPLIT").is_some() {
        return None;
    }
    let (q, k, v) = (&graph.tensors[t.src(0)?], &graph.tensors[t.src(1)?], &graph.tensors[t.src(2)?]);
    let (nq, nh, nb, kv, hsv) = (q.ne[1] as u64, q.ne[2] as u64, q.ne[3] as u64, k.ne[1] as u32, v.ne[0] as u64);
    if nq * nh * nb >= 96 || kv < 2 * FA_CHUNK || kv > 4096 || q.ne[0] % 32 != 0 || k.ne[0] != q.ne[0] || hsv % 32 != 0 {
        return None;
    }
    let nchunks = ceil_div(kv, FA_CHUNK);
    Some(FaSplit { nchunks, bytes: 4 * nq * nh * nb * nchunks as u64 * (hsv + 2) })
}

/// Where a producer that writes int8 blocks puts them: `xq` (one byte per element) then `xd` (an f32
/// scale per 32 elements), in the tensor's own allocation.
pub(super) fn q8_ranges(td: &Tn) -> (Range, Range) {
    q8_ranges_n(td, td.nelements() as u64)
}

/// The same for `n` activations held in `td`'s allocation (a sidecar: 9/8 of a byte per activation).
pub(super) fn q8_ranges_n(td: &Tn, n: u64) -> (Range, Range) {
    (Range::new(td.range.space, td.range.off, n), Range::new(td.range.space, td.range.off + n, n / 8))
}

impl<'a> Ctx<'a> {
    /// Quantize the activations of a q8_0 mat-vec into scratch, then the dp4a mat-vec over them.
    fn tree_q8_dot(&self, tree: &TreeKernels, id: TensorId, plan: DotPlan) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        let (m, k, n) = (ta.ne[1], ta.ne[0], plan.n);
        // The producer of the activations may have quantized them already (`Graph::q8_out`).
        let root_b = self.graph.root(b).0;
        let (xq, xd, quant) = if self.graph.q8_out.contains(&root_b) {
            let (xq, xd) = q8_ranges(&tb);
            (xq, xd, None)
        } else if let Some(&side) = self.graph.q8_side.get(&root_b) {
            let (xq, xd) = q8_ranges_n(&self.tn(side), tb.nelements() as u64);
            (xq, xd, None)
        } else {
            let off = self.layout.scratch[self.pos[&id]].ok_or_else(|| Error::Invalid("q8 dot scratch was not planned".into()))?;
            let xq = Range::new(Space::Step, self.arena_base + off, plan.xq_bytes);
            let xd = Range::new(Space::Step, self.arena_base + off + align(plan.xq_bytes), plan.xd_bytes);
            // one thread per four activations; the columns are rows of length k at stride tb.nb[1]
            let groups = n * k / 4;
            let blob = tree.blob("(k:quantize-q8)")?;
            (xq, xd, Some(self.launch(blob, vec![tb.range, xq, xd], words(&[groups, k, tb.nb[1]]), self.tree_grid(ceil_div(groups, WG)))?))
        };

        let rows = std::env::var("TL_TREE_DOT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let dot = tree.blob(&format!("(k:matvec-q8 {n} {rows})"))?;
        let sg = self.caps.subgroup_size;
        let push = [m, k, ta.nb[1], k / 4, k / 32, td.nb[1]];
        let grid = [ceil_div(m, (128 / sg) * rows), 1, 1];
        let d = self.launch(dot, vec![ta.range, xq, xd, td.range], words(&push), grid)?;
        Ok(Some(quant.into_iter().chain([d]).collect()))
    }

    pub(super) fn lower_tree(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        match t.op.as_str() {
            "ADD" | "SUB" | "MUL" | "DIV" => self.tree_binary(tree, id),
            "SCALE" | "SQR" | "SQRT" | "SIN" | "COS" | "LOG" | "UNARY" | "CPY" | "CONT" | "DUP" => self.tree_unary(tree, id),
            "REPEAT" => self.tree_repeat(tree, id),
            "CONCAT" => self.tree_concat(tree, id),
            "ARANGE" => self.tree_arange(tree, id),
            "GET_ROWS" => self.tree_get_rows(tree, id),
            "NORM" => self.tree_norm(tree, id),
            "RMS_NORM" | "RMS_NORM_MUL" => self.tree_rms_norm(tree, id),
            "SOFT_MAX" => self.tree_soft_max(tree, id),
            "SUM_ROWS" | "MEAN" => self.tree_sum_rows(tree, id),
            "MUL_MAT" => self.tree_mul_mat(tree, id),
            "ROPE" => self.tree_rope(tree, id),
            "SET_ROWS" => self.tree_set_rows(tree, id),
            "ARGMAX" => self.tree_argmax(tree, id),
            "FLASH_ATTN_EXT" => self.tree_flash_attn(tree, id),
            "FUSED_EW" => self.tree_fused(tree, id),
            "FUSED_ROW" => self.tree_fused_row(tree, id),
            "FUSED_NR" => self.tree_fused_nr(tree, id),
            "FUSED_GU" => self.tree_fused_gu(tree, id),
            "CUSTOM" => self.paged_custom(tree, id).map(Some),
            _ => Ok(None),
        }
    }

    /// A chain of row operations (`fuse_tree.rs`): inputs, then the tensors it also writes, then its result.
    fn tree_fused_row(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let f = self.graph.fused.get(&id).ok_or_else(|| Error::Invalid(format!("{} `{}` has no fusion details", t.op, t.name)))?;
        let mut bindings = Vec::new();
        let mut outs: Vec<String> = Vec::new();
        let side = self.graph.q8_side.get(&id).copied();
        for i in 0..t.src.len() {
            if Some(self.src(t, i)?) == side {
                continue; // the sidecar: bound after the result
            }
            let r = self.tn(self.src(t, i)?).range;
            bindings.push(r);
            if i >= f.n_inputs {
                outs.push(format!("({} f32)", f.taps[i - f.n_inputs]));
            }
        }
        let td = self.tn(id);
        let q8 = self.graph.q8_out.contains(&id);
        if q8 {
            let (xq, xd) = q8_ranges(&td);
            bindings.extend([xq, xd]);
        } else {
            bindings.push(td.range);
        }
        outs.push(format!("({} {})", f.last_stage, if q8 { "q8" } else { "f32" }));
        if let Some(side) = side {
            let (xq, xd) = q8_ranges_n(&self.tn(side), td.nelements() as u64);
            bindings.extend([xq, xd]);
            outs.push(format!("({} q8)", f.last_stage));
        }
        let args = f.args.replace("{OUTS}", &format!("'({})", outs.join(" ")));
        let rows = f.push[0];
        self.tree_launch(tree, &format!("(k:rowchain {args})"), bindings, &f.push, self.tree_grid(rows))
    }

    /// Gate and up projections with the gated product, as int8 blocks (`fuse_tree::gate_up`).
    fn tree_fused_gu(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (wa, wb, x) = (self.tn(self.src(t, 0)?), self.tn(self.src(t, 1)?), self.tn(self.src(t, 2)?));
        let td = self.tn(id);
        let (m, k) = (wa.ne[1], wa.ne[0]);
        let root_x = self.graph.root(self.src(t, 2)?).0;
        let (aq, ad, quant) = if self.graph.q8_out.contains(&root_x) {
            let (xq, xd) = q8_ranges(&x);
            (xq, xd, None)
        } else if let Some(&side) = self.graph.q8_side.get(&root_x) {
            let (xq, xd) = q8_ranges_n(&self.tn(side), x.nelements() as u64);
            (xq, xd, None)
        } else {
            let off = self.layout.scratch[self.pos[&id]].ok_or_else(|| Error::Invalid("gate/up scratch was not planned".into()))?;
            let (xq_bytes, xd_bytes) = (k as u64, k as u64 / 8);
            let xq = Range::new(Space::Step, self.arena_base + off, xq_bytes);
            let xd = Range::new(Space::Step, self.arena_base + off + align(xq_bytes), xd_bytes);
            let groups = k / 4;
            let blob = tree.blob("(k:quantize-q8)")?;
            (xq, xd, Some(self.launch(blob, vec![x.range, xq, xd], words(&[groups, k, x.nb[1]]), self.tree_grid(ceil_div(groups, WG)))?))
        };
        let rows: u32 = std::env::var("TL_TREE_GU_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let unroll: u32 = std::env::var("TL_TREE_GU_UNROLL").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let dot = tree.blob(&format!("(k:gateup-q8 {rows} {unroll})"))?;
        let (oq, od) = q8_ranges(&td);
        let d = self.launch(dot, vec![wa.range, wb.range, aq, ad, oq, od], words(&[m, k, wa.nb[1]]), [m / 32, 1, 1])?;
        Ok(Some(quant.into_iter().chain([d]).collect()))
    }

    /// Norm, rope and the cache write in one launch (`fuse_tree::norm_rope`).
    fn tree_fused_nr(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let f = self.graph.fused.get(&id).ok_or_else(|| Error::Invalid(format!("{} `{}` has no fusion details", t.op, t.name)))?;
        let a: Vec<&str> = f.args.split(' ').collect();
        let ncols: u32 = a[0].parse().unwrap();
        let (weighted, mode, dty, set_rows) = (a[1] == "1", a[2], a[3], a[4] == "1");
        let x = self.tn(self.src(t, 0)?);
        let td = self.tn(id);
        if x.nb[0] != 1 || td.nb[0] != 1 || (!set_rows && td.ty != Ty::F32) {
            return Ok(None);
        }
        let eps = f32::from_bits(t.op_params[0] as u32);
        let rows = x.nelements() / ncols;
        let mut bindings = vec![x.range];
        let mut push: Vec<u32> = vec![rows, x.ne[1], 0, 0, eps.to_bits()];
        let rope = mode != "#f";
        if rope {
            let n_dims = t.op_params[1];
            let (n_ctx_orig, freq_base, freq_scale) = (t.op_params[4], pf(t, 5), pf(t, 6));
            let (ext, attn, beta_fast, beta_slow) = (pf(t, 7), pf(t, 8), pf(t, 9), pf(t, 10));
            let corr = rope_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
            let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
            push[2] = n_dims as u32;
            push[3] = t.src(3).is_some() as u32;
            push.extend([freq_scale.to_bits(), ext.to_bits(), attn.to_bits(), corr[0].to_bits(), corr[1].to_bits(), theta_scale.to_bits()]);
        } else {
            push.extend([0; 6]);
        }
        if weighted {
            let w = self.tn(self.src(t, 1)?);
            bindings.push(w.range);
            push.push(w.ne[0]);
        }
        if rope {
            bindings.push(self.tn(self.src(t, 2)?).range);
            bindings.push(match t.src(3) {
                Some(ff) => self.tn(ff).range,
                None => unused(),
            });
        }
        if set_rows {
            let ix = self.tn(self.src(t, 4)?);
            bindings.push(ix.range);
            push.push(td.nb[1]);
        }
        bindings.push(td.range);
        let kind = format!("(k:norm-rope {ncols} {} {} '{dty} {})", if weighted { "#t" } else { "#f" }, if rope { format!("'{mode}") } else { "#f".into() }, if set_rows { "#t" } else { "#f" });
        self.tree_launch(tree, &kind, bindings, &push, self.tree_grid(rows))
    }

    /// A fused node (`fuse_tree.rs`): its kernel expression and push words were made with the node.
    fn tree_fused(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let f = self.graph.fused.get(&id).ok_or_else(|| Error::Invalid(format!("{} `{}` has no fusion details", t.op, t.name)))?;
        let mut bindings = Vec::new();
        for i in 0..t.src.len() {
            bindings.push(self.tn(self.src(t, i)?).range);
        }
        let td = self.tn(id);
        let (name, groups) = if self.graph.q8_out.contains(&id) {
            let (xq, xd) = q8_ranges(&td);
            bindings.extend([xq, xd]);
            ("k:ew-q8", ceil_div(td.nelements() / 4, WG))
        } else {
            bindings.push(td.range);
            ("k:ew", ceil_div(td.nelements(), WG))
        };
        self.tree_launch(tree, &format!("({name} {})", f.args), bindings, &f.push, self.tree_grid(groups))
    }

    /// Workgroups of an elementwise kernel over `n` elements (long grids spill into y).
    fn tree_grid(&self, groups: u32) -> [u32; 3] {
        let max = self.caps.max_groups[0].min(65535);
        if groups <= max {
            [groups.max(1), 1, 1]
        } else {
            [max, ceil_div(groups, max), 1]
        }
    }

    fn tree_launch(&self, tree: &TreeKernels, expr: &str, bindings: Vec<Range>, push: &[u32], grid: [u32; 3]) -> Result<Option<Vec<LaunchDesc>>> {
        let blob = tree.blob(expr)?;
        Ok(Some(vec![self.launch(blob, bindings, words(push), grid)?]))
    }

    fn tree_binary(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        let f32s = [ta.ty, tb.ty, td.ty].iter().all(|&x| x == Ty::F32);
        if !f32s || ta.ne != td.ne || (0..4).any(|k| tb.ne[k] == 0 || ta.ne[k] % tb.ne[k] != 0) {
            return Ok(None);
        }
        let op = t.op.to_ascii_lowercase();
        let mut push = vec![td.nelements(), td.ne[0], td.ne[1], td.ne[2]];
        for x in [ta.nb, tb.ne, tb.nb, td.nb] {
            push.extend(x);
        }
        self.tree_launch(tree, &format!("(k:binary '{op})"), vec![ta.range, tb.range, td.range], &push, self.tree_grid(ceil_div(td.nelements(), WG)))
    }

    /// Unary maps and copies (between f32, f16 and i32); `repeat` is [`Ctx::tree_repeat`].
    fn tree_unary(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        let copy = matches!(t.op.as_str(), "CPY" | "CONT" | "DUP");
        let (Some(sty), Some(dty)) = (tname(ta.ty), tname(td.ty)) else { return Ok(None) };
        if ta.ty == Ty::Q8_0 || td.ty == Ty::Q8_0 || (!copy && (ta.ty != Ty::F32 || td.ty != Ty::F32)) {
            return Ok(None);
        }
        let (f, p0, p1): (&str, f32, f32) = match t.op.as_str() {
            "SCALE" => ("scale", pf(t, 0), pf(t, 1)),
            "SQR" => ("sqr", 0.0, 0.0),
            "SQRT" => ("sqrt", 0.0, 0.0),
            "SIN" => ("sin", 0.0, 0.0),
            "COS" => ("cos", 0.0, 0.0),
            "LOG" => ("log", 0.0, 0.0),
            "CPY" | "CONT" | "DUP" => ("copy", 0.0, 0.0),
            "UNARY" => (
                match t.op_params[0] {
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
                    p => {
                        *self.stats.fallbacks.borrow_mut().entry(format!("UNARY#{p}")).or_default() += 1;
                        return Ok(None);
                    }
                },
                0.0,
                0.0,
            ),
            _ => return Ok(None),
        };
        let mode = if ta.ne == td.ne {
            "same"
        } else if copy && ta.nelements() == td.nelements() {
            "reshape"
        } else {
            return Ok(None);
        };
        self.tree_map(tree, f, mode, sty, dty, &ta, &td, p0, p1)
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_map(&self, tree: &TreeKernels, f: &str, mode: &str, sty: &str, dty: &str, ta: &Tn, td: &Tn, p0: f32, p1: f32) -> Result<Option<Vec<LaunchDesc>>> {
        let mut push = vec![td.nelements(), td.ne[0], td.ne[1], td.ne[2]];
        for x in [ta.ne, ta.nb, td.nb] {
            push.extend(x);
        }
        push.extend([p0.to_bits(), p1.to_bits()]);
        self.tree_launch(tree, &format!("(k:unary '{f} '{mode} '{sty} '{dty})"), vec![ta.range, td.range], &push, self.tree_grid(ceil_div(td.nelements(), WG)))
    }

    fn tree_repeat(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || (0..4).any(|k| ta.ne[k] == 0 || td.ne[k] % ta.ne[k] != 0) {
            return Ok(None);
        }
        self.tree_map(tree, "copy", "repeat", "f32", "f32", &ta, &td, 0.0, 0.0)
    }

    fn tree_concat(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        let dim = t.op_params[0];
        if !(0..4).contains(&dim) || [ta.ty, tb.ty, td.ty].iter().any(|&x| x != Ty::F32) {
            return Ok(None);
        }
        let mut push = vec![td.nelements(), td.ne[0], td.ne[1], td.ne[2], ta.ne[dim as usize]];
        for x in [ta.nb, tb.nb, td.nb] {
            push.extend(x);
        }
        self.tree_launch(tree, &format!("(k:concat {dim})"), vec![ta.range, tb.range, td.range], &push, self.tree_grid(ceil_div(td.nelements(), WG)))
    }

    fn tree_arange(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let td = self.tn(id);
        if td.ty != Ty::F32 {
            return Ok(None);
        }
        let push = [td.nelements(), t.op_params[0] as u32, t.op_params[2] as u32];
        self.tree_launch(tree, "(k:arange)", vec![td.range], &push, self.tree_grid(ceil_div(td.nelements(), WG)))
    }

    fn tree_get_rows(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, idx) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, ti, td) = (self.tn(a), self.tn(idx), self.tn(id));
        let Some(atype) = weight_name(ta.ty) else { return Ok(None) };
        if td.ty != Ty::F32 || ti.ty != Ty::I32 || ta.nb[0] != 1 {
            return Ok(None);
        }
        let mut push = vec![td.nelements(), td.ne[0], td.ne[1], td.ne[2]];
        push.extend(ta.nb);
        push.extend([ti.nb[0], ti.nb[1], ti.nb[2]]);
        push.extend(td.nb);
        self.tree_launch(tree, &format!("(k:get-rows '{atype})"), vec![ta.range, ti.range, td.range], &push, self.tree_grid(ceil_div(td.nelements(), WG)))
    }

    fn tree_norm(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || ta.nb[0] != 1 || !t.is_contiguous() {
            return Ok(None);
        }
        let push = [ta.nrows(), ta.ne[0], ta.ne[1], ta.ne[2], ta.nb[1], ta.nb[2], ta.nb[3], pf(t, 0).to_bits()];
        self.tree_launch(tree, "(k:norm)", vec![ta.range, td.range], &push, self.tree_grid(ta.nrows()))
    }

    /// `RMS_NORM`, or the fused `RMS_NORM_MUL` (src: x, weight; eps in `op_params[0]`).
    fn tree_rms_norm(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        let fused = t.op == "RMS_NORM_MUL";
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || ta.nb[0] != 1 || !t.is_contiguous() {
            return Ok(None);
        }
        let mut push = vec![ta.nrows(), ta.ne[0], ta.ne[1], ta.ne[2], ta.nb[1], ta.nb[2], ta.nb[3], pf(t, 0).to_bits()];
        let mut bindings = vec![ta.range];
        if fused {
            let tw = self.tn(self.src(t, 1)?);
            if tw.ty != Ty::F32 || tw.nb[0] != 1 || tw.ne[1..] != [1, 1, 1] {
                return Ok(None);
            }
            push.push(tw.ne[0]);
            bindings.push(tw.range);
        }
        // every reader is a dp4a mat-vec: write int8 blocks instead (Graph::q8_out)
        let q8 = self.graph.q8_out.contains(&id);
        if q8 {
            let (xq, xd) = q8_ranges(&td);
            bindings.extend([xq, xd]);
        } else {
            bindings.push(td.range);
        }
        self.tree_launch(tree, &format!("(k:rms-norm {} {})", if fused { "#t" } else { "#f" }, if q8 { "#t" } else { "#f" }), bindings, &push, self.tree_grid(ta.nrows()))
    }

    fn tree_soft_max(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        let mask = t.src(1).map(|m| self.tn(m));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || ta.nb[0] != 1 || !t.is_contiguous() || t.src(2).is_some() || pf(t, 1) != 0.0 {
            return Ok(None);
        }
        if let Some(m) = &mask {
            if m.ty != Ty::F32 || m.nb[0] != 1 || m.ne[0] < ta.ne[0] || m.ne[1] < ta.ne[1] || m.ne[2] == 0 || m.ne[3] == 0 {
                return Ok(None);
            }
        }
        let mut push = vec![ta.nrows(), ta.ne[0], ta.ne[1], ta.ne[2], ta.nb[1], ta.nb[2], ta.nb[3], pf(t, 0).to_bits()];
        let mut bindings = vec![ta.range];
        if let Some(m) = &mask {
            push.extend([m.nb[1], m.nb[2], m.nb[3], m.ne[2], m.ne[3]]);
            bindings.push(m.range);
        }
        bindings.push(td.range);
        self.tree_launch(tree, &format!("(k:soft-max {})", if mask.is_some() { "#t" } else { "#f" }), bindings, &push, self.tree_grid(ta.nrows()))
    }

    fn tree_sum_rows(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || ta.nb[0] != 1 || !t.is_contiguous() {
            return Ok(None);
        }
        let scale = if t.op == "MEAN" { 1.0 / ta.ne[0] as f32 } else { 1.0 };
        let push = [ta.nrows(), ta.ne[0], ta.ne[1], ta.ne[2], ta.nb[1], ta.nb[2], ta.nb[3], scale.to_bits()];
        self.tree_launch(tree, "(k:sum-rows)", vec![ta.range, td.range], &push, self.tree_grid(ta.nrows()))
    }

    fn tree_mul_mat(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        if let Some(plan) = q8_dot_plan(self.graph, &self.caps, id) {
            return self.tree_q8_dot(tree, id, plan);
        }
        let Some(atype) = weight_name(ta.ty) else { return Ok(None) };
        if tb.ty != Ty::F32 || td.ty != Ty::F32 || ta.nb[0] != 1 || tb.nb[0] != 1 || td.nb[0] != 1 || ta.ne[0] != tb.ne[0] {
            return Ok(None);
        }
        if ta.ne[2] == 0 || ta.ne[3] == 0 || tb.ne[2] % ta.ne[2] != 0 || tb.ne[3] % ta.ne[3] != 0 {
            return Ok(None);
        }
        let (m, mut n, k) = (ta.ne[1], tb.ne[1], tb.ne[0]);
        if td.ne != [m, n, tb.ne[2], tb.ne[3]] {
            return Ok(None);
        }
        // Weights shared by every batch (a single matrix) and dense activations: the batches are
        // just more columns, one bigger product instead of many small ones.
        let (mut ne12, mut ne13, mut r2, mut r3) = (tb.ne[2], tb.ne[3], tb.ne[2] / ta.ne[2], tb.ne[3] / ta.ne[3]);
        let dense = |x: &Tn| x.nb[2] == x.nb[1] * x.ne[1] && x.nb[3] == x.nb[2] * x.ne[2];
        if ta.ne[2] == 1 && ta.ne[3] == 1 && dense(&tb) && dense(&td) {
            n *= ne12 * ne13;
            (ne12, ne13, r2, r3) = (1, 1, 1, 1);
        }
        let batch = ne12 * ne13;

        // A few columns of one matrix: a subgroup per row of A, the weights read once.
        let sg = self.caps.subgroup_size;
        if n <= 8 && batch == 1 && k % 16 == 0 && sg > 0 && 128 % sg == 0 {
            let push = [m, k, ta.nb[1], tb.nb[1], td.nb[1]];
            // Rows per subgroup: the fewer columns, the more rows can share the activation loads.
            let rows: u32 = match n {
                1 => std::env::var("TL_TREE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(4),
                2 => 2,
                _ => 1,
            };
            let grid = [ceil_div(m, (128 / sg) * rows), 1, 1];
            return self.tree_launch(tree, &format!("(k:matvec '{atype} {n} {rows})"), vec![ta.range, tb.range, td.range], &push, grid);
        }

        // Tile by the number of columns (measured on TIPSv2: filling the GPU with smaller tiles
        // or larger register blocks was slower, see docs/design/tree.md).
        let (mut bm, mut bn, mut bk, mut tm, mut tn) = match n {
            0..=16 => (64, 16, 16, 4, 1),
            17..=64 => (32, 32, 16, 2, 2),
            65..=256 => (64, 64, 32, 4, 4),
            _ => (128, 64, 16, 8, 4),
        };
        // TL_TREE_MM="bm bn bk tm tn" forces one tile for every matmul (tuning experiments).
        if let Ok(v) = std::env::var("TL_TREE_MM") {
            let p: Vec<u32> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if let [a, b, c, d, e] = p[..] {
                (bm, bn, bk, tm, tn) = (a, b, c, d, e);
            }
        }
        if std::env::var_os("TL_TREE_SHAPES").is_some() {
            eprintln!("tree matmul {atype} m={m} n={n} k={k} batch={batch} -> tile {bm}x{bn}x{bk} regs {tm}x{tn}");
        }
        let push = [m, n, k, ta.nb[1], tb.nb[1], td.nb[1], ta.nb[2], ta.nb[3], tb.nb[2], tb.nb[3], td.nb[2], td.nb[3], ne12, r2, r3];
        // Tensor cores (cooperative matrices, f16 inputs with f32 accumulation) when the device has
        // them and there are enough columns to fill a 16x16 fragment; TL_TREE_NOCM=1 keeps the scalar
        // kernel (exact f32), TL_TREE_CM="bm bn bk wm wn" forces a tile.
        if self.caps.coopmat && self.caps.subgroup_size == 32 && n >= 16 && std::env::var_os("TL_TREE_NOCM").is_none() {
            // 64x64 tiles when there are enough of them to fill the GPU (two and a bit per SM), else
            // 32x32 tiles with four subgroups (measured on TIPSv2's n = 192 products, docs/design/tree.md).
            let tiles64 = ceil_div(m, 64) * ceil_div(n, 64) * batch;
            let cores = self.caps.shader_cores;
            let (mut cbm, mut cbn, mut cbk, mut wm, mut wn) = if n < 32 {
                (64, 16, 16, 2, 1)
            } else if tiles64 >= cores * 2 + cores / 4 {
                (64, 64, 16, 2, 2)
            } else {
                (32, 32, 32, 2, 2)
            };
            if let Ok(v) = std::env::var("TL_TREE_CM") {
                let p: Vec<u32> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if let [a, b, c, d, e] = p[..] {
                    (cbm, cbn, cbk, wm, wn) = (a, b, c, d, e);
                }
            }
            let grid = [ceil_div(m, cbm), ceil_div(n, cbn), batch];
            return self.tree_launch(tree, &format!("(k:matmul-cm '{atype} {cbm} {cbn} {cbk} {wm} {wn})"), vec![ta.range, tb.range, td.range], &push, grid);
        }
        let grid = [ceil_div(m, bm), ceil_div(n, bn), batch];
        self.tree_launch(tree, &format!("(k:matmul '{atype} {bm} {bn} {bk} {tm} {tn})"), vec![ta.range, tb.range, td.range], &push, grid)
    }

    fn tree_rope(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, pos) = (self.src(t, 0)?, self.src(t, 1)?);
        let ff = t.src(2);
        let (ta, tp, td) = (self.tn(a), self.tn(pos), self.tn(id));
        let mode = match t.op_params[2] {
            0 => "norm",
            2 => "neox",
            _ => return Ok(None),
        };
        let n_dims = t.op_params[1];
        if ta.ty != Ty::F32 || td.ty != Ty::F32 || tp.ty != Ty::I32 || ta.nb[0] != 1 || td.nb[0] != 1 || tp.nb[0] != 1 || ta.ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 {
            return Ok(None);
        }
        let (n_ctx_orig, freq_base, freq_scale) = (t.op_params[4], pf(t, 5), pf(t, 6));
        let (ext, attn, beta_fast, beta_slow) = (pf(t, 7), pf(t, 8), pf(t, 9), pf(t, 10));
        let corr = rope_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let n = ta.nrows() * (ta.ne[0] / 2);
        let push = [
            n,
            ta.ne[0],
            ta.ne[1],
            ta.ne[2],
            n_dims as u32,
            ta.nb[1],
            ta.nb[2],
            ta.nb[3],
            td.nb[1],
            td.nb[2],
            td.nb[3],
            ff.is_some() as u32,
            freq_scale.to_bits(),
            ext.to_bits(),
            attn.to_bits(),
            corr[0].to_bits(),
            corr[1].to_bits(),
            theta_scale.to_bits(),
        ];
        let ffr = match ff {
            Some(f) => {
                let tf = self.tn(f);
                if tf.ty != Ty::F32 {
                    return Ok(None);
                }
                tf.range
            }
            None => unused(),
        };
        self.tree_launch(tree, &format!("(k:rope '{mode})"), vec![ta.range, tp.range, ffr, td.range], &push, self.tree_grid(ceil_div(n, WG)))
    }

    fn tree_set_rows(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (a, idx) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, ti, td) = (self.tn(a), self.tn(idx), self.tn(id));
        if ta.ty != Ty::F32 || ti.ty != Ty::I32 || !matches!(td.ty, Ty::F32 | Ty::F16) {
            return Ok(None);
        }
        let mut push = vec![ta.nelements(), ta.ne[0], ta.ne[1], ta.ne[2]];
        push.extend(ta.nb);
        push.extend([ti.nb[0], ti.nb[1], ti.nb[2], ti.ne[1], ti.ne[2]]);
        push.extend(td.nb);
        self.tree_launch(tree, &format!("(k:set-rows '{})", tname(td.ty).unwrap()), vec![ta.range, ti.range, td.range], &push, self.tree_grid(ceil_div(ta.nelements(), WG)))
    }

    fn tree_argmax(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::I32 || ta.nb[0] != 1 || ta.ne[2] != 1 || ta.ne[3] != 1 {
            return Ok(None);
        }
        let push = [ta.ne[1], ta.ne[0], ta.nb[1]];
        self.tree_launch(tree, "(k:argmax)", vec![ta.range, td.range], &push, self.tree_grid(ta.ne[1]))
    }

    fn tree_flash_attn(&self, tree: &TreeKernels, id: TensorId) -> Result<Option<Vec<LaunchDesc>>> {
        let t = self.t(id);
        let (q, k, v) = (self.src(t, 0)?, self.src(t, 1)?, self.src(t, 2)?);
        let (tq, tk, tv, td) = (self.tn(q), self.tn(k), self.tn(v), self.tn(id));
        let mask = t.src(3).map(|m| self.tn(m));
        if tq.ty != Ty::F32 || td.ty != Ty::F32 || !matches!(tk.ty, Ty::F16 | Ty::F32) || !matches!(tv.ty, Ty::F16 | Ty::F32) {
            return Ok(None);
        }
        if t.src(4).is_some() || pf(t, 1) != 0.0 || tq.nb[0] != 1 || tk.nb[0] != 1 || tv.nb[0] != 1 || td.nb[0] != 1 {
            return Ok(None);
        }
        let (hsk, hsv, nq, nh, nb, kv) = (tq.ne[0], tv.ne[0], tq.ne[1], tq.ne[2], tq.ne[3], tk.ne[1]);
        if tk.ne[0] != hsk || tk.ne[2] == 0 || nh % tk.ne[2] != 0 || tk.ne[3] != nb || tv.ne[1] != kv || tv.ne[2] != tk.ne[2] || tv.ne[3] != nb {
            return Ok(None);
        }
        if td.ne != [hsv, nh, nq, nb] {
            return Ok(None);
        }
        let maxkv = match kv {
            0..=2048 => 2048,
            2049..=4096 => 4096,
            _ => return Ok(None),
        };
        if let Some(m) = &mask {
            if m.ty != Ty::F16 || m.nb[0] != 1 || m.ne[0] < kv || m.ne[1] < nq || m.ne[2] == 0 || m.ne[3] == 0 {
                return Ok(None);
            }
        }
        let (scale, cap) = (pf(t, 0), pf(t, 2));
        let (m1, m2, m3, mne2, mne3) = mask.as_ref().map_or((0, 0, 0, 1, 1), |m| (m.nb[1], m.nb[2], m.nb[3], m.ne[2], m.ne[3]));
        let push = [
            nq,
            nh,
            hsk,
            hsv,
            kv,
            nh / tk.ne[2],
            tq.nb[1],
            tq.nb[2],
            tq.nb[3],
            tk.nb[1],
            tk.nb[2],
            tk.nb[3],
            tv.nb[1],
            tv.nb[2],
            tv.nb[3],
            td.nb[1],
            td.nb[2],
            td.nb[3],
            m1,
            m2,
            m3,
            mne2,
            mne3,
            scale.to_bits(),
            cap.to_bits(),
        ];
        let mut bindings = vec![tq.range, tk.range, tv.range];
        if let Some(m) = &mask {
            bindings.push(m.range);
        }
        bindings.push(td.range);
        let (kt, vt, mk, sc) = (tname(tk.ty).unwrap(), tname(tv.ty).unwrap(), if mask.is_some() { "#t" } else { "#f" }, if cap != 0.0 { "#t" } else { "#f" });
        if let Some(split) = fa_split_plan(self.graph, id) {
            let off = self.layout.scratch[self.pos[&id]].ok_or_else(|| Error::Invalid("flash attention scratch was not planned".into()))?;
            let part = Range::new(Space::Step, self.arena_base + off, split.bytes);
            let push = [
                nq, nh, hsk, hsv, kv, nh / tk.ne[2], split.nchunks, tq.nb[1], tq.nb[2], tq.nb[3], tk.nb[1], tk.nb[2], tk.nb[3], tv.nb[1], tv.nb[2], tv.nb[3], m1, m2, m3, mne2, mne3, scale.to_bits(),
                cap.to_bits(),
            ];
            let mut bindings = vec![tq.range, tk.range, tv.range];
            if let Some(m) = &mask {
                bindings.push(m.range);
            }
            bindings.push(part);
            let a = self.launch(tree.blob(&format!("(k:flash-attn-part '{kt} '{vt} {mk} {sc} {FA_CHUNK} {hsk} {hsv})"))?, bindings, words(&push), [split.nchunks, nh, nq * nb])?;
            let b = if self.graph.q8_out.contains(&id) {
                let (xq, xd) = q8_ranges(&td);
                self.launch(tree.blob("(k:flash-attn-combine #t)")?, vec![part, xq, xd], words(&[nq, nh, hsv, split.nchunks]), [nh, nq * nb, 1])?
            } else {
                let push = [nq, nh, hsv, split.nchunks, td.nb[1], td.nb[2], td.nb[3]];
                self.launch(tree.blob("(k:flash-attn-combine)")?, vec![part, td.range], words(&push), [nh, nq * nb, 1])?
            };
            return Ok(Some(vec![a, b]));
        }
        let expr = format!("(k:flash-attn '{kt} '{vt} {mk} {sc} {maxkv})");
        self.tree_launch(tree, &expr, bindings, &push, [nq, nh, nb])
    }
}

/// What a `CUSTOM` node's `userdata` (ggml keeps it in `op_params[4..6]`) says: kind (4 bits), kv heads
/// (4), block size (8), max context length (16), scale (f32). See `Ctx::custom` in `cuda/lower.rs`.
struct PagedParams {
    kind: u8,
    kv_heads: u32,
    block_size: u32,
    max_ctx: u32,
    scale: f32,
}

fn paged_params(t: &Tensor) -> PagedParams {
    let ud = t.op_params[4] as u32 as u64 | (t.op_params[5] as u32 as u64) << 32;
    PagedParams {
        kind: (ud & 15) as u8,
        kv_heads: ((ud >> 4) & 15) as u32,
        block_size: ((ud >> 8) & 255) as u32,
        max_ctx: ((ud >> 16) & 0xffff) as u32,
        scale: f32::from_bits((ud >> 32) as u32),
    }
}

/// Scratch for the partial results of paged attention: [m, l, o[hs]] per (chunk, head, sequence).
pub(super) fn paged_scratch(graph: &Graph, id: TensorId) -> u64 {
    let t = &graph.tensors[id];
    let p = paged_params(t);
    if p.kind != 1 {
        return 0;
    }
    let Some(q) = t.src(0).map(|q| &graph.tensors[q]) else { return 0 };
    let nchunks = ceil_div(p.max_ctx, FA_CHUNK) as u64;
    4 * nchunks * q.ne[1] as u64 * q.ne[2] as u64 * (q.ne[0] as u64 + 2)
}

impl<'a> Ctx<'a> {
    /// Paged attention and the cache write (Scheme's `paged-attention` / `paged-cache-write`) on Tree
    /// kernels, over vLLM's pool layouts: keys `[blocks, kv_heads, hs/8, bs, 8]`, values
    /// `[blocks, kv_heads, hs, bs]`, f16. Attention is split over chunks of `FA_CHUNK` keys
    /// (`k:paged-attn-part`) and merged by the flash-attention combine kernel.
    pub(super) fn paged_custom(&self, tree: &TreeKernels, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let p = paged_params(t);
        let s = |i| self.src(t, i);
        let f16 = |x: &Tn| x.ty == Ty::F16;
        match p.kind {
            1 => {
                let (q, kc, vc, tables, lens) = (self.tn(s(0)?), self.tn(s(1)?), self.tn(s(2)?), self.tn(s(3)?), self.tn(s(4)?));
                let td = self.tn(id);
                let (hs, heads, seqs) = (q.ne[0], q.ne[1], q.ne[2]);
                if !(f16(&q) && f16(&kc) && f16(&vc) && f16(&td)) {
                    return unsupported(t, "paged attention takes f16 q, caches and output");
                }
                if tables.ty != Ty::I32 || lens.ty != Ty::I32 {
                    return unsupported(t, "block tables and context lengths are i32");
                }
                if q.nb[1] != hs || q.nb[2] != hs * heads || td.ne[..3] != q.ne[..3] || td.nb[1] != hs || td.nb[2] != hs * heads {
                    return unsupported(t, "paged attention takes a contiguous q and an output of its shape");
                }
                if hs % 32 != 0 || p.kv_heads == 0 || heads % p.kv_heads != 0 {
                    return unsupported(t, "paged attention takes a head size that is a multiple of 32 and heads that divide by the kv heads");
                }
                let nchunks = ceil_div(p.max_ctx, FA_CHUNK);
                let off = self.layout.scratch[self.pos[&id]].ok_or_else(|| Error::Invalid("paged attention scratch was not planned".into()))?;
                let part = Range::new(Space::Step, self.arena_base + off, paged_scratch(self.graph, id));
                let (kvh, kvb) = (hs * p.block_size, p.kv_heads * hs * p.block_size);
                let push = [heads, nchunks, heads / p.kv_heads, tables.ne[0], kvb, kvh, p.scale.to_bits()];
                let bindings = vec![q.range, kc.range, vc.range, tables.range, lens.range, part];
                let a = self.launch(tree.blob(&format!("(k:paged-attn-part {hs} {} {FA_CHUNK})", p.block_size))?, bindings, words(&push), [nchunks, heads, seqs])?;
                let b = self.launch(
                    tree.blob("(k:flash-attn-combine #f 'f16)")?,
                    vec![part, td.range],
                    words(&[1, heads, hs, nchunks, td.nb[1], 0, td.nb[2]]),
                    [heads, seqs, 1],
                )?;
                Ok(vec![a, b])
            }
            2 => {
                let (key, value, kc, vc, slots) = (self.tn(s(0)?), self.tn(s(1)?), self.tn(s(2)?), self.tn(s(3)?), self.tn(s(4)?));
                let tokens = key.ne[1];
                if !(f16(&key) && f16(&value) && f16(&kc) && f16(&vc)) {
                    return unsupported(t, "the cache write takes f16 keys, values and caches");
                }
                if slots.nelements() != 2 * tokens || key.nb[1] != key.ne[0] || value.nb[1] != value.ne[0] || key.ne != value.ne || p.kv_heads == 0 || key.ne[0] % p.kv_heads != 0 {
                    return unsupported(t, "the cache write takes dense keys and values and one i64 slot per token");
                }
                let hs = key.ne[0] / p.kv_heads;
                let n = key.nelements();
                let push = [n, hs, p.kv_heads, p.kv_heads * hs * p.block_size, hs * p.block_size];
                let blob = tree.blob(&format!("(k:paged-cache-write {})", p.block_size))?;
                Ok(vec![self.launch(blob, vec![key.range, value.range, kc.range, vc.range, slots.range], words(&push), self.tree_grid(ceil_div(n, WG)))?])
            }
            _ => unsupported(t, "unknown custom op"),
        }
    }
}
