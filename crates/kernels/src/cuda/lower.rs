//! Lowers graph nodes to CUDA launches of our own kernels (`native/*.cu`).
//!
//! Every node becomes one launch (no fusion) of a kernel with the uniform
//! signature `(Op op, src0..src3, dst)`: `op` is a fixed-size descriptor blob
//! holding the shapes/strides of up to five tensors plus scalar slots, packed
//! by [`OpBlob`] to mirror `struct Op` in `native/common.cuh`.

use std::sync::Arc;

use super::paged_attn::{reshape_and_cache_func, PagedGeometry, PagedPrepared, PagedScratch};
use super::{Device, Func, Kernel};
use crate::graph::{Graph, Tensor, TensorId, Ty};
use crate::plan::{BufId, Layout, Place};
use crate::{DType, Dims, Error, Result};

/// One kernel launch with its arguments resolved to places.
pub struct Dispatch {
    /// The uniform-signature kernel; `None` for a [`Custom`] launch.
    pub func: Option<Func>,
    pub custom: Option<Custom>,
    pub op: OpBlob,
    pub srcs: [Option<Place>; 4],
    pub dst: Place,
    pub dims: Dims,
    /// For error messages and debugging: the node's op and name.
    pub label: String,
}

/// Launches with their own argument lists (the vendored paged-attention
/// kernels), for nodes the graph carries as ggml `CUSTOM` ops: see
/// [`Ctx::custom`] for the encoding.
pub enum Custom {
    PagedAttention(Box<PagedLaunch>),
    CacheWrite(Box<CacheLaunch>),
}

pub struct PagedLaunch {
    pub geometry: PagedGeometry,
    pub prepared: PagedPrepared,
    pub scratch: PagedScratch,
    pub q: Place,
    pub k_cache: Place,
    pub v_cache: Place,
    pub block_tables: Place,
    pub context_lens: Place,
    pub out: Place,
    pub max_blocks: usize,
    pub scale: f32,
    pub kv_block_stride: usize,
    pub kv_head_stride: usize,
}

pub struct CacheLaunch {
    pub func: Func,
    pub dtype: DType,
    pub tokens: usize,
    pub head_size: usize,
    pub kv_heads: usize,
    pub block_size: usize,
    pub key: Place,
    pub value: Place,
    pub k_cache: Place,
    pub v_cache: Place,
    pub slots: Place,
}

pub const OP_BYTES: usize = 5 * 72 + 12 * 8;

/// The `struct Op` of `native/common.cuh`.
#[derive(Clone)]
pub struct OpBlob(pub [u8; OP_BYTES]);

impl OpBlob {
    fn new() -> Self {
        OpBlob([0; OP_BYTES])
    }

    fn put(&mut self, at: usize, bytes: &[u8]) {
        self.0[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// Slots 0..=3 are sources, 4 is the destination.
    fn tensor(&mut self, slot: usize, t: &Tensor) {
        let base = slot * 72;
        for i in 0..4 {
            self.put(base + 8 * i, &t.ne[i].to_ne_bytes());
            self.put(base + 32 + 8 * i, &(t.nb[i] as i64).to_ne_bytes());
        }
        self.put(base + 64, &(t.ty.0 as i32).to_ne_bytes());
    }

    fn int(&mut self, slot: usize, v: i64) {
        self.put(5 * 72 + 8 * slot, &v.to_ne_bytes());
    }

    fn float(&mut self, slot: usize, v: f32) {
        self.int(slot, v.to_bits() as i32 as i64);
    }
}

pub struct Ctx<'a> {
    pub dev: &'a Device,
    pub core: Arc<str>,
    pub graph: &'a Graph,
    pub layout: &'a Layout,
    pub max_shared: u32,
    pub sm_count: u32,
}

fn unsupported<T>(t: &Tensor, why: &str) -> Result<T> {
    Err(Error::Unsupported { op: t.op.clone(), node: t.name.clone(), why: why.to_string() })
}

fn plain(ty: Ty) -> bool {
    matches!(ty.0, 0 | 1 | 30 | 26 | 27)
}

fn quant_ok(ty: Ty) -> bool {
    plain(ty) && ty.0 != 26 && ty.0 != 27 || matches!(ty.0, 2 | 3 | 6 | 7 | 8 | 10..=14)
}

fn blocks(n: i64, per: i64) -> u32 {
    ((n + per - 1) / per).max(1) as u32
}

/// Threads for a one-block-per-row kernel: a power of two in 32..=256.
fn row_threads(n: i64) -> u32 {
    (n.max(1) as u64).next_power_of_two().clamp(32, 256) as u32
}

impl Ctx<'_> {
    fn t(&self, id: TensorId) -> &Tensor {
        &self.graph.tensors[id]
    }

    fn place(&self, id: TensorId) -> Place {
        self.layout.place[id]
    }

    fn src(&self, node: &Tensor, i: usize) -> Result<TensorId> {
        node.src(i).ok_or_else(|| Error::Invalid(format!("{} `{}` is missing source {i}", node.op, node.name)))
    }

    fn func(&self, library: &Arc<str>, entry: &str) -> Result<Func> {
        self.dev.resolve(&Kernel { library: library.clone(), entry: entry.to_string() })
    }

    fn core(&self, entry: &str) -> Result<Func> {
        self.func(&self.core, entry)
    }

    fn dispatch(&self, id: TensorId, func: Func, op: OpBlob, srcs: &[TensorId], dims: Dims) -> Vec<Dispatch> {
        let t = self.t(id);
        let mut places = [None; 4];
        for (i, s) in srcs.iter().enumerate() {
            places[i] = Some(self.place(*s));
        }
        vec![Dispatch { func: Some(func), custom: None, op, srcs: places, dst: self.place(id), dims, label: format!("{} {}", t.op, t.name) }]
    }

    /// A blob with the destination in slot 4 and `srcs` in 0.. (all sources must exist).
    fn blob(&self, id: TensorId, srcs: &[TensorId]) -> OpBlob {
        let mut b = OpBlob::new();
        for (i, s) in srcs.iter().enumerate() {
            b.tensor(i, self.t(*s));
        }
        b.tensor(4, self.t(id));
        b
    }

    fn elementwise(&self, id: TensorId, entry: &str, op: OpBlob, srcs: &[TensorId], n: i64) -> Result<Vec<Dispatch>> {
        let f = self.core(entry)?;
        Ok(self.dispatch(id, f, op, srcs, Dims::new([blocks(n, 256), 1, 1], [256, 1, 1])))
    }

    /// Lowers one node; view ops and empty tensors produce nothing.
    pub fn lower(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let t = self.t(id);
        if t.is_view_op() || t.nelements() == 0 {
            return Ok(vec![]);
        }
        let ops_ok = t.src.iter().flatten().all(|&s| {
            let st = self.t(s);
            plain(st.ty) || quant_ok(st.ty)
        });
        if !ops_ok || !(plain(t.ty)) {
            return unsupported(t, "operand type");
        }
        match t.op.as_str() {
            "CONCAT" => self.concat(id),
            "ADD" | "SUB" | "MUL" | "DIV" => self.bin(id),
            "REPEAT" => self.repeat(id),
            "SCALE" | "FILL" | "CLAMP" | "LEAKY_RELU" | "SQR" | "SQRT" | "SIN" | "COS" | "LOG" | "UNARY" => self.unary(id),
            "SUM_ROWS" | "MEAN" => self.sum_rows(id),
            "GET_ROWS" => self.get_rows(id),
            "SOFT_MAX" => self.soft_max(id),
            "DUP" | "CPY" | "CONT" => self.cpy(id),
            "NORM" | "RMS_NORM" => self.norm(id),
            "IM2COL" => self.im2col(id),
            "ARANGE" => self.arange(id),
            "UPSCALE" => self.upscale(id),
            "ROPE" => self.rope(id),
            "POOL_2D" => self.pool_2d(id),
            "SET_ROWS" => self.set_rows(id),
            "ARGMAX" => self.argmax(id),
            "CONV_2D_DW" => self.conv_2d_dw(id),
            "PAD" | "PAD_REFLECT_1D" => self.pad(id),
            "MUL_MAT" => self.mul_mat(id),
            "FLASH_ATTN_EXT" => self.flash_attn_ext(id),
            "CUSTOM" => self.custom(id),
            _ => unsupported(t, "op not implemented in the native CUDA backend"),
        }
    }

    /// Operands of ops that only read plain element types.
    fn need_plain(&self, op: &Tensor, srcs: &[TensorId]) -> Result<()> {
        if srcs.iter().any(|&s| !plain(self.t(s).ty)) {
            return unsupported(op, "quantized operand");
        }
        Ok(())
    }

    fn concat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        self.need_plain(op, &[s0, s1])?;
        let mut b = self.blob(id, &[s0, s1]);
        b.int(0, op.op_params[0] as i64);
        self.elementwise(id, "k_concat", b, &[s0, s1], op.nelements())
    }

    fn bin(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        self.need_plain(op, &[s0, s1])?;
        let mut b = self.blob(id, &[s0, s1]);
        b.int(0, match op.op.as_str() {
            "ADD" => 0,
            "SUB" => 1,
            "MUL" => 2,
            _ => 3,
        });
        self.elementwise(id, "k_bin", b, &[s0, s1], op.nelements())
    }

    fn repeat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        self.need_plain(op, &[s0])?;
        self.elementwise(id, "k_repeat", self.blob(id, &[s0]), &[s0], op.nelements())
    }

    fn unary(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        self.need_plain(op, &[s0])?;
        let mut b = self.blob(id, &[s0]);
        let code: i64 = match op.op.as_str() {
            "SCALE" => {
                b.float(1, op.op_param_f32(0));
                b.float(2, op.op_param_f32(1));
                100
            }
            "FILL" => {
                b.float(1, op.op_param_f32(0));
                101
            }
            "CLAMP" => {
                b.float(1, op.op_param_f32(0));
                b.float(2, op.op_param_f32(1));
                102
            }
            "LEAKY_RELU" => {
                b.float(1, op.op_param_f32(0));
                103
            }
            "SQR" => 104,
            "SQRT" => 105,
            "SIN" => 106,
            "COS" => 107,
            "LOG" => 108,
            "UNARY" => match op.op_params[0] {
                17 => return unsupported(op, "xielu"),
                n @ (0..=16 | 18..=21) => n as i64,
                n => return unsupported(op, &format!("unary op {n}")),
            },
            _ => unreachable!(),
        };
        b.int(0, code);
        self.elementwise(id, "k_unary", b, &[s0], op.nelements())
    }

    fn sum_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        self.need_plain(op, &[s0])?;
        let mut b = self.blob(id, &[s0]);
        b.int(0, i64::from(op.op == "MEAN"));
        let f = self.core("k_sum_rows")?;
        let dims = Dims::new([a.nrows() as u32, 1, 1], [row_threads(a.ne[0]), 1, 1]);
        Ok(self.dispatch(id, f, b, &[s0], dims))
    }

    fn get_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        if !matches!(self.t(s1).ty.0, 26 | 27) {
            return unsupported(op, "row indices must be i32 or i64");
        }
        self.elementwise(id, "k_get_rows", self.blob(id, &[s0, s1]), &[s0, s1], op.nelements())
    }

    fn soft_max(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if op.src(2).is_some() {
            return unsupported(op, "sinks are not supported by the soft_max lowering");
        }
        let mask = op.src(1);
        if let Some(m) = mask {
            if !matches!(self.t(m).ty.0, 0 | 1) {
                return unsupported(op, "mask must be f16 or f32");
            }
        }
        let (scale, max_bias) = (op.op_param_f32(0), op.op_param_f32(1));
        let n_head_log2 = 1u32 << (a.ne[2] as f32).log2().floor() as u32;
        let m0 = 2f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
        let srcs: Vec<TensorId> = std::iter::once(s0).chain(mask).collect();
        let mut b = self.blob(id, &srcs);
        b.float(0, scale);
        b.float(1, max_bias);
        b.float(2, m0);
        b.float(3, m1);
        b.int(4, n_head_log2 as i64);
        let f = self.core("k_soft_max")?;
        let dims = Dims::new([a.nrows() as u32, 1, 1], [row_threads(a.ne[0]), 1, 1]);
        Ok(self.dispatch(id, f, b, &srcs, dims))
    }

    fn cpy(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if a.nelements() != op.nelements() {
            return unsupported(op, "source and destination differ in size");
        }
        self.elementwise(id, "k_cpy", self.blob(id, &[s0]), &[s0], a.nelements())
    }

    fn norm(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        self.need_plain(op, &[s0])?;
        let mut b = self.blob(id, &[s0]);
        b.int(0, i64::from(op.op == "RMS_NORM"));
        b.float(1, op.op_param_f32(0));
        let f = self.core("k_norm")?;
        let dims = Dims::new([a.nrows() as u32, 1, 1], [row_threads(a.ne[0]), 1, 1]);
        Ok(self.dispatch(id, f, b, &[s0], dims))
    }

    fn im2col(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (kern, inp) = (self.t(s0), self.t(s1));
        self.need_plain(op, &[s1])?;
        let p = &op.op_params;
        let is_2d = p[6] == 1;
        let mut b = self.blob(id, &[s0, s1]);
        for i in 0..6 {
            b.int(i, p[i] as i64);
        }
        b.int(6, i64::from(is_2d));
        b.int(7, kern.ne[0]);
        b.int(8, if is_2d { kern.ne[1] } else { 1 });
        b.int(9, if is_2d { inp.ne[1] } else { 1 });
        self.elementwise(id, "k_im2col", b, &[s0, s1], op.nelements())
    }

    fn arange(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let mut b = self.blob(id, &[]);
        b.float(0, op.op_param_f32(0));
        b.float(1, op.op_param_f32(2));
        self.elementwise(id, "k_arange", b, &[], op.ne[0])
    }

    fn upscale(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        self.need_plain(op, &[s0])?;
        let flags = op.op_params[0];
        let (mode, antialias) = (flags & 0xFF, flags & (1 << 9) != 0);
        if !(0..=2).contains(&mode) {
            return unsupported(op, "unknown upscale mode");
        }
        let mut sf = [0, 1, 2, 3].map(|i| op.ne[i] as f32 / a.ne[i] as f32);
        let mut poffs = 0.5f32;
        if flags & (1 << 8) != 0 {
            poffs = 0.0;
            for i in 0..2 {
                if op.ne[i] > 1 && a.ne[i] > 1 {
                    sf[i] = (op.ne[i] - 1) as f32 / (a.ne[i] - 1) as f32;
                }
            }
        }
        let mut b = self.blob(id, &[s0]);
        for (i, v) in sf.iter().enumerate() {
            b.float(i, *v);
        }
        b.float(4, poffs);
        b.int(5, mode as i64);
        b.int(6, i64::from(antialias));
        self.elementwise(id, "k_upscale", b, &[s0], op.nelements())
    }

    fn rope(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, pos) = (self.t(s0), self.t(s1));
        let s2 = op.src(2);
        let p = &op.op_params;
        let mode = p[2];
        if mode != 0 && mode != 2 {
            return unsupported(op, "only normal and neox rope (mrope/vision are not supported)");
        }
        if pos.ty != Ty::I32 || pos.ne[0] != a.ne[2] {
            return unsupported(op, "needs one i32 position per token");
        }
        if let Some(f) = s2 {
            if self.t(f).ty != Ty::F32 || self.t(f).nb[0] != 4 {
                return unsupported(op, "frequency factors must be f32");
            }
        }
        let n_dims = p[1];
        let (freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) =
            (op.op_param_f32(5), op.op_param_f32(6), op.op_param_f32(7), op.op_param_f32(8), op.op_param_f32(9), op.op_param_f32(10));
        let n_ctx_orig = p[4] as f32;
        // ggml_rope_yarn_corr_dims
        let corr = |beta: f32| n_dims as f32 * (n_ctx_orig / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * freq_base.ln());
        let start = corr(beta_fast).floor();
        let end = corr(beta_slow).ceil();
        let (c0, c1) = (start.max(0.0), end.min(n_dims as f32 - 1.0));
        let mut b = self.blob(id, &[s0, s1]);
        if let Some(f) = s2 {
            b.tensor(2, self.t(f));
        }
        b.int(0, n_dims as i64);
        b.int(1, mode as i64);
        b.float(2, freq_scale);
        b.float(3, ext_factor);
        b.float(4, attn_factor);
        b.float(5, freq_base.powf(-2.0 / n_dims as f32));
        b.float(6, c0);
        b.float(7, c1);
        b.int(8, i64::from(s2.is_some()));
        let mut srcs = vec![s0, s1];
        srcs.extend(s2);
        let f = self.core("k_rope")?;
        let n = op.ne[0] / 2 * op.ne[1] * op.ne[2] * op.ne[3];
        Ok(self.dispatch(id, f, b, &srcs, Dims::new([blocks(n, 256), 1, 1], [256, 1, 1])))
    }

    fn pool_2d(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        self.need_plain(op, &[s0])?;
        let p = &op.op_params;
        if p[0] != 0 && p[0] != 1 {
            return unsupported(op, "unknown pooling op");
        }
        let mut b = self.blob(id, &[s0]);
        for i in 0..7 {
            b.int(i, p[i] as i64);
        }
        self.elementwise(id, "k_pool_2d", b, &[s0], op.nelements())
    }

    fn set_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        self.need_plain(op, &[s0])?;
        if !matches!(self.t(s1).ty.0, 26 | 27) {
            return unsupported(op, "row indices must be i32 or i64");
        }
        self.elementwise(id, "k_set_rows", self.blob(id, &[s0, s1]), &[s0, s1], self.t(s0).nelements())
    }

    fn argmax(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        self.need_plain(op, &[s0])?;
        if op.ty != Ty::I32 {
            return unsupported(op, "argmax writes i32");
        }
        let f = self.core("k_argmax")?;
        Ok(self.dispatch(id, f, self.blob(id, &[s0]), &[s0], Dims::new([a.nrows() as u32, 1, 1], [256, 1, 1])))
    }

    fn conv_2d_dw(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        self.need_plain(op, &[s0, s1])?;
        let mut b = self.blob(id, &[s0, s1]);
        for i in 0..6 {
            b.int(i, op.op_params[i] as i64);
        }
        self.elementwise(id, "k_conv_2d_dw", b, &[s0, s1], op.nelements())
    }

    fn pad(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        self.need_plain(op, &[s0])?;
        if a.ty != op.ty {
            return unsupported(op, "padding keeps the element type");
        }
        let p = &op.op_params;
        let mut b = self.blob(id, &[s0]);
        if op.op == "PAD" {
            if p[8] != 0 {
                return unsupported(op, "circular padding");
            }
            for (i, v) in [p[0], p[2], p[4], p[6]].into_iter().enumerate() {
                b.int(i, v as i64);
            }
            self.elementwise(id, "k_pad", b, &[s0], op.nelements())
        } else {
            b.int(0, p[0] as i64);
            self.elementwise(id, "k_pad_reflect_1d", b, &[s0], op.nelements())
        }
    }

    // ------------------------------------------------------------------ mul_mat

    fn mul_mat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, b_) = (self.t(s0), self.t(s1));
        if !quant_ok(a.ty) {
            return unsupported(op, &format!("weights of ggml type {}", a.ty.0));
        }
        if !plain(b_.ty) || b_.ty.0 == 26 || b_.ty.0 == 27 {
            return unsupported(op, "activations must be f32/f16/bf16");
        }
        if a.ne[0] != b_.ne[0] || op.ne[0] != a.ne[1] || op.ne[1] != b_.ne[1] || a.ne[2] == 0 || a.ne[3] == 0 {
            return unsupported(op, "inconsistent shapes");
        }
        let (r2, r3) = (op.ne[2] / a.ne[2], op.ne[3] / a.ne[3]);
        let mut b = self.blob(id, &[s0, s1]);
        b.int(0, r2);
        b.int(1, r3);
        let lib = mm_library(self.dev, a.ty)?;
        let (m, n) = (op.ne[0], op.ne[1]);
        if n <= 8 {
            let f = self.func(&lib, "mm_vec")?;
            let dims = Dims::new([blocks(m, 4), op.ne[2] as u32, op.ne[3] as u32], [128, 1, 1]);
            Ok(self.dispatch(id, f, b, &[s0, s1], dims))
        } else {
            let f = self.func(&lib, "mm_tile")?;
            let dims = Dims::new([blocks(m, 64), blocks(n, 64), (op.ne[2] * op.ne[3]) as u32], [256, 1, 1]);
            Ok(self.dispatch(id, f, b, &[s0, s1], dims))
        }
    }

    // ---------------------------------------------------------- flash attention

    fn flash_attn_ext(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (q, k, v) = (self.src(op, 0)?, self.src(op, 1)?, self.src(op, 2)?);
        let mask = op.src(3);
        if op.src(4).is_some() {
            return unsupported(op, "attention sinks");
        }
        let (tq, tk, tv) = (self.t(q), self.t(k), self.t(v));
        if tq.ty != Ty::F32 || op.ty != Ty::F32 {
            return unsupported(op, "needs f32 queries");
        }
        if !quant_ok(tk.ty) || !quant_ok(tv.ty) {
            return unsupported(op, "K/V type");
        }
        if let Some(m) = mask {
            if self.t(m).ty != Ty::F16 {
                return unsupported(op, "mask must be f16");
            }
        }
        if tv.ne[0] > 512 {
            return unsupported(op, "value head size above 512");
        }
        if tq.ne[0] != tk.ne[0] || tq.ne[2] % tk.ne[2] != 0 || tq.ne[3] % tk.ne[3] != 0 {
            return unsupported(op, "inconsistent head shapes");
        }
        let smem = ((tq.ne[0] + 128) * 4) as u32;
        if smem > self.max_shared {
            return unsupported(op, "head size needs more shared memory than the device has");
        }
        let (mut scale, max_bias, softcap) = (op.op_param_f32(0), op.op_param_f32(1), op.op_param_f32(2));
        if softcap != 0.0 {
            scale /= softcap;
        }
        let n_head_log2 = 1u32 << (tq.ne[2] as f32).log2().floor() as u32;
        let m0 = 2f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
        let mut b = OpBlob::new();
        b.tensor(0, tq);
        b.tensor(1, tk);
        b.tensor(2, tv);
        if let Some(m) = mask {
            b.tensor(3, self.t(m));
        }
        b.tensor(4, op);
        b.float(0, scale);
        b.float(1, max_bias);
        b.float(2, m0);
        b.float(3, m1);
        b.int(4, n_head_log2 as i64);
        b.float(5, softcap);
        let lib = fa_library(self.dev, tk.ty, tv.ty)?;
        if let Some((part_len, parts)) = fa_split_plan(self.graph, id, self.sm_count) {
            let pos = self.graph.nodes.iter().position(|&n| n == id).ok_or_else(|| Error::Invalid("node is not in the graph".into()))?;
            let scratch = self.layout.scratch[pos].ok_or_else(|| Error::Invalid("split attention scratch was not planned".into()))?;
            let partials = Place { buf: BufId::Arena, off: scratch };
            b.int(6, part_len);
            b.int(7, parts);
            let mut places = [None; 4];
            for (i, s) in [q, k, v].iter().chain(mask.iter()).enumerate() {
                places[i] = Some(self.place(*s));
            }
            let split = Dims::new([parts as u32, tq.ne[2] as u32, tq.ne[3] as u32], [128, 1, 1]).shared(smem);
            let reduce = Dims::new([tq.ne[2] as u32, 1, tq.ne[3] as u32], [128, 1, 1]);
            let label = format!("{} {}", op.op, op.name);
            return Ok(vec![
                Dispatch { func: Some(self.func(&lib, "fa_split")?), custom: None, op: b.clone(), srcs: places, dst: partials, dims: split, label: label.clone() },
                Dispatch {
                    func: Some(self.func(&lib, "fa_reduce")?),
                    custom: None,
                    op: b,
                    srcs: [Some(partials), None, None, None],
                    dst: self.place(id),
                    dims: reduce,
                    label,
                },
            ]);
        }
        let f = self.func(&lib, "fa_main")?;
        let dims = Dims::new([tq.ne[1] as u32, tq.ne[2] as u32, tq.ne[3] as u32], [128, 1, 1]).shared(smem);
        let mut srcs = vec![q, k, v];
        srcs.extend(mask);
        Ok(self.dispatch(id, f, b, &srcs, dims))
    }
}

impl Ctx<'_> {
    /// A ggml `CUSTOM` node (`ggml_custom_4d` with no function: nothing computes it
    /// on ggml) names one of the vendored paged-attention kernels in its
    /// `userdata`, which ggml keeps in `op_params[4..6]`. Low to high bits:
    /// kind (4), kv heads (4), block size (8), max context length (16), scale (f32).
    ///
    /// * kind 1, paged attention: sources q `f16 [hs, heads, seqs]`, key pool, value
    ///   pool (`f16`, vLLM layouts), block tables `u32 [max_blocks, seqs]`, context
    ///   lengths `u32 [seqs]`; the node is the output `f16 [hs, heads, seqs]`.
    /// * kind 2, cache write: sources key and value `f16 [hs * kv_heads, tokens]`, key pool,
    ///   value pool, slot mapping `i64 [tokens]` (a negative slot skips the token);
    ///   the node is a dummy.
    fn custom(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let t = self.t(id);
        let ud = t.op_params[4] as u32 as u64 | (t.op_params[5] as u32 as u64) << 32;
        let (kind, kv_heads, block_size, max_ctx) =
            ((ud & 15) as u8, ((ud >> 4) & 15) as usize, ((ud >> 8) & 255) as usize, ((ud >> 16) & 0xffff) as usize);
        let scale = f32::from_bits((ud >> 32) as u32);
        let src = |i| self.src(t, i);
        let label = format!("{} {}", t.op, t.name);
        let dispatch = |custom| Dispatch {
            func: None,
            custom: Some(custom),
            op: OpBlob::new(),
            srcs: [None; 4],
            dst: self.place(id),
            dims: Dims::new([1, 1, 1], [1, 1, 1]),
            label: label.clone(),
        };
        let f16 = |s: TensorId| self.t(s).ty.0 == 1;
        match kind {
            1 => {
                let (q, kc, vc, tables, lens) = (src(0)?, src(1)?, src(2)?, src(3)?, src(4)?);
                let qt = self.t(q);
                let (hs, heads, seqs) = (qt.ne[0] as usize, qt.ne[1] as usize, qt.ne[2] as usize);
                if !(f16(q) && f16(kc) && f16(vc) && f16(id)) {
                    return unsupported(t, "paged attention takes f16 q, caches and output");
                }
                if qt.nb[1] != (hs * 2) as u64 || qt.nb[2] != (hs * heads * 2) as u64 || self.t(id).ne[..3] != qt.ne[..3] {
                    return unsupported(t, "paged attention takes a contiguous q and an output of its shape");
                }
                let geometry = PagedGeometry {
                    dtype: DType::F16,
                    cache_dtype: DType::F16,
                    fp8_scales: false,
                    num_seqs: seqs,
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_size: hs,
                    block_size,
                    max_context_len: max_ctx,
                };
                let prepared = geometry.prepare(self.dev)?;
                let scratch = geometry.scratch(self.dev)?;
                let (kv_block_stride, kv_head_stride) =
                    super::paged_attn::PagedAttention::dense_strides(kv_heads, hs, block_size);
                Ok(vec![dispatch(Custom::PagedAttention(Box::new(PagedLaunch {
                    geometry,
                    prepared,
                    scratch,
                    q: self.place(q),
                    k_cache: self.place(kc),
                    v_cache: self.place(vc),
                    block_tables: self.place(tables),
                    context_lens: self.place(lens),
                    out: self.place(id),
                    max_blocks: self.t(tables).ne[0] as usize,
                    scale,
                    kv_block_stride,
                    kv_head_stride,
                })))])
            }
            2 => {
                let (key, value, kc, vc, slots) = (src(0)?, src(1)?, src(2)?, src(3)?, src(4)?);
                let kt = self.t(key);
                let tokens = kt.ne[1] as usize;
                if !(f16(key) && f16(value) && f16(kc) && f16(vc)) {
                    return unsupported(t, "the cache write takes f16 keys, values and caches");
                }
                if self.t(slots).nelements() != 2 * tokens as i64 || kt.nb[1] != kt.ne[0] as u64 * 2 {
                    return unsupported(t, "the cache write takes contiguous keys and one i64 slot per token");
                }
                Ok(vec![dispatch(Custom::CacheWrite(Box::new(CacheLaunch {
                    func: reshape_and_cache_func(self.dev, DType::F16, DType::F16, false)?,
                    dtype: DType::F16,
                    tokens,
                    head_size: kt.ne[0] as usize / kv_heads,
                    kv_heads,
                    block_size,
                    key: self.place(key),
                    value: self.place(value),
                    k_cache: self.place(kc),
                    v_cache: self.place(vc),
                    slots: self.place(slots),
                })))])
            }
            _ => unsupported(t, "unknown custom op"),
        }
    }
}

/// Split-KV decoding for a one-query flash attention node that would
/// otherwise run on few blocks: (keys per slice, slices). `TL_NATIVE_FA=plain`
/// turns it off.
fn fa_split_plan(graph: &Graph, id: TensorId, sm_count: u32) -> Option<(i64, i64)> {
    let node = &graph.tensors[id];
    if node.op != "FLASH_ATTN_EXT" || std::env::var("TL_NATIVE_FA").is_ok_and(|v| v == "plain") {
        return None;
    }
    let q = &graph.tensors[node.src(0)?];
    let k = &graph.tensors[node.src(1)?];
    let blocks = q.ne[2] * q.ne[3];
    if q.ne[1] != 1 || k.ne[1] < 256 {
        return None;
    }
    let want = (k.ne[1] * blocks + 4 * sm_count as i64 - 1) / (4 * sm_count as i64);
    let part_len = ((want + 31) / 32 * 32).max(64);
    let parts = (k.ne[1] + part_len - 1) / part_len;
    (parts >= 2).then_some((part_len, parts))
}

/// Arena scratch a node needs in addition to its output (split attention's partials).
pub fn scratch_bytes(graph: &Graph, id: TensorId, sm_count: u32) -> u64 {
    match fa_split_plan(graph, id, sm_count) {
        Some((_, parts)) => {
            let node = &graph.tensors[id];
            let q = &graph.tensors[node.src(0).unwrap()];
            let v = &graph.tensors[node.src(2).unwrap()];
            (q.ne[2] * q.ne[3] * parts * (v.ne[0] + 2) * 4) as u64
        }
        None => 0,
    }
}

const COMMON: &str = include_str!("native/common.cuh");
pub const CORE: &str = include_str!("native/core.cu");
const MM: &str = include_str!("native/mm.cu");
const FA: &str = include_str!("native/fa.cu");

/// The source of one native library: the NVRTC prelude, the shared header,
/// `defines` and the kernel file.
pub fn native_source(defines: &str, body: &str) -> String {
    format!("{}\n{COMMON}\n{defines}\n{body}", super::shaders::PRELUDE)
}

pub fn core_library(dev: &Device) -> Result<Arc<str>> {
    dev.library("tl-core", &[], || native_source("", CORE))
}

fn mm_library(dev: &Device, ty: Ty) -> Result<Arc<str>> {
    dev.library(&format!("tl-mm-{}", ty.0), &[], || native_source(&format!("#define MM_TY {}", ty.0), MM))
}

fn fa_library(dev: &Device, k: Ty, v: Ty) -> Result<Arc<str>> {
    dev.library(&format!("tl-fa-{}-{}", k.0, v.0), &[], || {
        native_source(&format!("#define KTY {}\n#define VTY {}", k.0, v.0), FA)
    })
}

/// Compiles every library variant (all matmul source types, the K/V type
/// pairs of flash attention); used by tests to surface kernel build errors.
pub fn compile_all_libraries(dev: &Device) -> Result<()> {
    core_library(dev)?;
    for ty in [0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14] {
        mm_library(dev, Ty(ty))?;
    }
    for ty in [0, 1, 30, 8, 2] {
        fa_library(dev, Ty(ty), Ty(ty))?;
    }
    Ok(())
}
