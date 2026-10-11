//! Lowers graph nodes to launches of ggml's Vulkan shaders.
//!
//! This is the host half of ggml-vulkan.cpp ported to Rust, for the shader
//! variants `scripts/build-ggml-vulkan-shaders.sh` precompiles: per op it picks
//! the variant, fills the push-constant struct the shader expects, computes the
//! workgroup grid, and names the buffer ranges the launch reads and writes. The
//! scheduler (`sched::Engine`) does the rest: hazards, barriers, memory, fences.
//!
//! Conventions inherited from ggml (all verified against its host code):
//! * strides are passed in elements (`nb / type_size`);
//! * ggml's shaders take the misalignment of each binding (the part of the
//!   byte offset below the device's offset alignment) in elements inside
//!   `misalign_offsets` / per-op fields, so blobs are `OffsetMode::Raw`;
//! * elementwise kernels run `(1, ceil(ne / 512), 1)` workgroups past 512
//!   elements because they index `z * 262144 + y * 512 + x`.
//!
//! Not ported (reported by name at compile time): the matrix-matrix matmul,
//! conv, im2col, soft_max, norms other than RMS, and the coopmat flash
//! attention paths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::ggml::SpvLibrary;
use super::tree::{TreeKernels, TreeStats};
use super::Caps;
use crate::graph::{Graph, Tensor, TensorId, Ty};
use crate::plan::{align, BufId, Layout, Place};
use crate::sched::{KernelBlob, LaunchDesc, Range, Space};
use crate::{Error, Result};

/// Loads and caches shader variants by name and specialization constants.
pub struct Shaders {
    lib: SpvLibrary,
    cache: Mutex<HashMap<(String, Vec<u32>), Arc<KernelBlob>>>,
}

impl Shaders {
    pub fn new(lib: SpvLibrary) -> Self {
        Shaders { lib, cache: Mutex::new(HashMap::new()) }
    }

    pub fn get(&self, name: &str, spec: &[u32]) -> Result<Arc<KernelBlob>> {
        let key = (name.to_string(), spec.to_vec());
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Ok(b.clone());
        }
        if !self.lib.contains(name) {
            return Err(Error::Invalid(format!("the ggml shader library has no `{name}`")));
        }
        let blob = Arc::new(self.lib.load(name, spec)?);
        self.cache.lock().unwrap().insert(key, blob.clone());
        Ok(blob)
    }
}

/// ggml's `ROUNDUP_POW2`.
fn roundup_pow2(m: u32, n: u32) -> u32 {
    (m + n - 1) & !(n - 1)
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b.max(1))
}

/// `init_fastdiv_values`: magic number and shift to divide by `d`.
fn fastdiv(d: u32) -> (u32, u32) {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = ((1u64 << 32) * ((1u64 << l) - d as u64) / d.max(1) as u64 + 1) as u32;
    (mp, l)
}

/// Workgroup counts for ggml's elementwise kernels with `wg_denoms = {512, 1, 1}`.
fn elementwise_grid(ne: u32) -> [u32; 3] {
    if ne > 262_144 {
        [1, 512, ceil_div(ne, 262_144)]
    } else if ne > 512 {
        [1, ceil_div(ne, 512), 1]
    } else {
        [1, 1, 1]
    }
}

/// Workgroup counts for the row-parallel kernels (one workgroup per row, `wg_denoms = {1, 1, 1}`).
fn rows_grid(nr: u32) -> [u32; 3] {
    if nr > 262_144 {
        [512, 512, ceil_div(nr, 262_144)]
    } else if nr > 512 {
        [512, ceil_div(nr, 512), 1]
    } else {
        [nr, 1, 1]
    }
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn unused() -> Range {
    Range::new(Space::Step, 0, 0)
}

fn ty_name(ty: Ty) -> String {
    ty.name().to_ascii_lowercase()
}

/// A tensor as a shader sees it.
#[derive(Clone, Copy)]
struct Tn {
    ty: Ty,
    ne: [u32; 4],
    /// Strides in elements.
    nb: [u32; 4],
    range: Range,
    /// Byte offset inside the backing buffer.
    off: u64,
}

impl Tn {
    fn nelements(&self) -> u32 {
        self.ne.iter().product()
    }

    fn nrows(&self) -> u32 {
        self.ne[1] * self.ne[2] * self.ne[3]
    }

    fn type_size(&self) -> u64 {
        self.ty.size()
    }
}

pub struct Ctx<'a> {
    pub shaders: &'a Shaders,
    pub caps: Caps,
    pub graph: &'a Graph,
    pub layout: &'a Layout,
    /// Step offset of the arena (the io region comes first).
    pub arena_base: u64,
    /// Node id -> position in `graph.nodes` (for per-node scratch).
    pub pos: &'a HashMap<TensorId, usize>,
    /// Generated kernels for the nodes that have a Tree generator (see `tree_ops.rs`).
    pub tree: Option<&'a TreeKernels>,
    pub stats: &'a TreeStats,
}

mod tree_ops;

fn unsupported<T>(t: &Tensor, why: &str) -> Result<T> {
    Err(Error::Unsupported { op: t.op.clone(), node: t.name.clone(), why: why.to_string() })
}

/// The `ggml_op_params` float at `i`.
fn pf(t: &Tensor, i: usize) -> f32 {
    t.op_param_f32(i)
}

/// ggml's `ggml_rope_yarn_corr_dims`.
fn rope_corr_dims(n_dims: i32, n_ctx_orig: i32, freq_base: f32, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
    let factor = |n_rot: f32| n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * freq_base.ln());
    let start = factor(beta_fast).floor();
    let end = factor(beta_slow).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

/// Flash attention (scalar path) parameters, shared by the scratch sizing and the lowering.
struct FaPlan {
    hsk: u32,
    hsv: u32,
    n: u32,
    kv: u32,
    gqa_ratio: u32,
    br: u32,
    bc: u32,
    workgroup_size: u32,
    d_split: u32,
    row_split: u32,
    shmem_staging: bool,
    aligned: bool,
    split_k: u32,
    split_kv: u32,
    workgroups_y: u32,
    workgroups_z: u32,
    scale: f32,
    max_bias: f32,
    softcap: f32,
    /// Bytes of split-K partials (0 when not split).
    scratch: u64,
}

struct FaTuning {
    workgroup_size: u32,
    block_rows: u32,
    block_cols: u32,
    d_split: u32,
    row_split: u32,
    shmem_staging: bool,
}

/// `get_fa_tuning_params_scalar` for non-AMD, non-Intel devices, with f16 K/V.
fn fa_tuning(caps: &Caps, hsk: u32, hsv: u32, n_rows: u32, n_kv: u32, k_ty: Ty, v_ty: Ty) -> FaTuning {
    let subgroup_size = caps.subgroup_size;
    let row_split = if n_rows < 4 || hsk <= 64 { 1 } else { 4 };
    let workgroup_size = if subgroup_size > 32 && (n_rows < 4 || hsk < if row_split == 1 { 128 } else { 64 }) {
        subgroup_size * 2
    } else {
        subgroup_size * 4
    };
    let d = hsk | hsv;
    let reduce_block_rows = d & 8 != 0 || n_kv < 1024;
    let (mut block_rows, block_cols);
    if n_rows == 1 {
        block_rows = 1;
        block_cols = 64;
    } else {
        block_rows = if row_split == 1 {
            if n_rows == 2 { 2 } else if n_rows <= 4 || reduce_block_rows { 4 } else { 8 }
        } else if n_rows <= 4 {
            4
        } else if n_rows <= 8 || reduce_block_rows {
            8
        } else {
            16
        };
        block_cols = if d & 8 != 0 { 64 } else { 32 };
    }
    let d_lsb = d ^ (d & (d - 1));
    let d_split = subgroup_size.min(8).min(d_lsb / 4);
    let shmem_staging = caps.vendor_id == 0x10de && hsk < 256 && hsv < 256;
    let mut t = FaTuning { workgroup_size, block_rows, block_cols, d_split, row_split, shmem_staging };
    if !reduce_block_rows && !fa_shmem_ok(caps, &t, hsk, hsv, k_ty, v_ty) {
        t.block_rows /= 2;
        block_rows = t.block_rows;
    }
    let _ = block_rows;
    t
}

/// `ggml_vk_flash_attn_scalar_shmem_support` (f16 shaders, no integer-dot K).
fn fa_shmem_ok(caps: &Caps, p: &FaTuning, hsk: u32, hsv: u32, _k: Ty, _v: Ty) -> bool {
    let float_size = if caps.fp16 { 2 } else { 4 };
    let tmpsh = p.workgroup_size * 4;
    let tmpshv4 = p.workgroup_size * 4 * float_size;
    let masksh = p.block_cols * (p.block_rows + 1) * float_size;
    let iq_shmem = 16 * float_size;
    let qf = p.block_rows * (hsk / 4 + 1) * 4 * float_size;
    let d = hsk.max(hsv);
    let kvsh = if p.shmem_staging { p.block_cols * (d / 4 + 1) * 4 * float_size } else { 4 * float_size };
    tmpsh + tmpshv4 + masksh + iq_shmem + qf + kvsh <= caps.shared_memory
}

/// Shape of a tensor without any placement.
struct Shape {
    ty: Ty,
    ne: [u32; 4],
    /// Strides in elements.
    nb: [u32; 4],
}

fn shape_of(graph: &Graph, id: TensorId) -> Shape {
    let t = &graph.tensors[id];
    let ts = t.ty.size();
    Shape {
        ty: t.ty,
        ne: [t.ne[0] as u32, t.ne[1] as u32, t.ne[2] as u32, t.ne[3] as u32],
        nb: [(t.nb[0] / ts) as u32, (t.nb[1] / ts) as u32, (t.nb[2] / ts) as u32, (t.nb[3] / ts) as u32],
    }
}

/// Plans scalar flash attention for node `id`: tuning, grouping of query heads, split-K.
fn fa_plan(graph: &Graph, caps: &Caps, id: TensorId) -> Result<FaPlan> {
    let t = &graph.tensors[id];
    let src = |i: usize| t.src(i).ok_or_else(|| Error::Invalid(format!("{} `{}` is missing source {i}", t.op, t.name)));
    let (tq, tk, tv, td) = (shape_of(graph, src(0)?), shape_of(graph, src(1)?), shape_of(graph, src(2)?), shape_of(graph, id));
    let mask = t.src(3).map(|m| shape_of(graph, m));
    let (hsk, hsv) = (tk.ne[0], tv.ne[0]);
    let kv = tk.ne[1];
    let (neq1, neq2, neq3) = (tq.ne[1], tq.ne[2], tq.ne[3]);
    let (nek2, nev2) = (tk.ne[2], tv.ne[2]);
    let nem2 = mask.as_ref().map_or(0, |m| m.ne[2]);
    let mut n = neq1;
    let qk_ratio = neq2 / nek2.max(1);
    let workgroups_x = neq1;
    let mut workgroups_y = neq2;
    let workgroups_z = neq3;
    let mut gqa_ratio = 1;

    let big = fa_tuning(caps, hsk, hsv, 512, kv, tk.ty, tv.ty);
    let max_gqa = big.block_rows.min(32);
    if n <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa && qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1 {
        gqa_ratio = qk_ratio;
        n = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }
    let tun = fa_tuning(caps, hsk, hsv, n, kv, tk.ty, tv.ty);
    let q_stride = tq.nb[1];
    let mut k_stride = tk.nb[1];
    let mut v_stride = tv.nb[1];
    if tk.ty == Ty::F32 {
        k_stride /= 4;
    }
    if tv.ty == Ty::F32 {
        v_stride /= 4;
    }
    let aligned = kv % tun.block_cols == 0 && q_stride & 7 == 0 && k_stride & 7 == 0 && v_stride & 7 == 0;

    let mut scale = t.op_param_f32(0);
    let max_bias = t.op_param_f32(1);
    let softcap = t.op_param_f32(2);
    if softcap != 0.0 {
        scale /= softcap;
    }

    let br = tun.block_rows;
    let tr = ceil_div(n, br);
    // ggml uses the device's shader core count; without a query, assume a mid-size GPU.
    let cores = caps.shader_cores.max(1);
    let mut split_k = 1;
    let mut split_kv = kv;
    if gqa_ratio > 1 && workgroups_x <= br {
        split_k = (cores * 2 / (workgroups_x * workgroups_y * workgroups_z).max(1)).max(1);
    } else if gqa_ratio <= 1 {
        let total = tr * workgroups_y * workgroups_z;
        if total < cores * 2 {
            split_k = cores * 2 / total.max(1);
        }
    }
    if split_k > 1 {
        split_kv = roundup_pow2((kv / split_k).max(1), tun.block_cols);
        split_k = ceil_div(kv, split_kv);
    }
    let (ne1, ne2, ne3) = (td.ne[1] as u64, td.ne[2] as u64, td.ne[3] as u64);
    let scratch = if split_k > 1 { (hsv as u64 * ne1 * 4 + ne1 * 4 * 2) * split_k as u64 * ne2 * ne3 } else { 0 };
    Ok(FaPlan {
        hsk,
        hsv,
        n,
        kv,
        gqa_ratio,
        br,
        bc: tun.block_cols,
        workgroup_size: tun.workgroup_size,
        d_split: tun.d_split,
        row_split: tun.row_split,
        shmem_staging: tun.shmem_staging,
        aligned,
        split_k,
        split_kv,
        workgroups_y,
        workgroups_z,
        scale,
        max_bias,
        softcap,
        scratch,
    })
}

/// A scalar-path matrix-matrix multiplication: tile, alignment, split-K.
struct MmPlan {
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    warptile: [u32; 11],
    /// Tile edge (the shader's `wg_denoms[0..2]`).
    den: [u32; 2],
    aligned: bool,
    split_k: u32,
    /// Bytes of split-K partial results (0 when not split).
    scratch: u64,
}

/// `ggml_vk_guess_matmul_pipeline` + `ggml_vk_guess_split_k` for devices without cooperative
/// matrices (the l/m/s tiles are all enabled; quantized weights use the BK=32 tiles).
fn mm_plan(graph: &Graph, caps: &Caps, id: TensorId) -> Result<MmPlan> {
    let t = &graph.tensors[id];
    let src = |i: usize| t.src(i).ok_or_else(|| Error::Invalid(format!("{} `{}` is missing source {i}", t.op, t.name)));
    let (ta, tb, td) = (shape_of(graph, src(0)?), shape_of(graph, src(1)?), shape_of(graph, id));
    let (m, n, k) = (ta.ne[1], tb.ne[1], tb.ne[0]);
    let batch = tb.ne[2] * tb.ne[3];
    let sg8 = caps.subgroup_size.max(8);
    let sg32 = caps.subgroup_size.max(32);
    let s_wm = if caps.subgroup_size == 8 { 8 } else { 32 };
    let bk = if ta.ty.is_quantized() { 32 } else { 16 };
    let l = [128, 128, 128, bk, sg8 * 2, 64, 2, 4, 4, 1, sg8];
    let mt = [128, 64, 64, bk, sg8, 32, 2, 4, 2, 1, sg8];
    let st = [sg32, 32, 32, bk, s_wm, 32, 2, 2, 2, 1, sg8];
    // The tile must fit the shared memory: (BM + BN) * (BK + 1) values of 16 or 32 bits.
    let value_size = if caps.fp16 { 2 } else { 4 };
    let fits = |w: &[u32; 11]| (w[1] + w[2]) * (w[3] + 1) * value_size <= caps.shared_memory;
    let (mm_l, mm_m) = (fits(&l), fits(&mt));
    let (warptile, den, align) = if (m <= 32 || n <= 32) || (!mm_m && !mm_l) {
        (st, [32, 32], 32)
    } else if m <= 64 || n <= 64 || !mm_l {
        (mt, [64, 64], 64)
    } else {
        (l, [128, 128], 128)
    };
    let _ = td;
    let aligned = k % align == 0 && m > 8 && n > 8;

    // ggml_vk_guess_split_k
    let mut split_k = 1;
    let cores = caps.shader_cores;
    if cores != 0 && m >= den[0] && n >= den[1] && k >= 2048 {
        let tiles = ceil_div(m, den[0]) * ceil_div(n, den[1]);
        if tiles <= cores / 2 {
            split_k = cores / tiles;
        } else if tiles <= cores * 2 / 3 {
            split_k = 3;
        }
        split_k = split_k.min(8);
        while split_k > 1 {
            let k_split = roundup_pow2(ceil_div(k, split_k), 256);
            if k_split * (split_k - 1) < k {
                break;
            }
            split_k -= 1;
        }
    }
    let d_bytes = 4 * td.ne.iter().map(|&x| x as u64).product::<u64>();
    let scratch = if split_k > 1 { d_bytes * split_k as u64 } else { 0 };
    Ok(MmPlan { m, n, k, batch, warptile, den, aligned, split_k, scratch })
}

/// Whether mat-vec `id` takes the dp4a path (its activations are quantized to int8).
pub fn q8_dot_ok(graph: &Graph, caps: &Caps, id: TensorId) -> bool {
    tree_ops::q8_dot_plan(graph, caps, id).is_some()
}

/// Whether flash attention `id` runs as split-KV chunks plus a merge (see tree_ops).
pub fn fa_split(graph: &Graph, id: TensorId) -> Option<()> {
    tree_ops::fa_split_plan(graph, id).map(|_| ())
}

/// Scratch bytes node `id` asks the planner for (split-K partials of flash attention and matmul).
pub fn scratch_bytes(graph: &Graph, id: TensorId, caps: &Caps, tree: bool) -> u64 {
    if tree {
        if graph.tensors[id].op == "FUSED_GU" {
            let t = &graph.tensors[id];
            let x = t.src(2).map_or(0, |x| graph.root(x).0);
            let k = t.src(0).map_or(0, |w| graph.tensors[w].ne[0]) as u64;
            return if graph.q8_out.contains(&x) || graph.q8_side.contains_key(&x) { 0 } else { align(k) + k / 8 };
        }
        if let Some(p) = tree_ops::q8_dot_plan(graph, caps, id) {
            let quantized = graph.tensors[id].src(1).is_some_and(|b| graph.q8_out.contains(&graph.root(b).0) || graph.q8_side.contains_key(&graph.root(b).0));
            return if quantized { 0 } else { align(p.xq_bytes) + p.xd_bytes };
        }
        if let Some(p) = tree_ops::fa_split_plan(graph, id) {
            return p.bytes.max(fa_plan(graph, caps, id).map_or(0, |p| p.scratch));
        }
    }
    match graph.tensors[id].op.as_str() {
        "FLASH_ATTN_EXT" => fa_plan(graph, caps, id).map_or(0, |p| p.scratch),
        "MUL_MAT" => mm_plan(graph, caps, id).map_or(0, |p| if mm_uses_matrix_kernel(graph, id) { p.scratch } else { 0 }),
        _ => 0,
    }
}

/// True when the node takes the matrix-matrix path (more than one column, or batched columns).
fn mm_uses_matrix_kernel(graph: &Graph, id: TensorId) -> bool {
    let t = &graph.tensors[id];
    let (Some(a), Some(b)) = (t.src(0), t.src(1)) else { return false };
    let (ta, tb) = (&graph.tensors[a], &graph.tensors[b]);
    let cols = tb.ne[1];
    let batched = tb.ne[2] * tb.ne[3] != 1;
    let _ = ta;
    cols > 8 || (cols > 1 && batched)
}

impl<'a> Ctx<'a> {
    fn t(&self, id: TensorId) -> &Tensor {
        &self.graph.tensors[id]
    }

    fn src(&self, node: &Tensor, i: usize) -> Result<TensorId> {
        node.src(i).ok_or_else(|| Error::Invalid(format!("{} `{}` is missing source {i}", node.op, node.name)))
    }

    fn place(&self, p: Place) -> (Space, u64) {
        match p.buf {
            BufId::Weights(i) => (Space::Model(i as u32), p.off),
            BufId::State => (Space::Session(0), p.off),
            BufId::Io => (Space::Step, p.off),
            BufId::Arena => (Space::Step, self.arena_base + p.off),
        }
    }

    fn tn(&self, id: TensorId) -> Tn {
        let t = self.t(id);
        let (space, off) = self.place(self.layout.place[id]);
        let ts = t.ty.size();
        let nbytes = t.nbytes();
        Tn {
            ty: t.ty,
            ne: [t.ne[0] as u32, t.ne[1] as u32, t.ne[2] as u32, t.ne[3] as u32],
            nb: [(t.nb[0] / ts) as u32, (t.nb[1] / ts) as u32, (t.nb[2] / ts) as u32, (t.nb[3] / ts) as u32],
            range: Range::new(space, off, nbytes),
            off,
        }
    }

    /// The misalignment of a binding in elements, as ggml's `get_misalign_bytes / type_size`.
    fn misalign(&self, t: &Tn) -> u32 {
        ((t.off % self.caps.storage_offset_align) / t.type_size()) as u32
    }

    /// For kernels with no misalignment fields: the binding must start on the alignment.
    fn aligned(&self, node: &Tensor, t: &Tn) -> Result<()> {
        if t.off % self.caps.storage_offset_align != 0 {
            return unsupported(node, "operand is not aligned to the device's storage buffer offset alignment");
        }
        Ok(())
    }

    fn launch(&self, blob: Arc<KernelBlob>, mut bindings: Vec<Range>, mut push: Vec<u8>, grid: [u32; 3]) -> Result<LaunchDesc> {
        // ggml sizes its pipeline layouts by the host struct, which can end in fields only some
        // variants declare (matmul's `padded_N`); ours is sized by the shader, so drop the tail.
        if push.len() > blob.sig.push_bytes as usize {
            push.truncate(blob.sig.push_bytes as usize);
        }
        if bindings.len() > blob.sig.bindings.len() {
            return Err(Error::Invalid(format!("`{}` takes {} bindings, got {}", blob.entry, blob.sig.bindings.len(), bindings.len())));
        }
        bindings.resize(blob.sig.bindings.len(), unused());
        if push.len() != blob.sig.push_bytes as usize {
            return Err(Error::Invalid(format!("`{}` takes {} push bytes, got {}", blob.entry, blob.sig.push_bytes, push.len())));
        }
        let max = self.caps.max_groups;
        let grid = [grid[0].max(1), grid[1].max(1), grid[2].max(1)];
        if grid.iter().zip(max).any(|(g, m)| *g > m) {
            return Err(Error::Invalid(format!("grid {grid:?} is over the device limit {max:?}")));
        }
        Ok(LaunchDesc { kernel: blob, bindings, push, grid })
    }

    /// `vk_op_binary_push_constants` (b is zeros when absent).
    fn binary_pc(&self, a: &Tn, b: Option<&Tn>, d: &Tn, p: [f32; 2], p3: i32) -> Vec<u8> {
        self.binary_pc_ne(a.nelements(), a, b, d, p, p3)
    }

    /// [`Ctx::binary_pc`] with an explicit element count (concat counts the destination's).
    fn binary_pc_ne(&self, ne: u32, a: &Tn, b: Option<&Tn>, d: &Tn, p: [f32; 2], p3: i32) -> Vec<u8> {
        let mut w = vec![ne];
        let zero = Tn { ty: Ty::F32, ne: [0; 4], nb: [0; 4], range: unused(), off: 0 };
        for t in [a, b.unwrap_or(&zero), d] {
            w.extend(t.ne);
            w.extend(t.nb);
        }
        let bm = b.map_or(0, |b| self.misalign(b));
        w.push((self.misalign(a) << 16) | (bm << 8) | self.misalign(d));
        w.extend([p[0].to_bits(), p[1].to_bits(), p3 as u32]);
        words(&w)
    }

    /// `vk_op_unary_push_constants` with `init_pushconst_fastdiv`.
    fn unary_pc(&self, a: &Tn, d: &Tn, ne: u32, p: [f32; 4]) -> Vec<u8> {
        let mut w = vec![ne];
        for t in [a, d] {
            w.extend(t.ne);
            w.extend(t.nb);
        }
        w.push((self.misalign(a) << 16) | self.misalign(d));
        w.extend(p.map(f32::to_bits));
        let fd = |t: &Tn| {
            let (m012, l012) = fastdiv(t.ne[2] * t.ne[1] * t.ne[0]);
            let (m01, l01) = fastdiv(t.ne[1] * t.ne[0]);
            let (m0, l0) = fastdiv(t.ne[0]);
            [m012, m01, m0, l012 | (l01 << 8) | (l0 << 16)]
        };
        w.extend(fd(a));
        w.extend(fd(d));
        words(&w)
    }

    /// Lowers one node.
    pub fn lower(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        if t.is_view_op() || t.nelements() == 0 {
            return Ok(vec![]);
        }
        if let Some(tree) = self.tree {
            if let Some(l) = self.lower_tree(tree, id)? {
                self.stats.tree_nodes.set(self.stats.tree_nodes.get() + 1);
                return Ok(l);
            }
            *self.stats.fallbacks.borrow_mut().entry(t.op.clone()).or_default() += 1;
            if tree.only {
                return unsupported(t, "no Tree kernel for this node");
            }
        }
        match t.op.as_str() {
            "ADD" | "SUB" | "MUL" | "DIV" => self.binary(id),
            "SCALE" | "SQR" | "SQRT" | "SIN" | "COS" | "LOG" | "CLAMP" | "LEAKY_RELU" | "FILL" | "UNARY" => self.unary(id),
            "RMS_NORM" => self.rms_norm(id),
            "RMS_NORM_MUL" => self.rms_norm_mul(id),
            "RMS_NORM_MUL_ROPE" => self.rms_norm_mul_rope(id),
            "NORM" => self.norm(id),
            "SUM_ROWS" | "MEAN" => self.sum_rows(id),
            "SOFT_MAX" => self.soft_max(id),
            "CONCAT" => self.concat(id),
            "IM2COL" => self.im2col(id),
            "POOL_2D" => self.pool_2d(id),
            "CONV_2D_DW" => self.conv_2d_dw(id),
            "PAD" => self.pad(id),
            "UPSCALE" => self.upscale(id),
            "MUL_MAT" => self.mul_mat(id),
            "ROPE" => self.rope(id),
            "FLASH_ATTN_EXT" => self.flash_attn(id),
            "SET_ROWS" => self.set_rows(id),
            "GET_ROWS" => self.get_rows(id),
            "CPY" | "CONT" | "DUP" => self.copy(id),
            "REPEAT" => self.repeat(id),
            "ARANGE" => self.arange(id),
            "ARGMAX" => self.argmax(id),
            _ => unsupported(t, "op not implemented on Vulkan"),
        }
    }

    fn binary(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        let ok = |ty: Ty| ty == Ty::F32 || ty == Ty::F16;
        if !ok(ta.ty) || !ok(tb.ty) || !ok(td.ty) {
            return unsupported(t, "operand type");
        }
        let op = t.op.to_ascii_lowercase();
        let name = format!("{op}_{}_{}_{}", ty_name(ta.ty), ty_name(tb.ty), ty_name(td.ty));
        let norepeat = ta.ne == tb.ne;
        let blob = self.shaders.get(&name, &[norepeat as u32])?;
        let pc = self.binary_pc(&ta, Some(&tb), &td, [0.0; 2], 0);
        Ok(vec![self.launch(blob, vec![ta.range, tb.range, td.range], pc, elementwise_grid(ta.nelements()))?])
    }

    fn unary(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        if !self.t(a).is_contiguous() || !t.is_contiguous() {
            return unsupported(t, "operand is not contiguous");
        }
        let (name, params): (&str, [f32; 4]) = match t.op.as_str() {
            "SCALE" => ("scale", [pf(t, 0), pf(t, 1), 0.0, 0.0]),
            "SQR" => ("sqr", [0.0; 4]),
            "SQRT" => ("sqrt", [0.0; 4]),
            "SIN" => ("sin", [0.0; 4]),
            "COS" => ("cos", [0.0; 4]),
            "LOG" => ("log", [0.0; 4]),
            "CLAMP" => ("clamp", [pf(t, 0), pf(t, 1), 0.0, 0.0]),
            "LEAKY_RELU" => ("leaky_relu", [pf(t, 0), 0.0, 0.0, 0.0]),
            "FILL" => ("fill", [pf(t, 0), 0.0, 0.0, 0.0]),
            "UNARY" => {
                let name = match t.op_params[0] {
                    0 => "abs",
                    1 => "sgn",
                    2 => "neg",
                    3 => "step",
                    4 => "tanh",
                    5 => "elu",
                    6 => "relu",
                    7 => "sigmoid",
                    8 => "gelu",
                    9 => "gelu_quick",
                    10 => "silu",
                    11 => "hardswish",
                    12 => "hardsigmoid",
                    13 => "exp",
                    14 => "expm1",
                    16 => "gelu_erf",
                    18 => "floor",
                    19 => "ceil",
                    20 => "round",
                    21 => "trunc",
                    _ => return unsupported(t, "unary op"),
                };
                (name, [0.0; 4])
            }
            _ => unreachable!(),
        };
        let blob = self.shaders.get(&format!("{name}_f32"), &[])?;
        let ne = td.nelements();
        let pc = self.unary_pc(&ta, &td, ne, params);
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, elementwise_grid(ne))?])
    }

    fn rms_norm(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        let blob = self.shaders.get("rms_norm_f32", &[0, 0])?;
        // Plain rms_norm passes its source as the (unused) multiplier.
        let pc = self.binary_pc(&ta, Some(&ta), &td, [pf(t, 0), 0.0], 0);
        Ok(vec![self.launch(blob, vec![ta.range, ta.range, td.range], pc, [ta.ne[1], ta.ne[2], ta.ne[3]])?])
    }

    /// Fused `rms_norm` + `mul` (see `fuse.rs`): src0 is the normalized input, src1 the weight.
    fn rms_norm_mul(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, w) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tw, td) = (self.tn(a), self.tn(w), self.tn(id));
        let blob = self.shaders.get("rms_norm_f32", &[0, 1])?;
        let pc = self.binary_pc(&ta, Some(&tw), &td, [pf(t, 0), 0.0], 0);
        Ok(vec![self.launch(blob, vec![ta.range, tw.range, td.range], pc, [ta.ne[1], ta.ne[2], ta.ne[3]])?])
    }

    fn norm(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        let blob = self.shaders.get("norm_f32", &[])?;
        let pc = self.unary_pc(&ta, &td, td.nelements(), [pf(t, 0), 0.0, 0.0, 0.0]);
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, rows_grid(ta.nrows()))?])
    }

    fn sum_rows(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        let weight = if t.op == "MEAN" { 1.0 / ta.ne[0] as f32 } else { 1.0 };
        let (m12, l12) = fastdiv(ta.ne[1] * ta.ne[2]);
        let (m1, l1) = fastdiv(ta.ne[1]);
        let pc = words(&[
            ta.ne[0],
            ta.ne[1],
            ta.ne[2],
            ta.nb[1],
            ta.nb[2],
            ta.nb[3],
            td.nb[1],
            td.nb[2],
            td.nb[3],
            weight.to_bits(),
            (self.misalign(&ta) << 16) | self.misalign(&td),
            m12,
            l12,
            m1,
            l1,
        ]);
        let blob = self.shaders.get("sum_rows_f32", &[self.caps.subgroup_size])?;
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, rows_grid(ta.nrows()))?])
    }

    fn soft_max(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let mask = t.src(1).map(|m| self.tn(m));
        let sinks = t.src(2).map(|m| self.tn(m));
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        if ta.ne[0] > 16384 {
            return unsupported(t, "rows longer than 16384 need the three-pass soft_max");
        }
        let f16_mask = match &mask {
            Some(m) if m.ty == Ty::F16 => true,
            Some(m) if m.ty == Ty::F32 => false,
            Some(_) => return unsupported(t, "mask type"),
            None => false,
        };
        for x in [Some(&ta), Some(&td), mask.as_ref(), sinks.as_ref()].into_iter().flatten() {
            self.aligned(t, x)?;
        }
        let scale = pf(t, 0);
        let max_bias = pf(t, 1);
        let n_head_log2 = 1u32 << (ta.ne[2] as f32).log2().floor() as u32;
        let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
        let (ne12, ne13, nb11, nb12, nb13) = mask.map_or((0, 0, 0, 0, 0), |m| (m.ne[2], m.ne[3], m.nb[1], m.nb[2], m.nb[3]));
        let pc = words(&[
            ta.ne[0],
            if mask.is_some() { ta.ne[1] } else { 0 },
            ta.ne[0],
            ta.ne[1],
            ta.ne[2],
            ne12,
            ne13,
            nb11,
            nb12,
            nb13,
            scale.to_bits(),
            max_bias.to_bits(),
            m0.to_bits(),
            m1.to_bits(),
            n_head_log2,
            ta.nrows(),
            sinks.is_some() as u32,
        ]);
        let blob = self.shaders.get(if f16_mask { "soft_max_f32_f16" } else { "soft_max_f32" }, &[self.caps.subgroup_size])?;
        let r = |x: Option<Tn>| x.map_or_else(unused, |x| x.range);
        Ok(vec![self.launch(blob, vec![ta.range, r(mask), r(sinks), td.range], pc, rows_grid(ta.nrows()))?])
    }

    fn concat(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        if ta.ty != td.ty || tb.ty != td.ty || td.ty.is_quantized() {
            return unsupported(t, "concat types");
        }
        let name = match td.ty.size() {
            1 => "concat_i8",
            2 => "concat_i16",
            4 => "concat_i32",
            8 => "concat_i64",
            _ => return unsupported(t, "concat element size"),
        };
        let blob = self.shaders.get(name, &[])?;
        let pc = self.binary_pc_ne(td.nelements(), &ta, Some(&tb), &td, [0.0; 2], t.op_params[0]);
        Ok(vec![self.launch(blob, vec![ta.range, tb.range, td.range], pc, elementwise_grid(td.nelements()))?])
    }


    fn im2col(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (k, x) = (self.src(t, 0)?, self.src(t, 1)?);
        let (tk, tx, td) = (self.tn(k), self.tn(x), self.tn(id));
        if tx.ty != Ty::F32 || !matches!(td.ty, Ty::F32 | Ty::F16) {
            return unsupported(t, "im2col types (f32 input, f32 or f16 output)");
        }
        self.aligned(t, &tx)?;
        self.aligned(t, &td)?;
        let p = |i: usize| t.op_params[i];
        let is_2d = p(6) == 1;
        let ic = tx.ne[if is_2d { 2 } else { 1 }];
        let ih = if is_2d { tx.ne[1] } else { 1 };
        let iw = tx.ne[0];
        let kh = if is_2d { tk.ne[1] } else { 1 };
        let kw = tk.ne[0];
        let oh = if is_2d { td.ne[2] } else { 1 };
        let ow = td.ne[1];
        let offset_delta = tx.nb[if is_2d { 2 } else { 1 }];
        let batch_offset = tx.nb[if is_2d { 3 } else { 2 }];
        let batch = tx.ne[if is_2d { 3 } else { 2 }];
        let chw = ic * kh * kw;
        // The destination address slot is 8 bytes (a buffer address in the BDA variants, unused here).
        let pc = words(&[
            0,
            0,
            batch_offset,
            offset_delta,
            ic,
            iw,
            ih,
            ow,
            oh,
            kw,
            kh,
            oh * batch,
            chw,
            p(0) as u32,
            p(1) as u32,
            p(2) as u32,
            p(3) as u32,
            p(4) as u32,
            p(5) as u32,
            batch * ic,
        ]);
        let name = if td.ty == Ty::F16 { "im2col_f32_f16" } else { "im2col_f32" };
        let blob = self.shaders.get(name, &[self.caps.subgroup_size])?;
        // Cap the X workgroups to limit concurrent channel reads; the shader loops over X.
        let min_cap = if self.caps.vendor_id == 0x1002 { 512 } else { 4096 };
        let x_elements = chw.min(min_cap.max(ow * kh * kw));
        let grid = [ceil_div(x_elements, 512), ow.min(self.caps.max_groups[1]), (oh * batch).min(self.caps.max_groups[2])];
        Ok(vec![self.launch(blob, vec![tx.range, td.range], pc, grid)?])
    }

    fn pool_2d(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        if !self.t(a).is_contiguous() {
            return unsupported(t, "pooling needs a contiguous source");
        }
        self.aligned(t, &ta)?;
        self.aligned(t, &td)?;
        let p = |i: usize| t.op_params[i];
        let (n, oc, oh, ow) = (td.ne[3], td.ne[2], td.ne[1], td.ne[0]);
        let pelements = n * oc * oh * ow;
        let pc = words(&[ta.ne[0], ta.ne[1], ow, oh, oc, pelements, p(0) as u32, p(2) as u32, p(1) as u32, p(4) as u32, p(3) as u32, p(6) as u32, p(5) as u32]);
        let blob = self.shaders.get("pool2d_f32", &[])?;
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, [ceil_div(pelements, 512), 1, 1])?])
    }

    fn conv_2d_dw(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (k, x) = (self.src(t, 0)?, self.src(t, 1)?);
        let (tk, tx, td) = (self.tn(k), self.tn(x), self.tn(id));
        if tx.ty != Ty::F32 || td.ty != Ty::F32 || !matches!(tk.ty, Ty::F32 | Ty::F16) {
            return unsupported(t, "conv_2d_dw types");
        }
        if !self.t(x).is_contiguous() {
            return unsupported(t, "depthwise convolution needs a contiguous input (channels-last layout is not lowered)");
        }
        for v in [&tk, &tx, &td] {
            self.aligned(t, v)?;
        }
        let p = |i: usize| t.op_params[i] as u32;
        let pc = words(&[td.nelements(), td.ne[3], td.ne[2], td.ne[0], td.ne[1], tx.ne[0], tx.ne[1], tk.ne[0], tk.ne[1], p(0), p(1), p(2), p(3), p(4), p(5)]);
        let name = if tk.ty == Ty::F16 { "conv2d_dw_whcn_f16_f32" } else { "conv2d_dw_whcn_f32" };
        let blob = self.shaders.get(name, &[])?;
        Ok(vec![self.launch(blob, vec![tk.range, tx.range, td.range], pc, elementwise_grid(td.nelements()))?])
    }

    fn pad(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        let mut w = vec![td.nelements()];
        for x in [&ta, &td] {
            w.extend(x.ne);
            w.extend(x.nb);
        }
        w.push((self.misalign(&ta) << 16) | self.misalign(&td));
        w.push(t.op_params[8] as u32);
        for i in 0..8 {
            w.push(t.op_params[i] as u32);
        }
        let blob = self.shaders.get("pad_f32", &[])?;
        Ok(vec![self.launch(blob, vec![ta.range, td.range], words(&w), elementwise_grid(td.nelements()))?])
    }

    fn upscale(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "only f32 is lowered");
        }
        let flags = t.op_params[0] as u32;
        let mode = flags & 0xff;
        if mode > 2 || flags & 0x200 != 0 {
            return unsupported(t, "upscale mode (nearest, bilinear and bicubic without antialiasing are lowered)");
        }
        let mut sf = [0, 1, 2, 3].map(|i| td.ne[i] as f32 / ta.ne[i] as f32);
        let mut pixel_offset = 0.5f32;
        if flags & 0x100 != 0 {
            // GGML_SCALE_FLAG_ALIGN_CORNERS
            for i in 0..2 {
                if td.ne[i] > 1 && ta.ne[i] > 1 {
                    sf[i] = (td.ne[i] - 1) as f32 / (ta.ne[i] - 1) as f32;
                }
            }
            pixel_offset = 0.0;
        }
        let pc = words(&[
            td.nelements(),
            self.misalign(&ta),
            self.misalign(&td),
            ta.ne[0],
            ta.ne[1],
            ta.nb[0],
            ta.nb[1],
            ta.nb[2],
            ta.nb[3],
            td.ne[0],
            td.ne[1],
            td.ne[2],
            td.ne[3],
            sf[0].to_bits(),
            sf[1].to_bits(),
            sf[2].to_bits(),
            sf[3].to_bits(),
            pixel_offset.to_bits(),
        ]);
        let blob = self.shaders.get("upscale_f32", &[mode])?;
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, elementwise_grid(td.nelements()))?])
    }

    fn mul_mat(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        if tb.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "activations must be f32");
        }
        let (k, n, cols) = (ta.ne[0], ta.ne[1], tb.ne[1]);
        let (ne02, ne12, ne13) = (ta.ne[2], tb.ne[2], tb.ne[3]);
        if mm_uses_matrix_kernel(self.graph, id) {
            return self.mul_mat_mm(id);
        }
        if !self.t(a).is_contiguous_rows() || tb.nb[0] != 1 {
            return unsupported(t, "weights or activations are not dense");
        }
        self.aligned(t, &ta)?;
        self.aligned(t, &tb)?;
        self.aligned(t, &td)?;
        let (r2, r3) = (ne12 / ne02.max(1), tb.ne[3] / ta.ne[3].max(1));

        // Variant and workgroup, as ggml_vk_get_dequantize_mul_mat_vec.
        let sg = self.caps.subgroup_size;
        let tname = ty_name(ta.ty);
        let (rows_per_wg, mut large) = match ta.ty.0 {
            0 => (1, false),
            1 | 30 => (2, false),
            2 | 3 | 6 | 7 | 8 => (1, false),
            10..=14 => (2, false),
            _ => return unsupported(t, "weight type"),
        };
        if self.caps.vendor_id == 0x10de {
            // Prefer larger workgroups when M is small, to keep more SMs busy.
            large = if ta.ty == Ty::Q6_K { n < 4096 && k >= 1024 } else { n <= 8192 && k >= 1024 };
        }
        // ggml's reduction modes: SUBGROUP (one subgroup per workgroup: add, no shared memory)
        // is `_subgroup_no_shmem`; HYBRID (several subgroups: add, then shared memory across
        // them) is `_subgroup`; SHMEM has no suffix.
        let suffix = match (self.caps.subgroup_arithmetic, large) {
            (true, false) => "_subgroup_no_shmem",
            (true, true) => "_subgroup",
            (false, _) => "",
        };
        let wg_size = if large { sg * 4 } else { sg };
        let blob = self.shaders.get(&format!("mul_mat_vec_{tname}_f32_f32{suffix}"), &[wg_size, rows_per_wg, cols])?;

        let batch_n = cols > 1;
        let (bx, by, bd) = if batch_n { (0, k, n) } else { (k * n, k * cols, n * cols) };
        let pc = words(&[k, k, k, n, bx, by, bd, 0, 0, ne02, ne12, r2, r3]);
        let groups_y = ne12 * ne13;
        let max_x = self.caps.max_groups[0];
        let mut groups_x = ceil_div(n, rows_per_wg);
        let mut groups_z = 1;
        if n > max_x {
            groups_z = 64;
            groups_x = ceil_div(n, groups_z);
        }
        // Weights, activations, destination, then the two optional bias buffers.
        Ok(vec![self.launch(blob, vec![ta.range, tb.range, td.range], pc, [groups_x, groups_y, groups_z])?])
    }


    /// Matrix-matrix multiplication on the scalar `matmul_*` shaders, B read as f32 straight
    /// from the activations (no conversion pass).
    fn mul_mat_mm(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, b) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, tb, td) = (self.tn(a), self.tn(b), self.tn(id));
        if tb.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "activations must be f32");
        }
        if !matches!(ta.ty.0, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 10..=14) {
            return unsupported(t, "weight type for matrix-matrix multiplication");
        }
        let dense_a = ta.nb[1] as u64 * ta.type_size() == self.t(a).row_size()
            && (ta.ne[2] == 1 || ta.nb[2] == ta.ne[1] * ta.nb[1])
            && (ta.ne[3] == 1 || ta.nb[3] == ta.ne[2] * ta.nb[2]);
        let dense_b = tb.nb[0] == 1
            && tb.nb[1] == tb.ne[0]
            && (tb.ne[2] == 1 || tb.nb[2] == tb.ne[1] * tb.nb[1])
            && (tb.ne[3] == 1 || tb.nb[3] == tb.ne[2] * tb.nb[2]);
        if !dense_a || !dense_b || td.nb[0] != 1 {
            return unsupported(t, "operands are not dense");
        }
        self.aligned(t, &ta)?;
        self.aligned(t, &tb)?;
        self.aligned(t, &td)?;
        let p = mm_plan(self.graph, &self.caps, id)?;
        let tile = match p.den[0] {
            128 => "l",
            64 => "m",
            _ => "s",
        };
        let _ = tile;
        let mut spec = p.warptile.to_vec();
        spec.push(p.aligned as u32);
        let blob = self.shaders.get(&format!("matmul_{}_f32", ty_name(ta.ty)), &spec)?;

        let (m, n, k) = (p.m, p.n, p.k);
        let (ne02, ne12, ne13) = (ta.ne[2], tb.ne[2], tb.ne[3]);
        let (r2, r3) = (ne12 / ne02.max(1), ne13 / ta.ne[3].max(1));
        let stride_d = td.nb[1];
        let stride_batch_d = stride_d * td.ne[1];
        let (stride_batch_x, stride_batch_y) = (ta.ne[0] * ta.ne[1], tb.ne[0] * tb.ne[1]);
        let max_z = self.caps.max_groups[2];
        if p.batch > max_z {
            return unsupported(t, "batch count over the device's z workgroup limit");
        }
        let gx = ceil_div(m, p.den[0]);
        let gy = ceil_div(n, p.den[1]);
        let pc = |k_split: u32| {
            words(&[m, n, k, k, k, stride_d, stride_batch_x, stride_batch_y, stride_batch_d, 0, p.batch, k_split, ne02, ne12, r2, r3, n])
        };
        if p.split_k == 1 {
            return Ok(vec![self.launch(blob, vec![ta.range, tb.range, td.range], pc(k), [gx, gy, p.batch])?]);
        }
        // Split-K: partial products into scratch, then a reduction into the destination.
        let off = self.layout.scratch[self.pos[&id]].ok_or_else(|| Error::Invalid("matmul scratch was not planned".into()))?;
        let split = Range::new(Space::Step, self.arena_base + off, p.scratch);
        let k_split = roundup_pow2(ceil_div(k, p.split_k), 256);
        let main = self.launch(blob, vec![ta.range, tb.range, split], pc(k_split), [gx * p.split_k, gy, p.batch])?;
        let reduce = self.shaders.get("split_k_reduce", &[])?;
        let total = m * n * p.batch;
        let red = self.launch(reduce, vec![split, td.range], words(&[total, p.split_k]), [ceil_div(total, 256 * 4), 1, 1])?;
        Ok(vec![main, red])
    }

    /// `vk_op_rope_push_constants`: `a` is the rotated tensor's input layout, `d_nb` the output
    /// row strides (elements), `set_rows_stride` nonzero when the output is scattered into a cache.
    fn rope_words(&self, t: &Tensor, has_ff: bool, a: &Tn, d_nb: [u32; 3], set_rows_stride: u32, d_misalign: u32) -> Result<Vec<u32>> {
        let n_dims = t.op_params[1];
        let mode = t.op_params[2];
        let n_ctx_orig = t.op_params[4];
        let freq_base = pf(t, 5);
        let freq_scale = pf(t, 6);
        let ext_factor = pf(t, 7);
        let attn_factor = pf(t, 8);
        let beta_fast = pf(t, 9);
        let beta_slow = pf(t, 10);
        if !matches!(mode, 0 | 2) {
            return unsupported(t, "rope mode (only normal and neox are lowered)");
        }
        let corr = rope_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        Ok(vec![
            mode as u32,
            a.nrows(),
            n_dims as u32,
            freq_scale.to_bits(),
            freq_base.to_bits(),
            ext_factor.to_bits(),
            attn_factor.to_bits(),
            corr[0].to_bits(),
            corr[1].to_bits(),
            theta_scale.to_bits(),
            has_ff as u32,
            0,
            0,
            0,
            0, // sections
            0, // is_imrope
            0, // is_back
            set_rows_stride,
            a.ne[0],
            a.ne[1],
            a.ne[2],
            a.nb[1],
            a.nb[2],
            a.nb[3],
            d_nb[0],
            d_nb[1],
            d_nb[2],
            self.misalign(a),
            d_misalign,
        ])
    }

    fn rope_grid(&self, a: &Tn) -> [u32; 3] {
        let nrows = a.nrows();
        let (mut x, mut z) = (nrows, 1);
        if nrows > self.caps.max_groups[0] {
            z = ceil_div(nrows, 32768);
            x = 32768;
        }
        [x, ceil_div(a.ne[0], 512), z]
    }

    fn rope(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, pos) = (self.src(t, 0)?, self.src(t, 1)?);
        let ff = t.src(2);
        let (ta, tp, td) = (self.tn(a), self.tn(pos), self.tn(id));
        if ta.ty != Ty::F32 || (td.ty != Ty::F32 && td.ty != Ty::F16) {
            return unsupported(t, "rope types");
        }
        let variant = if t.op_params[2] == 2 { "neox" } else { "norm" };
        let w = self.rope_words(t, ff.is_some(), &ta, [td.nb[1], td.nb[2], td.nb[3]], 0, self.misalign(&td))?;
        let name = format!("rope_{variant}_f32{}", if td.ty == Ty::F16 { "_f16" } else { "" });
        let blob = self.shaders.get(&name, &[])?;
        let ffr = ff.map_or_else(unused, |f| self.tn(f).range);
        // src0, positions, frequency factors, dst, set_rows indices.
        Ok(vec![self.launch(blob, vec![ta.range, tp.range, ffr, td.range], words(&w), self.rope_grid(&ta))?])
    }

    /// Fused `rms_norm` + `mul` + `rope` (see `fuse.rs`): src x, w, pos, ff.
    fn rms_norm_mul_rope(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (x, w, pos) = (self.src(t, 0)?, self.src(t, 1)?, self.src(t, 2)?);
        let ff = t.src(3);
        let (tx, tw, tp, td) = (self.tn(x), self.tn(w), self.tn(pos), self.tn(id));
        if tx.ty != Ty::F32 || tw.ty != Ty::F32 || !matches!(td.ty, Ty::F32 | Ty::F16) {
            return unsupported(t, "rms_norm + mul + rope types");
        }
        self.aligned(t, &tx)?;
        self.aligned(t, &tw)?;
        // The multiplied rows, as the shader's binary block sees them: dense f32, never stored.
        let dense = [1, tx.ne[0], tx.ne[0] * tx.ne[1], tx.ne[0] * tx.ne[1] * tx.ne[2]];
        let mul = Tn { ty: Ty::F32, ne: tx.ne, nb: dense, range: unused(), off: 0 };
        let mut pc = self.binary_pc(&tx, Some(&tw), &mul, [f32::from_bits(t.op_params[15] as u32), 0.0], 0);
        let rw = self.rope_words(t, ff.is_some(), &mul, [td.nb[1], td.nb[2], td.nb[3]], 0, self.misalign(&td))?;
        pc.extend(words(&rw));
        let name = format!("rms_norm_mul_rope_f32_{}", ty_name(td.ty));
        let blob = self.shaders.get(&name, &[0, 1])?;
        let r = |x: Option<Tn>| x.map_or_else(unused, |x| x.range);
        // x, weight, (unused), positions, frequency factors, dst.
        let bindings = vec![tx.range, tw.range, unused(), tp.range, r(ff.map(|f| self.tn(f))), td.range];
        Ok(vec![self.launch(blob, bindings, pc, [tx.ne[1], tx.ne[2], tx.ne[3]])?])
    }

    fn flash_attn(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (q, k, v) = (self.src(t, 0)?, self.src(t, 1)?, self.src(t, 2)?);
        let (tq, tk, tv, td) = (self.tn(q), self.tn(k), self.tn(v), self.tn(id));
        if tq.ty != Ty::F32 || td.ty != Ty::F32 {
            return unsupported(t, "query and output must be f32");
        }
        if !matches!(tk.ty, Ty::F16 | Ty::F32) || !matches!(tv.ty, Ty::F16 | Ty::F32) {
            return unsupported(t, "K/V type (only f16 and f32 are lowered)");
        }
        let mask = t.src(3).map(|m| self.tn(m));
        if let Some(m) = &mask {
            if m.ty != Ty::F16 {
                return unsupported(t, "mask must be f16");
            }
        }
        if t.src(4).is_some() {
            return unsupported(t, "attention sinks");
        }
        for x in [&tq, &tk, &tv] {
            self.aligned(t, x)?;
        }
        let p = fa_plan(self.graph, &self.caps, id)?;
        let sg = self.caps.subgroup_size;
        let flags = (mask.is_some() as u32) * 2 | ((p.softcap != 0.0) as u32) * 4;
        let ty_bytes = |ty: Ty| if ty == Ty::F32 { 16 } else { ty.size() as u32 };
        let spec = [
            p.workgroup_size,
            p.br,
            p.bc,
            p.hsk,
            p.hsv,
            !p.aligned as u32,
            p.d_split,
            p.row_split,
            sg,
            p.shmem_staging as u32,
            flags,
            0, // limit_occupancy_shmem
            tk.ty.0,
            tv.ty.0,
            ty_bytes(tk.ty),
            ty_bytes(tv.ty),
        ];
        let name = if self.caps.fp16 { "flash_attn_f32_f16" } else { "flash_attn_f32_f16_fp32" };
        let blob = self.shaders.get(name, &spec)?;

        let n_head_kv = tq.ne[2];
        let n_head_log2 = 1u32 << (n_head_kv as f32).log2().floor() as u32;
        let m0 = 2.0f32.powf(-p.max_bias / n_head_log2 as f32);
        let m1 = 2.0f32.powf(-(p.max_bias / 2.0) / n_head_log2 as f32);
        let (nem1, nem2, nem3) = mask.map_or((0, 0, 0), |m| (m.ne[1], m.ne[2], m.ne[3]));
        let pc = words(&[
            p.n,
            p.kv,
            td.ne[1],
            td.ne[2],
            td.ne[3],
            tq.ne[2],
            tq.ne[3],
            tk.ne[2],
            tk.ne[3],
            tv.ne[2],
            tv.ne[3],
            nem1,
            nem2,
            nem3,
            tq.nb[1],
            tq.nb[2] * 4,
            tq.nb[3] * 4,
            if tk.ty == Ty::F32 { tk.nb[1] / 4 } else { tk.nb[1] },
            tk.nb[2] * tk.ty.size() as u32,
            tk.nb[3] * tk.ty.size() as u32,
            if tv.ty == Ty::F32 { tv.nb[1] / 4 } else { tv.nb[1] },
            tv.nb[2] * tv.ty.size() as u32,
            tv.nb[3] * tv.ty.size() as u32,
            p.scale.to_bits(),
            p.max_bias.to_bits(),
            p.softcap.to_bits(),
            n_head_log2,
            m0.to_bits(),
            m1.to_bits(),
            p.gqa_ratio,
            p.split_kv,
            p.split_k,
        ]);
        let mask_r = mask.map_or_else(unused, |m| m.range);
        let wg_x = if p.gqa_ratio > 1 { tq.ne[1] } else { ceil_div(tq.ne[1], p.br) };
        if p.split_k > 1 {
            let pos = self.pos[&id];
            let off = self.layout.scratch[pos].ok_or_else(|| Error::Invalid("flash attention scratch was not planned".into()))?;
            let split = Range::new(Space::Step, self.arena_base + off, p.scratch);
            let grid_x = if p.gqa_ratio > 1 { p.split_k * tq.ne[1] } else { ceil_div(tq.ne[1], p.br) * p.split_k };
            let main = self.launch(blob, vec![tq.range, tk.range, tv.range, mask_r, unused(), split], pc, [grid_x, p.workgroups_y, p.workgroups_z])?;
            let reduce = self.shaders.get("fa_split_k_reduce", &[sg])?;
            let pc2 = words(&[p.hsv, td.ne[1], td.ne[2], td.ne[3], p.split_k, 0]);
            let red = self.launch(reduce, vec![split, unused(), td.range], pc2, [td.ne[1], ceil_div(p.hsv, sg), td.ne[2] * td.ne[3]])?;
            return Ok(vec![main, red]);
        }
        Ok(vec![self.launch(blob, vec![tq.range, tk.range, tv.range, mask_r, unused(), td.range], pc, [wg_x, p.workgroups_y, p.workgroups_z])?])
    }

    fn set_rows(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, idx) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, ti, td) = (self.tn(a), self.tn(idx), self.tn(id));
        let itype = match ti.ty {
            Ty::I64 => "i64",
            Ty::I32 => "i32",
            _ => return unsupported(t, "index type"),
        };
        if !matches!(ta.ty, Ty::F32 | Ty::F16) || !matches!(td.ty, Ty::F32 | Ty::F16) {
            return unsupported(t, "set_rows types");
        }
        let blob = self.shaders.get(&format!("set_rows_{}_{}_{itype}", ty_name(ta.ty), ty_name(td.ty)), &[1])?;
        let pc = self.binary_pc(&ta, Some(&ti), &td, [0.0; 2], 0);
        let ne = ceil_div(ta.nelements(), 512);
        let grid = if ne > 262_144 {
            [512, 512, ceil_div(ne, 262_144)]
        } else if ne > 512 {
            [512, ceil_div(ne, 512), 1]
        } else {
            [ne, 1, 1]
        };
        // The destination is read-modify-write: only some rows change.
        Ok(vec![self.launch(blob, vec![ta.range, ti.range, td.range], pc, grid)?])
    }

    fn get_rows(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let (a, idx) = (self.src(t, 0)?, self.src(t, 1)?);
        let (ta, ti, td) = (self.tn(a), self.tn(idx), self.tn(id));
        if td.ty != Ty::F32 || ti.ty != Ty::I32 {
            return unsupported(t, "get_rows needs i32 indices and an f32 result");
        }
        if !matches!(ta.ty.0, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 10..=14) {
            return unsupported(t, "get_rows source type");
        }
        if self.misalign(&ta) != 0 || self.misalign(&ti) != 0 || self.misalign(&td) != 0 {
            return unsupported(t, "get_rows operands must be aligned");
        }
        let blob = self.shaders.get(&format!("get_rows_{}_f32", ty_name(ta.ty)), &[])?;
        let pc = self.binary_pc(&ta, Some(&ti), &td, [0.0; 2], 0);
        let denom = if matches!(ta.ty, Ty::F32 | Ty::F16 | Ty::BF16) { 512 } else { 1024 };
        let grid = [ceil_div(ta.ne[0], denom), ti.ne[0].min(self.caps.max_groups[1]), (ti.ne[1] * ti.ne[2]).min(self.caps.max_groups[2])];
        Ok(vec![self.launch(blob, vec![ta.range, ti.range, td.range], pc, grid)?])
    }

    fn copy(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        // ggml_vk_get_cpy_pipeline: conversions between f32 and f16/bf16/i32 have their
        // own shaders; same-type copies move 4 (or 2) bytes at a time with the f32 (f16) shader.
        let (src, dst) = match (ta.ty, td.ty) {
            (Ty::F32, Ty::F32) | (Ty::F32, Ty::F16) | (Ty::F16, Ty::F16) | (Ty::F16, Ty::F32) => (ty_name(ta.ty), ty_name(td.ty)),
            (Ty::F32, Ty::BF16) | (Ty::BF16, Ty::F32) | (Ty::F32, Ty::I32) | (Ty::I32, Ty::F32) => (ty_name(ta.ty), ty_name(td.ty)),
            (x, y) if x == y && !x.is_quantized() && x.size() == 4 => ("f32".to_string(), "f32".to_string()),
            (x, y) if x == y && !x.is_quantized() && x.size() == 2 => ("f16".to_string(), "f16".to_string()),
            _ => return unsupported(t, "copy types"),
        };
        let contig = self.t(a).is_contiguous() && t.is_contiguous();
        let name = format!("{}cpy_{src}_{dst}", if contig { "contig_" } else { "" });
        let blob = self.shaders.get(&name, &[])?;
        let ne = ta.nelements();
        let pc = self.unary_pc(&ta, &td, ne, [0.0; 4]);
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, elementwise_grid(ne))?])
    }

    fn repeat(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        let name = match ta.ty.size() {
            4 if !ta.ty.is_quantized() => "repeat_i32",
            2 if !ta.ty.is_quantized() => "repeat_i16",
            _ => return unsupported(t, "repeat type"),
        };
        let blob = self.shaders.get(name, &[])?;
        let ne = td.nelements();
        let pc = self.unary_pc(&ta, &td, ne, [0.0; 4]);
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, elementwise_grid(ne))?])
    }

    fn arange(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let td = self.tn(id);
        if td.ty != Ty::F32 {
            return unsupported(t, "arange is f32");
        }
        self.aligned(t, &td)?;
        let blob = self.shaders.get("arange_f32", &[])?;
        let ne = td.nelements();
        // vk_op_push_constants { KX, KY, param1..4 }: start and step.
        let pc = words(&[ne, 1, pf(t, 0).to_bits(), pf(t, 2).to_bits(), 0, 0]);
        Ok(vec![self.launch(blob, vec![td.range], pc, [ceil_div(ne, 512), 1, 1])?])
    }

    fn argmax(&self, id: TensorId) -> Result<Vec<LaunchDesc>> {
        let t = self.t(id);
        let a = self.src(t, 0)?;
        let (ta, td) = (self.tn(a), self.tn(id));
        if ta.ty != Ty::F32 || td.ty != Ty::I32 {
            return unsupported(t, "argmax is f32 -> i32");
        }
        self.aligned(t, &ta)?;
        self.aligned(t, &td)?;
        let blob = self.shaders.get("argmax_f32", &[self.caps.subgroup_size])?;
        let pc = words(&[ta.ne[0], ta.ne[1], 0, 0, 0, 0]);
        let nr = ta.nrows();
        let grid = if nr > 262_144 {
            [512, 512, ceil_div(nr, 262_144)]
        } else if nr > 512 {
            [512, ceil_div(nr, 512), 1]
        } else {
            [nr, 1, 1]
        };
        Ok(vec![self.launch(blob, vec![ta.range, td.range], pc, grid)?])
    }
}
