//! Typed wrappers for the vendored paged-attention kernels (vLLM's CUDA
//! kernels as shipped by mistral.rs; see ../shaders.rs for what was trimmed).
//! Kernel selection, specialization and grid/shared-memory sizing mirror
//! upstream's host launcher in `pagedattention.cuh` (not itself compiled: see
//! that file's doc comment) exactly, the same way [`crate::metal::paged_attn`]
//! mirrors it for Metal. The two should stay numerically identical; the
//! struct fields here match [`crate::metal::paged_attn`]'s one for one.
//!
//! Cache layouts (elements, `x = 16 bytes / sizeof(cache)`):
//! * key cache   `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
//! * value cache `[num_blocks, num_kv_heads, head_size, block_size]`

use std::sync::Arc;

use crate::cuda::shaders::Module;
use crate::cuda::{Arg, BufRef, Device, Kernel};
use crate::{DType, Dims, Error, Result};

/// vLLM's CUDA default (`NUM_THREADS = 128` in `paged_attention_v{1,2}_launcher`);
/// the Metal port uses 256, its own choice for Apple GPUs, so the two differ here.
const NUM_THREADS: u32 = 128;
const WARP_SIZE: u32 = 32;
const PARTITION_SIZE: usize = 512;
const HEAD_SIZES: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];
const BLOCK_SIZES: [usize; 3] = [8, 16, 32];

fn ty(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        // vLLM's CUDA kernels take fp16 as a raw uint16_t bit pattern (`Vec<uint16_t,N>`
        // specializations stand in for `half`), unlike the Metal port which uses `half` directly.
        DType::F16 => "uint16_t",
        DType::BF16 => "__nv_bfloat16",
        DType::F8E4M3 => "uint8_t",
    }
}

fn kv_dt(fp8: bool) -> &'static str {
    if fp8 { "vllm::Fp8KVCacheDataType::kFp8E4M3" } else { "vllm::Fp8KVCacheDataType::kAuto" }
}

/// Innermost key-cache dimension: 16 bytes worth of elements.
pub fn x(cache: DType) -> usize {
    16 / cache.size()
}

fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(msg.into()))
}

fn check_dtypes(dtype: DType, cache: DType, scales: bool) -> Result<()> {
    if dtype == DType::F8E4M3 {
        return invalid("activations cannot be fp8");
    }
    if (cache == DType::F8E4M3) != scales {
        return invalid("an fp8 cache needs k/v scales, and only an fp8 cache takes them");
    }
    if cache != dtype && cache != DType::F8E4M3 {
        return invalid(format!("cache dtype {cache:?} must equal the activation dtype {dtype:?} or be fp8"));
    }
    Ok(())
}

fn kernel(lib: Arc<str>, entry: String) -> Kernel {
    Kernel { library: lib, entry }
}

// ---------------------------------------------------------------------------
// reshape_and_cache: scatter [tokens, heads, head_size] K/V into the paged cache
// ---------------------------------------------------------------------------

pub struct ReshapeAndCache<'a> {
    pub dtype: DType,
    pub cache_dtype: DType,
    pub num_tokens: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
    /// Elements between consecutive tokens of `key` / `value`.
    pub key_stride: usize,
    pub value_stride: usize,
    pub key: BufRef<'a>,
    pub value: BufRef<'a>,
    pub key_cache: BufRef<'a>,
    pub value_cache: BufRef<'a>,
    /// `i64[num_tokens]`; negative = padding, skipped.
    pub slot_mapping: BufRef<'a>,
    pub kv_scales: Option<(BufRef<'a>, BufRef<'a>)>,
}

pub fn reshape_and_cache(dev: &Device, p: &ReshapeAndCache) -> Result<()> {
    check_dtypes(p.dtype, p.cache_dtype, p.kv_scales.is_some())?;
    let x = x(p.cache_dtype);
    if p.head_size % x != 0 {
        return invalid(format!("head_size {} must be a multiple of x = {x}", p.head_size));
    }
    let (kv, cache) = (ty(p.dtype), ty(p.cache_dtype));
    let kvdt = kv_dt(p.kv_scales.is_some());
    let expr = format!("vllm::reshape_and_cache_kernel<{kv},{cache},{kvdt}>");
    let lib = dev.library(&format!("rac:{kv}:{cache}:{kvdt}"), &[&expr], || Module::ReshapeAndCache.source(""))?;
    let k = kernel(lib, expr);

    let (ks, vs): (Arg, Arg) = match p.kv_scales {
        Some((a, b)) => (a.into(), b.into()),
        None => (Arg::NullPtr, Arg::NullPtr),
    };
    let args = [
        p.key.into(),
        p.value.into(),
        p.key_cache.into(),
        p.value_cache.into(),
        p.slot_mapping.into(),
        Arg::I32(p.key_stride as i32),
        Arg::I32(p.value_stride as i32),
        Arg::I32(p.num_heads as i32),
        Arg::I32(p.head_size as i32),
        Arg::I32(p.block_size as i32),
        Arg::I32(x as i32),
        ks,
        vs,
    ];
    let threads = (p.num_heads * p.head_size).min(512) as u32;
    dev.launch(&k, &args, Dims::new([p.num_tokens as u32, 1, 1], [threads, 1, 1]))
}

// ---------------------------------------------------------------------------
// paged_attention (v1: one pass; v2: partitioned + reduce, for long contexts)
// ---------------------------------------------------------------------------

pub struct PagedAttention<'a> {
    pub dtype: DType,
    pub cache_dtype: DType,
    pub num_seqs: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
    pub max_context_len: usize,
    pub max_num_blocks_per_seq: usize,
    pub scale: f32,
    /// `tanh(qk / c) * c`; 1.0 disables (the kernel's sentinel).
    pub softcapping: f32,
    /// Elements between sequences of `q`.
    pub q_stride: usize,
    /// Elements between cache blocks / kv heads in the key cache.
    pub kv_block_stride: usize,
    pub kv_head_stride: usize,

    /// `[num_seqs, num_heads, head_size]`
    pub out: BufRef<'a>,
    pub q: BufRef<'a>,
    pub k_cache: BufRef<'a>,
    pub v_cache: BufRef<'a>,
    /// `u32[num_seqs, max_num_blocks_per_seq]`
    pub block_tables: BufRef<'a>,
    /// `u32[num_seqs]`
    pub context_lens: BufRef<'a>,
    pub kv_scales: Option<(BufRef<'a>, BufRef<'a>)>,
    /// `f32[num_heads]`
    pub alibi_slopes: Option<BufRef<'a>>,
    /// `f32[num_heads]` attention sinks.
    pub sinks: Option<BufRef<'a>>,
}

impl PagedAttention<'_> {
    /// Strides of a densely packed cache with the given geometry.
    pub fn dense_strides(num_kv_heads: usize, head_size: usize, block_size: usize) -> (usize, usize) {
        (num_kv_heads * head_size * block_size, head_size * block_size)
    }

    fn use_v1(&self) -> bool {
        let partitions = self.max_context_len.div_ceil(PARTITION_SIZE);
        (partitions == 1 || self.num_seqs * self.num_heads > 512) && PARTITION_SIZE % self.block_size == 0
    }
}

pub fn paged_attention(dev: &Device, p: &PagedAttention) -> Result<()> {
    check_dtypes(p.dtype, p.cache_dtype, p.kv_scales.is_some())?;
    if !HEAD_SIZES.contains(&p.head_size) {
        return invalid(format!("unsupported head_size {} (have {HEAD_SIZES:?})", p.head_size));
    }
    if !BLOCK_SIZES.contains(&p.block_size) {
        return invalid(format!("unsupported block_size {} (have {BLOCK_SIZES:?})", p.block_size));
    }
    if p.num_kv_heads == 0 || p.num_heads % p.num_kv_heads != 0 {
        return invalid("num_heads must be a multiple of num_kv_heads");
    }
    let v1 = p.use_v1();
    let (t, c, hs, bs) = (ty(p.dtype), ty(p.cache_dtype), p.head_size, p.block_size);
    let kvdt = kv_dt(p.kv_scales.is_some());

    let v1_expr = format!("vllm::paged_attention_v1_kernel<{t},{c},{kvdt},{hs},{bs},{NUM_THREADS}>");
    let v2_expr = format!("vllm::paged_attention_v2_kernel<{t},{c},{kvdt},{hs},{bs},{NUM_THREADS},{PARTITION_SIZE}>");
    let reduce_expr = format!("vllm::paged_attention_v2_reduce_kernel<{t},{hs},{NUM_THREADS},{PARTITION_SIZE}>");
    let exprs: Vec<&str> = if v1 { vec![&v1_expr] } else { vec![&v2_expr, &reduce_expr] };
    let lib = dev.library(&format!("pa:{t}:{c}:{kvdt}:{hs}:{bs}:{v1}"), &exprs, || Module::PagedAttention.source(""))?;

    let (ks, vs): (Arg, Arg) = match p.kv_scales {
        Some((a, b)) => (a.into(), b.into()),
        None => (Arg::NullPtr, Arg::NullPtr),
    };
    let alibi: Arg = p.alibi_slopes.into();
    let sinks: Arg = p.sinks.into();

    let elem = size_of::<f32>();
    let num_warps = (NUM_THREADS / WARP_SIZE) as usize;
    let outputs_size = (num_warps / 2) * hs * elem;

    if v1 {
        let padded = p.max_context_len.div_ceil(bs) * bs;
        let shared = (padded * elem).max(outputs_size);
        let k = kernel(lib, v1_expr);
        let args = [
            p.out.into(),
            p.q.into(),
            p.k_cache.into(),
            p.v_cache.into(),
            Arg::I32(p.num_kv_heads as i32),
            Arg::F32(p.scale),
            Arg::F32(p.softcapping),
            p.block_tables.into(),
            p.context_lens.into(),
            Arg::I32(p.max_num_blocks_per_seq as i32),
            alibi,
            Arg::I32(p.q_stride as i32),
            Arg::I32(p.kv_block_stride as i32),
            Arg::I32(p.kv_head_stride as i32),
            ks,
            vs,
            sinks,
        ];
        return dev.launch(
            &k,
            &args,
            Dims::new([p.num_heads as u32, p.num_seqs as u32, 1], [NUM_THREADS, 1, 1]).shared(shared as u32),
        );
    }

    // v2: per-partition partial results, then a reduction over partitions.
    let rows = p.num_seqs * p.num_heads;
    let partitions = p.max_context_len.div_ceil(PARTITION_SIZE);
    let exp_sums = dev.alloc(rows * partitions * elem)?;
    let max_logits = dev.alloc(rows * partitions * elem)?;
    let tmp_out = dev.alloc(rows * partitions * hs * p.dtype.size())?;

    let shared = (PARTITION_SIZE * elem).max(outputs_size);
    let main = kernel(lib.clone(), v2_expr);
    let args = [
        (&exp_sums).into(),
        (&max_logits).into(),
        (&tmp_out).into(),
        p.q.into(),
        p.k_cache.into(),
        p.v_cache.into(),
        Arg::I32(p.num_kv_heads as i32),
        Arg::F32(p.scale),
        Arg::F32(p.softcapping),
        p.block_tables.into(),
        p.context_lens.into(),
        Arg::I32(p.max_num_blocks_per_seq as i32),
        alibi,
        Arg::I32(p.q_stride as i32),
        Arg::I32(p.kv_block_stride as i32),
        Arg::I32(p.kv_head_stride as i32),
        ks,
        vs,
        sinks,
    ];
    dev.launch(
        &main,
        &args,
        Dims::new([p.num_heads as u32, p.num_seqs as u32, partitions as u32], [NUM_THREADS, 1, 1]).shared(shared as u32),
    )?;

    let reduce = kernel(lib, reduce_expr);
    let reduce_shared = 2 * partitions * elem;
    let rargs = [
        p.out.into(),
        (&exp_sums).into(),
        (&max_logits).into(),
        (&tmp_out).into(),
        p.context_lens.into(),
        Arg::I32(partitions as i32),
        sinks,
    ];
    dev.launch(
        &reduce,
        &rargs,
        Dims::new([p.num_heads as u32, p.num_seqs as u32, 1], [NUM_THREADS, 1, 1]).shared(reduce_shared as u32),
    )
}

// ---------------------------------------------------------------------------
// copy_blocks / gather_kv_cache
// ---------------------------------------------------------------------------

pub struct CopyBlocks<'a> {
    pub dtype: DType,
    pub key_cache: BufRef<'a>,
    pub value_cache: BufRef<'a>,
    /// `i64[num_pairs, 2]` of (src block, dst block).
    pub block_mapping: BufRef<'a>,
    pub num_pairs: usize,
    pub numel_per_block_key: usize,
    pub numel_per_block_value: usize,
}

pub fn copy_blocks(dev: &Device, p: &CopyBlocks) -> Result<()> {
    if p.numel_per_block_key != p.numel_per_block_value {
        return invalid("key and value blocks must have the same size");
    }
    let t = ty(p.dtype);
    let expr = format!("copy_blocks_kernel<{t}>");
    let lib = dev.library(&format!("cb:{t}"), &[&expr], || Module::CopyBlocks.source(""))?;
    let k = kernel(lib, expr);
    let args = [
        p.key_cache.into(),
        p.value_cache.into(),
        p.block_mapping.into(),
        Arg::I32(p.numel_per_block_key as i32),
        Arg::I32(p.numel_per_block_value as i32),
    ];
    dev.launch(&k, &args, Dims::new([p.num_pairs as u32, 1, 1], [p.numel_per_block_key.min(1024) as u32, 1, 1]))
}

pub struct GatherKvCache<'a> {
    pub cache_dtype: DType,
    pub out_dtype: DType,
    pub key_cache: BufRef<'a>,
    pub value_cache: BufRef<'a>,
    /// `[num_tokens, num_kv_heads, head_size]`
    pub k_out: BufRef<'a>,
    pub v_out: BufRef<'a>,
    pub kv_scales: Option<(BufRef<'a>, BufRef<'a>)>,
    /// `i32[num_seqs, block_table_stride]`
    pub block_table: BufRef<'a>,
    /// `i32[num_seqs + 1]` prefix sums of sequence lengths.
    pub cu_seq_lens: BufRef<'a>,
    pub num_tokens: usize,
    pub num_seqs: usize,
    pub block_size: usize,
    pub block_table_stride: usize,
    pub num_kv_heads: usize,
    pub head_size: usize,
}

pub fn gather_kv_cache(dev: &Device, p: &GatherKvCache) -> Result<()> {
    check_dtypes(p.out_dtype, p.cache_dtype, p.kv_scales.is_some())?;
    let (c, o) = (ty(p.cache_dtype), ty(p.out_dtype));
    let kvdt = kv_dt(p.kv_scales.is_some());
    let expr = format!("vllm::gather_kv_cache_kernel<{c},{o},{kvdt}>");
    let lib = dev.library(&format!("gkv:{c}:{o}:{kvdt}"), &[&expr], || Module::GatherKvCache.source(""))?;
    let k = kernel(lib, expr);

    let (ks, vs): (Arg, Arg) = match p.kv_scales {
        Some((a, b)) => (a.into(), b.into()),
        None => (Arg::NullPtr, Arg::NullPtr),
    };
    let args = [
        p.key_cache.into(),
        p.value_cache.into(),
        p.k_out.into(),
        p.v_out.into(),
        ks,
        vs,
        p.block_table.into(),
        p.cu_seq_lens.into(),
        Arg::I32(p.num_tokens as i32),
        Arg::I32(p.num_seqs as i32),
        Arg::I32(p.block_size as i32),
        Arg::I32(p.block_table_stride as i32),
        Arg::I32(p.num_kv_heads as i32),
        Arg::I32(p.head_size as i32),
        Arg::I32(x(p.cache_dtype) as i32),
    ];
    let threads = (p.num_kv_heads * p.head_size).min(512) as u32;
    dev.launch(&k, &args, Dims::new([p.num_tokens as u32, 1, 1], [threads, 1, 1]))
}
