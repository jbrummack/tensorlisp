//! Typed wrappers for the vendored paged-attention kernels (vLLM's design as
//! ported to Metal by mistral.rs). Kernel selection, specialization constants
//! and grid/shared-memory sizing follow mistral.rs' host code.
//!
//! Cache layouts (elements, `x = 16 bytes / sizeof(cache)`):
//! * key cache   `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
//! * value cache `[num_blocks, num_kv_heads, head_size, block_size]`

use crate::metal::shaders::Module;
use crate::metal::{Arg, BufRef, Device, Kernel};
use crate::{Const, DType, Dims, Error, Result};

const NUM_THREADS: u32 = 256;
const NUM_SIMD_LANES: u32 = 32;
const PARTITION_SIZE: usize = 512;
const HEAD_SIZES: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];
const BLOCK_SIZES: [usize; 3] = [8, 16, 32];

fn ty(d: DType) -> &'static str {
    match d {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        DType::F8E4M3 => "uchar",
    }
}

fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(msg.into()))
}

/// Innermost key-cache dimension: 16 bytes worth of elements.
pub fn x(cache: DType) -> usize {
    16 / cache.size()
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

fn kernel(lib: std::sync::Arc<str>, entry: String, consts: Vec<(u32, Const)>) -> Kernel {
    Kernel { library: lib, entry, consts }
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
    let lib = dev.library(&format!("rac:{kv}:{cache}"), || {
        Module::ReshapeAndCache.source(&format!("instantiate_reshape_and_cache({kv}, {cache});"))
    })?;
    let k = kernel(
        lib,
        format!("reshape_and_cache_kv_{kv}_cache_{cache}"),
        vec![(10, Const::Bool(p.kv_scales.is_some()))],
    );
    let mut args: Vec<(u32, Arg)> = vec![
        (0, p.key.into()),
        (1, p.value.into()),
        (2, p.key_cache.into()),
        (3, p.value_cache.into()),
        (4, p.slot_mapping.into()),
        (7, Arg::I32(p.key_stride as i32)),
        (8, Arg::I32(p.value_stride as i32)),
        (9, Arg::I32(p.num_heads as i32)),
        (10, Arg::I32(p.head_size as i32)),
        (11, Arg::I32(p.block_size as i32)),
        (12, Arg::I32(x as i32)),
    ];
    if let Some((ks, vs)) = p.kv_scales {
        args.push((5, ks.into()));
        args.push((6, vs.into()));
    }
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
    let ps = if v1 { 0 } else { PARTITION_SIZE };
    let entry = format!("paged_attention_{t}_cache_{c}_hs{hs}_bs{bs}_nt{NUM_THREADS}_nsl{NUM_SIMD_LANES}_ps{ps}");

    let lib = dev.library(&format!("pa:{t}:{c}:{hs}:{bs}:{ps}"), || {
        let mut inst = format!(
            "instantiate_paged_attention_inner({t}, {c}, {hs}, {bs}, {NUM_THREADS}, {NUM_SIMD_LANES}, {ps});\n"
        );
        if !v1 {
            inst.push_str(&format!(
                "instantiate_paged_attention_v2_reduce_inner({t}, {hs}, {NUM_THREADS}, {NUM_SIMD_LANES}, {ps});\n"
            ));
        }
        Module::PagedAttention.source(&inst)
    })?;
    let consts = vec![
        (10, Const::Bool(!v1)),
        (20, Const::Bool(p.alibi_slopes.is_some())),
        (30, Const::Bool(p.kv_scales.is_some())),
        (40, Const::Bool(p.sinks.is_some())),
    ];
    let main = kernel(lib.clone(), entry, consts);

    let elem = std::mem::size_of::<f32>();
    let num_simds = (NUM_THREADS / NUM_SIMD_LANES) as usize;
    let outputs_size = (num_simds / 2) * hs * elem;
    let partitions = p.max_context_len.div_ceil(PARTITION_SIZE);

    // Buffers common to both variants (slots 3..), see the kernel signature.
    let mut args: Vec<(u32, Arg)> = vec![
        (3, p.q.into()),
        (4, p.k_cache.into()),
        (5, p.v_cache.into()),
        (8, Arg::I32(p.num_kv_heads as i32)),
        (9, Arg::F32(p.scale)),
        (10, Arg::F32(p.softcapping)),
        (11, p.block_tables.into()),
        (12, p.context_lens.into()),
        (13, Arg::I32(p.max_num_blocks_per_seq as i32)),
        (15, Arg::I32(p.q_stride as i32)),
        (16, Arg::I32(p.kv_block_stride as i32)),
        (17, Arg::I32(p.kv_head_stride as i32)),
    ];
    if let Some((ks, vs)) = p.kv_scales {
        args.push((6, ks.into()));
        args.push((7, vs.into()));
    }
    if let Some(a) = p.alibi_slopes {
        args.push((14, a.into()));
    }
    if let Some(s) = p.sinks {
        args.push((18, s.into()));
    }

    if v1 {
        let padded = p.max_context_len.div_ceil(bs) * bs;
        let shared = (padded * elem).max(outputs_size);
        args.push((2, p.out.into()));
        return dev.launch(
            &main,
            &args,
            Dims::new([p.num_heads as u32, p.num_seqs as u32, 1], [NUM_THREADS, 1, 1]).shared(shared as u32),
        );
    }

    // v2: per-partition partial results, then a reduction over partitions.
    let rows = p.num_seqs * p.num_heads;
    let exp_sums = dev.alloc(rows * partitions * elem)?;
    let max_logits = dev.alloc(rows * partitions * elem)?;
    let tmp_out = dev.alloc(rows * partitions * hs * p.dtype.size())?;
    args.push((0, (&exp_sums).into()));
    args.push((1, (&max_logits).into()));
    args.push((2, (&tmp_out).into()));
    let shared = (PARTITION_SIZE * elem).max(outputs_size);
    dev.launch(
        &main,
        &args,
        Dims::new([p.num_heads as u32, p.num_seqs as u32, partitions as u32], [NUM_THREADS, 1, 1])
            .shared(shared as u32),
    )?;

    let reduce = kernel(
        lib,
        format!("paged_attention_v2_reduce_{t}_hs{hs}_nt{NUM_THREADS}_nsl{NUM_SIMD_LANES}_ps{PARTITION_SIZE}"),
        vec![(40, Const::Bool(p.sinks.is_some()))],
    );
    let mut rargs: Vec<(u32, Arg)> = vec![
        (0, p.out.into()),
        (1, (&exp_sums).into()),
        (2, (&max_logits).into()),
        (3, (&tmp_out).into()),
        (4, p.context_lens.into()),
        (5, Arg::I32(partitions as i32)),
    ];
    if let Some(s) = p.sinks {
        rargs.push((6, s.into()));
    }
    dev.launch(
        &reduce,
        &rargs,
        Dims::new([p.num_heads as u32, p.num_seqs as u32, 1], [NUM_THREADS, 1, 1])
            .shared((2 * partitions * elem) as u32),
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
    let lib = dev.library(&format!("cb:{t}"), || Module::CopyBlocks.source(&format!("instantiate_copy_blocks({t});")))?;
    let k = kernel(lib, format!("copy_blocks_{t}"), vec![]);
    dev.launch(
        &k,
        &[
            (0, p.key_cache.into()),
            (1, p.value_cache.into()),
            (2, p.block_mapping.into()),
            (3, Arg::I32(p.numel_per_block_key as i32)),
            (4, Arg::I32(p.numel_per_block_value as i32)),
        ],
        Dims::new([p.num_pairs as u32, 1, 1], [p.numel_per_block_key.min(1024) as u32, 1, 1]),
    )
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
    let lib = dev.library(&format!("gkv:{c}:{o}"), || {
        Module::GatherKvCache.source(&format!("instantiate_gather_kv_cache({c}, {o});"))
    })?;
    let k = kernel(
        lib,
        format!("gather_kv_cache_cache_{c}_out_{o}"),
        vec![(10, Const::Bool(p.kv_scales.is_some()))],
    );
    let mut args: Vec<(u32, Arg)> = vec![
        (0, p.key_cache.into()),
        (1, p.value_cache.into()),
        (2, p.k_out.into()),
        (3, p.v_out.into()),
        (6, p.block_table.into()),
        (7, p.cu_seq_lens.into()),
        (8, Arg::I32(p.num_tokens as i32)),
        (9, Arg::I32(p.num_seqs as i32)),
        (10, Arg::I32(p.block_size as i32)),
        (11, Arg::I32(p.block_table_stride as i32)),
        (12, Arg::I32(p.num_kv_heads as i32)),
        (13, Arg::I32(p.head_size as i32)),
        (14, Arg::I32(x(p.cache_dtype) as i32)),
    ];
    if let Some((ks, vs)) = p.kv_scales {
        args.push((4, ks.into()));
        args.push((5, vs.into()));
    }
    let threads = (p.num_kv_heads * p.head_size).min(512) as u32;
    dev.launch(&k, &args, Dims::new([p.num_tokens as u32, 1, 1], [threads, 1, 1]))
}
