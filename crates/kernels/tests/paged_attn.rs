#![cfg(target_os = "macos")]

use half::{bf16, f16};
use tensorlisp_kernels::metal::paged_attn::*;
use tensorlisp_kernels::metal::{Buffer, Device};
use tensorlisp_kernels::DType;

/// No GPU is only acceptable when explicitly allowed; a silent skip would pass vacuously.
fn device() -> Option<Device> {
    match Device::system_default() {
        Ok(d) => Some(d),
        Err(_) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => None,
        Err(e) => panic!("{e} (set TL_ALLOW_NO_GPU=1 to skip)"),
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

trait Elem: Copy {
    const DTYPE: DType;
    const TOL: f32;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
}
impl Elem for f32 {
    const DTYPE: DType = DType::F32;
    const TOL: f32 = 1e-4;
    fn from_f32(v: f32) -> Self { v }
    fn to_f32(self) -> f32 { self }
}
impl Elem for f16 {
    const DTYPE: DType = DType::F16;
    const TOL: f32 = 3e-3;
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
    fn to_f32(self) -> f32 { f16::to_f32(self) }
}
impl Elem for bf16 {
    const DTYPE: DType = DType::BF16;
    const TOL: f32 = 2e-2;
    fn from_f32(v: f32) -> Self { bf16::from_f32(v) }
    fn to_f32(self) -> f32 { bf16::to_f32(self) }
}

struct Case {
    ctx: Vec<usize>,
    heads: usize,
    kv_heads: usize,
    hs: usize,
    bs: usize,
    softcap: f32,
}

/// Fills a paged cache through `reshape_and_cache`, runs `paged_attention`,
/// and compares against a straightforward f32 softmax(q k^T) v.
fn run<T: Elem>(case: Case) {
    let Some(dev) = device() else { return };
    let Case { ctx, heads, kv_heads, hs, bs, softcap } = case;
    let seqs = ctx.len();
    let mut rng = Rng(0x9E3779B97F4A7C15);

    // Physical block assignment: shuffled so the block table is non-trivial.
    let blocks_per_seq: Vec<usize> = ctx.iter().map(|c| c.div_ceil(bs)).collect();
    let max_blocks = *blocks_per_seq.iter().max().unwrap();
    let num_blocks = blocks_per_seq.iter().sum::<usize>() + 3;
    let mut free: Vec<u32> = (0..num_blocks as u32).collect();
    for i in (1..free.len()).rev() {
        free.swap(i, rng.below(i + 1));
    }
    let mut tables = vec![0u32; seqs * max_blocks];
    let mut slots = Vec::<i64>::new();
    for (s, &c) in ctx.iter().enumerate() {
        for b in 0..blocks_per_seq[s] {
            tables[s * max_blocks + b] = free.pop().unwrap();
        }
        for t in 0..c {
            slots.push((tables[s * max_blocks + t / bs] as usize * bs + t % bs) as i64);
        }
    }
    let total: usize = ctx.iter().sum();

    let row = kv_heads * hs;
    let k: Vec<T> = (0..total * row).map(|_| T::from_f32(rng.f32())).collect();
    let v: Vec<T> = (0..total * row).map(|_| T::from_f32(rng.f32())).collect();
    let q: Vec<T> = (0..seqs * heads * hs).map(|_| T::from_f32(rng.f32())).collect();

    let cache_elems = num_blocks * kv_heads * hs * bs;
    let k_cache = dev.alloc(cache_elems * size_of::<T>()).unwrap();
    let v_cache = dev.alloc(cache_elems * size_of::<T>()).unwrap();
    k_cache.write(&vec![T::from_f32(0.0); cache_elems]);
    v_cache.write(&vec![T::from_f32(0.0); cache_elems]);
    let (kb, vb, qb) = (dev.upload(&k).unwrap(), dev.upload(&v).unwrap(), dev.upload(&q).unwrap());
    let slot_buf = dev.upload(&slots).unwrap();

    reshape_and_cache(
        &dev,
        &ReshapeAndCache {
            dtype: T::DTYPE,
            cache_dtype: T::DTYPE,
            num_tokens: total,
            num_heads: kv_heads,
            head_size: hs,
            block_size: bs,
            key_stride: row,
            value_stride: row,
            key: (&kb).into(),
            value: (&vb).into(),
            key_cache: (&k_cache).into(),
            value_cache: (&v_cache).into(),
            slot_mapping: (&slot_buf).into(),
            kv_scales: None,
        },
    )
    .unwrap();

    // Gather round-trips the cache exactly (also exercises reshape_and_cache's layout).
    let cu: Vec<i32> = std::iter::once(0)
        .chain(ctx.iter().scan(0, |a, c| {
            *a += *c as i32;
            Some(*a)
        }))
        .collect();
    let tables_i32: Vec<i32> = tables.iter().map(|&t| t as i32).collect();
    let (cu_buf, tbl_i32_buf) = (dev.upload(&cu).unwrap(), dev.upload(&tables_i32).unwrap());
    let (k_out, v_out) = (dev.alloc(total * row * size_of::<T>()).unwrap(), dev.alloc(total * row * size_of::<T>()).unwrap());
    gather_kv_cache(
        &dev,
        &GatherKvCache {
            cache_dtype: T::DTYPE,
            out_dtype: T::DTYPE,
            key_cache: (&k_cache).into(),
            value_cache: (&v_cache).into(),
            k_out: (&k_out).into(),
            v_out: (&v_out).into(),
            kv_scales: None,
            block_table: (&tbl_i32_buf).into(),
            cu_seq_lens: (&cu_buf).into(),
            num_tokens: total,
            num_seqs: seqs,
            block_size: bs,
            block_table_stride: max_blocks,
            num_kv_heads: kv_heads,
            head_size: hs,
        },
    )
    .unwrap();

    let tbl_buf = dev.upload(&tables).unwrap();
    let ctx_u32: Vec<u32> = ctx.iter().map(|&c| c as u32).collect();
    let ctx_buf = dev.upload(&ctx_u32).unwrap();
    let out = dev.alloc(seqs * heads * hs * size_of::<T>()).unwrap();
    let scale = 1.0 / (hs as f32).sqrt();
    let (kv_block_stride, kv_head_stride) = PagedAttention::dense_strides(kv_heads, hs, bs);
    paged_attention(
        &dev,
        &PagedAttention {
            dtype: T::DTYPE,
            cache_dtype: T::DTYPE,
            num_seqs: seqs,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_size: hs,
            block_size: bs,
            max_context_len: *ctx.iter().max().unwrap(),
            max_num_blocks_per_seq: max_blocks,
            scale,
            softcapping: softcap,
            q_stride: heads * hs,
            kv_block_stride,
            kv_head_stride,
            out: (&out).into(),
            q: (&qb).into(),
            k_cache: (&k_cache).into(),
            v_cache: (&v_cache).into(),
            block_tables: (&tbl_buf).into(),
            context_lens: (&ctx_buf).into(),
            kv_scales: None,
            alibi_slopes: None,
            sinks: None,
        },
    )
    .unwrap();
    dev.sync().unwrap();

    let gk: Vec<T> = k_out.read(total * row);
    let gv: Vec<T> = v_out.read(total * row);
    assert!(gk.iter().zip(&k).all(|(a, b)| a.to_f32() == b.to_f32()), "gathered keys differ");
    assert!(gv.iter().zip(&v).all(|(a, b)| a.to_f32() == b.to_f32()), "gathered values differ");

    let got: Vec<T> = out.read(seqs * heads * hs);
    let group = heads / kv_heads;
    let mut max_err = 0f32;
    let mut base = 0;
    for (s, &c) in ctx.iter().enumerate() {
        for h in 0..heads {
            let kvh = h / group;
            let qv = &q[(s * heads + h) * hs..][..hs];
            let mut logits: Vec<f32> = (0..c)
                .map(|t| {
                    let kt = &k[((base + t) * kv_heads + kvh) * hs..][..hs];
                    let mut qk = qv.iter().zip(kt).map(|(a, b)| a.to_f32() * b.to_f32()).sum::<f32>() * scale;
                    if softcap != 1.0 {
                        qk = (qk / softcap).tanh() * softcap;
                    }
                    qk
                })
                .collect();
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            logits.iter_mut().for_each(|l| *l = (*l - m).exp());
            let z: f32 = logits.iter().sum();
            for d in 0..hs {
                let want: f32 = (0..c).map(|t| logits[t] / z * v[((base + t) * kv_heads + kvh) * hs + d].to_f32()).sum();
                max_err = max_err.max((want - got[(s * heads + h) * hs + d].to_f32()).abs());
            }
        }
        base += c;
    }
    assert!(max_err <= T::TOL, "max abs error {max_err} > {}", T::TOL);
}

fn short() -> Case {
    Case { ctx: vec![1, 17, 40], heads: 8, kv_heads: 8, hs: 128, bs: 16, softcap: 1.0 }
}

#[test]
fn v1_f32() {
    run::<f32>(short());
}
#[test]
fn v1_f16_gqa() {
    run::<f16>(Case { kv_heads: 2, ..short() });
}
#[test]
fn v1_bf16() {
    run::<bf16>(Case { hs: 64, bs: 8, ..short() });
}
#[test]
fn v1_softcap() {
    run::<f32>(Case { softcap: 30.0, ..short() });
}
#[test]
fn v1_odd_head_size_and_block32() {
    run::<f32>(Case { hs: 80, bs: 32, ..short() });
}
#[test]
fn v2_long_context_f32() {
    // > 512 tokens and few (seq*head) rows => the partitioned kernel + reduce.
    run::<f32>(Case { ctx: vec![700, 1300], heads: 4, kv_heads: 2, hs: 128, bs: 16, softcap: 1.0 });
}
#[test]
fn v2_long_context_f16() {
    run::<f16>(Case { ctx: vec![515, 5], heads: 4, kv_heads: 4, hs: 64, bs: 32, softcap: 1.0 });
}

#[test]
fn copy_blocks_copies_pairs() {
    let Some(dev) = device() else { return };
    let (nb, n) = (6usize, 40usize);
    let kc: Vec<f32> = (0..nb * n).map(|i| i as f32).collect();
    let vc: Vec<f32> = (0..nb * n).map(|i| -(i as f32)).collect();
    let (kb, vb): (Buffer, Buffer) = (dev.upload(&kc).unwrap(), dev.upload(&vc).unwrap());
    let map = dev.upload(&[0i64, 4, 2, 5]).unwrap();
    copy_blocks(
        &dev,
        &CopyBlocks {
            dtype: DType::F32,
            key_cache: (&kb).into(),
            value_cache: (&vb).into(),
            block_mapping: (&map).into(),
            num_pairs: 2,
            numel_per_block_key: n,
            numel_per_block_value: n,
        },
    )
    .unwrap();
    dev.sync().unwrap();
    let (k2, v2): (Vec<f32>, Vec<f32>) = (kb.read(nb * n), vb.read(nb * n));
    for (src, dst) in [(0, 4), (2, 5)] {
        assert_eq!(k2[dst * n..][..n], kc[src * n..][..n]);
        assert_eq!(v2[dst * n..][..n], vc[src * n..][..n]);
    }
    assert_eq!(k2[n..2 * n], kc[n..2 * n], "untouched block changed");
}
