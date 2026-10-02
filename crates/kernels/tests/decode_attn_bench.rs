//! One-query attention on T5Gemma2's decode geometry (4 query heads, 1 K/V
//! head, head size 256, f16 cache): vLLM's paged attention against the native
//! executor's flash attention (one block per head, and split-KV). Timing only,
//! plus an agreement check on the all-valid case:
//! `cargo test --release -p tensorlisp-kernels --features cuda --test decode_attn_bench -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use std::time::Instant;

use half::f16;
use tensorlisp_kernels::cuda::exec::{Executor, Shared};
use tensorlisp_kernels::cuda::paged_attn::*;
use tensorlisp_kernels::cuda::Device;
use tensorlisp_kernels::graph::{Graph, Storage, Tensor, Ty};
use tensorlisp_kernels::plan::{align, Leaves};
use tensorlisp_kernels::DType;

const HEADS: usize = 4;
const HS: usize = 256;
const BS: usize = 16;

fn rng(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn tensor(ty: Ty, ne: [i64; 4], op: &str, storage: Storage, src: Vec<Option<usize>>) -> Tensor {
    let mut nb = [ty.size(); 4].map(|b| b as u64);
    for i in 1..4 {
        nb[i] = nb[i - 1] * ne[i - 1] as u64;
    }
    let mut op_params = [0i32; 16];
    op_params[0] = (1.0f32 / (HS as f32).sqrt()).to_bits() as i32;
    Tensor { ty, ne, nb, op: op.into(), op_params, src, view_src: None, view_offs: 0, storage, name: op.into() }
}

fn time(n: usize, mut f: impl FnMut(usize)) -> f64 {
    f(20);
    let t = Instant::now();
    f(n);
    t.elapsed().as_secs_f64() * 1e6 / n as f64
}

#[test]
#[ignore]
fn decode_attention_paged_vs_flash() {
    let dev = std::sync::Arc::new(Device::system_default().expect("a CUDA device"));
    let executor = Executor::new(dev.clone()).unwrap();
    println!("{}", dev.name());
    println!("{:>30} {:>9} {:>9} {:>9} {:>9}", "(rows, valid self + memory)", "fa 1blk", "fa split", "paged", "paged/split");

    for (rows, n_self, n_mem) in [(1088usize, 64usize, 1024usize), (1088, 10, 10), (1088, 20, 270), (8192, 64, 8128)] {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let k: Vec<f16> = (0..rows * HS).map(|_| f16::from_f32(rng(&mut seed))).collect();
        let v: Vec<f16> = (0..rows * HS).map(|_| f16::from_f32(rng(&mut seed))).collect();
        let q: Vec<f32> = (0..HEADS * HS).map(|_| rng(&mut seed)).collect();
        // Cache rows 0..64 hold the decoder's tokens, 64.. the memory's.
        let valid = |j: usize| j < n_self || (64..64 + n_mem).contains(&j);
        let mask: Vec<f16> = (0..rows).map(|j| f16::from_f32(if valid(j) { 0.0 } else { f32::NEG_INFINITY })).collect();
        let live: Vec<usize> = (0..rows).filter(|&j| valid(j)).collect();

        // --- native flash attention over the full cache with the mask
        let tensors = vec![
            tensor(Ty::F32, [HS as i64, 1, HEADS as i64, 1], "NONE", Storage::Input(0), vec![]),
            tensor(Ty::F16, [HS as i64, rows as i64, 1, 1], "NONE", Storage::State(0), vec![]),
            tensor(Ty::F16, [HS as i64, rows as i64, 1, 1], "NONE", Storage::State(1), vec![]),
            tensor(Ty::F16, [rows as i64, 1, 1, 1], "NONE", Storage::Input(1), vec![]),
            tensor(Ty::F32, [HS as i64, HEADS as i64, 1, 1], "FLASH_ATTN_EXT", Storage::Temp, vec![Some(0), Some(1), Some(2), Some(3)]),
        ];
        let graph = Graph { tensors, nodes: vec![4], keep: vec![4] };
        let cache_bytes = (rows * HS * 2) as u64;
        let states = [0, align(cache_bytes)];
        let state = dev.alloc((align(cache_bytes) * 2) as usize).unwrap();
        state.write_at(0, bytemuck_f16(&k));
        state.write_at(states[1] as usize, bytemuck_f16(&v));
        let leaves = Leaves { weights: &[], states: &states };
        let shared = Shared { weights: &[], state: &state };
        let mut programs = Vec::new();
        for mode in ["plain", "split"] {
            // SAFETY: single-threaded test binary section; read when the program is compiled.
            unsafe { std::env::set_var("TL_NATIVE_FA", mode) };
            let program = executor.compile(graph.clone(), &leaves).unwrap();
            program.set_input(0, f32_bytes(&q)).unwrap();
            program.set_input(1, bytemuck_f16(&mask)).unwrap();
            programs.push(program);
        }
        let fa_plain = time(300, |n| programs[0].run_n(&shared, n).unwrap());
        let fa_split = time(300, |n| programs[1].run_n(&shared, n).unwrap());
        let native_out = {
            programs[1].run_n(&shared, 1).unwrap();
            programs[1].read(4, HEADS * HS * 4).unwrap()
        };

        // --- paged attention over the compacted valid keys
        let ctx = live.len();
        let blocks = ctx.div_ceil(BS);
        let x = x(DType::F16);
        let mut kc = vec![f16::ZERO; blocks * HS * BS];
        let mut vc = vec![f16::ZERO; blocks * HS * BS];
        for (t, &j) in live.iter().enumerate() {
            let (b, o) = (t / BS, t % BS);
            for d in 0..HS {
                kc[((b * (HS / x) + d / x) * BS + o) * x + d % x] = k[j * HS + d];
                vc[(b * HS + d) * BS + o] = v[j * HS + d];
            }
        }
        let (kb, vb) = (dev.upload(&kc).unwrap(), dev.upload(&vc).unwrap());
        let q16: Vec<f16> = q.iter().map(|&a| f16::from_f32(a)).collect();
        let qb = dev.upload(&q16).unwrap();
        let tables: Vec<u32> = (0..blocks as u32).collect();
        let (tb, cb) = (dev.upload(&tables).unwrap(), dev.upload(&[ctx as u32]).unwrap());
        let out = dev.alloc(HEADS * HS * 2).unwrap();
        let (kv_block_stride, kv_head_stride) = PagedAttention::dense_strides(1, HS, BS);
        let p = PagedAttention {
            dtype: DType::F16,
            cache_dtype: DType::F16,
            num_seqs: 1,
            num_heads: HEADS,
            num_kv_heads: 1,
            head_size: HS,
            block_size: BS,
            max_context_len: ctx,
            max_num_blocks_per_seq: blocks,
            scale: 1.0 / (HS as f32).sqrt(),
            softcapping: 1.0,
            q_stride: HEADS * HS,
            kv_block_stride,
            kv_head_stride,
            out: (&out).into(),
            q: (&qb).into(),
            k_cache: (&kb).into(),
            v_cache: (&vb).into(),
            block_tables: (&tb).into(),
            context_lens: (&cb).into(),
            kv_scales: None,
            alibi_slopes: None,
            sinks: None,
        };
        let scratch = p.scratch(&dev).unwrap();
        let paged = time(300, |n| {
            for _ in 0..n {
                paged_attention_with(&dev, &p, Some(&scratch)).unwrap();
            }
            dev.sync().unwrap();
        });

        // Same keys, same math: the outputs agree up to f16 rounding of q and the output.
        let got: Vec<f16> = out.read(HEADS * HS);
        let want: Vec<f32> = native_out.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        let err = got.iter().zip(&want).map(|(a, b)| (a.to_f32() - b).abs()).fold(0f32, f32::max);
        assert!(err < 2e-2, "paged and native flash attention disagree: {err}");

        println!(
            "{:>30} {fa_plain:>7.1}us {fa_split:>7.1}us {paged:>7.1}us {:>8.2}x   (ctx {ctx}, max diff {err:.1e})",
            format!("{rows}, {n_self} + {n_mem}"),
            paged / fa_split
        );
    }
}

fn bytemuck_f16(v: &[f16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
