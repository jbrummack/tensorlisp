//! The split-KV flash attention kernels (flash_attn_part + flash_attn_combine) against a CPU
//! reference, one query over a cache of various lengths and mask shapes.
#![cfg(feature = "native-vulkan")]

use std::sync::Arc;

use chez::{Scheme, Value};
use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::{Engine, KernelBlob, LaunchDesc, OffsetMode, Range, Space};
use tensorlisp_kernels::vulkan::{Options, Vulkan};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const OPT_LIB: &str = include_str!("../src/scheme/stdlib/tree_opt.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn blob(scheme: &Scheme, expr: &str) -> Arc<KernelBlob> {
    let Value::Bytevector(b) = scheme.eval(&if std::env::var("TL_TREE_PASSES").is_ok_and(|v| v == "none") { format!("(eval (quote (compile-kernel {expr})) $env)") } else { format!("(eval (quote (compile-kernel (optimize {expr}))) $env)") }).unwrap_or_else(|e| panic!("{expr}: {e}")) else {
        panic!("{expr}: no bytevector");
    };
    let w: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    Arc::new(spirv::blob(w, OffsetMode::Prefix, vec![]).unwrap())
}

fn words(u: &[u32]) -> Vec<u8> {
    let mut v = vec![0u8; 0];
    for x in u {
        v.extend(x.to_ne_bytes());
    }
    v
}

fn align(n: usize) -> u64 {
    (n as u64).div_ceil(256) * 256
}

/// One query, `nh` heads over 2 K/V heads, head size 64, f32 K and V, an f16 mask with `valid` keys open.
fn check(scheme: &Scheme, e: &Engine<Vulkan>, kv: usize, valid: usize, chunk: usize) {
    let (nh, nkv, hs) = (4usize, 2usize, 64usize);
    let nchunks = kv.div_ceil(chunk);
    let q = noise(nh * hs, 1);
    let k = noise(nkv * kv * hs, 2);
    let v = noise(nkv * kv * hs, 3);
    let mask: Vec<u16> = (0..kv).map(|j| if j < valid { 0x0000 } else { 0xFC00 }).collect();
    let scale = 0.125f32;

    // CPU reference
    let mut want = vec![0f32; nh * hs];
    for h in 0..nh {
        let kvh = h / (nh / nkv);
        let scores: Vec<f64> = (0..valid)
            .map(|j| (0..hs).map(|c| q[h * hs + c] as f64 * k[(kvh * kv + j) * hs + c] as f64).sum::<f64>() * scale as f64)
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = ps.iter().sum();
        for c in 0..hs {
            want[h * hs + c] = (0..valid).map(|j| ps[j] * v[(kvh * kv + j) * hs + c] as f64).sum::<f64>() as f32 / z as f32;
        }
    }

    let sizes = [4 * q.len(), 4 * k.len(), 4 * v.len(), 2 * kv, 4 * nh * nchunks * (hs + 2), 4 * nh * hs];
    let mut offs = [0u64; 6];
    let mut at = 0u64;
    for (i, s) in sizes.iter().enumerate() {
        offs[i] = at;
        at += align(*s);
    }
    let r = |i: usize| Range::new(Space::Step, offs[i], sizes[i] as u64);
    let nomask = std::env::var_os("TL_NOMASK").is_some();
    let part = blob(scheme, &format!("(k:flash-attn-part 'f32 'f32 {} #f {chunk} {hs} {hs})", if nomask { "#f" } else { "#t" }));
    let comb = blob(scheme, "(k:flash-attn-combine)");
    // push: nq nh hsk hsv kv gqa nchunks q1 q2 q3 k1 k2 k3 v1 v2 v3 m1 m2 m3 mne2 mne3 scale cap
    let p1 = words(&[1, nh as u32, hs as u32, hs as u32, kv as u32, (nh / nkv) as u32, nchunks as u32, (nh * hs) as u32, hs as u32, (nh * hs) as u32, hs as u32, (kv * hs) as u32, (nkv * kv * hs) as u32, hs as u32, (kv * hs) as u32, (nkv * kv * hs) as u32, kv as u32, 0, 0, 1, 1, scale.to_bits(), 0]);
    // push: nq nh hsv nchunks d1 d2 d3
    let p2 = words(&[1, nh as u32, hs as u32, nchunks as u32, hs as u32, (nh * hs) as u32, (nh * hs) as u32]);
    let l1 = LaunchDesc { kernel: part, bindings: if nomask { vec![r(0), r(1), r(2), r(4)] } else { vec![r(0), r(1), r(2), r(3), r(4)] }, push: p1, grid: [nchunks as u32, nh as u32, 1] };
    let l2 = LaunchDesc { kernel: comb, bindings: vec![r(4), r(5)], push: p2, grid: [nh as u32, 1, 1] };
    let plan = e.compile(&[l1, l2], at).unwrap();
    let model = e.upload_model(&[]).unwrap();
    let mask_bytes: Vec<u8> = mask.iter().flat_map(|m| m.to_ne_bytes()).collect();
    let writes: Vec<(u64, &[u8])> = vec![(offs[0], &bytes(&q)), (offs[1], &bytes(&k)), (offs[2], &bytes(&v)), (offs[3], &mask_bytes)]
        .into_iter()
        .map(|(o, d)| (o, Box::leak(d.to_vec().into_boxed_slice()) as &[u8]))
        .collect();
    let t = e.submit(&plan, &model, None, &writes).unwrap();
    let got: Vec<f32> = t.read(offs[5], sizes[5]).unwrap().chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
    if std::env::var_os("TL_DEBUG_PART").is_some() {
        let part: Vec<f32> = t.read(offs[4], sizes[4]).unwrap().chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        for h in 0..nh {
            for c in 0..nchunks {
                let base = (h * nchunks + c) * (hs + 2);
                println!("head {h} chunk {c}: m {} l {} o[0..3] {:?} o[32..35] {:?}", part[base], part[base + 1], &part[base + 2..base + 5], &part[base + 34..base + 37]);
            }
        }
        for h in 0..nh {
            let e = (0..hs).map(|c| (got[h * hs + c] - want[h * hs + c]).abs()).fold(0f32, f32::max);
            let bad: Vec<usize> = (0..hs).filter(|&c| (got[h * hs + c] - want[h * hs + c]).abs() > 1e-4).collect();
            println!("head {h} bad elements {} first {:?}; got[30..36] {:?} want {:?}", bad.len(), bad.first(), &got[h * hs + 30..h * hs + 36], &want[h * hs + 30..h * hs + 36]);
            println!("head {h}: error {e:.2e}, got {:?} want {:?}", &got[h * hs..h * hs + 2], &want[h * hs..h * hs + 2]);
        }
        println!("got {:?}
want {:?}", &got[..4], &want[..4]);
        println!("got h1 {:?}
want h1 {:?}", &got[hs..hs + 4], &want[hs..hs + 4]);
        let kvh0 = 0;
        let scores: Vec<f64> = (0..valid.min(chunk)).map(|j| (0..hs).map(|c| q[c] as f64 * k[(kvh0 * kv + j) * hs + c] as f64).sum::<f64>() * scale as f64).collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let l: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        println!("reference head 0 chunk 0: m {m} l {l}");
    }
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("kv {kv} valid {valid} chunk {chunk}: max abs error {err:.2e}");
    assert!(err < 1e-4, "kv {kv} valid {valid} chunk {chunk}: error {err}");
}

#[test]
fn split_attention_matches_the_cpu() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels)))").unwrap();
    if std::env::var_os("TL_PRINT_KERNEL").is_some() {
        let text = scheme
            .eval("(eval '(call-with-string-output-port (lambda (p) (pretty-print (k:flash-attn-part 'f32 'f32 #f #f 128 64 64) p))) (environment '(chezscheme) '(tl tree kernels)))")
            .unwrap();
        println!("{text:?}");
    }
    let vk = Vulkan::open(Options { validate: false }).unwrap();
    let e = Engine::new(Arc::new(vk));
    for (kv, valid, chunk) in [(1, 1, 128), (2, 2, 128), (128, 128, 128), (300, 300, 128), (300, 1, 128), (1000, 750, 128), (512, 129, 128), (777, 777, 64)] {
        check(&scheme, &e, kv, valid, chunk);
    }
}

fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = (b >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7bff
    } else {
        sign | ((exp as u16) << 10) | man as u16
    }
}

/// GPU time of the two kernels at Gemma's decode shapes (1 query, 8 heads over 1 K/V head of 256,
/// 2048 cached keys of which `valid` are open).
///   TL_VULKAN_PROFILE=1 cargo test --release -p tensorlisp --features native-vulkan --test tree_attention -- --ignored --nocapture
#[test]
#[ignore]
fn split_attention_speed() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels)))").unwrap();
    let vk = Arc::new(Vulkan::open(Options { validate: false }).unwrap());
    let e = Engine::new(vk.clone());
    let (nh, nkv, hs, kv, chunk) = (8usize, 1usize, 256usize, 2048usize, 128usize);
    let nchunks = kv.div_ceil(chunk);
    let q: Vec<u8> = noise(nh * hs, 1).iter().flat_map(|v| v.to_ne_bytes()).collect();
    let kb: Vec<u8> = noise(nkv * kv * hs, 2).iter().flat_map(|v| f16_bits(*v).to_ne_bytes()).collect();
    let vb: Vec<u8> = noise(nkv * kv * hs, 3).iter().flat_map(|v| f16_bits(*v).to_ne_bytes()).collect();
    let sizes = [q.len(), kb.len(), vb.len(), 2 * kv, 4 * nh * nchunks * (hs + 2), 4 * nh * hs];
    let mut offs = [0u64; 6];
    let mut at = 0u64;
    for (i, s) in sizes.iter().enumerate() {
        offs[i] = at;
        at += align(*s);
    }
    let r = |i: usize| Range::new(Space::Step, offs[i], sizes[i] as u64);
    let part = blob(&scheme, &format!("(k:flash-attn-part 'f16 'f16 #t #f {chunk} {hs} {hs})"));
    let comb = blob(&scheme, "(k:flash-attn-combine)");
    let model = e.upload_model(&[]).unwrap();
    println!("kernel GPU microseconds (per launch)");
    for valid in [11usize, 601, 2000] {
        let mask: Vec<u8> = (0..kv).flat_map(|j| (if j < valid { 0x0000u16 } else { 0xFC00 }).to_ne_bytes()).collect();
        let p1 = words(&[1, nh as u32, hs as u32, hs as u32, kv as u32, (nh / nkv) as u32, nchunks as u32, (nh * hs) as u32, hs as u32, (nh * hs) as u32, hs as u32, (kv * hs) as u32, (nkv * kv * hs) as u32, hs as u32, (kv * hs) as u32, (nkv * kv * hs) as u32, kv as u32, 0, 0, 1, 1, 0.0625f32.to_bits(), 0]);
        let p2 = words(&[1, nh as u32, hs as u32, nchunks as u32, hs as u32, (nh * hs) as u32, (nh * hs) as u32]);
        let l1 = LaunchDesc { kernel: part.clone(), bindings: vec![r(0), r(1), r(2), r(3), r(4)], push: p1, grid: [nchunks as u32, nh as u32, 1] };
        let l2 = LaunchDesc { kernel: comb.clone(), bindings: vec![r(4), r(5)], push: p2, grid: [nh as u32, 1, 1] };
        let plan = e.compile(&[l1, l2], at).unwrap();
        let writes: Vec<(u64, &[u8])> = vec![(offs[0], &q), (offs[1], &kb), (offs[2], &vb), (offs[3], &mask)];
        for _ in 0..20 {
            e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap();
        }
        let _ = vk.profile_take();
        for _ in 0..100 {
            e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap();
        }
        let rows = vk.profile_take();
        let line: Vec<String> = rows.iter().map(|(l, n, ns)| format!("{} {:.1} us", &l[..l.len().min(30)], ns / *n as f64 / 1e3)).collect();
        println!("valid {valid}: {}", line.join(" | "));
    }
}
