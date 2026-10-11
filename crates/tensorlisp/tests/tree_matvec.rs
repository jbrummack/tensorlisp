//! q8_0 mat-vec on Vulkan at the shapes of a Gemma decode step: ggml's shader against Tree's
//! (plain and dp4a), GPU time per launch and weight bandwidth.
//!   TL_VULKAN_PROFILE=1 cargo test --release -p tensorlisp --features native-vulkan --test tree_matvec -- --ignored --nocapture
#![cfg(feature = "native-vulkan")]

use std::sync::Arc;

use chez::{Scheme, Value};
use tensorlisp_kernels::graph::{Graph, Storage, Tensor, Ty};
use tensorlisp_kernels::plan::Leaves;
use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::{Engine, KernelBlob, LaunchDesc, OffsetMode, Range, Space};
use tensorlisp_kernels::vulkan::exec::Executor;
use tensorlisp_kernels::vulkan::{Options, Vulkan};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const OPT_LIB: &str = include_str!("../src/scheme/stdlib/tree_opt.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn q8_weights(m: usize, k: usize) -> Vec<u8> {
    let mut r = rng(1);
    let mut out = Vec::with_capacity(m * k / 32 * 34);
    for _ in 0..m * k / 32 {
        out.extend((0x2000u16 | (r() as u16 & 0x3ff)).to_le_bytes());
        out.extend((0..32).map(|_| r() as u8));
    }
    out
}

fn floats(n: usize, seed: u64) -> Vec<u8> {
    let mut r = rng(seed);
    (0..n).flat_map(|_| (((r() >> 40) as f32) / (1u64 << 23) as f32 - 1.0).to_ne_bytes()).collect()
}

fn blob(scheme: &Scheme, expr: &str) -> Arc<KernelBlob> {
    let Value::Bytevector(b) = scheme.eval(&format!("(eval (quote (compile-kernel (optimize {expr}))) $env)")).unwrap_or_else(|e| panic!("{expr}: {e}")) else {
        panic!("{expr}");
    };
    let w: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut blob = spirv::blob(w, OffsetMode::Prefix, vec![]).unwrap();
    blob.label = format!("tree:{expr}");
    Arc::new(blob)
}

fn words(u: &[u32]) -> Vec<u8> {
    u.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn align(n: usize) -> u64 {
    (n as u64).div_ceil(256) * 256
}

fn graph(m: usize, k: usize) -> Graph {
    let nbk = (34 * k / 32) as u64;
    let mk = |ty: Ty, ne: [i64; 4], nb: [u64; 4], op: &str, src: Vec<Option<usize>>, storage| Tensor {
        ty,
        ne,
        nb,
        op: op.into(),
        op_params: [0; 16],
        src,
        view_src: None,
        view_offs: 0,
        storage,
        name: String::new(),
    };
    let mut g = Graph::default();
    g.tensors.push(mk(Ty::Q8_0, [k as i64, m as i64, 1, 1], [34, nbk, nbk * m as u64, nbk * m as u64], "NONE", vec![], Storage::Weight(0)));
    let kb = 4 * k as u64;
    g.tensors.push(mk(Ty::F32, [k as i64, 1, 1, 1], [4, kb, kb, kb], "NONE", vec![], Storage::Input(0)));
    let mb = 4 * m as u64;
    g.tensors.push(mk(Ty::F32, [m as i64, 1, 1, 1], [4, mb, mb, mb], "MUL_MAT", vec![Some(0), Some(1)], Storage::Temp));
    g.nodes.push(2);
    g.keep.push(2);
    g
}

#[test]
#[ignore]
fn q8_matvec_shapes() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels)))").unwrap();
    let vk = Vulkan::open(Options { validate: false }).unwrap();
    let exec = Executor::new(vk).expect("ggml shader library");
    let e = &*exec.engine;
    let sg = 32u32;
    println!("GPU microseconds per launch");
    for (m, k) in [(12288usize, 1536usize), (1536, 12288), (6144, 1536), (1536, 6144), (2048, 1536), (1536, 2048), (256, 1536), (1536, 256), (262144, 1536)] {
        let w = q8_weights(m, k);
        let x = floats(k, 2);
        let mb = w.len() as f64 / 1e6;
        let model = e.upload_model(&[&w]).unwrap();
        let iters = 40usize;
        let busy = |label_part: &str, prof: Vec<(String, u64, f64)>| prof.iter().filter(|p| p.0.contains(label_part)).map(|p| p.2).sum::<f64>() / 1e3 / iters as f64;

        // ggml
        let prog = exec.compile(graph(m, k), &Leaves { weights: &[(0, 0)], states: &[] }).unwrap();
        prog.set_input(0, &x).unwrap();
        for _ in 0..iters + 1 {
            prog.run(&model, None).unwrap();
        }
        let _ = exec.backend().profile_take();
        for _ in 0..iters {
            prog.run(&model, None).unwrap();
        }
        let ggml = busy("mul_mat", exec.backend().profile_take());
        drop(prog);

        // tree: x at 0, xq, xd, out
        let (xo, xqo, xdo, oo) = (0u64, align(4 * k), align(4 * k) + align(k), align(4 * k) + align(k) + align(k / 8));
        let total = oo + align(4 * m);
        let wr = Range::new(Space::Model(0), 0, w.len() as u64);
        let rows = 4u32;
        let wgrid = |rows: u32| [(m as u32).div_ceil((128 / sg) * rows), 1, 1];
        let quant = blob(&scheme, "(k:quantize-q8)");
        let rx = Range::new(Space::Step, xo, 4 * k as u64);
        let rxq = Range::new(Space::Step, xqo, k as u64);
        let rxd = Range::new(Space::Step, xdo, (k / 8) as u64);
        let ro = Range::new(Space::Step, oo, 4 * m as u64);
        let n = (k / 4) as u32;
        let l_quant = LaunchDesc { kernel: quant, bindings: vec![rx, rxq, rxd], push: words(&[n, k as u32, k as u32]), grid: [n.div_ceil(256), 1, 1] };
        let mut time = |launches: Vec<LaunchDesc>, part: &str| {
            let plan = e.compile(&launches, total).unwrap();
            for _ in 0..iters + 1 {
                e.submit(&plan, &model, None, &[(xo, &x)]).unwrap().wait().unwrap();
            }
            let _ = exec.backend().profile_take();
            for _ in 0..iters {
                e.submit(&plan, &model, None, &[(xo, &x)]).unwrap().wait().unwrap();
            }
            busy(part, exec.backend().profile_take())
        };
        let mut plain_t = vec![];
        let mut dot_t = vec![];
        for r in [1u32, 2, 4] {
            let plain = blob(&scheme, &format!("(k:matvec 'q8_0 1 {r})"));
            let u = std::env::var("TL_U").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
            let dot = blob(&scheme, &format!("(k:matvec-q8 1 {r} {u})"));
            let l_plain = LaunchDesc { kernel: plain, bindings: vec![wr, rx, ro], push: words(&[m as u32, k as u32, (k / 32) as u32, k as u32, m as u32]), grid: wgrid(r) };
            let l_dot = LaunchDesc { kernel: dot, bindings: vec![wr, rxq, rxd, ro], push: words(&[m as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32]), grid: wgrid(r) };
            plain_t.push(time(vec![l_plain], "k:matvec 'q8_0"));
            dot_t.push(time(vec![l_quant.clone(), l_dot], "k:matvec-q8"));
        }
        let _ = rows;
        let t_quant = time(vec![l_quant], "k:quantize-q8");
        let gbs = |us: f64| mb / us * 1e3;
        println!(
            "{m:>7} {k:>6} {mb:>7.1} | ggml {ggml:>6.1} ({:>3.0}G) | plain R1/2/4 {:>6.1} {:>6.1} {:>6.1} | dp4a R1/2/4 {:>6.1} {:>6.1} {:>6.1} ({:>3.0}G best) | quant {t_quant:.1}",
            gbs(ggml),
            plain_t[0], plain_t[1], plain_t[2],
            dot_t[0], dot_t[1], dot_t[2],
            gbs(dot_t.iter().cloned().fold(f64::MAX, f64::min) - t_quant)
        );
    }
}
