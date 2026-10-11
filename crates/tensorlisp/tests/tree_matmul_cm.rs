//! Matrix products on tensor cores (cooperative matrices, `k:matmul-cm`) against the CPU and against
//! the scalar tiled `k:matmul`, correctness at awkward shapes and GPU time at the shapes of a
//! TIPSv2 prefill.
//!   TL_VULKAN_PROFILE=1 cargo test --release -p tensorlisp --features native-vulkan --test tree_matmul_cm -- --nocapture
#![cfg(feature = "native-vulkan")]

use std::sync::Arc;

use chez::{Scheme, Value};
use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::{KernelBlob, LaunchDesc, OffsetMode, Range, Space};
use tensorlisp_kernels::vulkan::exec::Executor;
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
            ((s >> 40) as f32) / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn words(u: &[u32]) -> Vec<u8> {
    u.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn align(n: usize) -> u64 {
    (n as u64).div_ceil(256) * 256
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

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            c[j * m + i] = (0..k).map(|l| a[i * k + l] as f64 * b[j * k + l] as f64).sum::<f64>() as f32;
        }
    }
    c
}

#[test]
fn tensor_core_matmul() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels)))").unwrap();
    let vk = Vulkan::open(Options { validate: false }).unwrap();
    if !vk.caps().coopmat {
        eprintln!("no cooperative matrices on this device: skipped");
        return;
    }
    let exec = Executor::new(vk).expect("ggml shader library");
    let e = &*exec.engine;
    let model = e.upload_model(&[]).unwrap();

    let run = |kernel: &Arc<KernelBlob>, tile: [usize; 2], m: usize, n: usize, k: usize, a: &[f32], b: &[f32], iters: usize| -> (Vec<f32>, f64) {
        let (la, lb, lc) = (4 * m * k, 4 * n * k, 4 * m * n);
        let (ob, oc) = (align(la), align(la) + align(lb));
        let step = |off: u64, len: usize| Range::new(Space::Step, off, len as u64);
        let push = [m as u32, n as u32, k as u32, k as u32, k as u32, m as u32, 0, 0, 0, 0, 0, 0, 1, 1, 1];
        let grid = [m.div_ceil(tile[0]) as u32, n.div_ceil(tile[1]) as u32, 1];
        let launch = LaunchDesc { kernel: kernel.clone(), bindings: vec![step(0, la), step(ob, lb), step(oc, lc)], push: words(&push), grid };
        let plan = e.compile(&[launch], oc + align(lc)).unwrap();
        let (da, db) = (bytes(a), bytes(b));
        let writes: Vec<(u64, &[u8])> = vec![(0, &da), (ob, &db)];
        let out = floats(&e.submit(&plan, &model, None, &writes).unwrap().read(oc, lc).unwrap());
        for _ in 0..iters {
            e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap();
        }
        let _ = exec.backend().profile_take();
        for _ in 0..iters {
            e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap();
        }
        let us: f64 = exec.backend().profile_take().iter().map(|p| p.2).sum::<f64>() / 1e3 / iters.max(1) as f64;
        (out, us)
    };

    // correctness, edges included (m, n, k not multiples of the tile)
    for (m, n, k) in [(64usize, 64usize, 64usize), (100, 37, 70), (17, 130, 33), (257, 5, 16)] {
        let (a, b) = (noise(m * k, 1), noise(n * k, 2));
        let want = cpu_matmul(&a, &b, m, n, k);
        for cfg in ["64 64 32 2 2", "32 32 16 1 1", "64 32 32 2 1", "128 64 32 2 2"] {
            let p: Vec<usize> = cfg.split(' ').map(|x| x.parse().unwrap()).collect();
            let kern = blob(&scheme, &format!("(k:matmul-cm 'f32 {cfg})"));
            let (got, _) = run(&kern, [p[0], p[1]], m, n, k, &a, &b, 0);
            let scale = want.iter().fold(0f32, |s, v| s.max(v.abs())).max(1e-6);
            let err = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            println!("  cm {cfg} {m}x{n}x{k}: max abs error {err:.2e} (scale {scale:.2e})");
            assert!(err <= 5e-3 * scale, "cm {cfg} {m}x{n}x{k}: error {err} vs scale {scale}");
        }
    }

    // time: the shapes of TIPSv2 B/14 prefill (batch of 1024 patches against 768-wide weights)
    if std::env::var_os("TL_VULKAN_PROFILE").is_none() {
        return;
    }
    println!("GPU microseconds per launch: scalar tiled k:matmul vs tensor cores");
    for (m, n, k) in [(768usize, 192usize, 768usize), (3072, 192, 768), (768, 192, 3072), (2304, 192, 768), (768, 1024, 768), (3072, 1024, 768)] {
        let (a, b) = (noise(m * k, 1), noise(n * k, 2));
        let scalar = blob(&scheme, "(k:matmul 'f32 64 64 32 4 4)");
        let (_, t_scalar) = run(&scalar, [64, 64], m, n, k, &a, &b, 20);
        let mut line = format!("{m:>5}x{n:>5}x{k:>5} scalar {t_scalar:>8.1}");
        for cfg in ["64 64 16 2 2", "64 32 16 2 1", "32 64 16 1 2", "32 32 16 1 1", "32 32 16 2 2", "32 32 32 2 2", "64 64 32 2 2", "32 64 32 1 2"] {
            let p: Vec<usize> = cfg.split(' ').map(|x| x.parse().unwrap()).collect();
            let kern = blob(&scheme, &format!("(k:matmul-cm 'f32 {cfg})"));
            let (_, t) = run(&kern, [p[0], p[1]], m, n, k, &a, &b, 20);
            line += &format!(" | {cfg}: {t:>6.1}");
        }
        println!("{line}");
    }
}
