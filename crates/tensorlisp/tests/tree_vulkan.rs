//! Tree kernels compiled to SPIR-V by the Scheme compiler ((tl tree), docs/design/tree.md),
//! checked against CPU references under the Vulkan validation layer, then timed against
//! ggml's own precompiled Vulkan shaders (soft_max_f32, matmul_f32_f32) with GPU timestamps.
//!
//!   TL_VULKAN_PROFILE=1 cargo test --release -p tensorlisp --features native-vulkan \
//!       --test tree_vulkan -- --nocapture
//!
//! Needs the precompiled ggml shader library (scripts/build-ggml-vulkan-shaders.sh) for the
//! comparison and a Vulkan device; `spirv-val` is used when it can be found.
#![cfg(feature = "native-vulkan")]

use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use chez::{Scheme, Value};
use tensorlisp_kernels::graph::{Graph, Storage, Tensor, Ty};
use tensorlisp_kernels::plan::Leaves;
use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::{Backend, Engine, KernelBlob, LaunchDesc, OffsetMode, Range, Space};
use tensorlisp_kernels::vulkan::exec::Executor;
use tensorlisp_kernels::vulkan::{validation_errors, Options, Vulkan};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const KERNELS: &str = include_str!("tree_kernels.ss");

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

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn words(u: &[u32]) -> Vec<u8> {
    u.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

fn align(n: usize) -> u64 {
    (n as u64).div_ceil(256) * 256
}

/// Compiles a `(kernel ...)` datum (Scheme source text) to a blob named `tree:<label>`.
fn compile(scheme: &Scheme, label: &str, datum: &str) -> Arc<KernelBlob> {
    let Value::Bytevector(b) = scheme.eval(&format!("(compile-kernel {datum})")).unwrap_or_else(|e| panic!("{label}: {e}")) else {
        panic!("{label}: the compiler did not return a bytevector");
    };
    let w: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    spirv_val(label, &b);
    let n_words = w.len();
    let mut blob = spirv::blob(w, OffsetMode::Raw, vec![]).unwrap();
    blob.label = format!("tree:{label}");
    println!("{label}: {} words, signature {:?}, workgroup {:?}", n_words, blob.sig.bindings, blob.sig.workgroup);
    Arc::new(blob)
}

/// Runs the SDK's validator on the module when it can be found.
fn spirv_val(label: &str, module: &[u8]) {
    let exe = std::env::var_os("VULKAN_SDK")
        .map(|d| std::path::PathBuf::from(d).join("Bin").join("spirv-val"))
        .into_iter()
        .chain(std::fs::read_dir("C:/VulkanSDK").into_iter().flatten().flatten().map(|e| e.path().join("Bin").join("spirv-val")))
        .chain(std::iter::once("spirv-val".into()))
        .find(|p| Command::new(p).arg("--version").output().is_ok());
    let Some(exe) = exe else {
        eprintln!("spirv-val not found, skipping structural validation of {label}");
        return;
    };
    let path = std::env::temp_dir().join(format!("tree-{}-{label}.spv", std::process::id()));
    std::fs::write(&path, module).unwrap();
    let out = Command::new(exe).args(["--target-env", "vulkan1.1"]).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "spirv-val rejected {label}:\n{}", String::from_utf8_lossy(&out.stderr));
}

fn step(off: u64, len: usize) -> Range {
    Range::new(Space::Step, off, len as u64)
}

fn tree_launch(k: &Arc<KernelBlob>, bindings: Vec<Range>, push: &[u32], grid: [u32; 3]) -> LaunchDesc {
    LaunchDesc { kernel: k.clone(), bindings, push: words(push), grid }
}

/// One launch with inputs written at their offsets, `out` read back.
fn run_one(e: &Engine<Vulkan>, launch: LaunchDesc, inputs: &[(u64, Vec<u8>)], step_bytes: u64, out: (u64, usize)) -> Vec<f32> {
    let plan = e.compile(&[launch], step_bytes).unwrap();
    let model = e.upload_model(&[]).unwrap();
    let writes: Vec<(u64, &[u8])> = inputs.iter().map(|(o, d)| (*o, d.as_slice())).collect();
    let t = e.submit(&plan, &model, None, &writes).unwrap();
    floats(&t.read(out.0, out.1).unwrap())
}

#[track_caller]
fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("  {what}: max abs error {err:.2e} (scale {scale:.2e})");
    assert!(err <= tol * scale, "{what}: max abs error {err} > {tol} * {scale}");
}

fn cpu_softmax(x: &[f32], ncols: usize) -> Vec<f32> {
    x.chunks(ncols)
        .flat_map(|r| {
            let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f64> = r.iter().map(|v| ((v - m) as f64).exp()).collect();
            let s: f64 = e.iter().sum();
            e.into_iter().map(move |v| (v / s) as f32)
        })
        .collect()
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

fn cpu_gelu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let x = (a + b) as f64;
            (0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())) as f32
        })
        .collect()
}

struct Kernels {
    add: Arc<KernelBlob>,
    add_gelu: Arc<KernelBlob>,
    softmax_shared: Arc<KernelBlob>,
    softmax_subgroup: Arc<KernelBlob>,
    mm_naive: Arc<KernelBlob>,
    mm_tiled: Arc<KernelBlob>,
    mm_tiled_small: Arc<KernelBlob>,
    /// Staged variants of the tiled matmul: (label, blob, tile BM x BN).
    mm_sweep: Vec<(String, Arc<KernelBlob>, [usize; 2])>,
}

/// (BM, BN, BK, TM, TN): all with 256 threads, differing in tile shape and register block.
const SWEEP: &[[usize; 5]] = &[
    [64, 64, 16, 4, 4],
    [128, 64, 16, 8, 4],
    [64, 128, 16, 4, 8],
    [128, 128, 16, 8, 8],
    [128, 32, 16, 8, 2],
    [32, 128, 16, 2, 8],
    [128, 16, 16, 8, 1],
    [64, 16, 16, 4, 1],
    [64, 64, 32, 4, 4],
    [128, 64, 8, 8, 4],
    [32, 32, 16, 2, 2],
];

fn softmax_launch(k: &Arc<KernelBlob>, nrows: usize, ncols: usize) -> (LaunchDesc, u64) {
    let len = 4 * nrows * ncols;
    let y = align(len);
    (tree_launch(k, vec![step(0, len), step(y, len)], &[ncols as u32], [nrows as u32, 1, 1]), 2 * y)
}

fn matmul_launch(k: &Arc<KernelBlob>, m: usize, n: usize, k_: usize, tile: [usize; 2]) -> (LaunchDesc, [u64; 3], u64) {
    let (la, lb, lc) = (4 * m * k_, 4 * n * k_, 4 * m * n);
    let (ob, oc) = (align(la), align(la) + align(lb));
    let grid = [m.div_ceil(tile[0]) as u32, n.div_ceil(tile[1]) as u32, 1];
    (tree_launch(k, vec![step(0, la), step(ob, lb), step(oc, lc)], &[m as u32, n as u32, k_ as u32], grid), [0, ob, oc], oc + align(lc))
}

fn correctness(e: &Engine<Vulkan>, k: &Kernels) {
    println!("correctness (validation layer on)");
    // Elementwise, including a length that is not a multiple of the workgroup.
    for n in [1000usize, 1, 4096] {
        let (a, b) = (noise(n, 1), noise(n, 2));
        let out = align(4 * n) * 2;
        let launch = |kern: &Arc<KernelBlob>| tree_launch(kern, vec![step(0, 4 * n), step(align(4 * n), 4 * n), step(out, 4 * n)], &[n as u32], [(n as u32).div_ceil(256), 1, 1]);
        let inputs = [(0, bytes(&a)), (align(4 * n), bytes(&b))];
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        close(&format!("add n={n}"), &run_one(e, launch(&k.add), &inputs, out + align(4 * n), (out, 4 * n)), &sum, 1e-6);
        close(&format!("add+gelu n={n}"), &run_one(e, launch(&k.add_gelu), &inputs, out + align(4 * n), (out, 4 * n)), &cpu_gelu_add(&a, &b), 1e-5);
    }
    for (nrows, ncols) in [(37usize, 1000usize), (5, 17), (3, 4096), (1, 262144)] {
        let x = noise(nrows * ncols, 3).into_iter().map(|v| v * 8.0).collect::<Vec<_>>();
        let want = cpu_softmax(&x, ncols);
        for (name, kern) in [("shared", &k.softmax_shared), ("subgroup", &k.softmax_subgroup)] {
            let (l, bytes_) = softmax_launch(kern, nrows, ncols);
            let got = run_one(e, l, &[(0, bytes(&x))], bytes_, (align(4 * nrows * ncols), 4 * nrows * ncols));
            close(&format!("softmax {name} {nrows}x{ncols}"), &got, &want, 1e-5);
        }
    }
    for (m, n, kk) in [(100usize, 70usize, 90usize), (64, 64, 16), (200, 33, 257), (1, 1, 1), (17, 129, 5)] {
        let (a, b) = (noise(m * kk, 4), noise(n * kk, 5));
        let want = cpu_matmul(&a, &b, m, n, kk);
        for (name, kern, tile) in [("naive", &k.mm_naive, [16, 16]), ("tiled 64x64", &k.mm_tiled, [64, 64]), ("tiled 32x32", &k.mm_tiled_small, [32, 32])] {
            let (l, off, bytes_) = matmul_launch(kern, m, n, kk, tile);
            let got = run_one(e, l, &[(off[0], bytes(&a)), (off[1], bytes(&b))], bytes_, (off[2], 4 * m * n));
            close(&format!("matmul {name} {m}x{n}x{kk}"), &got, &want, 2e-4);
        }
        for (label, kern, tile) in &k.mm_sweep {
            let (l, off, bytes_) = matmul_launch(kern, m, n, kk, *tile);
            let got = run_one(e, l, &[(off[0], bytes(&a)), (off[1], bytes(&b))], bytes_, (off[2], 4 * m * n));
            close(&format!("matmul {label} {m}x{n}x{kk}"), &got, &want, 2e-4);
        }
    }
}

/// A one-node ggml graph over f32 operands.
fn ggml_graph(op: &str, ne: &[[i64; 4]], op_params: [i32; 16], out_ne: [i64; 4], weight_first: bool) -> Graph {
    let nb = |ne: [i64; 4]| {
        let mut nb = [4u64; 4];
        for i in 1..4 {
            nb[i] = nb[i - 1] * ne[i - 1] as u64;
        }
        nb
    };
    let mk = |ne: [i64; 4], op: &str, src: Vec<Option<usize>>, storage| Tensor {
        ty: Ty::F32,
        ne,
        nb: nb(ne),
        op: op.into(),
        op_params,
        src,
        view_src: None,
        view_offs: 0,
        storage,
        name: String::new(),
    };
    let mut g = Graph::default();
    for (i, &n) in ne.iter().enumerate() {
        let storage = if weight_first && i == 0 { Storage::Weight(0) } else { Storage::Input(i - weight_first as usize) };
        g.tensors.push(mk(n, "NONE", vec![], storage));
    }
    let id = g.tensors.len();
    let mut node = mk(out_ne, op, (0..ne.len()).map(Some).collect(), Storage::Temp);
    node.op_params = op_params;
    g.tensors.push(node);
    g.nodes.push(id);
    g.keep.push(id);
    g
}

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); // warm up (pipeline creation)
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// With TL_VULKAN_PROFILE=1: the GPU time per kernel since the last call.
fn gpu_table(exec: &Executor) {
    if std::env::var_os("TL_VULKAN_PROFILE").is_some() {
        let r = exec.backend().profile_report();
        for line in r.lines().filter(|l| !l.starts_with("total") && !l.starts_with("kernel")) {
            println!("    GPU  {line}");
        }
        exec.backend().profile_clear();
    }
}

fn bench(scheme_exec: &Executor, k: &Kernels) {
    let e = &*scheme_exec.engine;
    let model0 = e.upload_model(&[]).unwrap();
    println!("\nbenchmark (wall ms per submit+wait; GPU time per kernel is in the profile table below)");

    for (nrows, ncols) in [(4096usize, 1024usize), (512, 8192), (64, 16384), (2048, 128)] {
        let x = bytes(&noise(nrows * ncols, 3));
        // ggml: SOFT_MAX over [ncols, nrows], scale 1, no mask.
        let mut params = [0i32; 16];
        params[0] = 1.0f32.to_bits() as i32;
        let g = ggml_graph("SOFT_MAX", &[[ncols as i64, nrows as i64, 1, 1]], params, [ncols as i64, nrows as i64, 1, 1], false);
        let prog = scheme_exec.compile(g, &Leaves { weights: &[], states: &[] }).unwrap();
        prog.set_input(0, &x).unwrap();
        let ggml_ms = time(50, || prog.run(&model0, None).unwrap());
        let mut rows = vec![("ggml soft_max_f32".to_string(), ggml_ms)];
        for (name, kern) in [("tree shared", &k.softmax_shared), ("tree subgroup", &k.softmax_subgroup)] {
            let (l, bytes_) = softmax_launch(kern, nrows, ncols);
            let plan = e.compile(&[l], bytes_).unwrap();
            let ms = time(50, || {
                e.submit(&plan, &model0, None, &[(0, &x)]).unwrap().wait().unwrap();
            });
            rows.push((name.to_string(), ms));
        }
        println!("softmax {nrows} rows x {ncols}: {}", rows.iter().map(|(n, ms)| format!("{n} {ms:.3}")).collect::<Vec<_>>().join(" | "));
        gpu_table(scheme_exec);
    }

    println!("
matmul, GPU microseconds per launch (timestamps), ggml's matmul_f32_f32 vs Tree variants");
    for (m, n, kk) in [(1024usize, 1024usize, 1024usize), (2048, 512, 2048), (1536, 256, 1536), (1536, 64, 1536), (12288, 16, 1536), (4096, 32, 4096), (3072, 192, 768), (768, 192, 768), (768, 192, 3072), (2304, 192, 768)] {
        let (a, b) = (bytes(&noise(m * kk, 4)), bytes(&noise(n * kk, 5)));
        let g = ggml_graph("MUL_MAT", &[[kk as i64, m as i64, 1, 1], [kk as i64, n as i64, 1, 1]], [0; 16], [m as i64, n as i64, 1, 1], true);
        let prog = scheme_exec.compile(g, &Leaves { weights: &[(0, 0)], states: &[] }).unwrap();
        let model = e.upload_model(&[&a]).unwrap();
        prog.set_input(0, &b).unwrap();
        let iters = 20;
        let per_launch = |prof: Vec<(String, u64, f64)>| prof.iter().map(|p| p.2).sum::<f64>() / 1e3 / iters as f64;
        for _ in 0..iters + 1 {
            prog.run(&model, None).unwrap();
        }
        let _ = scheme_exec.backend().profile_take();
        for _ in 0..iters {
            prog.run(&model, None).unwrap();
        }
        drop(prog);
        let ggml_us = per_launch(scheme_exec.backend().profile_take());

        let mut results: Vec<(String, f64)> = Vec::new();
        let mut run_tree = |label: &str, kern: &Arc<KernelBlob>, tile: [usize; 2]| {
            let (l, off, bytes_) = matmul_launch(kern, m, n, kk, tile);
            let plan = e.compile(&[l], bytes_).unwrap();
            for _ in 0..2 {
                e.submit(&plan, &model0, None, &[(off[0], &a), (off[1], &b)]).unwrap().wait().unwrap();
            }
            let _ = scheme_exec.backend().profile_take();
            for _ in 0..iters {
                e.submit(&plan, &model0, None, &[(off[0], &a), (off[1], &b)]).unwrap().wait().unwrap();
            }
            results.push((label.to_string(), per_launch(scheme_exec.backend().profile_take())));
        };
        run_tree("naive", &k.mm_naive, [16, 16]);
        for (label, kern, tile) in &k.mm_sweep {
            run_tree(label, kern, *tile);
        }
        let flops = 2.0 * (m * n * kk) as f64;
        let tflops = |us: f64| flops / us / 1e6;
        let naive = results[0].1;
        let mut tiled: Vec<&(String, f64)> = results[1..].iter().collect();
        tiled.sort_by(|x, y| x.1.total_cmp(&y.1));
        println!(
            "m={m} n={n} k={kk}: ggml {ggml_us:.0} us ({:.1} TFLOP/s) | tree best {} {:.0} us ({:.1} TFLOP/s, {:.2}x ggml) | 64x64x16 {:.0} us | naive {naive:.0} us",
            tflops(ggml_us),
            tiled[0].0,
            tiled[0].1,
            tflops(tiled[0].1),
            tiled[0].1 / ggml_us,
            results.iter().find(|r| r.0.ends_with("64x64x16_4x4")).map_or(0.0, |r| r.1),
        );
    }
}

#[test]
fn tree_kernels_match_references_and_ggml() {
    let scheme = Scheme::new().unwrap();
    scheme.eval(SPIRV_LIB).unwrap();
    scheme.eval(TREE_LIB).unwrap();
    scheme.eval(KERNELS).unwrap();

    let k = Kernels {
        add: compile(&scheme, "add", "add-kernel"),
        add_gelu: compile(&scheme, "add_gelu", "add-gelu-kernel"),
        softmax_shared: compile(&scheme, "softmax_shared", "(softmax-shared-kernel 256)"),
        softmax_subgroup: compile(&scheme, "softmax_subgroup", "(softmax-subgroup-kernel 256)"),
        mm_naive: compile(&scheme, "matmul_naive", "matmul-naive-kernel"),
        mm_tiled: compile(&scheme, "matmul_tiled_64", "(matmul-tiled-kernel 64 64 16 4 4)"),
        mm_tiled_small: compile(&scheme, "matmul_tiled_32", "(matmul-tiled-kernel 32 32 16 2 2)"),
        mm_sweep: SWEEP
            .iter()
            .map(|c| {
                let label = format!("{}x{}x{}_{}x{}", c[0], c[1], c[2], c[3], c[4]);
                let blob = compile(&scheme, &format!("mm_{label}"), &format!("(matmul-tiled-kernel {} {} {} {} {})", c[0], c[1], c[2], c[3], c[4]));
                (label, blob, [c[0], c[1]])
            })
            .collect(),
    };

    // Correctness on a validated device.
    match Vulkan::open(Options { validate: true }) {
        Ok(vk) => {
            println!("{} | subgroup {} | validation {}", vk.name(), vk.subgroup_size(), vk.validated());
            let e = Engine::new(Arc::new(vk));
            correctness(&e, &k);
            assert_eq!(validation_errors(), 0, "the Vulkan validation layer reported errors");
        }
        Err(tensorlisp_kernels::Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Vulkan device, skipping");
            return;
        }
        Err(err) => panic!("{err}"),
    }

    // Timing on a plain device, next to ggml's shaders.
    let vk = Vulkan::open(Options { validate: false }).unwrap();
    let exec = Executor::new(vk).expect("ggml shader library (scripts/build-ggml-vulkan-shaders.sh)");
    bench(&exec, &k);
    println!("\n{}", exec.backend().profile_report());
}
