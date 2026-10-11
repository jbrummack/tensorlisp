//! Tree kernels compiled to PTX by the Scheme compiler ((tl tree ptx), docs/design/tree.md),
//! JIT-loaded by the CUDA driver and run through the scheduler's CUDA backend: checked against
//! CPU references, every generator of `(tl tree kernels)` loaded, and timed.
//!
//!   cargo test --release -p tensorlisp --features native-cuda --test tree_ptx -- --nocapture
//!
//! Needs an NVIDIA driver (no CUDA toolkit: PTX goes straight to the driver's JIT).
#![cfg(feature = "native-cuda")]

use std::sync::Arc;
use std::time::Instant;

use chez::{Scheme, Value};
use tensorlisp_kernels::cuda::backend::Cuda;
use tensorlisp_kernels::sched::{Access, Code, Engine, KernelBlob, LaunchDesc, OffsetMode, Range, Signature, Space, Target};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");
const PTX_LIB: &str = include_str!("../src/scheme/stdlib/tree_ptx.ss");
const TEST_KERNELS: &str = include_str!("tree_kernels.ss");

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

fn step(off: u64, len: usize) -> Range {
    Range::new(Space::Step, off, len as u64)
}

fn launch(k: &Arc<KernelBlob>, bindings: Vec<Range>, push: &[u32], grid: [u32; 3]) -> LaunchDesc {
    LaunchDesc { kernel: k.clone(), bindings, push: words(push), grid }
}

fn run_one(e: &Engine<Cuda>, l: LaunchDesc, inputs: &[(u64, Vec<u8>)], step_bytes: u64, out: (u64, usize)) -> Vec<f32> {
    let plan = e.compile(&[l], step_bytes).unwrap();
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

/// Milliseconds per launch of `l`: the wall time of a plan of `2 * reps` launches minus that of
/// `reps` launches, divided by `reps`, so the upload of the inputs (the same in both) cancels.
fn time_launch(e: &Engine<Cuda>, l: &LaunchDesc, inputs: &[(u64, Vec<u8>)], step_bytes: u64, reps: usize) -> f64 {
    let model = e.upload_model(&[]).unwrap();
    let writes: Vec<(u64, &[u8])> = inputs.iter().map(|(o, d)| (*o, d.as_slice())).collect();
    let run = |n: usize| {
        let launches: Vec<LaunchDesc> = (0..n).map(|_| l.clone()).collect();
        let plan = e.compile(&launches, step_bytes).unwrap();
        e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap(); // warm up
        let t = Instant::now();
        e.submit(&plan, &model, None, &writes).unwrap().wait().unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    (run(2 * reps) - run(reps)) / reps as f64
}

#[test]
fn tree_kernels_run_as_ptx() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(import (tl tree ptx))").unwrap();
    scheme.eval(TEST_KERNELS).unwrap();

    let cuda = match Cuda::system_default() {
        Ok(c) => c,
        Err(tensorlisp_kernels::Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => {
            eprintln!("no CUDA device, skipping");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let e = Engine::new(Arc::new(cuda));

    // The test kernels live in the interaction environment, where `compile-kernel-ptx` is imported.
    let ptx = |label: &str, expr: &str| compile_ptx_top(&scheme, label, expr);
    let add = ptx("add", "add-kernel");
    let add_gelu = ptx("add_gelu", "add-gelu-kernel");
    let sm_shared = ptx("softmax_shared", "(softmax-shared-kernel 256)");
    let sm_subgroup = ptx("softmax_subgroup", "(softmax-subgroup-kernel 256)");
    let mm_naive = ptx("matmul_naive", "matmul-naive-kernel");
    let mm_tiled = ptx("matmul_tiled_64", "(matmul-tiled-kernel 64 64 16 4 4)");
    let mm_wide = ptx("matmul_tiled_128x64", "(matmul-tiled-kernel 128 64 16 8 4)");
    let mm_big = ptx("matmul_tiled_128x128", "(matmul-tiled-kernel 128 128 16 8 8)");
    println!("compiled and loaded 8 PTX kernels");

    println!("correctness");
    for n in [1000usize, 1, 4096] {
        let (a, b) = (noise(n, 1), noise(n, 2));
        let out = align(4 * n) * 2;
        let l = |k: &Arc<KernelBlob>| launch(k, vec![step(0, 4 * n), step(align(4 * n), 4 * n), step(out, 4 * n)], &[n as u32], [(n as u32).div_ceil(256), 1, 1]);
        let inputs = [(0, bytes(&a)), (align(4 * n), bytes(&b))];
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        close(&format!("add n={n}"), &run_one(&e, l(&add), &inputs, out + align(4 * n), (out, 4 * n)), &sum, 1e-6);
        close(&format!("add+gelu n={n}"), &run_one(&e, l(&add_gelu), &inputs, out + align(4 * n), (out, 4 * n)), &cpu_gelu_add(&a, &b), 1e-5);
    }
    for (nrows, ncols) in [(37usize, 1000usize), (5, 17), (3, 4096), (1, 262144)] {
        let x: Vec<f32> = noise(nrows * ncols, 3).into_iter().map(|v| v * 8.0).collect();
        let want = cpu_softmax(&x, ncols);
        for (name, k) in [("shared", &sm_shared), ("subgroup", &sm_subgroup)] {
            let len = 4 * nrows * ncols;
            let y = align(len);
            let l = launch(k, vec![step(0, len), step(y, len)], &[ncols as u32], [nrows as u32, 1, 1]);
            let got = run_one(&e, l, &[(0, bytes(&x))], 2 * y, (y, len));
            close(&format!("softmax {name} {nrows}x{ncols}"), &got, &want, 1e-5);
        }
    }
    for (m, n, kk) in [(100usize, 70usize, 90usize), (64, 64, 16), (200, 33, 257), (1, 1, 1), (17, 129, 5)] {
        let (a, b) = (noise(m * kk, 4), noise(n * kk, 5));
        let want = cpu_matmul(&a, &b, m, n, kk);
        for (name, k, tile) in [("naive", &mm_naive, [16, 16]), ("tiled 64x64", &mm_tiled, [64, 64]), ("tiled 128x64", &mm_wide, [128, 64]), ("tiled 128x128", &mm_big, [128, 128])] {
            let (la, lb, lc) = (4 * m * kk, 4 * n * kk, 4 * m * n);
            let (ob, oc) = (align(la), align(la) + align(lb));
            let grid = [m.div_ceil(tile[0]) as u32, n.div_ceil(tile[1]) as u32, 1];
            let l = launch(k, vec![step(0, la), step(ob, lb), step(oc, lc)], &[m as u32, n as u32, kk as u32], grid);
            let got = run_one(&e, l, &[(0, bytes(&a)), (ob, bytes(&b))], oc + align(lc), (oc, lc));
            close(&format!("matmul {name} {m}x{n}x{kk}"), &got, &want, 2e-4);
        }
    }

    println!("\ntiming (wall ms per launch, 200 launches per plan)");
    for (nrows, ncols) in [(4096usize, 1024usize), (512, 8192), (64, 16384), (2048, 128)] {
        let x = bytes(&noise(nrows * ncols, 3));
        let len = 4 * nrows * ncols;
        let y = align(len);
        let row = |k: &Arc<KernelBlob>| {
            let l = launch(k, vec![step(0, len), step(y, len)], &[ncols as u32], [nrows as u32, 1, 1]);
            time_launch(&e, &l, &[(0, x.clone())], 2 * y, 200)
        };
        println!("softmax {nrows} x {ncols}: shared {:.4} ms | subgroup {:.4} ms", row(&sm_shared), row(&sm_subgroup));
    }
    for (m, n, kk) in [(1024usize, 1024usize, 1024usize), (2048, 512, 2048), (1536, 256, 1536), (1536, 64, 1536)] {
        let (a, b) = (bytes(&noise(m * kk, 4)), bytes(&noise(n * kk, 5)));
        let (la, lb, lc) = (4 * m * kk, 4 * n * kk, 4 * m * n);
        let (ob, oc) = (align(la), align(la) + align(lb));
        let flops = 2.0 * (m * n * kk) as f64;
        let mut cells = Vec::new();
        for (name, k, tile) in [("64x64", &mm_tiled, [64, 64]), ("128x64", &mm_wide, [128, 64]), ("128x128", &mm_big, [128, 128])] {
            let grid = [m.div_ceil(tile[0]) as u32, n.div_ceil(tile[1]) as u32, 1];
            let l = launch(k, vec![step(0, la), step(ob, lb), step(oc, lc)], &[m as u32, n as u32, kk as u32], grid);
            let ms = time_launch(&e, &l, &[(0, a.clone()), (ob, b.clone())], oc + align(lc), std::env::var("TL_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(200));
            cells.push(format!("{name} {ms:.3} ms ({:.1} TFLOP/s)", flops / ms / 1e9));
        }
        println!("matmul {m}x{n}x{kk}: {}", cells.join(" | "));
    }
}

/// Like [`compile_ptx`] for expressions that name the test kernels (defined in the top-level
/// environment of the embedded Scheme).
fn compile_ptx_top(scheme: &Scheme, label: &str, expr: &str) -> Arc<KernelBlob> {
    let reply = scheme.eval(&format!("(compile-kernel-ptx {expr})")).unwrap_or_else(|e| panic!("{label}: {e}"));
    blob_from_reply(label, reply)
}

fn blob_from_reply(label: &str, reply: Value) -> Arc<KernelBlob> {
    let Value::List(items) = reply else { panic!("{label}: unexpected reply {reply:?}") };
    if let (Some(dir), Some(Value::String(text))) = (std::env::var_os("TL_DUMP_PTX"), items.first()) {
        let _ = std::fs::create_dir_all(&dir);
        let name: String = label.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ptx")), text).unwrap();
    }
    let [Value::String(text), Value::String(name), accesses, Value::Int(push), Value::List(wg)] = &items[..] else {
        panic!("{label}: unexpected reply {items:?}")
    };
    let bindings = match accesses {
        Value::List(a) => a
            .iter()
            .map(|v| match v {
                Value::Symbol(s) if s == "ro" => Access::Read,
                Value::Symbol(s) if s == "wo" => Access::Write,
                _ => Access::ReadWrite,
            })
            .collect(),
        _ => vec![],
    };
    let workgroup = [0, 1, 2].map(|i| match wg.get(i) {
        Some(Value::Int(n)) => *n as u32,
        _ => 1,
    });
    Arc::new(KernelBlob {
        target: Target::Cuda,
        code: Code::Ptx(text.clone()),
        entry: name.clone(),
        label: format!("ptx:{label}"),
        sig: Signature { bindings, push_bytes: *push as u32, workgroup, offsets: OffsetMode::Raw },
        spec: vec![],
    })
}

#[test]
fn every_generator_loads_as_ptx() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels) '(tl tree ptx)))").unwrap();
    let cuda = match Cuda::system_default() {
        Ok(c) => c,
        Err(tensorlisp_kernels::Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => return,
        Err(e) => panic!("{e}"),
    };
    let e = Engine::new(Arc::new(cuda));
    let exprs = [
        "(k:binary 'add)",
        "(k:unary 'gelu 'same)",
        "(k:unary 'sin 'same)",
        "(k:unary 'copy 'reshape)",
        "(k:unary 'copy 'repeat)",
        "(k:unary 'copy 'same 'f32 'f16)",
        "(k:unary 'copy 'same 'f16 'f32)",
        "(k:unary 'copy 'same 'i32 'f32)",
        "(k:concat 1)",
        "(k:arange)",
        "(k:get-rows 'f32)",
        "(k:get-rows 'f16)",
        "(k:get-rows 'q8_0)",
        "(k:norm)",
        "(k:rms-norm #f)",
        "(k:rms-norm #t)",
        "(k:soft-max #f)",
        "(k:soft-max #t)",
        "(k:sum-rows)",
        "(k:matmul 'f32 64 64 32 4 4)",
        "(k:matmul 'f16 64 64 32 4 4)",
        "(k:matmul 'q8_0 64 64 32 4 4)",
        "(k:matvec 'f32 1)",
        "(k:matvec 'f16 3)",
        "(k:matvec 'q8_0 1 4)",
        "(k:matvec 'q8_0 8)",
        "(k:rope 'norm)",
        "(k:rope 'neox)",
        "(k:set-rows 'f32)",
        "(k:set-rows 'f16)",
        "(k:argmax)",
        "(k:flash-attn 'f16 'f16 #t #f 2048)",
        "(k:flash-attn 'f32 'f32 #f #t 512)",
        "(k:rms-norm #t #t)",
        "(k:rms-norm #f #t)",
        "(k:ew '((f32 full) (f32 row) (f32 scalar)) 'f32 '((t0 (+ x0 x1)) (t1 (* t0 x2)) (t2 (unary gelu t1))) 't2 0)",
        "(k:ew '((i32 full)) 'f16 '((t0 x0)) 't0 0)",
        "(k:ew '() 'f32 '((t0 (+ c0 (* (f32 idx) c1))) (t1 (unary step t0))) 't1 2)",
        "(k:ew-q8 '((f32 full) (f32 full)) 'f32 '((t0 (unary gelu x0)) (t1 (* t0 x1))) 't1 0)",
        "(k:rowchain 1536 '((f32 full) (f32 full) (f32 full)) '((norm 1 0) (ew ((t0 (+ x2 x))) t0) (norm #f 1)) '((1 f32) (2 q8)) 2)",
        "(k:rowchain 256 '((f32 full) (f32 row)) '((norm 1 0)) '((0 f32)) 1)",
        "(k:flash-attn-part 'f16 'f16 #t #f 128 256 256)",
        "(k:flash-attn-part 'f32 'f32 #f #t 64 64 64)",
        "(k:flash-attn-combine)",
        "(k:flash-attn-combine #t)",
        "(k:norm-rope 256 #t 'neox 'f32 #f)",
        "(k:norm-rope 256 #t 'neox 'f16 #t)",
        "(k:norm-rope 512 #f #f 'f16 #t)",
        "(k:norm-rope 64 #t 'norm 'f32 #f)",
        "(k:gateup-q8 2 2)",
        "(k:gateup-q8 1 1)",
        "(k:matvec-q8 1 1 3)",
        "(k:matvec-q8 2 2 2)",
    ];
    for expr in exprs {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|e| panic!("{expr}: {e}"));
        let blob = blob_from_reply(expr, reply);
        // Loading is what validates the PTX: the driver compiles it for this GPU.
        let plan = e.compile(&[launch(&blob, vec![step(0, 4096); blob.sig.bindings.len()], &vec![0; blob.sig.push_bytes as usize / 4], [1, 1, 1])], 4096);
        plan.unwrap_or_else(|err| panic!("{expr}: {err}"));
        println!("{expr}: loaded ({} bindings, {} push bytes)", blob.sig.bindings.len(), blob.sig.push_bytes);
    }
}

// ---- learning from ggml-cuda: int8 dot products ---------------------------------------------

fn f16_bits(x: f32) -> u16 {
    // Normal-range values only (scales here are small positive numbers).
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3ff;
    assert!(exp > 0 && exp < 31, "scale out of the f16 normal range");
    sign | ((exp as u16) << 10) | mant as u16
}

fn f16_value(bits: u16) -> f32 {
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f32;
    let v = (1.0 + mant / 1024.0) * 2f32.powi(exp - 15);
    if bits & 0x8000 != 0 { -v } else { v }
}

/// m rows of k values as q8_0 blocks (f16 scale, 32 int8), as bytes, plus their dequantized values.
fn q8_0_matrix(m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s >> 33
    };
    let mut bytes = Vec::with_capacity(m * k / 32 * 34);
    let mut values = Vec::with_capacity(m * k);
    for _ in 0..m * k / 32 {
        let scale = f16_bits(0.001 + (next() % 1000) as f32 * 1e-5);
        bytes.extend(scale.to_le_bytes());
        let d = f16_value(scale);
        for _ in 0..32 {
            let q = (next() % 255) as i32 - 127;
            bytes.push(q as i8 as u8);
            values.push(q as f32 * d);
        }
    }
    (bytes, values)
}

/// The activation quantization of `k:quantize-q8`: per 32 values, scale amax/127, round half away.
fn quantize_q8(x: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut q = Vec::new();
    let mut d = Vec::new();
    for block in x.chunks(32) {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        d.push(amax / 127.0);
        for quad in block.chunks(4) {
            let mut w = 0u32;
            for (i, v) in quad.iter().enumerate() {
                let t = v * inv;
                let r = if t < 0.0 { -((-t + 0.5).floor()) } else { (t + 0.5).floor() } as i32;
                w |= ((r as u32) & 255) << (8 * i);
            }
            q.push(w);
        }
    }
    (q, d)
}

#[test]
fn q8_matvec_with_dp4a() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels) '(tl tree ptx)))").unwrap();
    let cuda = match Cuda::system_default() {
        Ok(c) => c,
        Err(tensorlisp_kernels::Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => return,
        Err(e) => panic!("{e}"),
    };
    let e = Engine::new(Arc::new(cuda));
    let ptx = |label: &str, expr: &str| {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{label}: {err}"));
        blob_from_reply(label, reply)
    };
    let quant = ptx("quantize_q8", "(k:quantize-q8)");
    let dot = ptx("matvec_q8dot_1x4", "(k:matvec-q8 1 4)");
    let dequant = ptx("matvec_q8_1x4", "(k:matvec 'q8_0 1 4)");

    println!("{:>8} {:>6} | {:>9} {:>9} | {:>9} {:>9} | rel err vs f32 path", "m", "k", "f32 path", "GB/s", "dp4a", "GB/s");
    for (m, k) in [(2048usize, 1536usize), (6144, 1536), (12288, 1536), (1536, 12288), (262144, 1536)] {
        let (wbytes, wvals) = q8_0_matrix(m, k, 7);
        let x = noise(k, 9);
        let (xq, xd) = quantize_q8(&x);

        // arena: weights | x | xq | xd | out
        let ow = 0u64;
        let ox = align(wbytes.len());
        let oq = ox + align(4 * k);
        let od = oq + align(4 * xq.len());
        let oo = od + align(4 * xd.len());
        let total = oo + align(4 * m);
        let rows_per_wg = 4 * (128 / 32);
        let grid = [m.div_ceil(rows_per_wg) as u32, 1, 1];

        // the quantization kernel reproduces the CPU's int8 blocks (to a rounding step at most)
        let nq = (k / 4) as u32;
        let qlaunch = launch(&quant, vec![step(ox, 4 * k), step(oq, 4 * xq.len()), step(od, 4 * xd.len())], &[nq, k as u32, k as u32], [nq.div_ceil(256), 1, 1]);
        let plan = e.compile(&[qlaunch.clone()], total).unwrap();
        let model = e.upload_model(&[]).unwrap();
        let xb = bytes(&x);
        let t = e.submit(&plan, &model, None, &[(ox, &xb)]).unwrap();
        let got_q: Vec<u32> = t.read(oq, 4 * xq.len()).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let got_d = floats(&t.read(od, 4 * xd.len()).unwrap());
        let off_by = got_q
            .iter()
            .zip(&xq)
            .flat_map(|(a, b)| (0..4).map(move |i| (((*a >> (8 * i)) & 255) as u8 as i8 as i32 - ((*b >> (8 * i)) & 255) as u8 as i8 as i32).abs()))
            .max()
            .unwrap();
        assert!(off_by <= 1, "quantized activations differ by {off_by}");
        close("quantize scales", &got_d, &xd, 1e-6);

        // reference: f32 path = dequantized weights . x ; dp4a path = int dot with quantized x
        let want_f32: Vec<f32> = (0..m).map(|r| (0..k).map(|i| wvals[r * k + i] as f64 * x[i] as f64).sum::<f64>() as f32).collect();
        let xq_i8: Vec<i32> = xq.iter().flat_map(|w| (0..4).map(move |i| ((w >> (8 * i)) & 255) as u8 as i8 as i32)).collect();
        let want_dot: Vec<f32> = (0..m)
            .map(|r| {
                (0..k / 32)
                    .map(|b| {
                        let wq: i32 = (0..32).map(|i| (wvals[r * k + b * 32 + i] / f16_value(u16::from_le_bytes([wbytes[(r * (k / 32) + b) * 34], wbytes[(r * (k / 32) + b) * 34 + 1]]))).round() as i32 * xq_i8[b * 32 + i]).sum();
                        f16_value(u16::from_le_bytes([wbytes[(r * (k / 32) + b) * 34], wbytes[(r * (k / 32) + b) * 34 + 1]])) as f64 * xd[b] as f64 * wq as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect();

        let a_range = step(ow, wbytes.len());
        let f32_launch = launch(&dequant, vec![a_range, step(ox, 4 * k), step(oo, 4 * m)], &[m as u32, k as u32, (k / 32) as u32, k as u32, m as u32], grid);
        let dot_launch = launch(&dot, vec![a_range, step(oq, 4 * xq.len()), step(od, 4 * xd.len()), step(oo, 4 * m)], &[m as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32], grid);
        let inputs_f32 = [(ow, wbytes.clone()), (ox, bytes(&x))];
        let inputs_dot = [(ow, wbytes.clone()), (oq, xq.iter().flat_map(|w| w.to_le_bytes()).collect()), (od, bytes(&xd))];

        let got_f32 = run_one(&e, f32_launch.clone(), &inputs_f32, total, (oo, 4 * m));
        let got_dot = run_one(&e, dot_launch.clone(), &inputs_dot, total, (oo, 4 * m));
        close(&format!("f32 path {m}x{k}"), &got_f32, &want_f32, 1e-4);
        close(&format!("dp4a path {m}x{k}"), &got_dot, &want_dot, 1e-4);
        let quant_err = got_dot.iter().zip(&want_f32).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / want_f32.iter().fold(0f32, |m, v| m.max(v.abs()));

        let reps = if m >= 262144 { 40 } else { 200 };
        let bytes_w = wbytes.len() as f64;
        let t_f32 = time_launch(&e, &f32_launch, &inputs_f32, total, reps);
        let t_dot = time_launch(&e, &dot_launch, &inputs_dot, total, reps);
        println!("{m:>8} {k:>6} | {t_f32:>7.4}ms {:>8.0} | {t_dot:>7.4}ms {:>8.0} | {quant_err:.2e}", bytes_w / t_f32 / 1e6, bytes_w / t_dot / 1e6);
    }
}

/// Bandwidth of the dp4a mat-vec on a DRAM-sized matrix as a function of its shape parameters.
#[test]
fn q8_matvec_sweep() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels) '(tl tree ptx)))").unwrap();
    let Ok(cuda) = Cuda::system_default() else { return };
    let e = Engine::new(Arc::new(cuda));
    let (m, k) = (262144usize, 1536usize);
    let (wbytes, _) = q8_0_matrix(m, k, 7);
    let x = noise(k, 9);
    let (xq, xd) = quantize_q8(&x);
    let ow = 0u64;
    let oq = align(wbytes.len());
    let od = oq + align(4 * xq.len());
    let oo = od + align(4 * xd.len());
    let total = oo + align(4 * m);
    let inputs = [(ow, wbytes.clone()), (oq, xq.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()), (od, bytes(&xd))];
    for expr in std::env::var("TL_SWEEP").unwrap_or_else(|_| "(k:matvec-q8 1 1)|(k:matvec-q8 1 2)|(k:matvec-q8 1 4)|(k:matvec-q8 1 8)".into()).split('|') {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{expr}: {err}"));
        let blob = blob_from_reply(expr, reply);
        let rows: usize = expr.trim_end_matches(')').rsplit(' ').next().unwrap().parse().unwrap();
        let grid = [m.div_ceil(rows * (blob.sig.workgroup[0] as usize / 32)) as u32, 1, 1];
        let l = launch(&blob, vec![step(ow, wbytes.len()), step(oq, 4 * xq.len()), step(od, 4 * xd.len()), step(oo, 4 * m)], &[m as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32], grid);
        let ms = time_launch(&e, &l, &inputs, total, 40);
        println!("{expr}: {ms:.4} ms, {:.0} GB/s of weights", wbytes.len() as f64 / ms / 1e6);
    }
}

/// How fast can this GPU stream memory through a Tree kernel at all? Row sums of a big f32 matrix.
#[test]
fn streaming_read_bandwidth() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels) '(tl tree ptx)))").unwrap();
    let Ok(cuda) = Cuda::system_default() else { return };
    let e = Engine::new(Arc::new(cuda));
    // Each thread sums 8 floats `stride` apart (coalesced across the warp, 8 loads in flight).
    let probe = "(kernel readbw (workgroup 256) (buffers (a ro f32) (d wo f32)) (push (stride u32) (nthreads u32))
                   (let ([i (global-id 0)])
                     (when (< i nthreads)
                       (var s f32 0.0)
                       (for (j 0 8) (set! s (+ s (load a (+ i (* j stride))))))
                       (store d i s))))";
    let blob = blob_from_reply("readbw", scheme.eval(&format!("(eval '(compile-kernel-ptx '{probe}) $env)")).unwrap());
    for n in [1usize << 26, 1 << 28, 1 << 29] {
        let data = bytes(&noise(1 << 20, 3).into_iter().cycle().take(n).collect::<Vec<_>>());
        let threads = n / 8;
        let oo = align(data.len());
        let total = oo + align(4 * threads);
        let l = launch(&blob, vec![step(0, data.len()), step(oo, 4 * threads)], &[(n / 8) as u32, threads as u32], [threads.div_ceil(256) as u32, 1, 1]);
        let ms = time_launch(&e, &l, &[(0, data.clone())], total, 20);
        println!("stream {} MB: {ms:.3} ms, {:.0} GB/s", data.len() >> 20, data.len() as f64 / ms / 1e6);
    }
}

/// Repacking q8_0 weights into aligned words + vector loads, against the u16-composed dp4a kernel.
#[test]
fn q8_matvec_repacked() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB, PTX_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels) '(tl tree ptx)))").unwrap();
    let Ok(cuda) = Cuda::system_default() else { return };
    let e = Engine::new(Arc::new(cuda));
    let ptx = |label: &str, expr: &str| {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{label}: {err}"));
        blob_from_reply(label, reply)
    };
    let repack = ptx("repack_q8", "(k:repack-q8)");
    let plain = ptx("matvec_q8dot", "(k:matvec-q8 1 4)");
    let packed = ptx("matvec_q8r", "(k:matvec-q8r 1 4)");
    println!("{:>8} {:>6} | {:>8} {:>6} | {:>8} {:>6}", "m", "k", "u16 dp4a", "GB/s", "repacked", "GB/s");
    for (m, k) in [(2048usize, 1536usize), (6144, 1536), (12288, 1536), (1536, 12288), (262144, 1536)] {
        let (wbytes, _) = q8_0_matrix(m, k, 7);
        let x = noise(k, 9);
        let (xq, xd) = quantize_q8(&x);
        let blocks = m * k / 32;
        let ow = 0u64;
        let oqs = align(wbytes.len());
        let ods = oqs + align(32 * blocks);
        let oxq = ods + align(2 * blocks);
        let oxd = oxq + align(4 * xq.len());
        let oo = oxd + align(4 * xd.len());
        let oo2 = oo + align(4 * m);
        let total = oo2 + align(4 * m);
        let rows_per_wg = 4 * (128 / 32);
        let grid = [m.div_ceil(rows_per_wg) as u32, 1, 1];
        let xin: Vec<u8> = xq.iter().flat_map(|w| w.to_le_bytes()).collect();
        let xdb = bytes(&xd);

        // repack on the GPU, then compare both mat-vecs and time them
        let nwords = (blocks * 8) as u32;
        let rl = launch(&repack, vec![step(ow, wbytes.len()), step(oqs, 32 * blocks), step(ods, 2 * blocks)], &[nwords], [nwords.div_ceil(256), 1, 1]);
        let plain_l = launch(&plain, vec![step(ow, wbytes.len()), step(oxq, xin.len()), step(oxd, xdb.len()), step(oo, 4 * m)], &[m as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32], grid);
        let packed_l = launch(&packed, vec![step(oqs, 32 * blocks), step(ods, 2 * blocks), step(oxq, xin.len()), step(oxd, xdb.len()), step(oo2, 4 * m)], &[m as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32], grid);
        let plan = e.compile(&[rl, plain_l.clone(), packed_l.clone()], total).unwrap();
        let model = e.upload_model(&[]).unwrap();
        let t = e.submit(&plan, &model, None, &[(ow, &wbytes), (oxq, &xin), (oxd, &xdb)]).unwrap();
        let (a, b) = (floats(&t.read(oo, 4 * m).unwrap()), floats(&t.read(oo2, 4 * m).unwrap()));
        close(&format!("repacked == u16 path {m}x{k}"), &b, &a, 1e-6);

        // timings: the packed weights are inputs here (as they would be, repacked at load time)
        let (qs_bytes, ds_bytes) = {
            let rl = launch(&repack, vec![step(ow, wbytes.len()), step(oqs, 32 * blocks), step(ods, 2 * blocks)], &[nwords], [nwords.div_ceil(256), 1, 1]);
            let plan = e.compile(&[rl], total).unwrap();
            let t = e.submit(&plan, &model, None, &[(ow, &wbytes)]).unwrap();
            (t.read(oqs, 32 * blocks).unwrap(), t.read(ods, 2 * blocks).unwrap())
        };
        let reps = if m >= 262144 { 40 } else { 200 };
        let bytes_w = wbytes.len() as f64;
        let t_plain = time_launch(&e, &plain_l, &[(ow, wbytes.clone()), (oxq, xin.clone()), (oxd, xdb.clone())], total, reps);
        let t_packed = time_launch(&e, &packed_l, &[(oqs, qs_bytes), (ods, ds_bytes), (oxq, xin.clone()), (oxd, xdb.clone())], total, reps);
        println!("{m:>8} {k:>6} | {t_plain:>6.4}ms {:>6.0} | {t_packed:>6.4}ms {:>6.0}", bytes_w / t_plain / 1e6, bytes_w / t_packed / 1e6);
    }
}
