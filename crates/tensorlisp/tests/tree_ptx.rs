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

// ---- tensor cores: mma.sync fragments -------------------------------------------------------------

/// `k:matmul-cm` (cooperative-matrix forms) compiled to PTX `mma.sync.m16n8k16`: against the CPU at
/// awkward shapes, then timed against the scalar tiled kernel.
#[test]
fn tensor_core_matmul_as_ptx() {
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
    let compile = |expr: &str| {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{expr}: {err}"));
        blob_from_reply(expr, reply)
    };
    let run = |k: &Arc<KernelBlob>, tile: [usize; 2], m: usize, n: usize, kk: usize, a: &[f32], b: &[f32]| {
        let (la, lb, lc) = (4 * m * kk, 4 * n * kk, 4 * m * n);
        let (ob, oc) = (align(la), align(la) + align(lb));
        let push = [m as u32, n as u32, kk as u32, kk as u32, kk as u32, m as u32, 0, 0, 0, 0, 0, 0, 1, 1, 1];
        let grid = [m.div_ceil(tile[0]) as u32, n.div_ceil(tile[1]) as u32, 1];
        let l = launch(k, vec![step(0, la), step(ob, lb), step(oc, lc)], &push, grid);
        let got = run_one(&e, l.clone(), &[(0, bytes(a)), (ob, bytes(b))], oc + align(lc), (oc, lc));
        (got, l, oc + align(lc), ob)
    };
    for (m, n, kk) in [(64usize, 64usize, 64usize), (100, 37, 70), (17, 130, 33), (257, 5, 16)] {
        let (a, b) = (noise(m * kk, 1), noise(n * kk, 2));
        let want = cpu_matmul(&a, &b, m, n, kk);
        for cfg in ["64 64 32 2 2", "32 32 16 1 1", "64 32 32 2 1", "32 32 32 2 2"] {
            let p: Vec<usize> = cfg.split(' ').map(|x| x.parse().unwrap()).collect();
            let k = compile(&format!("(k:matmul-cm 'f32 {cfg})"));
            let (got, ..) = run(&k, [p[0], p[1]], m, n, kk, &a, &b);
            close(&format!("mma {cfg} {m}x{n}x{kk}"), &got, &want, 5e-3);
        }
    }
    if std::env::var_os("TL_BENCH").is_none() {
        println!("timing skipped (set TL_BENCH=1)");
        return;
    }
    println!("\ntiming (wall ms per launch)");
    for (m, n, kk) in [(768usize, 192usize, 768usize), (3072, 1024, 768), (768, 1024, 3072)] {
        let (a, b) = (noise(m * kk, 1), noise(n * kk, 2));
        let mut cells = Vec::new();
        for (name, expr, tile) in [("scalar 64x64", "(k:matmul 'f32 64 64 32 4 4)".to_string(), [64, 64]), ("mma 64x64x16", "(k:matmul-cm 'f32 64 64 16 2 2)".to_string(), [64, 64]), ("mma 32x32x32", "(k:matmul-cm 'f32 32 32 32 2 2)".to_string(), [32, 32])] {
            let k = compile(&expr);
            let (_, l, step_bytes, ob) = run(&k, tile, m, n, kk, &a, &b);
            let ms = time_launch(&e, &l, &[(0, bytes(&a)), (ob, bytes(&b))], step_bytes, 100);
            cells.push(format!("{name} {ms:.3} ms ({:.1} TFLOP/s)", 2.0 * (m * n * kk) as f64 / ms / 1e9));
        }
        println!("matmul {m}x{n}x{kk}: {}", cells.join(" | "));
    }
}

/// The q8_0 weight x f32 activation product of prefill (`k:matmul-cm`: the tile is dequantized to f16 in shared
/// memory, `mma.sync`), at Gemma 4 E2B's shapes, against a CPU reference on the first rows; timed with
/// `TL_BENCH=1`. `TL_CM_CFGS="64 64 16 2 2;..."` overrides the tiles.
#[test]
fn q8_matmul_cm_throughput() {
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
    let bench = std::env::var_os("TL_BENCH").is_some();
    let cfgs: Vec<String> = match std::env::var("TL_CM_CFGS") {
        Ok(v) => v.split(';').map(str::to_string).collect(),
        Err(_) => ["64 64 16 2 2", "64 64 32 2 2", "128 64 32 4 2", "128 128 32 4 2"].map(str::to_string).to_vec(),
    };
    let shapes: &[(usize, usize, usize)] = if bench { &[(12288, 406, 1536), (6144, 406, 1536), (1536, 406, 6144), (2048, 406, 1536)] } else { &[(512, 100, 256)] };
    for &(m, n, k) in shapes {
        let (wbytes, wvals) = q8_0_matrix(m, k, 1);
        let b = noise(n * k, 2);
        let rows = 64.min(m);
        let want = cpu_matmul(&wvals[..rows * k], &b, rows, n, k);
        for cfg in &cfgs {
            let p: Vec<usize> = cfg.split(' ').map(|x| x.parse().unwrap()).collect();
            let blob = {
                let expr = format!("(k:matmul-cm 'q8_0 {cfg})");
                let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{expr}: {err}"));
                blob_from_reply(&expr, reply)
            };
            let (la, lb, lc) = (wbytes.len(), 4 * n * k, 4 * m * n);
            let (ob, oc) = (align(la), align(la) + align(lb));
            let push = [m as u32, n as u32, k as u32, (k / 32) as u32, k as u32, m as u32, 0, 0, 0, 0, 0, 0, 1, 1, 1];
            let grid = [m.div_ceil(p[0]) as u32, n.div_ceil(p[1]) as u32, 1];
            let l = launch(&blob, vec![step(0, la), step(ob, lb), step(oc, lc)], &push, grid);
            let inputs = [(0u64, wbytes.clone()), (ob, bytes(&b))];
            let step_bytes = oc + align(lc);
            let got = run_one(&e, l.clone(), &inputs, step_bytes, (oc, lc));
            // output is [n][m]: compare the first `rows` rows of every column
            let mut g = Vec::new();
            for col in 0..n {
                g.extend_from_slice(&got[col * m..col * m + rows]);
            }
            let mut w = Vec::new();
            for col in 0..n {
                w.extend_from_slice(&want[col * rows..(col + 1) * rows]);
            }
            close(&format!("q8 cm {cfg} {m}x{n}x{k}"), &g, &w, 5e-3);
            if bench {
                let ms = time_launch(&e, &l, &inputs, step_bytes, 50);
                println!("q8 cm {cfg} {m}x{n}x{k}: {ms:.3} ms {:.1} TFLOP/s", 2.0 * (m * n * k) as f64 / ms / 1e9);
            }
        }
    }
}

fn words_of(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `k:matmul-q8mma`: q8_0 weights x int8-quantized activations on the int8 tensor cores (`mma.sync`
/// m16n8k32), against an exact reference (integer dot per block, scaled in f64); timed with `TL_BENCH=1`
/// next to the f16 tensor-core kernel. `TL_Q8MMA_CFGS="64 64 2 2;..."` overrides the tiles (BM BN WM WN).
#[test]
fn q8_matmul_on_int8_tensor_cores() {
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
    let bench = std::env::var_os("TL_BENCH").is_some();
    // TL_Q8MMA_KERNEL=w: the wide-staging variant (k % 256 == 0)
    let kernel = if std::env::var("TL_Q8MMA_KERNEL").is_ok_and(|v| v == "w") { "k:matmul-q8mma-w" } else { "k:matmul-q8mma" };
    let wide = kernel.ends_with("-w");
    let cfgs: Vec<String> = match std::env::var("TL_Q8MMA_CFGS") {
        Ok(v) => v.split(';').map(str::to_string).collect(),
        Err(_) if wide => ["64 64 2 2", "32 128 1 4", "64 32 2 1"].map(str::to_string).to_vec(),
        Err(_) => ["64 64 2 2", "128 64 4 2", "64 128 2 2", "128 128 4 2"].map(str::to_string).to_vec(),
    };
    let shapes: &[(usize, usize, usize)] = if bench {
        &[(12288, 406, 1536), (6144, 406, 1536), (1536, 406, 6144), (2048, 406, 1536)]
    } else if wide {
        &[(256, 100, 256), (130, 37, 512), (70, 5, 256), (33, 130, 768)]
    } else {
        &[(70, 37, 96), (256, 100, 256), (130, 5, 64)]
    };
    for &(m, n, k) in shapes {
        let (wbytes, _) = q8_0_matrix(m, k, 1);
        let x = noise(n * k, 2);
        let (mut xq, mut xd) = (Vec::new(), Vec::new());
        for col in 0..n {
            let (q, d) = quantize_q8(&x[col * k..(col + 1) * k]);
            xq.extend(q);
            xd.extend(d);
        }
        // exact reference on the first rows
        let rows = 64.min(m);
        let mut want = vec![0f32; rows * n];
        for col in 0..n {
            for r in 0..rows {
                let mut s = 0f64;
                for b in 0..k / 32 {
                    let blk = &wbytes[(r * (k / 32) + b) * 34..][..34];
                    let ws = f16_value(u16::from_le_bytes([blk[0], blk[1]])) as f64;
                    let mut dot = 0i64;
                    for i in 0..32 {
                        let q = (xq[col * k / 4 + b * 8 + i / 4] >> (8 * (i % 4))) as u8 as i8;
                        dot += blk[2 + i] as i8 as i64 * q as i64;
                    }
                    s += ws * xd[col * (k / 32) + b] as f64 * dot as f64;
                }
                want[col * rows + r] = s as f32;
            }
        }
        for cfg in &cfgs {
            let p: Vec<usize> = cfg.split(' ').map(|x| x.parse().unwrap()).collect();
            let expr = format!("({kernel} {cfg})");
            let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{expr}: {err}"));
            let blob = blob_from_reply(&expr, reply);
            let (la, lq, ld, lc) = (wbytes.len(), 4 * xq.len(), 4 * xd.len(), 4 * m * n);
            let (oq, od) = (align(la), align(la) + align(lq));
            let oc = od + align(ld);
            let push = [m as u32, n as u32, k as u32, (k / 32) as u32, (k / 4) as u32, (k / 32) as u32, m as u32];
            let grid = [m.div_ceil(p[0]) as u32, n.div_ceil(p[1]) as u32, 1];
            let l = launch(&blob, vec![step(0, la), step(oq, lq), step(od, ld), step(oc, lc)], &push, grid);
            let inputs = [(0u64, wbytes.clone()), (oq, words_of(&xq)), (od, bytes(&xd))];
            let step_bytes = oc + align(lc);
            let got = run_one(&e, l.clone(), &inputs, step_bytes, (oc, lc));
            let mut g = Vec::new();
            for col in 0..n {
                g.extend_from_slice(&got[col * m..col * m + rows]);
            }
            close(&format!("q8 mma {cfg} {m}x{n}x{k}"), &g, &want, 1e-4);
            if bench {
                let ms = time_launch(&e, &l, &inputs, step_bytes, 50);
                println!("q8 mma {cfg} {m}x{n}x{k}: {ms:.3} ms {:.1} TOP/s", 2.0 * (m * n * k) as f64 / ms / 1e9);
            }
        }
    }
}

fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    if exp < -14 {
        // subnormal: multiples of 2^-24
        return sign | (a * 16777216.0).round() as u16;
    }
    let mant = (b & 0x7f_ffff) as u64;
    let mut m = (mant + 0x0fff + ((mant >> 13) & 1)) >> 13; // round to nearest even
    let mut e = (exp + 15) as u32;
    if m == 0x400 {
        m = 0;
        e += 1;
    }
    sign | ((e as u16) << 10) | m as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        sign * m * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0.0 { sign * f32::INFINITY } else { f32::NAN }
    } else {
        sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| to_f16(*x).to_ne_bytes()).collect()
}

/// Flash attention on tensor cores (`k:flash-attn-mma`) against a CPU reference over f16-rounded inputs: prefill
/// shapes with a causal (optionally windowed) mask, query counts and cache lengths off the tile sizes, grouped heads.
/// `TL_BENCH=1` times Gemma 4 E2B's two attention shapes against the scalar kernel.
#[test]
fn flash_attention_on_tensor_cores() {
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
    let compile = |expr: &str| {
        let reply = scheme.eval(&format!("(eval '(compile-kernel-ptx {expr}) $env)")).unwrap_or_else(|err| panic!("{expr}: {err}"));
        blob_from_reply(expr, reply)
    };
    let bench = std::env::var_os("TL_BENCH").is_some();
    // (nq, kv, hsk, hsv, nh, nkv, window, masked)
    let cases: &[(usize, usize, usize, usize, usize, usize, usize, bool)] = if bench {
        &[(406, 406, 256, 256, 8, 1, 0, true), (406, 406, 512, 512, 8, 1, 0, true)]
    } else {
        &[
            (64, 64, 64, 64, 2, 1, 0, true),
            (37, 37, 64, 64, 4, 2, 0, true),
            (100, 100, 128, 128, 4, 2, 0, true),
            (41, 150, 256, 256, 8, 1, 0, true),
            (33, 33, 256, 256, 8, 1, 16, true),
            (70, 70, 512, 512, 8, 1, 0, true),
            (20, 99, 64, 64, 2, 1, 0, false),
        ]
    };
    for &(nq, kv, hsk, hsv, nh, nkv, window, masked) in cases {
        let q = noise(nh * nq * hsk, 1);
        let k: Vec<f32> = noise(nkv * kv * hsk, 2).iter().map(|x| from_f16(to_f16(x * 2.0))).collect();
        let v: Vec<f32> = noise(nkv * kv * hsv, 3).iter().map(|x| from_f16(to_f16(*x))).collect();
        let qr: Vec<f32> = q.iter().map(|x| from_f16(to_f16(*x))).collect();
        let allowed = |i: usize, j: usize| !masked || (j + nq <= kv + i && (window == 0 || j + window + nq > kv + i));
        let mask: Vec<u16> = (0..nq).flat_map(|i| (0..kv).map(move |j| (i, j))).map(|(i, j)| if allowed(i, j) { 0 } else { 0xfc00 }).collect();
        let scale = 1.0 / (hsk as f32).sqrt();
        let gqa = nh / nkv;
        let mut want = vec![0f32; nq * nh * hsv];
        for h in 0..nh {
            for i in 0..nq {
                let sc: Vec<f64> = (0..kv)
                    .map(|j| {
                        if !allowed(i, j) {
                            return f64::NEG_INFINITY;
                        }
                        let kb = ((h / gqa) * kv + j) * hsk;
                        (0..hsk).map(|c| qr[(h * nq + i) * hsk + c] as f64 * k[kb + c] as f64).sum::<f64>() * scale as f64
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ps: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let l: f64 = ps.iter().sum();
                for c in 0..hsv {
                    let o: f64 = (0..kv).map(|j| ps[j] * v[((h / gqa) * kv + j) * hsv + c] as f64).sum();
                    want[(i * nh + h) * hsv + c] = (o / l) as f32;
                }
            }
        }
        let expr = format!("(k:flash-attn-mma {} #f {hsk} {hsv})", if masked { "#t" } else { "#f" });
        let kern = compile(&expr);
        let (lq, lk, lv, lm, ld) = (4 * q.len(), 2 * k.len(), 2 * v.len(), 2 * mask.len(), 4 * want.len());
        let (ok, ov, om, od) = (align(lq), align(lq) + align(lk), align(lq) + align(lk) + align(lv), align(lq) + align(lk) + align(lv) + align(lm));
        let push = [
            nq as u32, nh as u32, hsk as u32, hsv as u32, kv as u32, gqa as u32,
            hsk as u32, (nq * hsk) as u32, 0,
            hsk as u32, (kv * hsk) as u32, 0,
            hsv as u32, (kv * hsv) as u32, 0,
            hsv as u32, (nh * hsv) as u32, 0,
            kv as u32, 0, 0, 1, 1,
            scale.to_bits(), 0,
        ];
        let mut bindings = vec![step(0, lq), step(ok, lk), step(ov, lv)];
        if masked {
            bindings.push(step(om, lm));
        }
        bindings.push(step(od, ld));
        let l = launch(&kern, bindings, &push, [nq.div_ceil(16) as u32, nh as u32, 1]);
        let mbytes: Vec<u8> = mask.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let inputs = [(0, bytes(&q)), (ok, f16_bytes(&k)), (ov, f16_bytes(&v)), (om, mbytes)];
        let step_bytes = od + align(ld);
        let got = run_one(&e, l.clone(), &inputs, step_bytes, (od, ld));
        close(&format!("flash mma nq {nq} kv {kv} hs {hsk}/{hsv} nh {nh}/{nkv} window {window} masked {masked}"), &got, &want, 3e-3);
        if bench {
            let ms = time_launch(&e, &l, &inputs, step_bytes, 50);
            let flops = 4.0 * (nh * nq * kv * hsk) as f64 / 2.0;
            println!("flash mma nq {nq} hs {hsk}: {ms:.3} ms ({:.1} TFLOP/s causal-effective)", flops / ms / 1e9);
            let scalar = compile("(k:flash-attn 'f16 'f16 #t #f 2048)");
            let mut sl = l.clone();
            sl.kernel = scalar;
            sl.grid = [nq as u32, nh as u32, 1];
            let sms = time_launch(&e, &sl, &inputs, step_bytes, 50);
            println!("flash scalar nq {nq} hs {hsk}: {sms:.3} ms");
        }
    }
}
