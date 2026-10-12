//! Steady-state timing of whole pipelines on the native device (kernels compiled by the first run):
//!   cargo test --release -p tensorlisp --features native-metal-tree --test native_bench -- --ignored --nocapture
#![cfg(native_device)]

use std::time::Instant;

use tensorlisp::{Device, Model, RawInput};

#[test]
#[ignore]
fn t5gemma2_generate() {
    // TL_BENCH_MODEL=t5gemma2-270m-q8_0.gguf picks another file of models/t5gemma2-270m.
    let file = std::env::var("TL_BENCH_MODEL").unwrap_or_else(|_| "t5gemma2-270m-f16.gguf".into());
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/t5gemma2-270m").join(file);
    if !path.exists() {
        eprintln!("model not found, skipping");
        return;
    }
    let model = Model::load(&path, Device::Native).unwrap();
    for i in 0..8 {
        let t0 = Instant::now();
        let out = model.pipeline("generate", vec![("prompt".into(), RawInput::Text("The capital of France is".into()))]).unwrap();
        let text = out.iter().find(|(n, _)| n == "text").unwrap().1.as_text().unwrap().to_string();
        eprintln!("run {i}: {:.1} ms  {text:?}", t0.elapsed().as_secs_f64() * 1e3);
    }
}

/// One `mul-mat` per shape (weights m x k, activations k x n), timed per run, as TFLOPS:
///   cargo test --release -p tensorlisp [--features native-metal-tree] --test native_bench matmul_shapes -- --ignored --nocapture
#[test]
#[ignore]
fn matmul_shapes() {
    use tensorlisp::{DType, Program, RunOptions, gguf::GgufWriter};
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut noise = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };
    // TL_BENCH_ONLY=f16:4096x4096x256 runs one shape, TL_BENCH_RUNS=n times (for GPU traces).
    let only = std::env::var("TL_BENCH_ONLY").ok();
    let extra_runs: usize = std::env::var("TL_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    for dtype in [DType::F32, DType::F16] {
        for (m, k, n) in [(768usize, 768usize, 192usize), (3072, 768, 192), (768, 3072, 192), (2304, 768, 192), (768, 768, 1024), (4096, 4096, 256), (4096, 4096, 1)] {
            if only.as_ref().is_some_and(|o| *o != format!("{dtype}:{m}x{k}x{n}")) {
                continue;
            }
            let program = format!("(model main (inputs [x f32 ({k} {n})]) (outputs [y (ggml-mul-mat (weight \"w\") x) 2]))");
            let mut w = GgufWriter::new();
            w.set_program(&Program::Text(program)).unwrap();
            let data = noise(m * k);
            let bytes = if dtype == DType::F32 { data.iter().flat_map(|v| v.to_le_bytes()).collect() } else { dtype.from_f32(&data, k).unwrap() };
            w.add_tensor("w", dtype, &[m, k], bytes).unwrap();
            let path = std::env::temp_dir().join(format!("tl-bench-{}-{m}-{k}-{n}.gguf", dtype));
            w.write(&path).unwrap();
            let model = Model::load(&path, Device::Native).unwrap();
            let x = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, k]), noise(k * n)).unwrap();
            let runs = extra_runs;
            let mut best = f64::MAX;
            for i in 0..runs + 3 {
                let t0 = Instant::now();
                model.run(&[("x", x.view())]).unwrap();
                if i >= 3 {
                    best = best.min(t0.elapsed().as_secs_f64());
                }
            }
            let _ = std::fs::remove_file(&path);
            let _ = RunOptions::default();
            eprintln!("{dtype} m={m} k={k} n={n}: {:.3} ms  {:.2} TFLOPS", best * 1e3, 2.0 * (m * k * n) as f64 / best / 1e12);
        }
    }
}
