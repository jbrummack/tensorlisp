//! A whole model on Tree-generated kernels: the TIPSv2 text encoder (f32 weights) on the native
//! Vulkan device, run repeatedly and timed. Mode comes from the environment:
//!
//!   cargo test --release -p tensorlisp --features native-vulkan --test tree_model -- --nocapture
//!   TL_NATIVE_TREE=1    ... Tree kernels where a generator exists, ggml's shaders elsewhere
//!   TL_NATIVE_TREE=only ... Tree kernels only (fails on a node without one)
//!   TL_VULKAN_PROFILE=1 ... per-kernel GPU time table when the device closes
//!
//! Needs models/tipsv2-b14/tipsv2-b14-text-f32.gguf (skipped without it).
#![cfg(feature = "native-vulkan")]

use std::time::Instant;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, Model, RunOptions};

fn noise(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    Array::from_shape_fn(IxDyn(shape), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    })
}

fn cosine(a: &ArrayD<f32>, b: &ArrayD<f32>) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn tipsv2_text_encoder_timing() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tipsv2-b14/tipsv2-b14-text-f32.gguf");
    if !path.exists() {
        eprintln!("model not found, skipping");
        return;
    }
    let mut ids = noise(&[3, 64], 1).mapv(|v| (v * 30000.0).floor());
    let mut paddings = ArrayD::<f32>::zeros(IxDyn(&[3, 64]));
    for (row, len) in [64usize, 20, 5].into_iter().enumerate() {
        for j in len..64 {
            ids[[row, j]] = 0.0;
            paddings[[row, j]] = 1.0;
        }
    }
    let inputs = [("ids", ids.view()), ("paddings", paddings.view())];

    let cpu = Model::load(&path, Device::Cpu).unwrap();
    let want = cpu.run_with(&inputs, &RunOptions::default()).unwrap().outputs;

    let native = Model::load(&path, Device::Native).unwrap();
    let got = native.run_with(&inputs, &RunOptions::default()).unwrap().outputs;
    for ((name, w), (_, g)) in want.iter().zip(&got) {
        let cos = cosine(g, w);
        println!("{name}: cosine to the CPU backend {cos:.8}");
        assert!(cos > 0.99999, "{name}: cosine {cos}");
    }

    let iters = 50;
    let t = Instant::now();
    for _ in 0..iters {
        native.run_with(&inputs, &RunOptions::default()).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("mode {:?}: {ms:.3} ms per run ({iters} runs)", std::env::var("TL_NATIVE_TREE").unwrap_or_else(|_| "ggml shaders".into()));
}
