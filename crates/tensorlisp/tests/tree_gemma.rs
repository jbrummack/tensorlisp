//! Gemma 4 E2B decode step on the native Vulkan device: time per step with ggml's shaders against
//! Tree kernels (and the fusions on top), at a short and a long context.
//!   TL_NATIVE_TREE=only TL_TREE_DOT=1 TL_VULKAN_PROFILE=1 \
//!     cargo test --release -p tensorlisp --features native-vulkan --test tree_gemma -- --ignored --nocapture
//! Needs models/gemma4-e2b/gemma4-e2b-q8_0.gguf (ports/gemma4/README.md).
#![cfg(native_device)]

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, LoadOptions, Model, Program, RunOptions};

fn load() -> Option<Model> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma4-e2b/gemma4-e2b-q8_0.gguf");
    if !path.exists() {
        return None;
    }
    let ports = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ports/gemma4");
    let source = std::fs::read_to_string(ports.join("gemma4.ss")).unwrap();
    let options = LoadOptions { program: Some(Program::Text(source)), ..Default::default() };
    Some(Model::load_with(path, Device::Native, options).unwrap())
}

fn ints(v: usize) -> ArrayD<f32> {
    Array::from_shape_fn(IxDyn(&[1, 1]), |_| v as f32)
}

fn step(model: &Model, token: usize, pos: usize) -> usize {
    let (t, p) = (ints(token), ints(pos));
    let out = model.run_entry("step", &[("token", t.view()), ("pos", p.view())], &RunOptions::default()).unwrap();
    out.outputs.into_iter().find(|(n, _)| n == "next").unwrap().1.iter().next().map(|&v| v as usize).unwrap()
}

/// Median-ish: the best of three rounds of `steps` decode steps starting at position `at`.
fn ms_per_step(model: &Model, at: usize, steps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let mut tok = 2;
        let t = Instant::now();
        for i in 0..steps {
            tok = step(model, tok, at + i);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    best
}

#[test]
#[ignore]
fn decode_step_time() {
    let Some(model) = load() else { return };
    step(&model, 2, 0);
    let steps: usize = std::env::var("TL_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let mode = std::env::var("TL_NATIVE_TREE").unwrap_or_else(|_| "ggml shaders".into());
    println!("[{mode}] position 10:   {:.3} ms per step", ms_per_step(&model, 10, steps));
    println!("[{mode}] position 600:  {:.3} ms per step", ms_per_step(&model, 600, steps));
}
