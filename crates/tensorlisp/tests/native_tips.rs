//! TIPSv2 (the real port in ports/tipsv2) on the native Metal device against
//! the ggml CPU backend. Skipped when the converted weights are not under
//! models/tipsv2-b14 (see ports/tipsv2/README.md for how to produce them).
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, Model, RunOptions, Taps};

fn model_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tipsv2-b14").join(name);
    p.exists().then_some(p)
}

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

fn compare(path: &PathBuf, inputs: &[(&str, ArrayD<f32>)], min_cosine: f64) {
    let cpu = Model::load(path, Device::Cpu).unwrap();
    let native = Model::load(path, Device::Native).unwrap();
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    let options = RunOptions { taps: Taps::All, ..Default::default() };
    let want = cpu.run_with(&views, &options).unwrap();
    let got = native.run_with(&views, &options).unwrap();
    assert_eq!(want.outputs.len(), got.outputs.len());
    for ((name, w), (_, g)) in want.taps.iter().chain(&want.outputs).zip(got.taps.iter().chain(&got.outputs)) {
        let cos = cosine(g, w);
        assert!(cos >= min_cosine, "{name}: cosine {cos}");
    }
}

#[test]
fn text_encoder_f32() {
    let Some(path) = model_path("tipsv2-b14-text-f32.gguf") else { return };
    let mut ids = noise(&[3, 64], 1).mapv(|v| (v * 30000.0).floor());
    let mut paddings = ArrayD::<f32>::zeros(IxDyn(&[3, 64]));
    for (row, len) in [64usize, 20, 5].into_iter().enumerate() {
        for j in len..64 {
            ids[[row, j]] = 0.0;
            paddings[[row, j]] = 1.0;
        }
    }
    compare(&path, &[("ids", ids), ("paddings", paddings)], 0.99999);
}

#[test]
fn vision_encoder_f16_with_flash_attention() {
    let Some(path) = model_path("tipsv2-b14-vision-f16.gguf") else { return };
    compare(&path, &[("image", noise(&[1, 3, 224, 224], 2))], 0.9999);
}

fn port_path(model: &str, file: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models").join(model).join(file);
    p.exists().then_some(p)
}

#[test]
fn yolo11n_head_matches_cpu() {
    let Some(path) = port_path("yolo11n", "yolo11n.gguf") else { return };
    compare(&path, &[("image", noise(&[1, 3, 320, 320], 3))], 0.9999);
}

#[test]
fn t5gemma2_generates_the_same_text_as_ggml_metal() {
    use tensorlisp::RawInput;
    let Some(path) = port_path("t5gemma2-270m", "t5gemma2-270m-f16.gguf") else { return };
    let text = |device| {
        let model = Model::load(&path, device).unwrap();
        let out = model.pipeline("generate", vec![("prompt".into(), RawInput::Text("The capital of France is".into()))]).unwrap();
        let (_, v) = out.iter().find(|(n, _)| n == "text").unwrap();
        v.as_text().unwrap().to_string()
    };
    let (native, ggml) = (text(Device::Native), text(Device::Gpu));
    assert_eq!(native, ggml);
    assert!(native.contains("Paris"), "{native}");
}

#[test]
fn ppocrv6_detector_and_recognizer_match_the_cpu() {
    let Some(path) = port_path("ppocrv6", "ppocrv6-f16.gguf") else { return };
    let cpu = Model::load(&path, Device::Cpu).unwrap();
    let native = Model::load(&path, Device::Native).unwrap();
    for (entry, shape) in [("det", [1usize, 3, 96, 160]), ("rec", [1, 3, 48, 160])] {
        let x = noise(&shape, 4) - 0.5;
        let run = |m: &Model| m.run_entry(entry, &[("image", x.view())], &RunOptions::default()).unwrap().outputs;
        for ((name, w), (_, g)) in run(&cpu).iter().zip(&run(&native)) {
            let cos = cosine(g, w);
            assert!(cos >= 0.9999, "{entry}/{name}: cosine {cos}");
        }
    }
}
