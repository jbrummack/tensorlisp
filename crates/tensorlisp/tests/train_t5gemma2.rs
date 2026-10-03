//! LoRA training of T5Gemma 2 on ggml's autodiff, against the PyTorch reference of
//! ports/t5gemma2/train_reference.py. Needs the model and the reference's output:
//!
//!   python ports/t5gemma2/train_reference.py models/t5gemma2-270m/model.safetensors $TL_TRAIN_REF
//!   TL_TRAIN_REF=... cargo test --release -p tensorlisp --test train_t5gemma2 -- --ignored --nocapture
//!
//! `TL_TRAIN_DEVICE=cpu|gpu` picks the ggml backend (default cpu).

use std::{fs, path::PathBuf};

use ndarray::{ArrayD, IxDyn};
use tensorlisp::{AdapterFormat, Device, LoadOptions, Model, Program, TrainOptions};

fn read_npy(path: &PathBuf) -> ArrayD<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).unwrap();
    assert!(header.contains("<f4") && header.contains("False"), "{header}");
    let shape_text = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = shape_text.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
    let data: Vec<f32> = bytes[start + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
}

fn rel_err(a: &ArrayD<f32>, b: &ArrayD<f32>) -> f32 {
    let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    num / den.max(1e-12)
}

fn load(device: Device) -> Option<(Model, PathBuf)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let gguf = std::env::var("TL_TRAIN_GGUF").map(PathBuf::from).unwrap_or_else(|_| root.join("models/t5gemma2-270m/t5gemma2-270m-f16.gguf"));
    let reference = PathBuf::from(std::env::var("TL_TRAIN_REF").ok()?);
    if !gguf.exists() {
        return None;
    }
    let program = fs::read_to_string(root.join("ports/t5gemma2/t5gemma2.ss")).unwrap()
        + "\n"
        + &fs::read_to_string(root.join("ports/t5gemma2/train.ss")).unwrap();
    let model = Model::load_with(&gguf, device, LoadOptions { program: Some(Program::Text(program)), ..Default::default() }).unwrap();
    Some((model, reference))
}

fn device() -> Device {
    match std::env::var("TL_TRAIN_DEVICE").as_deref() {
        Ok("gpu") => Device::Gpu,
        _ => Device::Cpu,
    }
}

/// Tolerance scale: f32 weights on the CPU are exact to float noise, f16 weights or a GPU add rounding
/// of the activations (and Adam amplifies gradient noise).
fn tolerance(strict: f32, loose: f32) -> f32 {
    let f32_weights = std::env::var("TL_TRAIN_GGUF").is_ok_and(|p| p.contains("f32"));
    if f32_weights && device() == Device::Cpu { strict } else { loose }
}

const INPUTS: [&str; 8] = ["ids", "gather", "pos", "mask", "tokens", "dpos", "targets", "weights"];

fn batch(reference: &PathBuf) -> Vec<(&'static str, ArrayD<f32>)> {
    INPUTS.iter().map(|&n| (n, read_npy(&reference.join(format!("batch/{n}.npy"))))).collect()
}

fn set_init(model: &Model, reference: &PathBuf) {
    for (name, shape) in model.params() {
        let init = read_npy(&reference.join(format!("init/{name}.npy")));
        assert_eq!(init.shape(), shape.as_slice(), "{name}");
        model.set_state(&name, &init).unwrap();
    }
}

#[test]
#[ignore]
fn loss_and_gradients_match_pytorch() {
    let Some((model, reference)) = load(device()) else { return };
    let batch = batch(&reference);
    let shapes: Vec<(&str, Vec<usize>)> = batch.iter().map(|(n, a)| (*n, a.shape().to_vec())).collect();
    set_init(&model, &reference);
    let t0 = std::time::Instant::now();
    let mut trainer = model.trainer(&shapes, TrainOptions::default()).unwrap();
    println!("trainer built in {:.2?}", t0.elapsed());
    let views: Vec<_> = batch.iter().map(|(n, a)| (*n, a.view())).collect();
    let t0 = std::time::Instant::now();
    let loss = trainer.eval_grads(&views).unwrap();
    println!("forward + backward in {:.2?}", t0.elapsed());
    let t0 = std::time::Instant::now();
    trainer.eval_grads(&views).unwrap();
    println!("second forward + backward in {:.2?}", t0.elapsed());
    let expected = read_npy(&reference.join("loss.npy"))[[0]];
    println!("loss {loss} (pytorch {expected})");
    assert!((loss - expected).abs() < 2e-3 * expected.abs(), "loss {loss} vs {expected}");

    let mut worst = (0f32, String::new());
    let mut sum = 0f32;
    let names = trainer.param_names();
    let mut by_module: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
    for name in &names {
        let g = trainer.grad(name).unwrap();
        let e = read_npy(&reference.join(format!("grads/{name}.npy")));
        let err = rel_err(&g, &e);
        let parts: Vec<&str> = name.split('.').collect(); // lora.<stack>.layers.<i>.<...>.<A|B>
        by_module.entry(format!("{} {:>2} {}", parts[1], parts[3], parts.last().unwrap())).or_default().push(err);
        sum += err;
        if err > worst.0 {
            worst = (err, name.clone());
        }
    }
    if std::env::var("TL_TRAIN_VERBOSE").is_ok() {
        for (k, v) in &by_module {
            println!("  {k}: mean rel err {:.1e}", v.iter().sum::<f32>() / v.len() as f32);
        }
    }
    println!("{} gradients: mean relative error {:.2e}, worst {:.2e} ({})", names.len(), sum / names.len() as f32, worst.0, worst.1);
    let tol = tolerance(5e-3, 5e-2);
    assert!(worst.0 < tol, "gradient of {} off by {} (tolerance {tol})", worst.1, worst.0);
}

#[test]
#[ignore]
fn adamw_trajectory_matches_pytorch() {
    let Some((model, reference)) = load(device()) else { return };
    let batch = batch(&reference);
    let shapes: Vec<(&str, Vec<usize>)> = batch.iter().map(|(n, a)| (*n, a.shape().to_vec())).collect();
    set_init(&model, &reference);
    let opts = TrainOptions { lr: 1e-3, ..Default::default() };
    let mut trainer = model.trainer(&shapes, opts).unwrap();
    let views: Vec<_> = batch.iter().map(|(n, a)| (*n, a.view())).collect();
    let expected = read_npy(&reference.join("trajectory/loss.npy"));
    for (i, e) in expected.iter().enumerate() {
        let s = trainer.step(&views).unwrap();
        println!("step {i}: loss {} (pytorch {e})", s.loss);
        assert!(s.optimized);
        assert!((s.loss - e).abs() < 5e-2 * e.abs().max(1.0), "step {i}: {} vs {e}", s.loss);
    }
    let mut worst = 0f32;
    for (name, value) in trainer.params_values().unwrap() {
        worst = worst.max(rel_err(&value, &read_npy(&reference.join(format!("trajectory/{name}.npy")))));
    }
    println!("parameters after {} steps: worst relative error {worst:.2e}", expected.len());
    assert!(worst < tolerance(2e-2, 8e-2), "parameters off by {worst}");
}

#[test]
#[ignore]
fn adapter_round_trip() {
    let Some((model, reference)) = load(device()) else { return };
    let batch = batch(&reference);
    let shapes: Vec<(&str, Vec<usize>)> = batch.iter().map(|(n, a)| (*n, a.shape().to_vec())).collect();
    set_init(&model, &reference);
    let mut trainer = model.trainer(&shapes, TrainOptions { lr: 1e-3, ..Default::default() }).unwrap();
    let views: Vec<_> = batch.iter().map(|(n, a)| (*n, a.view())).collect();
    for _ in 0..2 {
        trainer.step(&views).unwrap();
    }
    let dir = std::env::temp_dir().join("tl-adapter-test");
    let format = AdapterFormat::default();
    model.save_adapter(&dir, &format).unwrap();
    println!("{}", std::fs::read_to_string(dir.join("adapter_config.json")).unwrap());

    // A fresh model (B = 0) gets the trained adapter back, bit for bit.
    let (fresh, _) = load(device()).unwrap();
    let loaded = fresh.load_adapter(&dir, &format).unwrap();
    assert_eq!(loaded, model.params().len());
    for (name, _) in model.params() {
        assert_eq!(model.state(&name).unwrap(), fresh.state(&name).unwrap(), "{name}");
    }
    // ... and a wrong prefix is an error, not a silent no-op.
    let wrong = AdapterFormat { key_prefix: "x.".into(), ..format };
    assert!(fresh.load_adapter(&dir, &wrong).is_err());
}
