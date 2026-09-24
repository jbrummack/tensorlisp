//! End-to-end: safetensors -> convert -> check -> compare -> pack -> quantize -> run.
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

use ndarray::{Array1, Array2, ArrayD};
use safetensors::{Dtype, tensor::TensorView};
use serde_json::Value;

#[path = "../src/npy.rs"]
#[allow(dead_code)]
mod npy;

const PROGRAM: &str = r#"
(define (linear x name)
  (ggml-add (ggml-mul-mat (weight (string-append name ".weight")) x)
            (weight (string-append name ".bias"))))

(model (inputs [x f32 (32 batch)])
  (define h (tap "hidden" (ggml-relu (linear x "fc1")) 2))
  (outputs [logits (linear h "fc2") 2]))
"#;

fn tl(dir: &Path, args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_tl")).args(args).current_dir(dir).output().unwrap();
    eprintln!("tl {}\n{}{}", args.join(" "), String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    out
}

fn tl_json(dir: &Path, args: &[&str]) -> (i32, Value) {
    let mut all = vec!["--json"];
    all.extend_from_slice(args);
    let out = tl(dir, &all);
    (out.status.code().unwrap(), serde_json::from_slice(&out.stdout).unwrap())
}

/// The entry of a JSON array whose `key` field is `value`.
fn find<'a>(list: &'a Value, key: &str, value: &str) -> &'a Value {
    list.as_array().unwrap().iter().find(|t| t[key] == value).unwrap_or_else(|| panic!("no {value} in {list}"))
}

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

fn setup() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tl-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("ref")).unwrap();

    let w1 = Array2::from_shape_vec((64, 32), values(64 * 32, 0.1)).unwrap();
    let b1 = Array1::from_vec(values(64, 0.2));
    let w2 = Array2::from_shape_vec((10, 64), values(640, 0.3)).unwrap() * 0.1;
    let b2 = Array1::from_vec(values(10, 0.4));

    // fc1.weight as bf16, the rest f32, all under a "model." prefix.
    let bf16: Vec<u8> = w1.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
    let f32_bytes = |a: &[f32]| a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let (b1b, w2b, b2b) = (f32_bytes(b1.as_slice().unwrap()), f32_bytes(w2.as_slice().unwrap()), f32_bytes(b2.as_slice().unwrap()));
    let tensors = vec![
        ("model.fc1.weight", TensorView::new(Dtype::BF16, vec![64, 32], &bf16).unwrap()),
        ("model.fc1.bias", TensorView::new(Dtype::F32, vec![64], &b1b).unwrap()),
        ("model.fc2.weight", TensorView::new(Dtype::F32, vec![10, 64], &w2b).unwrap()),
        ("model.fc2.bias", TensorView::new(Dtype::F32, vec![10], &b2b).unwrap()),
    ];
    std::fs::write(dir.join("mlp.safetensors"), safetensors::serialize(tensors, None).unwrap()).unwrap();
    std::fs::write(dir.join("net.ss"), PROGRAM).unwrap();

    // Reference with the bf16-rounded weights, like PyTorch would compute it.
    let w1 = w1.mapv(|v| half::bf16::from_f32(v).to_f32());
    let x = Array2::from_shape_vec((4, 32), values(128, 1.0)).unwrap();
    let hidden = (x.dot(&w1.t()) + &b1).mapv(|v| v.max(0.0));
    let logits = hidden.dot(&w2.t()) + &b2;
    npy::write(&dir.join("x.npy"), &x.into_dyn()).unwrap();
    npy::write(&dir.join("ref/hidden.npy"), &hidden.into_dyn()).unwrap();
    npy::write(&dir.join("ref/logits.npy"), &logits.into_dyn()).unwrap();
    dir
}

#[test]
fn porting_workflow() {
    let dir = setup();

    // convert: strip the prefix, keep bf16.
    let (code, v) = tl_json(&dir, &["convert", "mlp.safetensors", "--strip-prefix", "model.", "-o", "weights.gguf"]);
    assert_eq!(code, 0);
    let fc1 = find(&v["tensors"], "name", "fc1.weight");
    assert_eq!(fc1["type"], "bf16");
    assert_eq!(fc1["source_name"], "model.fc1.weight");

    // inspect: no program yet, shapes in numpy order.
    let (_, v) = tl_json(&dir, &["inspect", "weights.gguf"]);
    assert!(v["program"]["error"].as_str().unwrap().contains("TL_VER"));
    assert_eq!(find(&v["tensors"], "name", "fc1.weight")["shape"], serde_json::json!([64, 32]));

    // check with an external program.
    let (code, v) = tl_json(&dir, &["check", "weights.gguf", "--program", "net.ss", "-i", "x=4,32"]);
    assert_eq!(code, 0);
    assert_eq!(v["outputs"][0]["shape"], serde_json::json!([4, 10]));
    assert_eq!(v["taps"], serde_json::json!(["hidden"]));
    assert_eq!(v["nodes"].as_array().unwrap().len(), 5);

    // compare against the references. ggml's bf16 matmul also rounds the
    // activations to bf16, hence bf16-sized tolerances.
    let args = [
        "compare", "weights.gguf", "--program", "net.ss", "--device", "cpu", "-i", "x=x.npy", "--reference-dir", "ref",
        "--atol", "0.01", "--rtol", "0.01",
    ];
    let (code, v) = tl_json(&dir, &args);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["comparisons"][0]["name"], "hidden");
    assert_eq!(v["comparisons"][1]["name"], "logits");

    // A broken port (relu missing) fails first at the tap.
    std::fs::write(dir.join("buggy.ss"), PROGRAM.replace("(ggml-relu (linear x \"fc1\"))", "(linear x \"fc1\")")).unwrap();
    let args = [
        "compare", "weights.gguf", "--program", "buggy.ss", "--device", "cpu", "-i", "x=x.npy", "--reference-dir", "ref",
        "--atol", "0.01", "--rtol", "0.01",
    ];
    let (code, v) = tl_json(&dir, &args);
    assert_eq!(code, 2);
    assert_eq!(v["first_failure"], "tap hidden");

    // Program errors are reported, not crashes.
    std::fs::write(dir.join("bad-shape.ss"), "(model (inputs [x f32]) (outputs [y (ggml-add x (weight \"fc2.bias\"))]))").unwrap();
    let (code, v) = tl_json(&dir, &["check", "weights.gguf", "--program", "bad-shape.ss", "-i", "x=4,32"]);
    assert_eq!(code, 1);
    assert!(v["error"].as_str().unwrap().contains("GGML_ASSERT"), "{v}");

    // pack, then the model runs without --program.
    let (code, _) = tl_json(&dir, &["pack", "weights.gguf", "--program", "net.ss", "-o", "mlp.gguf"]);
    assert_eq!(code, 0);
    let (code, v) = tl_json(&dir, &["run", "mlp.gguf", "--device", "cpu", "-i", "x=x.npy", "-o", "out", "--taps", "all"]);
    assert_eq!(code, 0);
    assert_eq!(v["outputs"][0]["shape"], serde_json::json!([4, 10]));
    let logits: ArrayD<f32> = npy::read(&dir.join("out/logits.npy")).unwrap();
    let reference: ArrayD<f32> = npy::read(&dir.join("ref/logits.npy")).unwrap();
    assert!(logits.iter().zip(reference.iter()).all(|(a, b)| (a - b).abs() < 1e-2));
    assert!(dir.join("out/hidden.npy").exists());

    // quantize keeps the program; results stay close.
    let (code, v) = tl_json(&dir, &["quantize", "mlp.gguf", "mlp-q8.gguf", "-t", "q8_0", "--min-elements", "0"]);
    assert_eq!(code, 0);
    assert_eq!(find(&v["tensors"], "name", "fc1.weight")["to"], "q8_0");
    assert_eq!(find(&v["tensors"], "name", "fc1.bias")["kept_because"], "1-D (use --include-1d)");
    let (code, v) = tl_json(&dir, &["compare", "mlp-q8.gguf", "--device", "cpu", "-i", "x=x.npy", "-r", "logits=ref/logits.npy", "--atol", "0.1", "--rtol", "0.1"]);
    assert_eq!(code, 0, "{v}");
    assert!(v["comparisons"][0]["cosine_similarity"].as_f64().unwrap() > 0.999);

    std::fs::remove_dir_all(dir).unwrap();
}
