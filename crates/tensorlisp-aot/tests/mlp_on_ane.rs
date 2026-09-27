#![cfg(target_vendor = "apple")]

//! End-to-end proof of the AOT pipeline: lowers tensorlisp's own `mlp` test
//! fixture (see `crates/tensorlisp/tests/mlp.rs`) into a MIL program,
//! serializes it into a Core ML model, runs it through
//! `coreml_rs::runtime::Model` (Neural Engine included), and checks the
//! result against the same plain-ndarray reference forward pass the CPU
//! test checks against -- so a wrong lowering (bad transpose, wrong
//! broadcast, wrong output node) shows up as a numeric mismatch, not a
//! silent success.

use std::path::PathBuf;

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use ndarray::{Array1, Array2};
use tensorlisp::gguf::GgufWriter;
use tensorlisp::{Device, Model, Program as TlProgram, Taps};

const MLP: &str = r#"
(define (linear x name)
  (ggml-add (ggml-mul-mat (weight (string-append name ".weight")) x)
            (weight (string-append name ".bias"))))

(model (inputs [x f32 (4 batch)])
  (define h (ggml-relu (linear x "fc1")))
  (outputs [logits (linear h "fc2") 2]))
"#;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

struct Weights {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl Weights {
    fn new() -> Self {
        Weights {
            w1: Array2::from_shape_vec((8, 4), values(32, 0.1)).unwrap(),
            b1: Array1::from_vec(values(8, 0.2)),
            w2: Array2::from_shape_vec((3, 8), values(24, 0.3)).unwrap(),
            b2: Array1::from_vec(values(3, 0.4)),
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = (x.dot(&self.w1.t()) + &self.b1).mapv(|v| v.max(0.0));
        h.dot(&self.w2.t()) + &self.b2
    }
}

fn write_model(weights: &Weights) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&TlProgram::Text(MLP.into())).unwrap();
    w.add_f32("fc1.weight", weights.w1.view().into_dyn()).unwrap();
    w.add_f32("fc1.bias", weights.b1.view().into_dyn()).unwrap();
    w.add_f32("fc2.weight", weights.w2.view().into_dyn()).unwrap();
    w.add_f32("fc2.bias", weights.b2.view().into_dyn()).unwrap();
    let path = std::env::temp_dir().join(format!("tensorlisp-aot-test-{}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

#[test]
fn mlp_matches_reference_on_ane() {
    let weights = Weights::new();
    let path = write_model(&weights);

    // The ggml reference graph for a fixed batch of 2 -- ground truth for
    // both the op sequence and every node's computed shape.
    let batch = 2usize;
    let tl_model = Model::load(&path, Device::Cpu).unwrap();
    let info = tl_model.graph(&[("x", vec![batch, 4])], &Taps::default()).unwrap();

    let (func, output_names) = tensorlisp_aot::lower_graph(&path, &info, &[("x", vec![batch as u64, 4])], Opset::Ios17)
        .expect("lower ggml graph to MIL");

    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    let spec = program.to_model("main").unwrap();

    let loaded = CoreMlModel::load(
        &spec,
        &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() },
    )
    .unwrap_or_else(|e| panic!("failed to load MIL model on the ANE: {e}\n{spec:#?}"));

    let x = Array2::from_shape_vec((batch, 4), values(batch * 4, 1.0)).unwrap();
    let x_flat: Vec<f32> = x.iter().copied().collect();
    let outputs = loaded
        .predict(&[("x", ArrayRef::new(&[batch, 4], &x_flat).unwrap())])
        .unwrap();

    let expected = weights.forward(&x);
    let actual = &outputs[&output_names["logits"]];
    let actual_f32 = actual.to_f32();
    assert_eq!(actual_f32.len(), expected.len());
    for (a, e) in actual_f32.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3, "mismatch: got {actual_f32:?}, want {expected:?}");
    }

    std::fs::remove_file(path).ok();
}
