#![cfg(target_vendor = "apple")]

//! Proves `POOL_2D` and `UPSCALE` against real CPU values without needing
//! yolo11n's attention block (`c2psa`) implemented first: both of
//! yolo11n's real usages of these ops (SPPF's max pools, the FPN's nearest
//! upsamples) sit downstream of `c2psa` in the graph, which needs
//! `TRANSPOSE`/`PERMUTE`/`CONT` -- not yet implemented. These are small,
//! synthetic, single-output models (same proof shape as `mlp_on_ane.rs`)
//! exercising exactly one op each against tensorlisp's own real CPU run, no
//! yolo11n weights required.

use std::path::PathBuf;

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use tensorlisp::gguf::GgufWriter;
use tensorlisp::{Device, Model, Program as TlProgram, Taps};

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

fn write_model(program: &str) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&TlProgram::Text(program.into())).unwrap();
    let path = std::env::temp_dir().join(format!("tensorlisp-aot-test-{}-{}.gguf", std::process::id(), fastrand_suffix()));
    w.write(&path).unwrap();
    path
}

// Cheap unique-ish suffix so the two tests in this file (run in parallel by
// default) don't race on the same temp path.
fn fastrand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64
}

fn assert_single_output_matches_cpu(program: &str, shape_whcn: [usize; 4]) {
    let path = write_model(program);
    let [w, h, c, n] = shape_whcn;

    let tl_model = Model::load(&path, Device::Cpu).unwrap();
    let input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, c, h, w]), values(n * c * h * w, 1.0)).unwrap();

    let info = tl_model.graph(&[("x", vec![n, c, h, w])], &Taps::default()).unwrap();
    let (func, output_names) =
        tensorlisp_aot::lower_graph(&path, &info, &[("x", vec![n as u64, c as u64, h as u64, w as u64])], Opset::Ios17)
            .expect("lower ggml graph to MIL");

    let mut mil_program = Program::new();
    mil_program.add_function("main", func).unwrap();
    let spec = mil_program.to_model("main").unwrap();
    let loaded = CoreMlModel::load(
        &spec,
        &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() },
    )
    .unwrap_or_else(|e| panic!("failed to load MIL model on the ANE: {e}\n{spec:#?}"));

    let x_flat: Vec<f32> = input.iter().copied().collect();
    let outputs = loaded.predict(&[("x", ArrayRef::new(&[n, c, h, w], &x_flat).unwrap())]).unwrap();
    let mil_y = outputs[&output_names["y"]].to_f32();

    let cpu_out = tl_model.run(&[("x", input.view())]).unwrap();
    let cpu_y: Vec<f32> = cpu_out["y"].iter().copied().collect();

    assert_eq!(mil_y.len(), cpu_y.len());
    let max_diff = mil_y.iter().zip(cpu_y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "mismatch, max abs diff {max_diff}");

    std::fs::remove_file(path).ok();
}

#[test]
fn max_pool_matches_cpu() {
    // Same shape as SPPF's pools: 5x5 kernel, stride 1, "same" padding 2.
    let program = r#"
      (import (tl nn))
      (model (inputs [x f32 (8 8 3 batch)])
        (outputs [y (nn:max-pool x 5 'stride 1 'padding 2) 4]))
    "#;
    assert_single_output_matches_cpu(program, [8, 8, 3, 2]);
}

#[test]
fn upsample_nearest_matches_cpu() {
    // Same as yolo11n's FPN upsamples: nearest-neighbor, factor 2.
    let program = r#"
      (import (tl nn))
      (model (inputs [x f32 (8 8 3 batch)])
        (outputs [y (nn:upsample-nearest x 2) 4]))
    "#;
    assert_single_output_matches_cpu(program, [8, 8, 3, 2]);
}
