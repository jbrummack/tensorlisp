#![cfg(target_vendor = "apple")]

//! Checks PP-OCRv6 (`ports/ppocrv6/ppocrv6.ss`, entries `det` and `rec`)
//! against real taps, one slice at a time, same staged approach as
//! `tests/yolo11n_on_ane.rs`. Exercises PAD, MEAN and HARDSIGMOID for the
//! first time (NORM/GELU_ERF are already proven by
//! `tests/tipsv2_vision_on_ane.rs`).
use std::path::PathBuf;

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use tensorlisp::{Device, Model, RunOptions, Taps};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/ppocrv6/ppocrv6.gguf")
}

fn deterministic_image(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.0129).sin() * 0.5 + 0.5).collect()
}

/// `|mil - cpu| <= atol + rtol * |cpu|` (numpy's own `allclose` formula) --
/// every model is now lowered at fp16 (see `lib.rs`'s own `DTYPE`), whose
/// relative precision is roughly constant regardless of magnitude, unlike
/// a fixed absolute bound.
#[track_caller]
fn assert_close(label: &str, mil: &[f32], cpu: &[f32]) {
    assert_eq!(mil.len(), cpu.len(), "{label}: mil {} vs cpu {}", mil.len(), cpu.len());
    const ATOL: f32 = 0.25;
    const RTOL: f32 = 0.1;
    let max_diff = mil.iter().zip(cpu.iter()).map(|(a, b)| (a - b).abs() - RTOL * b.abs()).fold(f32::MIN, f32::max);
    assert!(max_diff <= ATOL, "{label} mismatch, max (|mil-cpu| - {RTOL}*|cpu|) = {max_diff} (atol {ATOL})");
}

/// Lowers `entry`'s graph up to `tap`, runs it on the ANE, runs the same
/// tap on the CPU, and asserts they match within `1e-2`.
fn assert_tap_matches_cpu(entry: &str, tap: &str, input_shape: &[usize]) {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }
    let tl_model = Model::load(&path, Device::Cpu).unwrap();
    let input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input_shape), deterministic_image(input_shape.iter().product())).unwrap();

    let info = tl_model.graph_entry(entry, &[("image", input_shape.to_vec())], &Taps::Names(vec![tap.to_string()])).unwrap();
    let inputs_u64: Vec<(&str, Vec<u64>)> = vec![("image", input_shape.iter().map(|&d| d as u64).collect())];
    let (func, output_names) =
        tensorlisp_aot::lower_until(&path, &info, &inputs_u64, Opset::Ios17, tap).unwrap_or_else(|e| panic!("lower up to {tap}: {e}"));

    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    let spec = program.to_model("main").unwrap();
    let loaded = CoreMlModel::load(&spec, &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() })
        .unwrap_or_else(|e| panic!("failed to load MIL model: {e}"));

    let image_flat: Vec<f32> = input.iter().copied().collect();
    let mil_out = loaded.predict(&[("image", ArrayRef::new(input_shape, &image_flat).unwrap())]).unwrap();
    let mil_tap = mil_out[&output_names[tap]].to_f32();

    let cpu_out = tl_model.run_entry(entry, &[("image", input.view())], &RunOptions { taps: Taps::Names(vec![tap.to_string()]) }).unwrap();
    let cpu_tap: Vec<f32> = cpu_out.taps.iter().find(|(n, _)| n == tap).unwrap().1.iter().copied().collect();

    assert_close(tap, &mil_tap, &cpu_tap);
}

/// 128 x 128 (a multiple of 32, small for a fast test).
const DET_SIZE: [usize; 4] = [1, 3, 128, 128];

/// det.stage1 is the LCNetV4 backbone's stem + first stage: exercises
/// `stem`'s `PAD` (`F.pad (0,1,0,1)`) and, if this stage has SE blocks,
/// `squeeze-excite`'s `MEAN`/`HARDSIGMOID`.
#[test]
fn det_stage1_matches_cpu() {
    assert_tap_matches_cpu("det", "det.stage1", &DET_SIZE);
}

#[test]
fn det_stage4_matches_cpu() {
    assert_tap_matches_cpu("det", "det.stage4", &DET_SIZE);
}

/// det.neck: RepLKFPN (intraclass multi-scale convs, top-down/bottom-up
/// sums, upsample, concat) -- no new ops beyond the backbone's, but the
/// first real test of this much of the graph together.
#[test]
fn det_neck_matches_cpu() {
    assert_tap_matches_cpu("det", "det.neck", &DET_SIZE);
}

/// 48 tall (fixed), 160 wide -- narrower than `rec-min-width` (320), so
/// the `rec` model's own conditional zero-pad-to-320 (`ggml-pad`) is
/// actually exercised, not skipped.
const REC_SIZE: [usize; 4] = [1, 3, 48, 160];

#[test]
fn rec_backbone_matches_cpu() {
    assert_tap_matches_cpu("rec", "rec.backbone", &REC_SIZE);
}

#[test]
fn rec_pooled_matches_cpu() {
    assert_tap_matches_cpu("rec", "rec.pooled", &REC_SIZE);
}

/// rec.encoder is the SVTR transformer stack (`attn:multi-head`, no
/// `'flash`) on top of the CNN backbone -- the first real test of
/// attention in this port.
#[test]
fn rec_encoder_matches_cpu() {
    assert_tap_matches_cpu("rec", "rec.encoder", &REC_SIZE);
}

fn assert_full_model_matches_cpu(entry: &str, output_name: &str, input_shape: &[usize]) {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }
    let tl_model = Model::load(&path, Device::Cpu).unwrap();
    let input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(input_shape), deterministic_image(input_shape.iter().product())).unwrap();

    let info = tl_model.graph_entry(entry, &[("image", input_shape.to_vec())], &Taps::default()).unwrap();
    let inputs_u64: Vec<(&str, Vec<u64>)> = vec![("image", input_shape.iter().map(|&d| d as u64).collect())];
    let (func, output_names) = tensorlisp_aot::lower_graph(&path, &info, &inputs_u64, Opset::Ios17).unwrap_or_else(|e| panic!("lower full graph: {e}"));

    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    let spec = program.to_model("main").unwrap();
    let loaded = CoreMlModel::load(&spec, &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() })
        .unwrap_or_else(|e| panic!("failed to load MIL model: {e}"));

    let image_flat: Vec<f32> = input.iter().copied().collect();
    let mil_out = loaded.predict(&[("image", ArrayRef::new(input_shape, &image_flat).unwrap())]).unwrap();
    let mil = mil_out[&output_names[output_name]].to_f32();

    let cpu_out = tl_model.run_entry(entry, &[("image", input.view())], &RunOptions::default()).unwrap();
    let cpu: Vec<f32> = cpu_out.outputs.iter().find(|(n, _)| n == output_name).unwrap().1.iter().copied().collect();

    assert_close(output_name, &mil, &cpu);
}

#[test]
fn det_full_model_matches_cpu() {
    assert_full_model_matches_cpu("det", "prob", &DET_SIZE);
}

#[test]
fn rec_full_model_matches_cpu() {
    assert_full_model_matches_cpu("rec", "probs", &REC_SIZE);
}
