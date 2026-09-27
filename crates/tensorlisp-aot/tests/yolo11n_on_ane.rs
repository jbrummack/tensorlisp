#![cfg(target_vendor = "apple")]

//! Checks new op lowerings against real yolo11n taps one slice at a time,
//! per the staged plan: prove each newly-supported op against a real CPU
//! value before trusting it in the whole model.

use std::path::PathBuf;

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use tensorlisp::{Device, Model, RunOptions, Taps};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/yolo11n/yolo11n.gguf")
}

fn deterministic_image(size: usize) -> Vec<f32> {
    (0..3 * size * size).map(|i| (i as f32 * 0.0137).sin() * 0.5 + 0.5).collect()
}

/// Lowers the graph up to `tap`, runs it on the ANE, runs the same tap on
/// the CPU, and asserts they match within `1e-3`. Shared by every
/// per-tap proof test in this file (see `layer_00_conv_silu_matches_cpu` for
/// the original, unfactored version this was pulled from).
fn assert_tap_matches_cpu(tap: &str, size: usize) {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }
    let tl_model = Model::load(&path, Device::Cpu).unwrap();
    let input = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 3, size, size]),
        deterministic_image(size),
    )
    .unwrap();

    let info = tl_model.graph(&[("image", vec![1, 3, size, size])], &Taps::Names(vec![tap.to_string()])).unwrap();

    let (func, output_names) = tensorlisp_aot::lower_until(
        &path,
        &info,
        &[("image", vec![1, 3, size as u64, size as u64])],
        Opset::Ios17,
        tap,
    )
    .unwrap_or_else(|e| panic!("lower up to {tap}: {e}"));

    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    let spec = program.to_model("main").unwrap();
    let loaded = CoreMlModel::load(
        &spec,
        &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() },
    )
    .unwrap_or_else(|e| panic!("failed to load MIL model: {e}"));

    let image_flat: Vec<f32> = input.iter().copied().collect();
    let mil_out = loaded
        .predict(&[("image", ArrayRef::new(&[1, 3, size, size], &image_flat).unwrap())])
        .unwrap();
    let mil_tap = mil_out[&output_names[tap]].to_f32();

    let cpu_out = tl_model.run_with(&[("image", input.view())], &RunOptions { taps: Taps::Names(vec![tap.to_string()]) }).unwrap();
    let cpu_tap: Vec<f32> = cpu_out.taps.iter().find(|(n, _)| n == tap).unwrap().1.iter().copied().collect();

    assert_eq!(mil_tap.len(), cpu_tap.len());
    let max_diff = mil_tap.iter().zip(cpu_tap.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "{tap} mismatch, max abs diff {max_diff}");
}

#[test]
fn layer_00_conv_silu_matches_cpu() {
    assert_tap_matches_cpu("layer.00", 64); // multiple of 32, small for a fast test
}

/// layer.02 is yolo11n's first C3k2 block: channel-split (`VIEW`) into two
/// halves, a chain of Bottlenecks on one half, then `concat-channels`
/// (`CONCAT`) of every intermediate before the final conv. Exercises `VIEW`
/// and `CONCAT` end to end against real weights for the first time.
#[test]
fn layer_02_c3k2_matches_cpu() {
    assert_tap_matches_cpu("layer.02", 64);
}

/// layer.09 is SPPF: cv1, three chained 5x5 stride-1 "same"-padded max
/// pools, concat all four, cv2. Exercises `POOL_2D` for the first time.
#[test]
fn layer_09_sppf_matches_cpu() {
    assert_tap_matches_cpu("layer.09", 64);
}

// layer.11 (the first FPN upsample) sits downstream of layer.10 (`c2psa`,
// which needs TRANSPOSE/PERMUTE/CONT -- not implemented yet), so `UPSCALE`
// is proven separately, on a small synthetic model with no attention
// dependency: see `tests/pool_upsample_on_ane.rs::upsample_nearest_matches_cpu`.
