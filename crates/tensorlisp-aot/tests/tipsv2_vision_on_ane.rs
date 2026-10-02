#![cfg(target_vendor = "apple")]

//! Checks the TIPSv2 vision encoder (`ports/tipsv2/vision.ss`) against real
//! taps, one slice at a time, same staged approach as
//! `tests/yolo11n_on_ane.rs`. Exercises NORM, GELU_ERF and IM2COL for the
//! first time.
//!
//! The port's own `flash-attention` toggle (top of `vision.ss`) defaults to
//! `#t` (ggml's fused, f16-K/V attention kernel) -- not device-gated the
//! way `nn:conv2d`'s method choice is, so graphing on `Device::Cpu` doesn't
//! avoid it, and there's no MIL/ANE equivalent lowered for
//! `GGML_OP_FLASH_ATTN_EXT`. The port's own README says `#f` gives the
//! exact PyTorch-comparable path on the CPU, so this test repacks the real
//! vision weights with that one line flipped (into a temp file, at test
//! time -- no new file is committed) rather than testing a config the port
//! doesn't itself vouch for as equivalent.
use std::path::{Path, PathBuf};

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use tensorlisp::gguf::{GgufFile, GgufWriter};
use tensorlisp::{Device, Model, Program as TlProgram, Taps};

const VISION_SS: &str = include_str!("../../../ports/tipsv2/vision.ss");

fn weights_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tipsv2-b14/tipsv2-b14-vision-f32.gguf")
}

/// The real vision weights, repacked with `flash-attention` forced to `#f`
/// -- same tensors, byte for byte, just a different program (see the
/// module doc comment for why).
fn model_path() -> Option<PathBuf> {
    let src = weights_path();
    if !src.exists() {
        return None;
    }
    let source = VISION_SS.replace("(define flash-attention #t)", "(define flash-attention #f)");
    assert_ne!(source, VISION_SS, "flash-attention toggle line not found -- did vision.ss change?");
    // `(outputs ...)` entries aren't ggml-named the way `tap` calls are (see
    // `full_model_matches_cpu`'s own doc comment for why that matters), so
    // `lower_graph`'s `find_named_node` can't locate any of these three by
    // name -- wrap each in its own uniquely-named tap so it's addressable
    // the same way every other test in this file already tests a tap.
    let source = source.replace(
        "(outputs [cls (token 0) 2]\n           [register (token 1) 2]\n           [patches (tensor:slice final 1 2 (* gw gh)) 3]))",
        "(outputs [cls (tap \"out.cls\" (token 0) 2) 2]\n           [register (tap \"out.register\" (token 1) 2) 2]\n           [patches (tap \"out.patches\" (tensor:slice final 1 2 (* gw gh)) 3) 3]))",
    );
    assert_ne!(source.find("out.cls"), None, "outputs clause not found -- did vision.ss change?");

    let file = GgufFile::open(&src).unwrap();
    let mut w = GgufWriter::new();
    w.copy_metadata(&file);
    w.set_program(&TlProgram::Text(source)).unwrap();
    let (file_ref, src_ref) = (&file, &src);
    for (idx, info) in file.tensor_infos().into_iter().enumerate() {
        let idx = idx as i64;
        w.add_tensor_with(&info.name, info.dtype, &info.shape, move || file_ref.read_tensor_bytes(src_ref, idx)).unwrap();
    }
    // Unique per test *thread*, not just per process: cargo test runs
    // `#[test]` functions from the same binary concurrently, and every one
    // of them calls `model_path()` -- a name keyed on PID alone would let
    // two tests race on the same file (one rewriting or deleting it out
    // from under another still reading it).
    let out = std::env::temp_dir().join(format!("tensorlisp-tipsv2-vision-aot-{}-{:?}.gguf", std::process::id(), std::thread::current().id()));
    w.write(&out).unwrap();
    Some(out)
}

fn deterministic_image(size: usize) -> Vec<f32> {
    (0..3 * size * size).map(|i| (i as f32 * 0.0091).sin() * 0.5 + 0.5).collect()
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

/// 448 (32 x 32 patches) is the checkpoint's native grid -- any other
/// multiple of 14 would also need `vision:resize-positions`'s bilinear
/// antialiased UPSCALE, which isn't lowered.
const SIZE: usize = 448;

fn assert_tap_matches_cpu(path: &Path, tap: &str) {
    let tl_model = Model::load(path, Device::Cpu).unwrap();
    let input = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, SIZE, SIZE]), deterministic_image(SIZE)).unwrap();

    let info = tl_model.graph(&[("image", vec![1, 3, SIZE, SIZE])], &Taps::Names(vec![tap.to_string()])).unwrap();
    let (func, output_names) = tensorlisp_aot::lower_until(path, &info, &[("image", vec![1, 3, SIZE as u64, SIZE as u64])], Opset::Ios17, tap)
        .unwrap_or_else(|e| panic!("lower up to {tap}: {e}"));

    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    let spec = program.to_model("main").unwrap();
    let loaded = CoreMlModel::load(&spec, &LoadOptions { compute_units: ComputeUnits::CpuAndNeuralEngine, ..Default::default() })
        .unwrap_or_else(|e| panic!("failed to load MIL model: {e}"));

    let image_flat: Vec<f32> = input.iter().copied().collect();
    let mil_out = loaded.predict(&[("image", ArrayRef::new(&[1, 3, SIZE, SIZE], &image_flat).unwrap())]).unwrap();
    let mil_tap = mil_out[&output_names[tap]].to_f32();

    let cpu_out = tl_model.run_with(&[("image", input.view())], &tensorlisp::RunOptions { taps: Taps::Names(vec![tap.to_string()]) }).unwrap();
    let cpu_tap: Vec<f32> = cpu_out.taps.iter().find(|(n, _)| n == tap).unwrap().1.iter().copied().collect();

    assert_close(tap, &mil_tap, &cpu_tap);
}

#[test]
fn embed_matches_cpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: {:?} not present", weights_path());
        return;
    };
    assert_tap_matches_cpu(&path, "embed");
    std::fs::remove_file(path).ok();
}

#[test]
fn block_00_matches_cpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: {:?} not present", weights_path());
        return;
    };
    assert_tap_matches_cpu(&path, "block.00");
    std::fs::remove_file(path).ok();
}

#[test]
fn block_01_matches_cpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: {:?} not present", weights_path());
        return;
    };
    assert_tap_matches_cpu(&path, "block.01");
    std::fs::remove_file(path).ok();
}

#[test]
fn block_11_matches_cpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: {:?} not present", weights_path());
        return;
    };
    assert_tap_matches_cpu(&path, "block.11");
    std::fs::remove_file(path).ok();
}

/// The complete encoder, ANE-compiled end to end: every block, the final
/// norm, and all three declared outputs (`cls`, `register`, `patches`),
/// each checked via its own tap (`out.cls`/`out.register`/`out.patches`,
/// added by `model_path`'s own source patch). Plain `(outputs ...)`
/// entries aren't ggml-named the way a `tap` call names its own result, so
/// `find_named_node` can't address them directly (this model has three
/// outputs, so `lower_graph`'s one-output "just use the last node"
/// fallback doesn't apply either) -- and separately, the tapped `"norm"`
/// tensor itself isn't usable as a stand-in: `find_named_node`'s
/// "most-wrapped node with this name" heuristic (see its own doc comment)
/// picks the *wrong* node for it, since three different expressions here
/// (`token(0)`, `token(1)`, the patches slice) all further wrap that same
/// tensor and so all get ggml names starting with `"norm ("` -- harmless
/// for every other tap in this file (none of theirs are reused more than
/// once), but why a plain `"norm"` tap isn't a meaningful check on its own.
#[test]
fn full_model_matches_cpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: {:?} not present", weights_path());
        return;
    };
    for tap in ["out.cls", "out.register", "out.patches"] {
        assert_tap_matches_cpu(&path, tap);
    }
    std::fs::remove_file(path).ok();
}
