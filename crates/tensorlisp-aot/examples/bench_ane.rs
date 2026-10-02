//! Ad hoc latency benchmark for the AOT/CoreML path, at the same input
//! sizes the CPU/Metal numbers in each port's own README were measured at
//! (`ports/yolo11/README.md`, `ports/tipsv2/README.md`,
//! `ports/ppocrv6/README.md`), so the numbers this prints are directly
//! comparable to those tables. Not a permanent test (no assertions, no
//! correctness check -- see `tests/*_on_ane.rs` for that): a one-off "how
//! fast is this on the ANE" report.
//!
//! Run with `cargo run -p tensorlisp-aot --release --example bench_ane`.
use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_rs::mil::{Opset, Program};
use coreml_rs::runtime::{ArrayRef, ComputeUnits, LoadOptions, Model as CoreMlModel};
use tensorlisp::gguf::{GgufFile, GgufWriter};
use tensorlisp::{Device, Model, Program as TlProgram, Taps};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn deterministic(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.0129 + seed).sin() * 0.5 + 0.5).collect()
}

/// Loads `spec` under each of the three compute-unit settings, times
/// `iters` `predict()` calls after one warmup call each, and prints the
/// mean.
fn bench(label: &str, spec: &coreml_rs::proto::specification::Model, inputs: &[(&str, ArrayRef)], iters: u32) {
    for (name, units) in [("CPU", ComputeUnits::CpuOnly), ("GPU", ComputeUnits::CpuAndGpu), ("ANE", ComputeUnits::CpuAndNeuralEngine)] {
        let load_start = Instant::now();
        let loaded = match CoreMlModel::load(spec, &LoadOptions { compute_units: units, ..Default::default() }) {
            Ok(m) => m,
            Err(e) => {
                println!("{label} [{name}]: failed to load: {e}");
                continue;
            }
        };
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

        // Warmup (compiles kernels, first-call overhead).
        if let Err(e) = loaded.predict(inputs) {
            println!("{label} [{name}]: failed to run: {e}");
            continue;
        }

        let start = Instant::now();
        for _ in 0..iters {
            loaded.predict(inputs).unwrap();
        }
        let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("{label} [{name}]: load {load_ms:.1} ms, mean of {iters}: {mean_ms:.1} ms");
    }
}

fn build_spec(path: &Path, entry: &str, inputs: &[(&str, Vec<u64>)]) -> coreml_rs::proto::specification::Model {
    let tl_model = Model::load(path, Device::Cpu).unwrap();
    let input_shapes: Vec<(&str, Vec<usize>)> = inputs.iter().map(|(n, s)| (*n, s.iter().map(|&d| d as usize).collect())).collect();
    let info = tl_model.graph_entry(entry, &input_shapes, &Taps::default()).unwrap();
    let (func, _) = tensorlisp_aot::lower_graph(path, &info, inputs, Opset::Ios17).unwrap_or_else(|e| panic!("lower {entry}: {e}"));
    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    program.to_model("main").unwrap()
}

/// Like `build_spec`, but builds up to a named tap instead of the model's
/// own (possibly ambiguous or unnamed) outputs.
fn build_spec_tap(path: &Path, entry: &str, tap: &str, inputs: &[(&str, Vec<u64>)]) -> coreml_rs::proto::specification::Model {
    let tl_model = Model::load(path, Device::Cpu).unwrap();
    let input_shapes: Vec<(&str, Vec<usize>)> = inputs.iter().map(|(n, s)| (*n, s.iter().map(|&d| d as usize).collect())).collect();
    let info = tl_model.graph_entry(entry, &input_shapes, &Taps::Names(vec![tap.to_string()])).unwrap();
    let (func, _) = tensorlisp_aot::lower_until(path, &info, inputs, Opset::Ios17, tap).unwrap_or_else(|e| panic!("lower {entry}/{tap}: {e}"));
    let mut program = Program::new();
    program.add_function("main", func).unwrap();
    program.to_model("main").unwrap()
}

fn yolo11n() {
    let path = models_dir().join("yolo11n/yolo11n.gguf");
    if !path.exists() {
        println!("yolo11n: skipping, {path:?} not present");
        return;
    }
    const SIZE: usize = 640;
    let spec = build_spec(&path, "main", &[("image", vec![1, 3, SIZE as u64, SIZE as u64])]);
    let image = deterministic(3 * SIZE * SIZE, 0.0);
    let inputs = [("image", ArrayRef::new(&[1, 3, SIZE, SIZE], &image).unwrap())];
    println!("--- yolo11n, 640x640 (README: Metal direct ~134ms, Metal im2col ~48ms, CPU direct ~110-170ms) ---");
    bench("yolo11n", &spec, &inputs, 10);
}

/// Repacks the real vision weights with `flash-attention` forced `#f` (see
/// `tests/tipsv2_vision_on_ane.rs`'s own doc comment for why).
fn tipsv2_vision_path() -> Option<PathBuf> {
    let src = models_dir().join("tipsv2-b14/tipsv2-b14-vision-f32.gguf");
    if !src.exists() {
        return None;
    }
    let vision_ss = include_str!("../../../ports/tipsv2/vision.ss");
    let source = vision_ss.replace("(define flash-attention #t)", "(define flash-attention #f)");
    assert_ne!(source, vision_ss, "flash-attention toggle line not found -- did vision.ss change?");
    // `lower_graph` can't resolve any of this model's three (unnamed, in
    // ggml's own terms) outputs -- see `tests/tipsv2_vision_on_ane.rs`'s
    // own doc comment on why -- so this taps just `patches` (the one
    // output that shares essentially all of its computation with the
    // other two, `cls`/`register`; representative of the model's real
    // per-image latency without needing three separate MIL builds).
    let source = source.replace(
        "[patches (tensor:slice final 1 2 (* gw gh)) 3]))",
        "[patches (tap \"bench.patches\" (tensor:slice final 1 2 (* gw gh)) 3) 3]))",
    );
    assert_ne!(source.find("bench.patches"), None, "outputs clause not found -- did vision.ss change?");
    let file = GgufFile::open(&src).unwrap();
    let mut w = GgufWriter::new();
    w.copy_metadata(&file);
    w.set_program(&TlProgram::Text(source)).unwrap();
    let (file_ref, src_ref) = (&file, &src);
    for (idx, info) in file.tensor_infos().into_iter().enumerate() {
        let idx = idx as i64;
        w.add_tensor_with(&info.name, info.dtype, &info.shape, move || file_ref.read_tensor_bytes(src_ref, idx)).unwrap();
    }
    let out = std::env::temp_dir().join(format!("tensorlisp-bench-tipsv2-vision-{}.gguf", std::process::id()));
    w.write(&out).unwrap();
    Some(out)
}

fn tipsv2_vision() {
    let Some(path) = tipsv2_vision_path() else {
        println!("tipsv2 vision: skipping, weights not present");
        return;
    };
    const SIZE: usize = 448;
    let spec = build_spec_tap(&path, "main", "bench.patches", &[("image", vec![1, 3, SIZE as u64, SIZE as u64])]);
    let image = deterministic(3 * SIZE * SIZE, 1.0);
    let inputs = [("image", ArrayRef::new(&[1, 3, SIZE, SIZE], &image).unwrap())];
    println!("--- tipsv2 vision, 448x448 x1 image (README: Metal f32 168ms/2img (~84ms/img), f16 90ms/1img, CPU f32 ~750ms/img) ---");
    bench("tipsv2-vision", &spec, &inputs, 10);
    std::fs::remove_file(path).ok();
}

fn ppocrv6_det() {
    let path = models_dir().join("ppocrv6/ppocrv6.gguf");
    if !path.exists() {
        println!("ppocrv6 det: skipping, {path:?} not present");
        return;
    }
    const W: usize = 1216;
    const H: usize = 1600;
    let spec = build_spec(&path, "det", &[("image", vec![1, 3, H as u64, W as u64])]);
    let image = deterministic(3 * W * H, 2.0);
    let inputs = [("image", ArrayRef::new(&[1, 3, H, W], &image).unwrap())];
    println!("--- ppocrv6 det, 1216x1600 (README: Metal ~2.2s, CPU ~6.3-7.4s, PyTorch CPU ~8.7s) ---");
    bench("ppocrv6-det", &spec, &inputs, 3);
}

fn ppocrv6_rec() {
    let path = models_dir().join("ppocrv6/ppocrv6.gguf");
    if !path.exists() {
        println!("ppocrv6 rec: skipping, {path:?} not present");
        return;
    }
    const W: usize = 1197;
    const H: usize = 48;
    let spec = build_spec(&path, "rec", &[("image", vec![1, 3, H as u64, W as u64])]);
    let image = deterministic(3 * W * H, 3.0);
    let inputs = [("image", ArrayRef::new(&[1, 3, H, W], &image).unwrap())];
    println!("--- ppocrv6 rec, 48x1197 (README: Metal ~52ms, CPU ~270ms, PyTorch CPU ~74ms) ---");
    bench("ppocrv6-rec", &spec, &inputs, 10);
}

fn main() {
    yolo11n();
    tipsv2_vision();
    ppocrv6_det();
    ppocrv6_rec();
}
