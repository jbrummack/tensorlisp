//! The native Metal device (tensorlisp's own executor) against the ggml CPU
//! backend and straightforward Rust references, op family by op family.
#![cfg(native_device)]

use std::path::PathBuf;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{DType, Device, Model, Program, RunOptions, gguf::GgufWriter};

fn noise(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    Array::from_shape_fn(IxDyn(shape), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    })
}

struct Weight<'a> {
    name: &'a str,
    dtype: DType,
    data: &'a ArrayD<f32>,
}

fn write(name: &str, program: &str, weights: &[Weight]) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    for Weight { name, dtype, data } in weights {
        let flat: Vec<f32> = data.as_standard_layout().iter().copied().collect();
        let n_per_row = *data.shape().last().unwrap();
        let bytes = if *dtype == DType::F32 {
            flat.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            dtype.from_f32(&flat, n_per_row).unwrap()
        };
        w.add_tensor(name, *dtype, data.shape(), bytes).unwrap();
    }
    let path = std::env::temp_dir().join(format!("tensorlisp-native-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

fn run(model: &Model, entry: &str, inputs: &[(&str, &ArrayD<f32>)]) -> Vec<(String, ArrayD<f32>)> {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    model.run_entry(entry, &views, &RunOptions::default()).unwrap().outputs
}

/// |got - want| <= tol * max|want| (+ a hair), the scale of the reference.
#[track_caller]
fn close(what: &str, got: &ArrayD<f32>, want: &ArrayD<f32>, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    eprintln!("{what}: max abs error {err:.3e} (scale {scale:.3e}, limit {tol:.0e} of it)");
    assert!(err <= tol * scale + 1e-6, "{what}: max abs error {err} > {tol} * {scale}");
}

/// Runs `entry` on the CPU backend and the native device and compares every output.
#[track_caller]
fn same_as_cpu(path: &PathBuf, entry: &str, inputs: &[(&str, &ArrayD<f32>)], tol: f32) {
    let cpu = Model::load(path, Device::Cpu).unwrap();
    let native = Model::load(path, Device::Native).unwrap();
    let want = run(&cpu, entry, inputs);
    let got = run(&native, entry, inputs);
    assert_eq!(want.len(), got.len());
    for ((name, w), (_, g)) in want.iter().zip(&got) {
        close(&format!("{entry}/{name}"), g, w, tol);
    }
}

#[test]
fn matmul_kernels_by_type_and_batch() {
    // y = x @ W^T with W of every type the matrix-vector / small-batch / matrix-matrix kernels cover.
    let (k, rows) = (256, 96);
    let w = noise(&[rows, k], 1);
    let ns = [1usize, 3, 5, 8, 9, 40];
    let entries: String = ns
        .iter()
        .map(|n| format!("(model n{n} (inputs [x f32 ({k} {n})]) (outputs [y (ggml-mul-mat (weight \"w\") x) 2]))\n"))
        .collect();
    for dtype in ["f32", "f16", "bf16", "q4_0", "q8_0", "q4_K", "q6_K"].map(|n| DType::parse(n).unwrap()) {
        let path = write(&format!("mm-{dtype}"), &entries, &[Weight { name: "w", dtype, data: &w }]);
        let model = Model::load(&path, Device::Native).unwrap();
        // The dequantized weights are the exact reference for this type.
        let stored = {
            let flat: Vec<f32> = w.iter().copied().collect();
            let bytes = if dtype == DType::F32 { flat.iter().flat_map(|v| v.to_le_bytes()).collect() } else { dtype.from_f32(&flat, k).unwrap() };
            let deq = if dtype == DType::F32 { flat } else { dtype.to_f32(&bytes).unwrap() };
            ndarray::Array2::from_shape_vec((rows, k), deq).unwrap()
        };
        for &n in &ns {
            let x = noise(&[n, k], 10 + n as u64);
            let got = run(&model, &format!("n{n}"), &[("x", &x)]).remove(0).1;
            let want = x.clone().into_dimensionality::<ndarray::Ix2>().unwrap().dot(&stored.t());
            // Matrix-matrix kernels round operands to half.
            close(&format!("{dtype} n={n}"), &got, &want.into_dyn(), 4e-3);
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn elementwise_norms_and_reductions() {
    let program = r#"
      (import (tl nn))
      (model norms (inputs [x f32 (8 5)] [y f32 (8 5)])
        (outputs [ln (nn:layer-norm x "ln" 'eps 1e-6) 2]
                 [rms (nn:rms-norm x "rms") 2]
                 [gelu (nn:gelu-tanh-exact x) 2]
                 [erf (ggml-gelu-erf x) 2]
                 [relu (ggml-relu x) 2]
                 [silu (ggml-silu x) 2]
                 [arith (ggml-div (ggml-sub (ggml-mul x y) (ggml-sqr x)) (ggml-scale-bias (ggml-exp y) 1.0 2.0)) 2]
                 [bias (ggml-scale-bias x 0.5 3.0) 2]
                 [mean (ggml-sum-rows x) 2]
                 [sincos (ggml-add (ggml-sin x) (ggml-cos y)) 2]))
      (model embed (inputs [ids i32 (3 2)])
        (outputs [e (nn:embedding "table" ids) 3]))
      (model pos (inputs [x f32 (8 5)])
        (outputs [p (nn:sinusoidal-positions 8 5) 2]
                 [c (tensor:concat (list x (nn:sinusoidal-positions 8 5)) 1) 2]
                 [r (ggml-repeat-4d (ggml-sum-rows x) 8 5 1 1) 2]))
      (model mean (inputs [x f32 (4 3 2)] [valid f32 (3 2)])
        (outputs [m (nn:masked-mean x valid) 2]))
    "#
    .replace("(import (tl nn))", "(import (tl nn) (tl tensor))");
    let (g, b, r) = (noise(&[8], 1), noise(&[8], 2), noise(&[8], 3));
    let table = noise(&[10, 4], 4);
    let path = write(
        "elementwise",
        &program,
        &[
            Weight { name: "ln.weight", dtype: DType::F32, data: &g },
            Weight { name: "ln.bias", dtype: DType::F32, data: &b },
            Weight { name: "rms.weight", dtype: DType::F32, data: &r },
            Weight { name: "table", dtype: DType::F16, data: &table },
        ],
    );
    let (x, y) = (noise(&[5, 8], 5) * 3.0, noise(&[5, 8], 6));
    same_as_cpu(&path, "norms", &[("x", &x), ("y", &y)], 1e-4);
    let ids = ndarray::arr2(&[[1.0f32, 7.0, 3.0], [0.0, 9.0, 9.0]]).into_dyn();
    same_as_cpu(&path, "embed", &[("ids", &ids)], 0.0);
    same_as_cpu(&path, "pos", &[("x", &x)], 1e-4);
    let valid = ndarray::arr2(&[[1.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]]).into_dyn();
    same_as_cpu(&path, "mean", &[("x", &noise(&[2, 3, 4], 7)), ("valid", &valid)], 1e-5);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn attention_masks_and_flash() {
    // 4 query heads sharing 2 K/V heads; 5 queries (vector flash kernel), then 40 (matrix flash kernel),
    // K/V lengths that are and are not multiples of the kernels' cache block; one
    // query over 1000 and 777 keys takes the CUDA split-KV decode path (batch 1 has only 5
    // valid keys, so most of its slices are fully masked).
    for (nq, nk) in [(5usize, 6usize), (40, 70), (40, 64), (3, 100), (1, 1000), (1, 777)] {
        let program = format!(
            r#"
          (import (tl attn))
          (model sdpa (inputs [q f32 (64 {nq} 4 2)] [k f32 (64 {nk} 2 2)] [v f32 (64 {nk} 2 2)] [valid f32 ({nk} 2)])
            (define mask (attn:padding-mask valid {nq}))
            (outputs [plain (attn:sdpa q k v 'mask mask) 3]
                     [nomask (attn:sdpa q k v) 3]
                     [flash (attn:sdpa q k v 'mask mask 'flash #t) 3]
                     [flash-nomask (attn:sdpa q k v 'flash #t) 3]))
        "#
        );
        let path = write(&format!("attn-{nq}-{nk}"), &program, &[]);
        let (q, k, v) = (noise(&[2, 4, nq, 64], 1), noise(&[2, 2, nk, 64], 2), noise(&[2, 2, nk, 64], 3));
        let mut valid = ArrayD::<f32>::ones(IxDyn(&[2, nk]));
        for j in nk * 3 / 4..nk {
            valid[[0, j]] = 0.0;
        }
        for j in 5..nk {
            valid[[1, j]] = 0.0;
        }
        // Flash attention keeps K/V in half precision.
        same_as_cpu(&path, "sdpa", &[("q", &q), ("k", &k), ("v", &v), ("valid", &valid)], 8e-3);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn patch_embedding_through_im2col() {
    let program = r#"
      (import (tl nn))
      (model patch (inputs [x f32 (28 14 3 2)])
        (outputs [p (nn:patch-embed x "pe" 14) 3]
                 [c (nn:conv2d x "c3" 'stride 2 'method 'im2col) 4]))
    "#;
    let (pw, pb) = (noise(&[16, 3, 14, 14], 1), noise(&[16], 2));
    let (cw, cb) = (noise(&[8, 3, 3, 3], 3), noise(&[8], 4));
    let path = write(
        "patch",
        program,
        &[
            Weight { name: "pe.weight", dtype: DType::F16, data: &pw },
            Weight { name: "pe.bias", dtype: DType::F32, data: &pb },
            Weight { name: "c3.weight", dtype: DType::F16, data: &cw },
            Weight { name: "c3.bias", dtype: DType::F32, data: &cb },
        ],
    );
    same_as_cpu(&path, "patch", &[("x", &noise(&[2, 3, 14, 28], 5))], 4e-3);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unsupported_ops_are_reported_by_name() {
    let program = "(model (inputs [x f32 (8 3)]) (outputs [y (ggml-cumsum x) 2]))";
    let path = write("unsupported", program, &[]);
    let native = Model::load(&path, Device::Native).unwrap();
    let x = noise(&[3, 8], 1);
    let err = native.run(&[("x", x.view())]).unwrap_err().to_string();
    assert!(err.contains("CUMSUM") && err.contains("not implemented"), "{err}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn rope_pooling_depthwise_argmax() {
    let program = r#"
      (import (tl nn) (tl attn))
      (model rope (inputs [x f32 (8 2 3)] [pos i32 (3)])
        (outputs [neox (attn:rope x pos 'base 100.0 'scale 0.5) 3]
                 [norm (ggml-rope-ext x pos #f 8 0 0 100.0 1.0 0.0 1.0 0.0 0.0) 3]))
      (model pool (inputs [x f32 (9 7 3 2)])
        (outputs [max (nn:max-pool x 3 'stride 1 'padding 1) 4]
                 [avg (ggml-pool-2d x GGML_OP_POOL_AVG 2 2 2 2 0.0 0.0) 4]
                 [dw (nn:conv2d-depthwise x "dw") 4]
                 [arg (ggml-argmax (ggml-reshape-2d x 9 42)) 1]))
    "#;
    let dw = noise(&[3, 1, 3, 3], 9);
    let db = noise(&[3], 10);
    let path = write(
        "rope-pool",
        program,
        &[
            Weight { name: "dw.weight", dtype: DType::F32, data: &dw },
            Weight { name: "dw.bias", dtype: DType::F32, data: &db },
        ],
    );
    let pos = ndarray::arr1(&[0.0f32, 3.0, 7.0]).into_dyn();
    same_as_cpu(&path, "rope", &[("x", &noise(&[3, 2, 8], 4)), ("pos", &pos)], 1e-5);
    same_as_cpu(&path, "pool", &[("x", &noise(&[2, 3, 7, 9], 5))], 1e-5);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn state_and_set_rows_persist_between_runs() {
    let program = r#"
      (define-state "sum" f32 4)
      (define-state "rows" f32 2 3)
      (define-state "half" f16 2 3)
      (model add (inputs [x f32 (4 1)] [at i32 (1 1)])
        (define x1 (ggml-reshape-1d x 4))
        (effect (ggml-cpy (ggml-add (state "sum") x1) (state "sum")))
        (effect (ggml-set-rows (state "rows") (ggml-view-2d x1 2 1 8 0) (ggml-reshape-1d at 1)))
        (effect (ggml-set-rows (state "half") (ggml-view-2d x1 2 1 8 0) (ggml-reshape-1d at 1)))
        (outputs [sum (ggml-scale (state "sum") 1.0) 1] [rows (ggml-scale (state "rows") 1.0) 2]
                 [half (ggml-cpy (state "half") (ggml-new-tensor-2d GGML_TYPE_F32 2 3)) 2]))
    "#;
    let path = write("state", program, &[]);
    let cpu = Model::load(&path, Device::Cpu).unwrap();
    let native = Model::load(&path, Device::Native).unwrap();
    for (step, at) in [0.0f32, 2.0, 1.0, 2.0].into_iter().enumerate() {
        let x = noise(&[1, 4], 20 + step as u64);
        let at = ndarray::arr2(&[[at]]).into_dyn();
        let want = run(&cpu, "add", &[("x", &x), ("at", &at)]);
        let got = run(&native, "add", &[("x", &x), ("at", &at)]);
        for ((n, w), (_, g)) in want.iter().zip(&got) {
            close(&format!("step {step} {n}"), g, w, 1e-3);
        }
    }
    native.reset_state();
    let at = ndarray::arr2(&[[0.0f32]]).into_dyn();
    let x = noise(&[1, 4], 30);
    let sum = run(&native, "add", &[("x", &x), ("at", &at)]).remove(0).1;
    close("after reset", &sum, &x.clone().into_shape_with_order(IxDyn(&[4])).unwrap(), 1e-6);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn padding_with_leading_and_trailing_amounts() {
    let program = r#"
      (model pad (inputs [x f32 (5 4 2)])
        (outputs [lr (ggml-pad-ext x 2 1 3 0 0 0 0 0) 3]
                 [right (ggml-pad x 3 2 0 0) 3]
                 [lead3 (ggml-pad-ext x 0 0 0 0 1 2 0 0) 3]
                 [reflect (ggml-pad-reflect-1d x 2 1) 3]))
    "#;
    let path = write("pad", program, &[]);
    same_as_cpu(&path, "pad", &[("x", &noise(&[2, 4, 5], 3))], 0.0);
    std::fs::remove_file(path).unwrap();
}
