//! The stdlib ((tl tensor), (tl nn), (tl attn), (tl vision), (tl util))
//! against straightforward Rust implementations of the PyTorch semantics.
use std::path::PathBuf;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, Model, Program, RawInput, RunOptions, gguf::GgufWriter};

/// Deterministic values in [-1, 1).
fn noise(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    Array::from_shape_fn(IxDyn(shape), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    })
}

fn load(name: &str, program: &str, weights: &[(&str, &ArrayD<f32>)]) -> (Model, PathBuf) {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    for (n, a) in weights {
        w.add_f32(n, a.view()).unwrap();
    }
    let path = std::env::temp_dir().join(format!("tensorlisp-stdlib-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    (Model::load(&path, Device::Cpu).unwrap(), path)
}

fn run(model: &Model, entry: &str, inputs: &[(&str, &ArrayD<f32>)]) -> Vec<ArrayD<f32>> {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    model.run_entry(entry, &views, &RunOptions::default()).unwrap().outputs.into_iter().map(|(_, a)| a).collect()
}

#[track_caller]
fn close(got: &ArrayD<f32>, want: &ArrayD<f32>, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "shape");
    let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(max <= tol, "max abs error {max} > {tol}");
}

#[test]
fn imports_bind_namespaced_names() {
    let program = r#"
      (import (tl nn) (only (tl tensor) dim) (rename (tl attn) (sdpa attention)))
      (model (inputs [x f32 (3 2)])
        (outputs [y (nn:linear x "fc") 2] [d (ggml-scale x (inexact (dim x 1))) 2]))
    "#;
    let w = noise(&[4, 3], 1);
    let b = noise(&[4], 2);
    let (model, path) = load("imports", program, &[("fc.weight", &w), ("fc.bias", &b)]);
    let x = noise(&[2, 3], 3);
    let out = run(&model, "main", &[("x", &x)]);
    let want = x.clone().into_dimensionality::<ndarray::Ix2>().unwrap().dot(&w.clone().into_dimensionality::<ndarray::Ix2>().unwrap().t())
        + b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
    close(&out[0], &want.into_dyn(), 1e-5);
    close(&out[1], &(&x * 2.0), 0.0);
    std::fs::remove_file(path).unwrap();

    for (program, message) in [
        ("(import (chezscheme)) (model (inputs [x f32]) (outputs [y x]))", "only the stdlib can be imported"),
        ("(import (prefix (rnrs io ports) io:)) (model (inputs [x f32]) (outputs [y x]))", "only the stdlib can be imported"),
        ("(model (inputs [x f32]) (outputs [y x])) (import (tl nn))", "imports must come before"),
        ("(model (inputs [x f32]) (outputs [y (linear x \"fc\")]))", "linear"),
        ("(import (tl nn)) (model (inputs [x f32]) (outputs [y (nn:layer-norm x \"n\" 'epsilon 1)]))", "unknown option epsilon"),
    ] {
        let (model, path) = {
            let mut w = GgufWriter::new();
            w.set_program(&Program::Text(program.into())).unwrap();
            let path = std::env::temp_dir().join(format!("tensorlisp-stdlib-{}-err.gguf", std::process::id()));
            w.write(&path).unwrap();
            (Model::load(&path, Device::Cpu), path)
        };
        let err = match model {
            Err(e) => e.to_string(),
            Ok(m) => m.graph(&[("x", vec![2])], &Default::default()).err().unwrap().to_string(),
        };
        assert!(err.contains(message), "{program}: {err}");
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn norms_and_embedding() {
    let program = r#"
      (import (tl nn))
      (model norms (inputs [x f32 (8 5)])
        (outputs [ln (nn:layer-norm x "ln" 'eps 1e-6) 2]
                 [rms (nn:rms-norm x "rms") 2]
                 [gemma (nn:rms-norm x "rms" 'offset 1) 2]
                 [gelu (nn:gelu-tanh-exact x) 2]))
      (model embed (inputs [ids i32 (3 2)])
        (outputs [e (nn:embedding "table" ids) 3]))
      (model mean (inputs [x f32 (4 3 2)] [valid f32 (3 2)])
        (outputs [m (nn:masked-mean x valid) 2]))
    "#;
    let (g, b, r) = (noise(&[8], 1), noise(&[8], 2), noise(&[8], 3));
    let table = noise(&[10, 4], 4);
    let (model, path) = load("norms", program, &[("ln.weight", &g), ("ln.bias", &b), ("rms.weight", &r), ("table", &table)]);
    let x = noise(&[5, 8], 5) * 3.0;
    let out = run(&model, "norms", &[("x", &x)]);
    let rows = |f: &dyn Fn(ndarray::ArrayView1<f32>) -> Vec<f32>| {
        let v: Vec<f32> = x.rows().into_iter().flat_map(|r| f(r)).collect();
        ArrayD::from_shape_vec(IxDyn(&[5, 8]), v).unwrap()
    };
    let ln = rows(&|r| {
        let mean = r.mean().unwrap();
        let var = r.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        r.iter().zip(g.iter().zip(&b)).map(|(v, (g, b))| (v - mean) / (var + 1e-6).sqrt() * g + b).collect()
    });
    close(&out[0], &ln, 1e-5);
    let rms = |offset: f32| {
        rows(&|row| {
            let ms = row.mapv(|v| v * v).mean().unwrap();
            row.iter().zip(&r).map(|(v, w)| v / (ms + 1e-6).sqrt() * (w + offset)).collect()
        })
    };
    close(&out[1], &rms(0.0), 1e-5);
    close(&out[2], &rms(1.0), 1e-5);
    let gelu = x.mapv(|v| 0.5 * v * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh()));
    close(&out[3], &gelu, 1e-5);

    let ids = ndarray::arr2(&[[1.0f32, 7.0, 3.0], [0.0, 9.0, 9.0]]).into_dyn();
    let out = run(&model, "embed", &[("ids", &ids)]);
    let want = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |i| table[[ids[[i[0], i[1]]] as usize, i[2]]]);
    close(&out[0], &want, 0.0);

    let x = noise(&[2, 3, 4], 6);
    let valid = ndarray::arr2(&[[1.0f32, 1.0, 0.0], [1.0, 0.0, 0.0]]).into_dyn();
    let out = run(&model, "mean", &[("x", &x), ("valid", &valid)]);
    let want = ArrayD::from_shape_fn(IxDyn(&[2, 4]), |i| {
        let n: f32 = (0..3).map(|l| valid[[i[0], l]]).sum();
        (0..3).map(|l| x[[i[0], l, i[1]]] * valid[[i[0], l]]).sum::<f32>() / n
    });
    close(&out[0], &want, 1e-6);
    std::fs::remove_file(path).unwrap();
}

/// torch Conv2d (NCHW), groups 1 or depthwise.
fn conv_ref(x: &ArrayD<f32>, w: &ArrayD<f32>, b: &ArrayD<f32>, stride: usize, pad: usize, depthwise: bool) -> ArrayD<f32> {
    let (n, c, h, wd) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (o, k) = (w.shape()[0], w.shape()[2]);
    let (ho, wo) = ((h + 2 * pad - k) / stride + 1, (wd + 2 * pad - k) / stride + 1);
    ArrayD::from_shape_fn(IxDyn(&[n, o, ho, wo]), |i| {
        let (bi, oi, y, xx) = (i[0], i[1], i[2], i[3]);
        let mut acc = b[[oi]];
        for ci in 0..(if depthwise { 1 } else { c }) {
            let channel = if depthwise { oi } else { ci };
            for ky in 0..k {
                for kx in 0..k {
                    let (iy, ix) = ((y * stride + ky) as isize - pad as isize, (xx * stride + kx) as isize - pad as isize);
                    if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < wd {
                        acc += w[[oi, ci, ky, kx]] * x[[bi, channel, iy as usize, ix as usize]];
                    }
                }
            }
        }
        acc
    })
}

#[test]
fn convolutions_and_pooling() {
    let program = r#"
      (import (tl nn))
      (model conv (inputs [x f32 (9 7 3 2)])
        (outputs [im2col (nn:conv2d x "c3" 'stride 2 'method 'im2col) 4]
                 [direct (nn:conv2d x "c3" 'stride 2 'method 'direct) 4]
                 [pointwise (nn:conv2d x "c1" 'method 'im2col) 4]
                 [depthwise (nn:conv2d-depthwise x "dw") 4]
                 [maxpool (nn:max-pool x 3 'stride 1 'padding 1) 4]
                 [up (nn:upsample-nearest x 2) 4]))
    "#;
    let (w3, b3) = (noise(&[5, 3, 3, 3], 1), noise(&[5], 2));
    let (w1, b1) = (noise(&[4, 3, 1, 1], 3), noise(&[4], 4));
    let (wd, bd) = (noise(&[3, 1, 3, 3], 5), noise(&[3], 6));
    let (model, path) = load(
        "conv",
        program,
        &[("c3.weight", &w3), ("c3.bias", &b3), ("c1.weight", &w1), ("c1.bias", &b1), ("dw.weight", &wd), ("dw.bias", &bd)],
    );
    let x = noise(&[2, 3, 7, 9], 7);
    let out = run(&model, "conv", &[("x", &x)]);
    let want = conv_ref(&x, &w3, &b3, 2, 1, false);
    close(&out[0], &want, 1e-5);
    close(&out[1], &want, 1e-5);
    close(&out[2], &conv_ref(&x, &w1, &b1, 1, 0, false), 1e-5);
    close(&out[3], &conv_ref(&x, &wd, &bd, 1, 1, true), 1e-5);
    let maxpool = ArrayD::from_shape_fn(IxDyn(&[2, 3, 7, 9]), |i| {
        let mut m = f32::NEG_INFINITY;
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let (y, xx) = (i[2] as isize + dy, i[3] as isize + dx);
                if (0..7).contains(&y) && (0..9).contains(&xx) {
                    m = m.max(x[[i[0], i[1], y as usize, xx as usize]]);
                }
            }
        }
        m
    });
    close(&out[4], &maxpool, 0.0);
    close(&out[5], &ArrayD::from_shape_fn(IxDyn(&[2, 3, 14, 18]), |i| x[[i[0], i[1], i[2] / 2, i[3] / 2]]), 0.0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn attention() {
    // 4 query heads sharing 2 K/V heads, head dim 8.
    let program = r#"
      (import (tl attn))
      (model sdpa (inputs [q f32 (8 5 4 2)] [k f32 (8 6 2 2)] [v f32 (8 6 2 2)] [valid f32 (6 2)])
        (define mask (attn:padding-mask valid 5))
        (outputs [plain (attn:sdpa q k v 'mask mask) 3]
                 [flash (attn:sdpa q k v 'mask mask 'flash #t) 3]))
      (model masks (inputs [x f32 (1)])
        (outputs [causal (attn:causal-mask 5) 2]
                 [sliding (attn:causal-mask 5 'window 2) 2]
                 [window (attn:window-mask 5 'left 2 'right 3) 2]))
      (model rope (inputs [x f32 (8 2 3)] [pos i32 (3)])
        (outputs [neox (attn:rope x pos 'base 100.0 'scale 0.5) 3]))
    "#;
    let (model, path) = load("attn", program, &[]);
    let (q, k, v) = (noise(&[2, 4, 5, 8], 1), noise(&[2, 2, 6, 8], 2), noise(&[2, 2, 6, 8], 3));
    let valid = ndarray::arr2(&[[1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).into_dyn();
    let out = run(&model, "sdpa", &[("q", &q), ("k", &k), ("v", &v), ("valid", &valid)]);
    let scale = 1.0 / 8f32.sqrt();
    let want = ArrayD::from_shape_fn(IxDyn(&[2, 5, 32]), |i| {
        let (b, l, c) = (i[0], i[1], i[2]);
        let (h, d) = (c / 8, c % 8);
        let kv = h / 2; // consecutive query heads share a K/V head
        let scores: Vec<f32> = (0..6)
            .map(|j| {
                let s: f32 = (0..8).map(|e| q[[b, h, l, e]] * k[[b, kv, j, e]]).sum::<f32>() * scale;
                if valid[[b, j]] > 0.0 { s } else { f32::NEG_INFINITY }
            })
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = scores.iter().map(|s| (s - max).exp()).sum();
        (0..6).map(|j| (scores[j] - max).exp() / z * v[[b, kv, j, d]]).sum()
    });
    close(&out[0], &want, 1e-5);
    close(&out[1], &want, 5e-3); // f16 K/V

    let out = run(&model, "masks", &[("x", &ndarray::arr1(&[0.0f32]).into_dyn())]);
    let mask = |allowed: &dyn Fn(i32, i32) -> bool| {
        ArrayD::from_shape_fn(IxDyn(&[5, 5]), |i| if allowed(i[0] as i32, i[1] as i32) { 0.0 } else { -1e30 })
    };
    close(&out[0], &mask(&|q, k| k <= q), 0.0);
    close(&out[1], &mask(&|q, k| k <= q && q - k < 2), 0.0);
    close(&out[2], &mask(&|q, k| (q - k >= 0 && q - k < 2) || (k - q > 0 && k - q < 3)), 0.0);

    let x = noise(&[3, 2, 8], 4);
    let pos = ndarray::arr1(&[0.0f32, 3.0, 7.0]).into_dyn();
    let out = run(&model, "rope", &[("x", &x), ("pos", &pos)]);
    let want = ArrayD::from_shape_fn(IxDyn(&[3, 2, 8]), |i| {
        let (l, h, d) = (i[0], i[1], i[2]);
        let j = d % 4;
        let theta = pos[[l]] * 0.5 * 100f32.powf(-((2 * j) as f32) / 8.0);
        let (c, s) = (theta.cos(), theta.sin());
        if d < 4 { x[[l, h, d]] * c - x[[l, h, d + 4]] * s } else { x[[l, h, d]] * c + x[[l, h, d - 4]] * s }
    });
    close(&out[0], &want, 1e-5);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn positions_and_detection_heads() {
    let program = r#"
      (import (tl nn) (tl vision) (tl tensor))
      (model positions (inputs [x f32 (1)])
        (outputs [tips (nn:sinusoidal-positions 6 4) 2]
                 [t2t (nn:sinusoidal-positions 6 4 'divisor 'n 'order 'cos-sin 'max-timescale 100.0) 2]))
      (model boxes (inputs [raw f32 (6 64)])
        ;; A 3 x 2 grid at 8 pixels per cell.
        (outputs [boxes (vision:decode-ltrb (vision:dfl raw) 3 2 8) 2]))
      (model grid (inputs [tokens f32 (4 6)])
        (define grid (vision:tokens->grid tokens 3 2))
        (outputs [back (vision:grid->tokens grid) 2] [second-channel (tensor:slice grid 2 1 1) 2]))
    "#;
    let (model, path) = load("vision", program, &[]);
    let out = run(&model, "positions", &[("x", &ndarray::arr1(&[0.0f32]).into_dyn())]);
    let positions = |divisor: f32, timescale: f32, cos_first: bool| {
        ArrayD::from_shape_fn(IxDyn(&[4, 6]), |i| {
            let (p, c) = (i[0] as f32, i[1]);
            let f = (-((c % 3) as f32) * timescale.ln() / divisor).exp();
            if (c < 3) != cos_first { (p * f).sin() } else { (p * f).cos() }
        })
    };
    close(&out[0], &positions(2.0, 10000.0, false), 1e-5);
    close(&out[1], &positions(3.0, 100.0, true), 1e-5);

    let raw = noise(&[64, 6], 5) * 4.0;
    let out = run(&model, "boxes", &[("raw", &raw)]);
    let want = ArrayD::from_shape_fn(IxDyn(&[4, 6]), |i| {
        let a = i[1];
        let dist: Vec<f32> = (0..4)
            .map(|side| {
                let logits: Vec<f32> = (0..16).map(|bin| raw[[side * 16 + bin, a]]).collect();
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = logits.iter().map(|l| (l - max).exp()).sum();
                logits.iter().enumerate().map(|(bin, l)| (l - max).exp() / z * bin as f32).sum()
            })
            .collect();
        let (ax, ay) = ((a % 3) as f32 + 0.5, (a / 3) as f32 + 0.5);
        8.0 * match i[0] {
            0 => ax + (dist[2] - dist[0]) / 2.0,
            1 => ay + (dist[3] - dist[1]) / 2.0,
            2 => dist[0] + dist[2],
            _ => dist[1] + dist[3],
        }
    });
    close(&out[0], &want, 1e-4);

    let tokens = noise(&[6, 4], 6);
    let out = run(&model, "grid", &[("tokens", &tokens)]);
    close(&out[0], &tokens, 0.0);
    // Channel 1 of the [3, 2, 4] grid: token n's channel 1, as [2, 3] (numpy).
    close(&out[1], &ArrayD::from_shape_fn(IxDyn(&[2, 3]), |i| tokens[[i[0] * 3 + i[1], 1]]), 0.0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn util_helpers_in_pipelines() {
    let program = r##"
      (import (tl util))
      (model (inputs [x f32]) (outputs [y x]))
      (pipeline text ([s string])
        (results [replaced (util:string-replace s "<img>" (util:string-repeat "#" 3))]
                 [joined (util:string-join '("a" "b" "c") ", ")]
                 [padded (util:pad-list '(1 2) 4 0)]
                 [last (util:last '(4 5 6))]
                 [ints (util:->integers '(1.0 2.6 -0.4))]))
    "##;
    let (model, path) = load("util", program, &[]);
    let r = model.pipeline("text", vec![("s".into(), RawInput::Text("see <img> and <img>".into()))]).unwrap();
    let text = |i: usize| r[i].1.as_text().unwrap().to_string();
    let array = |i: usize| r[i].1.as_array().unwrap().iter().copied().collect::<Vec<_>>();
    assert_eq!(text(0), "see ### and ###");
    assert_eq!(text(1), "a, b, c");
    assert_eq!(array(2), [1.0, 2.0, 0.0, 0.0]);
    assert_eq!(array(3), [6.0]);
    assert_eq!(array(4), [1.0, 3.0, 0.0]);
    std::fs::remove_file(path).unwrap();
}
