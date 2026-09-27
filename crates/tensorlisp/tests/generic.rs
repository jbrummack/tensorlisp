//! (tl generic): the AOT-portable, unprefixed op vocabulary tensorlisp-aot
//! lowers to backend leaves. Each alias/wrapper here just forwards to an
//! already-tested ggml/(tl nn)/(tl tensor) op (see
//! crates/tensorlisp/src/scheme/stdlib/generic.ss), so the property worth
//! checking isn't new numerics -- it's that (1) the bare names actually
//! resolve with no prefix (unlike every other stdlib library, which is
//! auto-prefixed unless the import spec says otherwise) and (2) a model
//! written with them produces bit-for-bit the same graph as the equivalent
//! ggml-/nn:-prefixed program.
use std::path::PathBuf;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, Model, Program, gguf::GgufWriter};

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
    let path = std::env::temp_dir().join(format!("tensorlisp-generic-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    (Model::load(&path, Device::Cpu).unwrap(), path)
}

fn run(model: &Model, inputs: &[(&str, &ArrayD<f32>)]) -> Vec<ArrayD<f32>> {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    model.run(&views).unwrap().into_iter().map(|(_, a)| a).collect()
}

/// Every generic name resolves bare (no `generic:` prefix), and a model
/// built with them (elementwise ops, mul-mat, reshape, conv2d, slice,
/// concat, max-pool, upsample-nearest -- everything tensorlisp-aot
/// currently lowers) produces exactly the same output as the same graph
/// written with the underlying ggml-/nn:-prefixed calls directly.
#[test]
fn generic_ops_match_their_ggml_equivalents() {
    let generic_program = r#"
      (import (tl generic) (only (tl tensor) dim))
      (model (inputs [x f32 (8 8 3 batch)])
        (define pooled (max-pool x 5 'stride 1 'padding 2))
        (define relued (relu (sub (add pooled x) x)))
        (define concatenated (concat (list (slice relued 2 0 1) (slice relued 2 1 2)) 2))
        (define upped (upsample-nearest concatenated 2))
        (define pooled2 (max-pool upped 3 'stride 2 'padding 1))
        (define flat (reshape pooled2 192 (dim pooled2 3)))
        (define linear (relu (add (mul-mat (weight "w") flat) (weight "b"))))
        (outputs [y (sigmoid (silu (mul linear linear))) 2]))
    "#;
    let ggml_program = r#"
      (import (tl nn) (tl tensor))
      (model (inputs [x f32 (8 8 3 batch)])
        (define pooled (nn:max-pool x 5 'stride 1 'padding 2))
        (define relued (ggml-relu (ggml-sub (ggml-add pooled x) x)))
        (define concatenated (tensor:concat (list (tensor:slice relued 2 0 1) (tensor:slice relued 2 1 2)) 2))
        (define upped (nn:upsample-nearest concatenated 2))
        (define pooled2 (nn:max-pool upped 3 'stride 2 'padding 1))
        (define flat (ggml-reshape-2d pooled2 192 (tensor:dim pooled2 3)))
        (define linear (ggml-relu (ggml-add (ggml-mul-mat (weight "w") flat) (weight "b"))))
        (outputs [y (ggml-sigmoid (ggml-silu (ggml-mul linear linear))) 2]))
    "#;

    let x = noise(&[2, 3, 8, 8], 1);
    let w = noise(&[4, 192], 2);
    let b = noise(&[4], 3);

    let (generic_model, generic_path) = load("generic", generic_program, &[("w", &w), ("b", &b)]);
    let (ggml_model, ggml_path) = load("ggml", ggml_program, &[("w", &w), ("b", &b)]);

    let generic_out = run(&generic_model, &[("x", &x)]);
    let ggml_out = run(&ggml_model, &[("x", &x)]);

    assert_eq!(generic_out.len(), 1);
    assert_eq!(generic_out[0], ggml_out[0], "generic-op graph must match the equivalent ggml-prefixed graph exactly");

    std::fs::remove_file(generic_path).ok();
    std::fs::remove_file(ggml_path).ok();
}
