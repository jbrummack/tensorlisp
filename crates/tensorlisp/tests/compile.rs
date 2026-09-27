//! `tensorlisp::compile_generic_to_ggml`: the simplest pass of the nanopass
//! AOT compiler, a syntax-level rename of `(tl generic)` calls into their
//! ggml/(tl nn)/(tl tensor) equivalents (see `$tl-compile-generic-to-ggml`
//! in `src/scheme/core.ss`). The property worth checking is the compiler
//! pipeline itself: compiling a generic-op program and running the result
//! must match running the original generic-op program directly -- proving
//! "trace through a dynamically selected compiler, then run its output" is
//! sound, on the easiest possible target, before a harder one (a real leaf
//! IR, e.g. CoreML MIL) needs real codegen instead of a rename.
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
    let path = std::env::temp_dir().join(format!("tensorlisp-compile-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    (Model::load(&path, Device::Cpu).unwrap(), path)
}

fn run(model: &Model, inputs: &[(&str, &ArrayD<f32>)]) -> Vec<ArrayD<f32>> {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    model.run(&views).unwrap().into_iter().map(|(_, a)| a).collect()
}

#[test]
fn compiled_ggml_source_matches_the_generic_program() {
    let generic_program = r#"
      (import (tl generic) (tl tensor))
      (model (inputs [x f32 (8 8 3 batch)])
        (define pooled (max-pool x 5 'stride 1 'padding 2))
        (define relued (relu (sub (add pooled x) x)))
        (define concatenated (concat (list (slice relued 2 0 1) (slice relued 2 1 2)) 2))
        (define upped (upsample-nearest concatenated 2))
        (define pooled2 (max-pool upped 3 'stride 2 'padding 1))
        (define flat (reshape pooled2 192 (tensor:dim pooled2 3)))
        (define linear (relu (add (mul-mat (weight "w") flat) (weight "b"))))
        (outputs [y (sigmoid (silu (mul linear linear))) 2]))
    "#;

    let compiled = tensorlisp::compile_generic_to_ggml(generic_program).expect("compile generic source to ggml source");
    // The `dim` call (from `(tl tensor)`, not `(tl generic)`) must survive
    // untouched, and every generic op name must be gone from the output.
    assert!(compiled.contains("nn:max-pool"), "compiled source: {compiled}");
    assert!(compiled.contains("ggml-mul"), "compiled source: {compiled}");
    for generic_name in ["(add ", "(sub ", "(mul ", "(relu ", "(silu ", "(sigmoid ", "(mul-mat ", "(slice ", "(concat "] {
        assert!(!compiled.contains(generic_name), "leftover generic call {generic_name:?} in: {compiled}");
    }

    let x = noise(&[2, 3, 8, 8], 1);
    let w = noise(&[4, 192], 2);
    let b = noise(&[4], 3);

    let (generic_model, generic_path) = load("generic", generic_program, &[("w", &w), ("b", &b)]);
    let (compiled_model, compiled_path) = load("compiled", &compiled, &[("w", &w), ("b", &b)]);

    let generic_out = run(&generic_model, &[("x", &x)]);
    let compiled_out = run(&compiled_model, &[("x", &x)]);

    assert_eq!(generic_out.len(), 1);
    assert_eq!(generic_out[0], compiled_out[0], "compiled program must match the generic program exactly");

    std::fs::remove_file(generic_path).ok();
    std::fs::remove_file(compiled_path).ok();
}
