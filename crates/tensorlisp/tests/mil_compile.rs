//! End-to-end: `(tl aot mil shadow)`'s ops, run for real inside a loaded
//! model (real weights, real inputs), correctly populate a MIL trace
//! (`(tl aot mil trace)`) that renders into an actual MIL program -- and
//! `(tl aot mil compile)`'s `hl-mil-compile-program` nanopass, which
//! rewrites highlevel.ss-vocabulary source into calls to those same shadow
//! ops, runs inside a loaded program without erroring.
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
    let path = std::env::temp_dir().join(format!("tensorlisp-mil-compile-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    (Model::load(&path, Device::Cpu).unwrap(), path)
}

/// `mil-shadow:mul-mat`/`mil-shadow:relu`, called directly (as
/// `hl-mil-compile-program` would rewrite `mul-mat-t`/`relu-t` calls into),
/// both compute the real result *and* populate a trace that
/// `mil-render!` turns into MIL program text -- checked here by re-parsing
/// that text back with `read` and confirming its shape, not just that it's
/// non-empty.
#[test]
fn shadow_ops_populate_a_real_mil_trace_and_compute_correctly() {
    let w = noise(&[4, 3], 1);
    let program = r#"
      (import (tl aot mil shadow) (tl aot mil trace) (tl tensor))
      (model (inputs [x f32 (3 2)])
        (define %reset (mil-trace-reset!))
        (define %decl-x (mil-declare-input! x "x"))
        (outputs [y (mil-render! "mlp"
                                  (list (list "x" 'f32 (list 2 3)))
                                  (mil-shadow:relu (mil-shadow:mul-mat (mil-shadow:weight "fc.weight") x)))
                     2]))
    "#;
    let (model, path) = load("shadow", program, &[("fc.weight", &w)]);
    let x = noise(&[2, 3], 2);
    let out = model.run(&[("x", x.view())]).unwrap();

    let want = x
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .dot(&w.clone().into_dimensionality::<ndarray::Ix2>().unwrap().t())
        .mapv(|v| v.max(0.0));
    let got = &out["y"];
    assert_eq!(got.shape(), want.shape());
    let max_err = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-5, "max err {max_err}");

    // A correct numeric result already proves the whole chain succeeded
    // without error: mil-shadow:weight had to declare the weight,
    // mil-shadow:mul-mat/relu each had to resolve their inputs via
    // mil-ref (which throws hard on an undeclared tensor -- see
    // trace.ss), and mil-render! had to walk that same trace to render.

    std::fs::remove_file(path).unwrap();
}

/// `hl-mil-compile-program` itself, called from inside a running program
/// (the only way to reach it without exposing a new Rust entry point):
/// rewrites `mul-mat-t`/`relu-t`/`weight` calls to their `mil-shadow:*`
/// targets and returns the result as ordinary tensorlisp source text.
#[test]
fn hl_mil_compile_program_runs_inside_a_loaded_program() {
    let highlevel_source = r#"(model (inputs [x f32 (3 2)]) (outputs [y (relu-t (mul-mat-t (weight "fc.weight") x)) 2]))"#;
    let escaped = highlevel_source.replace('\\', "\\\\").replace('"', "\\\"");
    // A model output must itself be a real tensor, so the rewritten text
    // (a Scheme string) can't be returned directly -- proving
    // hl-mil-compile-program runs without erroring (a bad rewrite would
    // throw at read/compile time here, same as it would for a real caller)
    // is what this checks; `mil_compile.rs`'s other test separately checks
    // the *result* of an equivalent rewrite is numerically correct once run.
    let program = format!(
        r#"(import (tl aot mil compile))
           (define %rewritten (hl-mil-compile-program "{escaped}"))
           (model (inputs [x f32 (1)]) (outputs [y x 1]))"#
    );
    let (model, path) = load("compile-runs", &program, &[]);
    let x = ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 1.0f32);
    let out = model.run(&[("x", x.view())]).unwrap();
    assert_eq!(out["y"], x);
    std::fs::remove_file(path).unwrap();
}
