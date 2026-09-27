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
    let path = std::env::temp_dir().join(format!("tensorlisp-mil-smoke-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    (Model::load(&path, Device::Cpu).unwrap(), path)
}

#[test]
fn conv_and_render() {
    let program = r#"
      (import (tl generic) (tl tensor))
      (model (inputs [x f32 (8 8 3 1)])
        (define y (conv2d x "c" 'stride 1))
        (define z (conv2d-depthwise y "d" 'stride 1))
        (outputs [out (%mil-render! "test" (list (list "x" 'f32 '(8 8 3 1))) z) 4]))
    "#;
    // %mil-render! is called from inside the model body (its tensors, and
    // %weights/%ctx, are only live while the graph is being built) and
    // returns its argument unchanged, so it can stand in for `z` as the
    // model's own declared output without changing what actually runs.

    let compiled = tensorlisp::compile_generic_to_mil(program).expect("compile");
    println!("{compiled}");

    let kernel_c = noise(&[4, 3, 3, 3], 10); // [C_out, C_in, kh, kw] ndarray order
    let kernel_d = noise(&[4, 1, 3, 3], 11); // depthwise: [C, 1, kh, kw]
    let x = noise(&[1, 3, 8, 8], 1);

    let (model, path) = load("conv", &compiled, &[("c.weight", &kernel_c), ("d.weight", &kernel_d)]);
    model.run(&[("x", x.view())]).unwrap();
    let mil_text = tensorlisp::mil_last_render().unwrap();
    println!("{mil_text}");

    assert!(mil_text.contains("(program test"));
    assert!(mil_text.contains("(inputs (%x (tensor f32 (8 8 3 1))))"));
    assert!(mil_text.contains("weights"));
    assert!(mil_text.contains("%c_weight"));
    assert!(mil_text.contains("%d_weight"));
    assert!(mil_text.contains("(conv "));

    std::fs::remove_file(path).ok();
}

#[test]
fn smoke_compile_and_run() {
    let program = r#"
      (import (tl generic) (tl tensor))
      (model (inputs [x f32 (8 8 3 batch)])
        (define pooled (max-pool x 5 'stride 1 'padding 2))
        (define added (add pooled x))
        (define concatenated (concat (list (slice added 2 0 1) (slice added 2 1 2)) 2))
        (define upped (upsample-nearest concatenated 2))
        (outputs [y (sigmoid (silu upped)) 4]))
    "#;

    let compiled = tensorlisp::compile_generic_to_mil(program).expect("compile generic source to mil trace source");
    println!("{compiled}");
    assert!(compiled.contains("mil-trace:"));

    let x = noise(&[2, 3, 8, 8], 1);
    let (model, path) = load("smoke", &compiled, &[]);
    let out = model.run(&[("x", x.view())]).unwrap();
    assert!(out.contains_key("y"));

    std::fs::remove_file(path).ok();
}
