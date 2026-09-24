//! Writes a two-layer MLP as a tensorlisp GGUF, loads it and runs it:
//! `cargo run -p tensorlisp --example mlp`.
use ndarray::{Array1, Array2};
use tensorlisp::{Device, Model, Program, gguf::GgufWriter};

const PROGRAM: &str = r#"
(define (linear x name)
  (ggml-add (ggml-mul-mat (weight (string-append name ".weight")) x)
            (weight (string-append name ".bias"))))

(model (inputs [x f32 (4 batch)])        ; ggml order: (features batch)
  (define h (ggml-relu (linear x "fc1")))
  (outputs [logits (linear h "fc2") 2]))  ; keep rank 2 even when batch = 1
"#;

fn main() -> anyhow::Result<()> {
    // Weights in PyTorch layout: Linear.weight is (out_features, in_features).
    let fill = |n: usize| (0..n).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>();
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(PROGRAM.into()))?;
    w.add_f32("fc1.weight", Array2::from_shape_vec((8, 4), fill(32))?.view().into_dyn())?;
    w.add_f32("fc1.bias", Array1::from_vec(fill(8)).view().into_dyn())?;
    w.add_f32("fc2.weight", Array2::from_shape_vec((3, 8), fill(24))?.view().into_dyn())?;
    w.add_f32("fc2.bias", Array1::from_vec(fill(3)).view().into_dyn())?;
    let path = std::env::temp_dir().join("tensorlisp-mlp.gguf");
    w.write(&path)?;

    let model = Model::load(&path, Device::Auto)?;
    println!("running on {}", model.device_name());
    let x = Array2::<f32>::ones((2, 4));
    let out = model.run(&[("x", x.view().into_dyn())])?;
    println!("logits = {}", out["logits"]);
    Ok(())
}
