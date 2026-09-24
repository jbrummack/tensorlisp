//! ggml assertions become errors instead of aborting the process.
use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use tensorlisp::{Device, Error, Model, Program, gguf::GgufWriter};

fn write(name: &str, program: &str, tensors: &[(&str, ArrayD<f32>)]) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    for (n, t) in tensors {
        w.add_f32(n, t.view()).unwrap();
    }
    let path = std::env::temp_dir().join(format!("tensorlisp-asserts-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

fn vector(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()
}

#[test]
fn assertion_while_building_is_an_error() {
    let bias = Array1::<f32>::ones(8).into_dyn();
    let path = write("build", "(model (inputs [x f32]) (outputs [y (ggml-add x (weight \"b\")) 1]))", &[("b", bias)]);
    let model = Model::load(&path, Device::Cpu).unwrap();

    // ggml_add requires b to broadcast over x.
    for _ in 0..2 {
        let err = model.run(&[("x", vector(&[1.0; 5]).view())]).unwrap_err().to_string();
        assert!(err.contains("ggml-add"), "{err}");
        assert!(err.contains("GGML_ASSERT(ggml_can_repeat(b, a))"), "{err}");
        assert!(err.contains("a = #<tensor f32 (5)>\n  b = #<tensor f32 (8)>"), "{err}");
    }

    // The same model keeps working for valid shapes.
    let out = model.run(&[("x", vector(&[1.0; 8]).view())]).unwrap();
    assert_eq!(out["y"], vector(&[2.0; 8]));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn assertion_during_compute_poisons_the_model() {
    let table = Array2::from_shape_fn((8, 4), |(r, c)| (r * 10 + c) as f32).into_dyn();
    let program = r#"
      (model (inputs [idx f32 (1)])
        (outputs [row (ggml-get-rows (weight "table") (ggml-cast idx GGML_TYPE_I32)) 2]))
    "#;
    let path = write("compute", program, &[("table", table)]);

    let model = Model::load(&path, Device::Cpu).unwrap();
    let out = model.run(&[("idx", vector(&[2.0]).view())]).unwrap();
    assert_eq!(out["row"].iter().copied().collect::<Vec<_>>(), [20.0, 21.0, 22.0, 23.0]);

    // A single out-of-range index is checked on the calling thread.
    let err = model.run(&[("idx", vector(&[100.0]).view())]).unwrap_err();
    assert!(matches!(err, Error::Ggml(ref m) if m.contains("i01 < ne01")), "{err}");
    let err = model.run(&[("idx", vector(&[2.0]).view())]).unwrap_err().to_string();
    assert!(err.contains("load it again"), "{err}");
    drop(model);

    // Loading again recovers.
    let model = Model::load(&path, Device::Cpu).unwrap();
    assert!(model.run(&[("idx", vector(&[3.0]).view())]).is_ok());
    std::fs::remove_file(path).unwrap();
}
