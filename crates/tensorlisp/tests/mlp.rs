use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use tensorlisp::{Device, Error, Model, Program, gguf::GgufWriter};

const MLP: &str = r#"
(define (linear x name)
  (ggml-add (ggml-mul-mat (weight (string-append name ".weight")) x)
            (weight (string-append name ".bias"))))

(model (inputs [x f32 (4 batch)])
  (define h (ggml-relu (linear x "fc1")))
  (outputs [logits (linear h "fc2") 2]))
"#;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

struct Weights {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl Weights {
    fn new() -> Self {
        Weights {
            w1: Array2::from_shape_vec((8, 4), values(32, 0.1)).unwrap(),
            b1: Array1::from_vec(values(8, 0.2)),
            w2: Array2::from_shape_vec((3, 8), values(24, 0.3)).unwrap(),
            b2: Array1::from_vec(values(3, 0.4)),
        }
    }

    /// The same network in plain ndarray, PyTorch Linear semantics.
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = (x.dot(&self.w1.t()) + &self.b1).mapv(|v| v.max(0.0));
        h.dot(&self.w2.t()) + &self.b2
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tensorlisp-test-{}-{name}.gguf", std::process::id()))
}

fn write_model(name: &str, program: &str, weights: &Weights) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    w.add_f32("fc1.weight", weights.w1.view().into_dyn()).unwrap();
    w.add_f32("fc1.bias", weights.b1.view().into_dyn()).unwrap();
    w.add_f32("fc2.weight", weights.w2.view().into_dyn()).unwrap();
    w.add_f32("fc2.bias", weights.b2.view().into_dyn()).unwrap();
    let path = temp_path(name);
    w.write(&path).unwrap();
    path
}

fn assert_close(actual: &ArrayD<f32>, expected: &Array2<f32>) {
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4, "{actual} != {expected}");
    }
}

fn run_mlp(device: Device) {
    let weights = Weights::new();
    let path = write_model(&format!("mlp-{device:?}"), MLP, &weights);
    let model = Model::load(&path, device).unwrap();
    assert_eq!(model.inputs().len(), 1);

    // Same shape twice reuses the allocated graph; a new shape rebuilds it.
    for batch in [2, 2, 1, 2] {
        let x = Array2::from_shape_vec((batch, 4), values(batch * 4, 1.0)).unwrap();
        let out = model.run(&[("x", x.view().into_dyn())]).unwrap();
        assert_close(&out["logits"], &weights.forward(&x));
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mlp_cpu() {
    run_mlp(Device::Cpu);
}

#[test]
fn mlp_gpu() {
    if ggml_sys::HAS_METAL || ggml_sys::HAS_CUDA {
        run_mlp(Device::Gpu);
    }
}

#[test]
fn input_errors() {
    let path = write_model("input-errors", MLP, &Weights::new());
    let model = Model::load(&path, Device::Cpu).unwrap();
    let ok = ArrayD::<f32>::zeros(IxDyn(&[1, 4]));
    let err = |inputs: &[(&str, ndarray::ArrayViewD<f32>)]| model.run(inputs).unwrap_err().to_string();

    assert!(err(&[]).contains("missing input \"x\""));
    assert!(err(&[("x", ok.view()), ("y", ok.view())]).contains("unknown input \"y\""));
    let wrong = ArrayD::<f32>::zeros(IxDyn(&[1, 5]));
    assert!(err(&[("x", wrong.view())]).contains("declares [_, 4]"), "{}", err(&[("x", wrong.view())]));
    assert!(model.run(&[("x", ok.view())]).is_ok());
    std::fs::remove_file(path).unwrap();
}

fn load_error(name: &str, program: &str) -> String {
    let path = write_model(name, program, &Weights::new());
    let result = Model::load(&path, Device::Cpu).and_then(|m| {
        let x = ArrayD::<f32>::zeros(IxDyn(&[1, 4]));
        m.run(&[("x", x.view())])
    });
    std::fs::remove_file(path).unwrap();
    match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{name}: expected an error"),
    }
}

#[test]
fn program_errors() {
    let e = load_error("no-model", "(define x 1)");
    assert!(e.contains("does not define a model"), "{e}");

    let e = load_error("missing-weight", "(model (inputs [x f32]) (outputs [y (weight \"nope\")]))");
    assert!(e.contains("no tensor with this name") && e.contains("nope"), "{e}");

    let e = load_error("not-a-tensor", "(model (inputs [x f32]) (outputs [y (ggml-relu 5)]))");
    assert!(e.contains("ggml-relu") && e.contains("expected a tensor"), "{e}");

    let e = load_error("outside-build", "(ggml-relu #f)");
    assert!(e.contains("only be used inside a model"), "{e}");

    // Programs can't reach ports, eval or the FFI.
    for (name, form) in [
        ("sandbox-file", "(open-input-file \"/etc/passwd\")"),
        ("sandbox-eval", "(eval '(+ 1 2) (interaction-environment))"),
        ("sandbox-ffi", "(foreign-procedure \"ggml_add\" (uptr uptr uptr) uptr)"),
        ("sandbox-raw", "(ggml_add 0 0 0)"),
    ] {
        let e = load_error(name, form);
        assert!(e.contains("not bound") || e.contains("invalid syntax"), "{name}: {e}");
    }
}

#[test]
fn not_a_tensorlisp_file() {
    let mut w = GgufWriter::new();
    w.set_str("general.name", "plain").unwrap();
    let path = temp_path("plain");
    w.write(&path).unwrap();
    let e = Model::load(&path, Device::Cpu).err().unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(e, Error::Program(ref m) if m.contains("TL_VER")), "{e}");
}

#[test]
fn outputs_can_share_intermediates() {
    let weights = Weights::new();
    let program = r#"
      (model (inputs [x f32 (4 _)])
        (define h (ggml-add (ggml-mul-mat (weight "fc1.weight") x) (weight "fc1.bias")))
        (outputs [pre h 2] [post (ggml-relu h) 2] [t (ggml-transpose h) 2]))
    "#;
    let path = write_model("shared", program, &weights);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let x = Array2::from_shape_vec((3, 4), values(12, 2.0)).unwrap();
    let out = model.run(&[("x", x.view().into_dyn())]).unwrap();
    let pre = x.dot(&weights.w1.t()) + &weights.b1;
    assert_close(&out["pre"], &pre);
    assert_close(&out["post"], &pre.mapv(|v| v.max(0.0)));
    // Transposed (non-contiguous) outputs come back materialized.
    assert_close(&out["t"], &pre.t().to_owned());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn i32_inputs_and_strides() {
    // Rows 0..8 of fc1.weight ([8, 4], ggml [4, 8]) looked up by i32 ids.
    let weights = Weights::new();
    let program = r#"
      (model (inputs [ids i32 (n)])
        (define rows (ggml-get-rows (weight "fc1.weight") ids))
        (define nb (strides (weight "fc1.weight")))
        ;; Second column of the table as a strided view: offset nb0 = 4 bytes, rows nb1 apart.
        (outputs [rows rows 2] [column (ggml-cont (ggml-view-2d (weight "fc1.weight") 1 8 (cadr nb) (car nb))) 2]))
    "#;
    let path = write_model("i32", program, &weights);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let ids = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 0.0, 7.0]).unwrap();
    let out = model.run(&[("ids", ids.view())]).unwrap();
    for (i, &id) in [2usize, 0, 7].iter().enumerate() {
        assert_eq!(out["rows"].index_axis(ndarray::Axis(0), i), weights.w1.row(id).into_dyn());
    }
    assert_eq!(out["column"].shape(), &[8, 1]);
    assert_eq!(out["column"].iter().copied().collect::<Vec<_>>(), weights.w1.column(1).to_vec());

    // Non-integers are rejected for i32 inputs.
    let bad = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap();
    let err = model.run(&[("ids", bad.view())]).unwrap_err().to_string();
    assert!(err.contains("is i32 but contains 1.5"), "{err}");
    std::fs::remove_file(path).unwrap();
}
