//! Proves `$tl-compile-generic-to-mil` (see `src/scheme/core.ss`) on the
//! real yolo11n model source (`ports/yolo11/yolo11.ss`), not just a
//! synthetic fixture: compiles its backbone (layers 0-9, up through SPPF --
//! everything before c2psa's attention block, which needs ops
//! (permute/transpose/attn:sdpa) this pass doesn't cover yet, the same
//! boundary `tensorlisp-aot`'s Rust lowering already documents), builds a
//! new GGUF with the *same* real weight tensors and the compiled program
//! text, and runs it for real.
#![cfg(not(target_os = "windows"))]

use std::path::PathBuf;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::gguf::{GgufFile, GgufWriter};
use tensorlisp::{Device, Model, Program};

fn gguf_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/yolo11n/yolo11n.gguf")
}

fn yolo_helpers() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ports/yolo11/yolo11.ss");
    let src = std::fs::read_to_string(path).expect("read ports/yolo11/yolo11.ss");
    // Everything before "--- model": class names, `conv`/`dwconv`/`channels`/
    // `concat-channels`/`block-count`/`block`/`bottleneck`/`c3k`/`c3k2`/`sppf`
    // -- real helper functions the backbone needs, verbatim from the model.
    src.split(";; --- model").next().expect("yolo11.ss has a ';; --- model' marker").to_owned()
}

#[test]
fn yolo11n_backbone_compiles_and_runs_as_mil() {
    let path = gguf_path();
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }

    let program = format!(
        r#"{helpers}
        (model (inputs [image f32 (64 64 3 1)])
          (define checked
            (unless (and (= 0 (remainder (dim image 0) 32)) (= 0 (remainder (dim image 1) 32)))
              (error 'yolo11 "image height and width must be multiples of 32" (list (dim image 1) (dim image 0)))))
          (define (named i) (number->string i))
          (let* ([x (conv image (named 0) 2)]
                 [x (conv x (named 1) 2)]
                 [x (c3k2 x (named 2))]
                 [x (conv x (named 3) 2)]
                 [x (c3k2 x (named 4))]
                 [x (conv x (named 5) 2)]
                 [x (c3k2 x (named 6))]
                 [x (conv x (named 7) 2)]
                 [x (c3k2 x (named 8))]
                 [x (sppf x (named 9))])
            (outputs [layer9 (%mil-render! "yolo11n-backbone" (list (list "image" 'f32 '(64 64 3 1))) x) 4])))
        "#,
        helpers = yolo_helpers()
    );

    let compiled = tensorlisp::compile_generic_to_mil(&program).expect("compile yolo11n backbone to a mil trace program");

    // Same real weights as models/yolo11n/yolo11n.gguf, new (traced) program.
    let src = GgufFile::open(&path).expect("open yolo11n.gguf");
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(compiled)).unwrap();
    for (i, info) in src.tensor_infos().iter().enumerate() {
        let bytes = src.read_tensor_bytes(&path, i as i64).unwrap();
        w.add_tensor(&info.name, info.dtype, &info.shape, bytes).unwrap();
    }
    let traced_path = std::env::temp_dir().join(format!("tensorlisp-yolo11n-mil-{}.gguf", std::process::id()));
    w.write(&traced_path).unwrap();

    let model = Model::load(&traced_path, Device::Cpu).expect("load traced yolo11n backbone");
    let image: ArrayD<f32> = Array::from_shape_fn(IxDyn(&[1, 3, 64, 64]), |idx| {
        ((idx[1] * 64 + idx[2]) * 64 + idx[3]) as f32 * 1e-5
    });
    model.run(&[("image", image.view())]).expect("run traced yolo11n backbone");

    let mil_text = tensorlisp::mil_last_render().unwrap();
    std::fs::remove_file(&traced_path).ok();

    println!("{mil_text}");
    assert!(mil_text.starts_with("(program yolo11n-backbone"), "{mil_text}");
    for op in ["(conv ", "(silu ", "(add ", "(concat ", "(max_pool "] {
        assert!(mil_text.contains(op), "missing {op:?} in traced yolo11n backbone: {mil_text}");
    }
    // Layers 0-9: 1 stem conv + 4 downsampling convs + 4 C3k2 blocks (each
    // several convs) + SPPF's cv1/cv2 -- comfortably more than a couple dozen.
    let conv_count = mil_text.matches("(conv ").count();
    assert!(conv_count > 20, "expected the backbone's many convs, got {conv_count}: {mil_text}");
}
