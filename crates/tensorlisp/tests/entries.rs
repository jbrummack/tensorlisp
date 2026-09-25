//! Named entries and pipelines that run them (e.g. a generation loop).
use std::path::PathBuf;

use tensorlisp::{Device, Model, Program, RawInput, RunOptions, Value, gguf::GgufWriter};

const TOKENIZER_JSON: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "[UNK]": 1, "a": 2, "photo": 3, "of": 4, "dog": 5}, "unk_token": "[UNK]"}
}"#;

fn write(name: &str, program: &str) -> PathBuf {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(program.into())).unwrap();
    w.set_asset("tokenizer.json", TOKENIZER_JSON.as_bytes()).unwrap();
    let path = std::env::temp_dir().join(format!("tensorlisp-entries-{}-{name}.gguf", std::process::id()));
    w.write(&path).unwrap();
    path
}

const PROGRAM: &str = r#"
  (define tok (tokenizer (asset "tokenizer.json")))
  (model double (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))
  (model triple (inputs [x f32 (4 batch)] [n i32 (1 batch)])
    (outputs [y (ggml-scale x 3.0) 2] [n (ggml-scale (ggml-cast n GGML_TYPE_F32) 1.0) 2]))

  ;; Doubles x `times` times through the double entry, then triples it once.
  (pipeline grow ([x array] [times string])
    (let loop ([x x] [n (string->number times)])
      (if (= n 0)
          (let ([r (run triple [x x] [n (list->array '(7))])])
            (results [x (output r 'y)]
                     [sum (fold-left + 0 (array->list (output r 'y)))]
                     [text (detokenize tok '(2 3 4 5))]
                     [dog (token-id tok "dog")]))
          (loop (output (run double [x x]) 'y) (- n 1)))))

  (pipeline broken ([x array])
    (results [y (output (run double [x (list->array '(1 2 3))]) 'y)]))
"#;

#[test]
fn entries_run_separately() {
    let path = write("entries", PROGRAM);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let names: Vec<&str> = model.entries().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["double", "triple"]);
    assert_eq!(model.inputs()[0].name, "x");
    assert_eq!(model.pipelines().iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["grow", "broken"]);

    let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0, 4.0]]).into_dyn();
    let n = ndarray::arr2(&[[5.0f32]]).into_dyn();
    // Alternate between entries: each keeps its own allocated graph.
    for _ in 0..2 {
        let d = model.run_entry("double", &[("x", x.view())], &RunOptions::default()).unwrap();
        assert_eq!(d.outputs[0].1, &x * 2.0);
        let t = model.run_entry("triple", &[("x", x.view()), ("n", n.view())], &RunOptions::default()).unwrap();
        assert_eq!(t.outputs[0].1, &x * 3.0);
    }
    // The default entry is the first one.
    assert_eq!(model.run_with(&[("x", x.view())], &RunOptions::default()).unwrap().outputs[0].1, &x * 2.0);
    let err = model.run_entry("nope", &[], &RunOptions::default()).unwrap_err().to_string();
    assert!(err.contains("no entry \"nope\"; entries: double, triple"), "{err}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn pipelines_run_entries_in_a_loop() {
    let path = write("pipeline", PROGRAM);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let example = || {
        vec![
            ("x".to_string(), RawInput::Array(ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn())),
            ("times".to_string(), RawInput::Text("3".into())),
        ]
    };
    for _ in 0..2 {
        let results = model.pipeline("grow", example()).unwrap();
        let get = |name: &str| &results.iter().find(|(n, _)| n == name).unwrap().1;
        // 3 doublings, then a tripling: x * 24, with the batch dimension run added.
        assert_eq!(get("x"), &Value::Array(ndarray::arr2(&[[24.0f32, 48.0, 72.0, 96.0]]).into_dyn()));
        assert_eq!(get("sum").as_array().unwrap().iter().copied().collect::<Vec<_>>(), [240.0]);
        assert_eq!(get("text").as_text(), Some("a photo of dog"));
        assert_eq!(get("dog").as_array().unwrap().iter().copied().collect::<Vec<_>>(), [5.0]);
    }

    // An error inside a run surfaces with the entry's name, and the model stays usable.
    let err = model
        .pipeline("broken", vec![("x".into(), RawInput::Array(ndarray::arr1(&[0.0f32]).into_dyn()))])
        .unwrap_err()
        .to_string();
    assert!(err.contains("run double") && err.contains("has shape [1, 3]"), "{err}");
    assert!(model.pipeline("grow", example()).is_ok());
    let err = model.pipeline("nope", vec![]).unwrap_err().to_string();
    assert!(err.contains("no pipeline \"nope\"; pipelines: grow, broken"), "{err}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn entry_definition_errors() {
    for (program, message) in [
        ("(model a (inputs [x f32]) (outputs [y x])) (model a (inputs [x f32]) (outputs [y x]))", "duplicate entry name"),
        ("(model (inputs [x f32]) (outputs [y x])) (model (inputs [x f32]) (outputs [y x]))", "only one unnamed model"),
        ("(model a (inputs [x f32]) (outputs [y x])) (pipeline a () (results))", "already has this name"),
        ("(model a (inputs [x f32]) (outputs [y x])) (define (f) (run a [x 1]))  (f)", "run can only be used in a pipeline"),
    ] {
        let path = write("errors", program);
        let err = Model::load(&path, Device::Cpu).err().unwrap().to_string();
        assert!(err.contains(message), "{program}: {err}");
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn state_persists_between_runs() {
    let program = r#"
      (define-state "sum" f32 4)
      (define-state "rows" f32 2 3)
      ;; sum += x; rows[at] = (x0, x1)
      (model add (inputs [x f32 (4 1)] [at i32 (1 1)])
        (define x1 (ggml-reshape-1d x 4))
        (effect (ggml-cpy (ggml-add (state "sum") x1) (state "sum")))
        (effect (ggml-set-rows (state "rows") (ggml-view-2d x1 2 1 8 0) (ggml-reshape-1d at 1)))
        (outputs [sum (ggml-scale (state "sum") 1.0) 1] [rows (ggml-scale (state "rows") 1.0) 2]))
      ;; An entry without outputs, only effects.
      (model clear-row (inputs [at i32 (1 1)])
        (effect (ggml-set-rows (state "rows") (ggml-scale (ggml-view-2d (state "sum") 2 1 8 0) 0.0) (ggml-reshape-1d at 1)))
        (outputs))
      (pipeline twice ([x array])
        (run add [x x] [at (list->array '(0))])
        (run clear-row [at (list->array '(0))])
        (results [sum (output (run add [x x] [at (list->array '(2))]) 'sum)]))
    "#;
    let path = write("state", program);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0, 4.0]]).into_dyn();
    let run = |at: f32| {
        let at = ndarray::arr2(&[[at]]).into_dyn();
        model.run_entry("add", &[("x", x.view()), ("at", at.view())], &RunOptions::default()).unwrap().outputs
    };
    assert_eq!(run(0.0)[0].1.iter().copied().collect::<Vec<_>>(), [1.0, 2.0, 3.0, 4.0]);
    let out = run(2.0);
    assert_eq!(out[0].1.iter().copied().collect::<Vec<_>>(), [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(out[1].1.iter().copied().collect::<Vec<_>>(), [1.0, 2.0, 0.0, 0.0, 1.0, 2.0]);

    model.reset_state();
    let results = model.pipeline("twice", vec![("x".into(), RawInput::Array(ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn()))]).unwrap();
    assert_eq!(results[0].1.as_array().unwrap().iter().copied().collect::<Vec<_>>(), [2.0, 4.0, 6.0, 8.0]);

    let bad = write("state-bad", "(define-state \"s\" f64 4) (model (inputs [x f32]) (outputs [y x]))");
    let err = Model::load(&bad, Device::Cpu).err().unwrap().to_string();
    assert!(err.contains("type must be f32 or f16"), "{err}");
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(bad).unwrap();
}
