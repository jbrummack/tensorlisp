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

const STREAM_PROGRAM: &str = r#"
  (model double (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))
  ;; Doubles every chunk as it arrives and reports each one right away.
  (pipeline running ([xs stream] [label string])
    (let loop ([n 0])
      (let ([x (stream-recv xs)])
        (cond
          [(string? x) (emit 'note x) (loop n)]
          [x (emit 'y (output (run double [x x]) 'y))
             (emit 'note label)
             (loop (+ n 1))]
          [else (results [count n])]))))
  (pipeline no-emit ([xs stream]) (emit 'y (stream-recv xs)) (results [done 1]))
"#;

#[test]
fn pipelines_stream_in_and_out() {
    use std::sync::mpsc;
    use tensorlisp::{Chunk, input_stream};

    let path = write("stream", STREAM_PROGRAM);
    let model = Model::load(&path, Device::Cpu).unwrap();
    assert_eq!(model.pipelines()[0].raw_inputs[0].kind, tensorlisp::RawKind::Stream);

    let (tx, rx) = input_stream(|| {});
    let (out_tx, out_rx) = mpsc::channel::<(String, Chunk)>();
    let emit: tensorlisp::Emit = std::sync::Arc::new(move |name, chunk| out_tx.send((name.to_string(), chunk)).is_ok());
    let worker = {
        let model = model.clone();
        std::thread::spawn(move || {
            model.pipeline_streaming(
                "running",
                vec![("xs".into(), RawInput::Stream(rx)), ("label".into(), RawInput::Text("L".into()))],
                Some(emit),
            )
        })
    };

    // Each chunk's result arrives before the next chunk is sent: the pipeline really is incremental.
    for i in 1..=3 {
        tx.send(Chunk::Array(ndarray::arr2(&[[i as f32, 0.0, 0.0, 1.0]]).into_dyn())).unwrap();
        let (name, chunk) = out_rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        assert_eq!(name, "y");
        let Chunk::Array(y) = chunk else { panic!("expected an array") };
        assert_eq!(y, ndarray::arr2(&[[2.0 * i as f32, 0.0, 0.0, 2.0]]).into_dyn());
        let (name, chunk) = out_rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        assert_eq!(name, "note");
        assert!(matches!(chunk, Chunk::Text(t) if t == "L"));
    }
    tx.send(Chunk::Text("hello".into())).unwrap();
    assert!(matches!(out_rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap(), (n, Chunk::Text(t)) if n == "note" && t == "hello"));
    drop(tx); // end of stream
    let results = worker.join().unwrap().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "count");
    assert_eq!(results[0].1.as_array().unwrap().iter().copied().collect::<Vec<_>>(), [3.0]);

    // Without output streams, emit is an error that names the problem.
    let (tx, rx) = input_stream(|| {});
    tx.send(Chunk::Text("x".into())).unwrap();
    let err = model.pipeline("no-emit", vec![("xs".into(), RawInput::Stream(rx))]).unwrap_err().to_string();
    assert!(err.contains("emit can only be used in a pipeline that was started with output streams"), "{err}");

    // A listener that went away stops the pipeline.
    let (tx, rx) = input_stream(|| {});
    tx.send(Chunk::Text("x".into())).unwrap();
    let emit: tensorlisp::Emit = std::sync::Arc::new(|_, _| false);
    let err = model.pipeline_streaming("no-emit", vec![("xs".into(), RawInput::Stream(rx))], Some(emit)).unwrap_err().to_string();
    assert!(err.contains("nobody is listening"), "{err}");
    std::fs::remove_file(path).unwrap();
}

const SESSION_PROGRAM: &str = r#"
  (define-state "sum" f32 4)
  (define-state "rows" f32 2 3)
  (model add (inputs [x f32 (4 1)] [at i32 (1 1)])
    (define x1 (ggml-reshape-1d x 4))
    (effect (ggml-cpy (ggml-add (state "sum") x1) (state "sum")))
    (effect (ggml-set-rows (state "rows") (ggml-view-2d x1 2 1 8 0) (ggml-reshape-1d at 1)))
    (outputs [sum (ggml-scale (state "sum") 1.0) 1] [rows (ggml-scale (state "rows") 1.0) 2]))
  (pipeline twice ([x array])
    (run add [x x] [at (list->array '(0))])
    (results [sum (output (run add [x x] [at (list->array '(2))]) 'sum)]))
"#;

fn sessions_on(device: Device) {
    use tensorlisp::{ModelState, RunOptions};
    let path = write("sessions", SESSION_PROGRAM);
    let model = Model::load(&path, device).unwrap();
    let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0, 4.0]]).into_dyn();
    // Runs `add` (sum += x; rows[at] = x[..2]) and returns (sum, rows).
    let add = |state: Option<&ModelState>, at: f32| -> (Vec<f32>, Vec<f32>) {
        let at = ndarray::arr2(&[[at]]).into_dyn();
        let options = RunOptions { state: state.cloned(), ..Default::default() };
        let out = model.run_entry("add", &[("x", x.view()), ("at", at.view())], &options).unwrap().outputs;
        (out[0].1.iter().copied().collect(), out[1].1.iter().copied().collect())
    };
    // After n adds the sum is n * x.
    let n_times = |n: f32| -> Vec<f32> { vec![n, 2.0 * n, 3.0 * n, 4.0 * n] };

    let (s1, s2) = (model.new_state().unwrap(), model.new_state().unwrap());
    assert_eq!(add(None, 0.0).0, n_times(1.0));
    assert_eq!(add(Some(&s1), 0.0).0, n_times(1.0)); // a new session starts at zero
    assert_eq!(add(Some(&s1), 0.0).0, n_times(2.0));
    assert_eq!(add(Some(&s2), 0.0).0, n_times(1.0));
    assert_eq!(add(None, 0.0).0, n_times(2.0)); // the default kept its own
    assert_eq!(add(Some(&s1), 0.0).0, n_times(3.0)); // interleaved, nobody mixes
    assert_eq!(add(Some(&s2), 0.0).0, n_times(2.0));
    // now: default 2, s1 3, s2 2

    // Views into state (set-rows) land in the right session: only row 0 was written before.
    let (sum, rows) = add(Some(&s2), 2.0); // s2: 3
    assert_eq!(sum, n_times(3.0));
    assert_eq!(rows, [1.0, 2.0, 0.0, 0.0, 1.0, 2.0]);
    let fresh = model.new_state().unwrap();
    assert_eq!(add(Some(&fresh), 2.0).1, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]); // never wrote row 0
    drop(fresh);

    // Resets are per session.
    s1.reset().unwrap();
    assert_eq!(add(Some(&s1), 0.0).0, n_times(1.0));
    assert_eq!(add(Some(&s2), 0.0).0, n_times(4.0));
    add(Some(&s1), 0.0); // s1: 2, resident
    model.reset_state(); // the default, while s1 is resident on the ggml path
    assert_eq!(add(None, 0.0).0, n_times(1.0));
    assert_eq!(add(Some(&s1), 0.0).0, n_times(3.0));

    // A pipeline runs all of its entries in the session it was started with.
    let arg = || vec![("x".into(), RawInput::Array(ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]).into_dyn()))];
    let out = model.pipeline_in(Some(&s2), "twice", arg(), None).unwrap(); // s2: 4 -> 6
    assert_eq!(out[0].1.as_array().unwrap().iter().copied().collect::<Vec<_>>(), n_times(6.0));
    assert_eq!(add(Some(&s2), 0.0).0, n_times(7.0));
    assert_eq!(add(None, 0.0).0, n_times(2.0));

    // Dropping the session that is resident hands the live state back to the default.
    drop(s2);
    assert_eq!(add(None, 0.0).0, n_times(3.0));
    drop(s1);
    assert_eq!(add(None, 0.0).0, n_times(4.0));

    // A fork starts as a copy and then goes its own way, whether or not the original is the
    // session whose data is live.
    let a = model.new_state().unwrap();
    for n in 1..=3 {
        assert_eq!(add(Some(&a), 0.0).0, n_times(n as f32));
    }
    add(None, 0.0); // a is not resident now
    let b = a.fork().unwrap();
    assert_eq!(add(Some(&b), 0.0).0, n_times(4.0));
    assert_eq!(add(Some(&a), 0.0).0, n_times(4.0)); // not affected by b
    let c = a.fork().unwrap(); // a is resident now
    assert_eq!(add(Some(&c), 0.0).0, n_times(5.0));
    assert_eq!(add(Some(&a), 0.0).0, n_times(5.0));
    assert_eq!(add(Some(&b), 0.0).0, n_times(5.0));
    drop(a);
    assert_eq!(add(Some(&c), 0.0).0, n_times(6.0)); // forks outlive the original

    // A state is only good for its own model.
    let other = Model::load(&path, device).unwrap();
    let options = RunOptions { state: Some(other.new_state().unwrap()), ..Default::default() };
    let at = ndarray::arr2(&[[0.0f32]]).into_dyn();
    let err = model.run_entry("add", &[("x", x.view()), ("at", at.view())], &options).unwrap_err().to_string();
    assert!(err.contains("belongs to another model"), "{err}");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn sessions_have_their_own_state() {
    sessions_on(Device::Cpu);
    if Model::load(write("probe", "(model (inputs [x f32]) (outputs [y x]))"), Device::Gpu).is_ok() {
        sessions_on(Device::Gpu);
    }
    if cfg!(target_os = "macos") {
        sessions_on(Device::Native);
    }
}

#[test]
fn a_pipeline_waiting_for_input_leaves_the_scheme_thread_free() {
    use std::{sync::mpsc, time::Duration};
    use tensorlisp::{Chunk, input_stream};

    let path = write("yield", STREAM_PROGRAM);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let start = |label: &str| {
        let (tx, rx) = input_stream(|| {});
        let (out_tx, out_rx) = mpsc::channel::<(String, Chunk)>();
        let emit: tensorlisp::Emit = std::sync::Arc::new(move |name, chunk| out_tx.send((name.to_string(), chunk)).is_ok());
        let model = model.clone();
        let label = label.to_string();
        let worker = std::thread::spawn(move || {
            model.pipeline_streaming(
                "running",
                vec![("xs".into(), RawInput::Stream(rx)), ("label".into(), RawInput::Text(label))],
                Some(emit),
            )
        });
        (tx, out_rx, worker)
    };
    let chunk = |v: f32| Chunk::Array(ndarray::arr2(&[[v, 0.0, 0.0, 1.0]]).into_dyn());
    let wait = Duration::from_secs(30);

    // Two pipelines, both waiting for their first chunk.
    let (tx_a, out_a, worker_a) = start("A");
    let (tx_b, out_b, worker_b) = start("B");
    std::thread::sleep(Duration::from_millis(300));

    // Meanwhile the Scheme thread serves other work: a pipeline that needs no input, and an entry run
    // with a shape it has never seen (building its graph takes the Scheme thread).
    let (done_tx, done_rx) = mpsc::channel();
    {
        let model = model.clone();
        std::thread::spawn(move || {
            let x = ndarray::Array2::<f32>::ones((3, 4)).into_dyn();
            let out = model.run_entry("double", &[("x", x.view())], &RunOptions::default());
            done_tx.send(out.map(|o| o.outputs[0].1.shape().to_vec())).unwrap();
        });
    }
    let shape = done_rx.recv_timeout(wait).expect("an entry ran while pipelines were waiting").unwrap();
    assert_eq!(shape, [3, 4]);

    // B is answered before A has had anything: they don't wait for each other.
    tx_b.send(chunk(2.0)).unwrap();
    let (name, Chunk::Array(y)) = out_b.recv_timeout(wait).unwrap() else { panic!() };
    assert_eq!((name.as_str(), y[[0, 0]]), ("y", 4.0));
    assert!(out_a.try_recv().is_err());
    tx_a.send(chunk(3.0)).unwrap();
    let (_, Chunk::Array(y)) = out_a.recv_timeout(wait).unwrap() else { panic!() };
    assert_eq!(y[[0, 0]], 6.0);

    drop(tx_a);
    drop(tx_b);
    for worker in [worker_a, worker_b] {
        let results = worker.join().unwrap().unwrap();
        assert_eq!(results[0].1.as_array().unwrap().iter().copied().collect::<Vec<_>>(), [1.0]);
    }
    std::fs::remove_file(path).unwrap();
}

const CALLBACK_PROGRAM: &str = r#"
  (model id (inputs [x f32 (4 1)]) (outputs [y (ggml-scale x 1.0) 1]))
  (pipeline chat ([prompt string])
    (let* ([reply (client-call "greet" prompt 3 (list->array '(1 2)))]
           [first (client-sample (list->array '(0.1 2.5 -1.0 2.5 0.3)) '(7 8) 3)]
           [second (client-sample (list->array '(5.0 1.0)) (list->array '(1)) 2 #t)])
      (client-call "log" "done")
      (results [reply reply] [first first] [second second])))
  (pipeline broken ([prompt string]) (client-call "fail") (results))
"#;

#[test]
fn pipelines_call_back_into_the_client() {
    use std::sync::{Arc, Mutex};
    use tensorlisp::{Callbacks, Chunk, PipelineOptions, SampleRequest};

    #[derive(Default)]
    struct Client {
        log: Mutex<Vec<String>>,
    }
    impl Callbacks for Client {
        fn call(&self, name: &str, args: Vec<Chunk>) -> Result<Option<Chunk>, String> {
            match name {
                "greet" => {
                    let (Chunk::Text(p), Chunk::Array(n), Chunk::Array(v)) = (&args[0], &args[1], &args[2]) else {
                        return Err("bad arguments".into());
                    };
                    assert_eq!((n.ndim(), n.iter().copied().collect::<Vec<_>>()), (0, vec![3.0]));
                    self.log.lock().unwrap().push(format!("greet {p} {:?}", v.iter().copied().collect::<Vec<_>>()));
                    Ok(Some(Chunk::Text(format!("hello {p}"))))
                }
                "log" => {
                    self.log.lock().unwrap().push("log".into());
                    Ok(None)
                }
                _ => Err(format!("no handler {name:?}")),
            }
        }

        fn sample(&self, r: SampleRequest) -> Result<i32, String> {
            self.log.lock().unwrap().push(format!(
                "sample {} {:?} {:?} full={}",
                r.step,
                r.candidates,
                r.history,
                r.full.as_ref().map_or(0, |f| f.len())
            ));
            // Pick the second-best candidate.
            Ok(r.candidates[1].0)
        }
    }

    let path = write("callbacks", CALLBACK_PROGRAM);
    let model = Model::load(&path, Device::Cpu).unwrap();
    let client = Arc::new(Client::default());
    let options = || PipelineOptions { callbacks: Some(client.clone()), ..Default::default() };
    let example = || vec![("prompt".to_string(), RawInput::Text("world".into()))];
    let results = model.pipeline_with("chat", example(), options()).unwrap();
    let get = |name: &str| &results.iter().find(|(n, _)| n == name).unwrap().1;
    assert_eq!(get("reply").as_text(), Some("hello world"));
    assert_eq!(get("first").as_array().unwrap().iter().copied().collect::<Vec<_>>(), [3.0]);
    assert_eq!(get("second").as_array().unwrap().iter().copied().collect::<Vec<_>>(), [1.0]);
    assert_eq!(
        *client.log.lock().unwrap(),
        [
            "greet world [1.0, 2.0]",
            "sample 0 [(1, 2.5), (3, 2.5), (4, 0.3)] [7, 8] full=0",
            "sample 1 [(0, 5.0), (1, 1.0)] [1] full=2",
            "log"
        ]
    );

    // Errors from the client, and a pipeline started without callbacks, name the problem.
    let err = model.pipeline_with("broken", example(), options()).unwrap_err().to_string();
    assert!(err.contains("no handler \"fail\""), "{err}");
    let err = model.pipeline("chat", example()).unwrap_err().to_string();
    assert!(err.contains("started without callbacks"), "{err}");
    std::fs::remove_file(path).unwrap();
}
