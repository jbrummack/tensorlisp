//! A server and a client in one process over a unix socket: install by streaming a GGUF, describe,
//! run an entry with f32/f16 tensors, run a pipeline, and remove.
use std::{path::PathBuf, rc::Rc};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tensorlisp::{Device, Program, gguf::GgufWriter};
use tensorlisp_serve::{
    Registry, convert, serve_stream,
    store::Store,
    tensorlisp_capnp::{DType, RunStatus, server, tensor, value},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

const TOKENIZER_JSON: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "[UNK]": 1, "a": 2, "photo": 3, "of": 4, "dog": 5}, "unk_token": "[UNK]"}
}"#;

const PROGRAM: &str = r#"
  (define tok (tokenizer (asset "tokenizer.json")))
  (model double (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))
  (define-state "sum" f32 4)
  ;; sum += x, per session
  (model acc (inputs [x f32 (4 1)])
    (effect (ggml-cpy (ggml-add (state "sum") (ggml-reshape-1d x 4)) (state "sum")))
    (outputs [sum (ggml-scale (state "sum") 1.0) 1]))
  (pipeline acc-twice ([x array])
    (run acc [x x])
    (results [sum (output (run acc [x x]) 'sum)]))
  (pipeline chat ([prompt string])
    (let* ([reply (client-call "greet" prompt 3 (list->array '(1 2)))]
           [first (client-sample (list->array '(0.1 2.5 -1.0 2.5 0.3)) '(7 8) 3)]
           [second (client-sample (list->array '(5.0 1.0)) (list->array '(1)) 2 #t)])
      (client-call "log" "done")
      (results [reply reply] [first first] [second second])))
  (pipeline broken ([prompt string]) (client-call "fail") (results))
  (pipeline chat-later ([xs stream])
    (stream-recv xs)
    (client-call "log" "later")
    (results [done 1]))
  (model triple (inputs [x f32 (4 batch)] [n i32 (1 batch)])
    (outputs [y (ggml-scale x 3.0) 2] [n (ggml-scale (ggml-cast n GGML_TYPE_F32) 1.0) 2]))
  (pipeline grow ([x array] [times string])
    (let loop ([x x] [n (string->number times)])
      (if (= n 0)
          (let ([r (run triple [x x] [n (list->array '(7))])])
            (results [x (output r 'y)] [text (detokenize tok '(2 3 4 5))]))
          (loop (output (run double [x x]) 'y) (- n 1)))))
;; Sends every audio or image chunk straight back.
  (pipeline echo ([a stream])
    (let loop ([n 0])
      (let ([c (stream-recv a)])
        (if c (begin (emit 'back c) (loop (+ n 1))) (results [count n])))))
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
"#;

fn make_gguf(dir: &std::path::Path) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text(PROGRAM.into())).unwrap();
    w.set_asset("tokenizer.json", TOKENIZER_JSON.as_bytes()).unwrap();
    let path = dir.join("source.gguf");
    w.write(&path).unwrap();
    std::fs::read(path).unwrap()
}

fn set_tensor(mut v: value::Builder<'_>, dtype: DType, dims: &[u64], values: &[f32]) {
    let mut t: tensor::Builder<'_> = v.reborrow().init_tensor();
    t.set_dtype(dtype);
    let mut d = t.reborrow().init_dims(dims.len() as u32);
    for (i, &x) in dims.iter().enumerate() {
        d.set(i as u32, x);
    }
    t.set_data(&convert::encode(dtype, values).unwrap());
}

fn read_values(t: tensor::Reader<'_>) -> (DType, Vec<u64>, Vec<f32>) {
    let dims = t.get_dims().unwrap().iter().collect();
    let dtype = t.get_dtype().unwrap();
    (dtype, dims, convert::decode(dtype, t.get_data().unwrap()).unwrap())
}

#[tokio::test(flavor = "current_thread")]
async fn install_run_and_remove() {
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-serve-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let gguf = make_gguf(&dir);

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            // Server.
            let registry = Registry::new(Store::open(dir.join("store")).unwrap(), Device::Cpu);
            let socket = dir.join("tl.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            tokio::task::spawn_local(async move {
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    serve_stream(registry.clone(), stream);
                }
            });

            // Client.
            let stream = tokio::net::UnixStream::connect(&socket).await.unwrap();
            let (r, w) = tokio::io::split(stream);
            let network = twoparty::VatNetwork::new(r.compat(), w.compat_write(), Side::Client, Default::default());
            let mut rpc = RpcSystem::new(Box::new(network), None);
            let srv: server::Client = rpc.bootstrap(Side::Server);
            tokio::task::spawn_local(rpc);

            let info = srv.info_request().send().promise.await.unwrap();
            assert_eq!(info.get().unwrap().get_devices().unwrap().get(0).get_name().unwrap().to_str().unwrap(), "cpu");

            // Install by streaming the file in small chunks.
            let mut req = srv.install_request();
            req.get().init_options().set_name("dbl");
            let upload = req.send().promise.await.unwrap().get().unwrap().get_upload().unwrap();
            for chunk in gguf.chunks(4096) {
                let mut push = upload.push_request();
                push.get().set_chunk(chunk);
                push.send().await.unwrap();
            }
            let model = upload.finish_request().send().promise.await.unwrap().get().unwrap().get_model().unwrap();

            let listing = srv.list_request().send().promise.await.unwrap();
            let models = listing.get().unwrap().get_models().unwrap();
            assert_eq!(models.len(), 1);
            let id = models.get(0).get_id().unwrap().to_str().unwrap().to_owned();
            assert_eq!(models.get(0).get_name().unwrap().to_str().unwrap(), "dbl");
            assert!(models.get(0).get_loaded());
            assert_eq!(id.len(), 64);

            // Installing the same bytes again is the same model.
            let again = {
                let mut req = srv.install_request();
                req.get().init_options().set_name("copy");
                let up = req.send().promise.await.unwrap().get().unwrap().get_upload().unwrap();
                let mut push = up.push_request();
                push.get().set_chunk(&gguf);
                push.send().await.unwrap();
                let m = up.finish_request().send().promise.await.unwrap().get().unwrap().get_model().unwrap();
                let d = m.describe_request().send().promise.await.unwrap();
                d.get().unwrap().get_info().unwrap().get_id().unwrap().to_str().unwrap().to_owned()
            };
            assert_eq!(again, id);

            // describe.
            let d = model.describe_request().send().promise.await.unwrap();
            let info = d.get().unwrap().get_info().unwrap();
            let entries: Vec<String> =
                info.get_entries().unwrap().iter().map(|e| e.get_name().unwrap().to_str().unwrap().to_owned()).collect();
            assert_eq!(entries, ["double", "acc", "triple"]);
            let double_x = info.get_entries().unwrap().get(0).get_inputs().unwrap().get(0);
            assert_eq!(double_x.get_name().unwrap().to_str().unwrap(), "x");
            let pipelines = info.get_pipelines().unwrap();
            let grow = pipelines.iter().find(|p| p.get_name().unwrap().to_str().unwrap() == "grow").unwrap();
            assert_eq!(grow.get_inputs().unwrap().len(), 2);

            // run with f16 in and out.
            let x = [1.0f32, 2.0, 3.0, 4.0];
            let mut req = model.run_request();
            {
                let mut p = req.get();
                p.set_entry("double");
                let mut inputs = p.reborrow().init_inputs(1);
                let mut i0 = inputs.reborrow().get(0);
                i0.set_name("x");
                set_tensor(i0.init_value(), DType::F16, &[1, 4], &x);
                p.init_options().set_output_dtype(DType::F16);
            }
            let res = req.send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
            let (dtype, dims, values) = read_values(t.unwrap());
            assert_eq!((dtype, dims, values), (DType::F16, vec![1, 4], vec![2.0, 4.0, 6.0, 8.0]));

            // run with the result left on the server, then read it back.
            let mut req = model.run_request();
            {
                let mut p = req.get();
                p.set_entry("double");
                let mut inputs = p.reborrow().init_inputs(1);
                let mut i0 = inputs.reborrow().get(0);
                i0.set_name("x");
                set_tensor(i0.init_value(), DType::F32, &[1, 4], &x);
                p.init_options().set_outputs_by_ref(true);
            }
            let res = req.send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            let value::Which::TensorRef(r) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
            let r = r.unwrap();
            let info = r.info_request().send().promise.await.unwrap();
            assert_eq!(info.get().unwrap().get_dims().unwrap().iter().collect::<Vec<_>>(), [1, 4]);
            let data = r.read_request().send().promise.await.unwrap();
            let bytes = data.get().unwrap().get_data().unwrap();
            assert_eq!(convert::decode(DType::F32, bytes).unwrap(), [2.0, 4.0, 6.0, 8.0]);

            // An error names the problem and leaves the connection usable.
            let mut req = model.run_request();
            req.get().set_entry("nope");
            let err = req.send().promise.await.err().unwrap().to_string();
            assert!(err.contains("no entry \"nope\"; entries: double, acc, triple"), "{err}");

            // pipeline: x doubled twice, then tripled; a tensor argument and a text argument.
            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("grow");
                let mut args = p.init_args(2);
                let mut a0 = args.reborrow().get(0);
                a0.set_name("x");
                set_tensor(a0.init_value(), DType::F32, &[1, 4], &x);
                let mut a1 = args.reborrow().get(1);
                a1.set_name("times");
                a1.init_value().set_text("2");
            }
            let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let res = run.result_request().send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            assert_eq!(outs.get(0).get_name().unwrap().to_str().unwrap(), "x");
            let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(read_values(t.unwrap()).2, [12.0, 24.0, 36.0, 48.0]);
            let value::Which::Text(text) = outs.get(1).get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(text.unwrap().to_str().unwrap(), "a photo of dog");
            let status = run.status_request().send().promise.await.unwrap();
            assert_eq!(status.get().unwrap().get_status().unwrap(), RunStatus::Finished);

            // A missing argument is reported when the run starts.
            let mut req = model.pipeline_request();
            req.get().set_name("grow");
            let err = req.send().promise.await.err().unwrap().to_string();
            assert!(err.contains("needs an argument \"x\""), "{err}");

            // open by alias, then remove: both names go, so the file does too.
            let mut open = srv.open_request();
            open.get().set_name("dbl");
            open.send().promise.await.unwrap();
            for name in ["dbl", "copy"] {
                let mut rm = srv.remove_request();
                rm.get().set_name(name);
                rm.send().promise.await.unwrap();
            }
            let mut open = srv.open_request();
            open.get().set_name("dbl");
            assert!(open.send().promise.await.is_err());
            let listing = srv.list_request().send().promise.await.unwrap();
            assert_eq!(listing.get().unwrap().get_models().unwrap().len(), 0);
        })
        .await;
    let _ = Rc::new(());
    std::fs::remove_dir_all(dir).unwrap();
}

/// Starts an in-process server and returns a client to it with the test model installed.
async fn setup(dir: &std::path::Path) -> (server::Client, tensorlisp_serve::tensorlisp_capnp::model::Client) {
    let registry = Registry::new(Store::open(dir.join("store")).unwrap(), Device::Cpu);
    let socket = dir.join("tl.sock");
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    tokio::task::spawn_local(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            serve_stream(registry.clone(), stream);
        }
    });
    let stream = tokio::net::UnixStream::connect(&socket).await.unwrap();
    let (r, w) = tokio::io::split(stream);
    let network = twoparty::VatNetwork::new(r.compat(), w.compat_write(), Side::Client, Default::default());
    let mut rpc = RpcSystem::new(Box::new(network), None);
    let srv: server::Client = rpc.bootstrap(Side::Server);
    tokio::task::spawn_local(rpc);

    let mut req = srv.install_request();
    req.get().init_options().set_name("m");
    let upload = req.send().promise.await.unwrap().get().unwrap().get_upload().unwrap();
    let mut push = upload.push_request();
    push.get().set_chunk(&make_gguf(dir));
    push.send().await.unwrap();
    let model = upload.finish_request().send().promise.await.unwrap().get().unwrap().get_model().unwrap();
    (srv, model)
}

fn chunk(i: usize) -> [f32; 4] {
    [i as f32, 0.0, 0.0, 1.0]
}

#[tokio::test(flavor = "current_thread")]
async fn pipelines_stream_in_and_out() {
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-serve-stream-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (_srv, model) = setup(&dir).await;

            let d = model.describe_request().send().promise.await.unwrap();
            let pipelines = d.get().unwrap().get_info().unwrap().get_pipelines().unwrap();
            let running = pipelines.iter().find(|p| p.get_name().unwrap().to_str().unwrap() == "running").unwrap();
            assert!(running.get_streaming_in());

            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("running");
                let mut args = p.reborrow().init_args(1);
                let mut a = args.reborrow().get(0);
                a.set_name("label");
                a.init_value().set_text("L");
                p.init_options().set_output_dtype(DType::F16);
            }
            let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let sink = {
                let mut r = run.input_request();
                r.get().set_name("xs");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let y = {
                let mut r = run.output_request();
                r.get().set_name("y");
                r.send().promise.await.unwrap().get().unwrap().get_source().unwrap()
            };
            let notes = {
                let mut r = run.output_request();
                r.get().set_name("note");
                r.send().promise.await.unwrap().get().unwrap().get_source().unwrap()
            };

            // While the pipeline waits for its first chunk, the runtime serves other calls, even ones
            // that need the Scheme thread (a shape the entry has not been built for).
            let mut req = model.run_request();
            {
                let mut p = req.get();
                p.set_entry("double");
                let mut inputs = p.init_inputs(1);
                let mut i0 = inputs.reborrow().get(0);
                i0.set_name("x");
                set_tensor(i0.init_value(), DType::F32, &[3, 4], &[1.0; 12]);
            }
            let res = tokio::time::timeout(std::time::Duration::from_secs(30), req.send().promise)
                .await
                .expect("a run while a pipeline waits for input")
                .unwrap();
            assert_eq!(first_values(&res).len(), 12);

            // Each chunk is answered before the next one goes in.
            for i in 1..=3 {
                let mut push = sink.push_request();
                set_tensor(push.get().init_chunk(), DType::F32, &[1, 4], &chunk(i));
                push.send().await.unwrap();
                let pulled = y.pull_request().send().promise.await.unwrap();
                let pulled = pulled.get().unwrap();
                assert!(!pulled.get_eof());
                let chunks = pulled.get_chunks().unwrap();
                assert_eq!(chunks.len(), 1);
                let value::Which::Tensor(t) = chunks.get(0).which().unwrap() else { panic!() };
                let (dtype, dims, values) = read_values(t.unwrap());
                let want = chunk(i).map(|v| v * 2.0).to_vec();
                assert_eq!((dtype, dims, values), (DType::F16, vec![1, 4], want));
            }
            let mut push = sink.push_request();
            push.get().init_chunk().set_text("hello");
            push.send().await.unwrap();

            // The notes: three labels and the pushed text, in order.
            let mut seen = Vec::new();
            while seen.len() < 4 {
                let mut r = notes.pull_request();
                r.get().set_max(8);
                let pulled = r.send().promise.await.unwrap();
                for c in pulled.get().unwrap().get_chunks().unwrap().iter() {
                    let value::Which::Text(t) = c.which().unwrap() else { panic!() };
                    seen.push(t.unwrap().to_str().unwrap().to_owned());
                }
            }
            assert_eq!(seen, ["L", "L", "L", "hello"]);

            sink.done_request().send().promise.await.unwrap();
            // Pushing after done is refused (a streaming call reports that on a later call, so
            // nothing is asserted on it here; the count below shows the chunk never arrived).
            let mut push = sink.push_request();
            set_tensor(push.get().init_chunk(), DType::F32, &[1, 4], &chunk(9));
            let _ = push.send().await;
            let end = y.pull_request().send().promise.await.unwrap();
            assert!(end.get().unwrap().get_eof());
            let res = run.result_request().send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            assert_eq!(outs.get(0).get_name().unwrap().to_str().unwrap(), "count");
            let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(read_values(t.unwrap()).2, [3.0]);

            // Audio and image chunks round trip (audio comes back as mono).
            let mut req = model.pipeline_request();
            req.get().set_name("echo");
            let echo = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let (sink, back) = {
                let mut r = echo.input_request();
                r.get().set_name("a");
                let sink = r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap();
                let mut r = echo.output_request();
                r.get().set_name("back");
                (sink, r.send().promise.await.unwrap().get().unwrap().get_source().unwrap())
            };
            let mut push = sink.push_request();
            {
                let mut a = push.get().init_chunk().init_audio();
                a.set_sample_rate(16000);
                a.set_channels(2);
                let mut samples = a.init_samples(4);
                for (i, v) in [1.0f32, 3.0, 2.0, 4.0].into_iter().enumerate() {
                    samples.set(i as u32, v);
                }
            }
            push.send().await.unwrap();
            let mut push = sink.push_request();
            {
                let mut img = push.get().init_chunk().init_image();
                img.set_width(2);
                img.set_height(1);
                img.reborrow().init_format().set_rgb8(());
                img.set_pixels(&[1, 2, 3, 4, 5, 6]);
            }
            push.send().await.unwrap();
            sink.done_request().send().promise.await.unwrap();
            let mut got = Vec::new();
            loop {
                let pulled = back.pull_request().send().promise.await.unwrap();
                let pulled = pulled.get().unwrap();
                for c in pulled.get_chunks().unwrap().iter() {
                    match c.which().unwrap() {
                        value::Which::Audio(a) => {
                            let a = a.unwrap();
                            got.push(format!("audio {} {} {:?}", a.get_sample_rate(), a.get_channels(), a.get_samples().unwrap().iter().collect::<Vec<_>>()));
                        }
                        value::Which::Image(i) => {
                            let i = i.unwrap();
                            got.push(format!("image {}x{} {:?}", i.get_width(), i.get_height(), i.get_pixels().unwrap()));
                        }
                        _ => panic!("unexpected chunk"),
                    }
                }
                if pulled.get_eof() {
                    break;
                }
            }
            assert_eq!(got, ["audio 16000 1 [2.0, 3.0]", "image 2x1 [1, 2, 3, 4, 5, 6]"]);

            // Unknown stream name.
            let mut r = run.input_request();
            r.get().set_name("nope");
            let err = r.send().promise.await.err().unwrap().to_string();
            assert!(err.contains("no input stream \"nope\"; streams: xs"), "{err}");

            // A run that is cancelled while it waits for input ends instead of blocking the runtime.
            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("running");
                let mut args = p.init_args(1);
                let mut a = args.reborrow().get(0);
                a.set_name("label");
                a.init_value().set_text("L");
            }
            let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            run.cancel_request().send().promise.await.unwrap();
            let status = run.status_request().send().promise.await.unwrap();
            assert_eq!(status.get().unwrap().get_status().unwrap(), RunStatus::Cancelled);
            // The Scheme thread is free again: a plain entry still runs.
            let mut req = model.run_request();
            {
                let mut p = req.get();
                p.set_entry("double");
                let mut inputs = p.init_inputs(1);
                let mut i0 = inputs.reborrow().get(0);
                i0.set_name("x");
                set_tensor(i0.init_value(), DType::F32, &[1, 4], &chunk(1));
            }
            let res = tokio::time::timeout(std::time::Duration::from_secs(30), req.send().promise).await.expect("run after cancel").unwrap();
            assert_eq!(res.get().unwrap().get_outputs().unwrap().len(), 1);

            // A run the client lets go of while the pipeline waits for input also ends.
            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("running");
                let mut args = p.init_args(1);
                let mut a = args.reborrow().get(0);
                a.set_name("label");
                a.init_value().set_text("L");
            }
            drop(req.send().promise.await.unwrap());
            tokio::task::yield_now().await;
            let mut req = model.run_request();
            {
                let mut p = req.get();
                p.set_entry("double");
                let mut inputs = p.init_inputs(1);
                let mut i0 = inputs.reborrow().get(0);
                i0.set_name("x");
                set_tensor(i0.init_value(), DType::F32, &[1, 4], &chunk(2));
            }
            tokio::time::timeout(std::time::Duration::from_secs(30), req.send().promise).await.expect("run after drop").unwrap();
        })
        .await;
    std::fs::remove_dir_all(dir).unwrap();
}

fn first_values(res: &capnp::capability::Response<tensorlisp_serve::tensorlisp_capnp::model::run_results::Owned>) -> Vec<f32> {
    let outs = res.get().unwrap().get_outputs().unwrap();
    let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
    read_values(t.unwrap()).2
}

#[tokio::test(flavor = "current_thread")]
async fn sessions_keep_separate_state() {
    use tensorlisp_serve::tensorlisp_capnp::session;
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-serve-session-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (_srv, model) = setup(&dir).await;
            let new_session = || {
                let model = model.clone();
                async move { model.session_request().send().promise.await.unwrap().get().unwrap().get_session().unwrap() }
            };
            let acc = |session: Option<session::Client>| {
                let model = model.clone();
                async move {
                    let mut req = model.run_request();
                    {
                        let mut p = req.get();
                        p.set_entry("acc");
                        let mut inputs = p.reborrow().init_inputs(1);
                        let mut i0 = inputs.reborrow().get(0);
                        i0.set_name("x");
                        set_tensor(i0.init_value(), DType::F32, &[1, 4], &chunk(1));
                        if let Some(s) = session {
                            p.set_session(s);
                        }
                    }
                    first_values(&req.send().promise.await.unwrap())
                }
            };
            let (s1, s2) = (new_session().await, new_session().await);
            assert_eq!(acc(Some(s1.clone())).await, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(acc(Some(s1.clone())).await, [2.0, 0.0, 0.0, 2.0]);
            assert_eq!(acc(Some(s2.clone())).await, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(acc(None).await, [1.0, 0.0, 0.0, 1.0]); // the model's own state
            assert_eq!(acc(Some(s1.clone())).await, [3.0, 0.0, 0.0, 3.0]);

            // info lists the state; reset clears one session only.
            let info = s1.info_request().send().promise.await.unwrap();
            let info = info.get().unwrap();
            assert_eq!(info.get_bytes_used(), 16);
            assert_eq!(info.get_state().unwrap().get(0).get_name().unwrap().to_str().unwrap(), "sum");
            s1.reset_request().send().promise.await.unwrap();
            assert_eq!(acc(Some(s1.clone())).await, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(acc(Some(s2.clone())).await, [2.0, 0.0, 0.0, 2.0]);
            let mut reset = s1.reset_request();
            reset.get().init_names(1).set(0, "sum");
            assert!(reset.send().promise.await.is_err());

            // A pipeline runs in the session it was given.
            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("acc-twice");
                let mut args = p.reborrow().init_args(1);
                let mut a = args.reborrow().get(0);
                a.set_name("x");
                set_tensor(a.init_value(), DType::F32, &[1, 4], &chunk(1));
                p.set_session(s2.clone());
            }
            let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let res = run.result_request().send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(read_values(t.unwrap()).2, [4.0, 0.0, 0.0, 4.0]);
            assert_eq!(acc(Some(s2.clone())).await, [5.0, 0.0, 0.0, 5.0]);
            assert_eq!(acc(None).await, [2.0, 0.0, 0.0, 2.0]);

            // fork: a session that starts as a copy and then diverges.
            let forked = s2.fork_request().send().promise.await.unwrap().get().unwrap().get_session().unwrap();
            assert_eq!(acc(Some(forked.clone())).await, [6.0, 0.0, 0.0, 6.0]);
            assert_eq!(acc(Some(s2.clone())).await, [6.0, 0.0, 0.0, 6.0]);
            assert_eq!(acc(Some(forked.clone())).await, [7.0, 0.0, 0.0, 7.0]);
            // The fork is a session like any other (it can be forked, reset, passed back).
            forked.reset_request().send().promise.await.unwrap();
            assert_eq!(acc(Some(forked.clone())).await, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(acc(Some(s2.clone())).await, [7.0, 0.0, 0.0, 7.0]);
            assert_eq!(acc(None).await, [3.0, 0.0, 0.0, 3.0]);

            // describe lists the state tensors.
            let d = model.describe_request().send().promise.await.unwrap();
            let state = d.get().unwrap().get_info().unwrap().get_state().unwrap();
            assert_eq!((state.len(), state.get(0).get_bytes()), (1, 16));

            // Letting go of a session frees it; the others and the default carry on.
            drop((s1, s2, forked));
            assert_eq!(acc(None).await, [4.0, 0.0, 0.0, 4.0]);

            // A session capability that is not ours is refused.
            let foreign: session::Client = capnp_rpc::new_client(Foreign);
            let mut req = model.run_request();
            req.get().set_entry("acc");
            req.get().set_session(foreign);
            let err = req.send().promise.await.err().unwrap().to_string();
            assert!(err.contains("not one of this server's"), "{err}");
        })
        .await;
    std::fs::remove_dir_all(dir).unwrap();
}

/// A client-side session that claims a token no server issued.
struct Foreign;

impl tensorlisp_serve::tensorlisp_capnp::session::Server for Foreign {
    async fn token(
        self: capnp::capability::Rc<Self>,
        _: tensorlisp_serve::tensorlisp_capnp::session::TokenParams,
        mut results: tensorlisp_serve::tensorlisp_capnp::session::TokenResults,
    ) -> Result<(), capnp::Error> {
        results.get().set_token(12345);
        Ok(())
    }
}

/// The client's side of `Callbacks`, recording what it was asked.
struct TestClient {
    log: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
}

impl tensorlisp_serve::tensorlisp_capnp::callbacks::Server for TestClient {
    async fn call(
        self: capnp::capability::Rc<Self>,
        params: tensorlisp_serve::tensorlisp_capnp::callbacks::CallParams,
        mut results: tensorlisp_serve::tensorlisp_capnp::callbacks::CallResults,
    ) -> Result<(), capnp::Error> {
        let p = params.get()?;
        let name = p.get_name()?.to_str()?.to_owned();
        let args = p.get_args()?;
        match name.as_str() {
            "greet" => {
                let value::Which::Text(prompt) = args.get(0).which()? else { panic!() };
                let value::Which::Tensor(n) = args.get(1).which()? else { panic!() };
                let (_, dims, values) = read_values(n?);
                assert_eq!((dims, values), (vec![], vec![3.0]));
                let value::Which::Tensor(v) = args.get(2).which()? else { panic!() };
                let prompt = prompt?.to_str()?.to_owned();
                self.log.borrow_mut().push(format!("greet {prompt} {:?}", read_values(v?).2));
                results.get().init_result().set_text(format!("hello {prompt}").as_str());
            }
            "log" => {
                self.log.borrow_mut().push("log".into());
                results.get().init_result().set_none(());
            }
            other => return Err(capnp::Error::failed(format!("no handler {other:?}"))),
        }
        Ok(())
    }

    async fn sample(
        self: capnp::capability::Rc<Self>,
        params: tensorlisp_serve::tensorlisp_capnp::callbacks::SampleParams,
        mut results: tensorlisp_serve::tensorlisp_capnp::callbacks::SampleResults,
    ) -> Result<(), capnp::Error> {
        let r = params.get()?.get_request()?;
        let ids: Vec<i32> = r.get_candidate_ids()?.iter().collect();
        let logits: Vec<f32> = r.get_candidate_logits()?.iter().collect();
        let history: Vec<i32> = r.get_history()?.iter().collect();
        let mut full = 0;
        if r.has_full_logits() {
            let data = r.get_full_logits()?.read_request().send().promise.await?;
            full = convert::decode(DType::F32, data.get()?.get_data()?)?.len();
        }
        self.log.borrow_mut().push(format!("sample {} {ids:?} {logits:?} {history:?} full={full}", r.get_step()));
        results.get().set_token(ids[1]); // the second-best candidate
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn pipelines_call_back_into_the_client() {
    use tensorlisp_serve::tensorlisp_capnp::callbacks;
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-serve-callback-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (_srv, model) = setup(&dir).await;
            let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let start = |name: &str, handlers: Option<callbacks::Client>| {
                let mut req = model.pipeline_request();
                {
                    let mut p = req.get();
                    p.set_name(name);
                    let mut args = p.reborrow().init_args(1);
                    let mut a = args.reborrow().get(0);
                    a.set_name("prompt");
                    a.init_value().set_text("world");
                    if let Some(h) = handlers {
                        p.set_callbacks(h);
                    }
                }
                async move { req.send().promise.await.unwrap().get().unwrap().get_run().unwrap() }
            };
            let client: callbacks::Client = capnp_rpc::new_client(TestClient { log: log.clone() });

            let run = start("chat", Some(client.clone())).await;
            let res = run.result_request().send().promise.await.unwrap();
            let outs = res.get().unwrap().get_outputs().unwrap();
            let by_name = |n: &str| outs.iter().find(|o| o.get_name().unwrap().to_str().unwrap() == n).unwrap();
            let value::Which::Text(reply) = by_name("reply").get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(reply.unwrap().to_str().unwrap(), "hello world");
            let value::Which::Tensor(first) = by_name("first").get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(read_values(first.unwrap()).2, [3.0]);
            let value::Which::Tensor(second) = by_name("second").get_value().unwrap().which().unwrap() else { panic!() };
            assert_eq!(read_values(second.unwrap()).2, [1.0]);
            assert_eq!(
                *log.borrow(),
                [
                    "greet world [1.0, 2.0]",
                    "sample 0 [1, 3, 4] [2.5, 2.5, 0.3] [7, 8] full=0",
                    "sample 1 [0, 1] [5.0, 1.0] [1] full=2",
                    "log"
                ]
            );

            // An error in the client's handler becomes the run's error.
            let run = start("broken", Some(client.clone())).await;
            let err = run.result_request().send().promise.await.err().unwrap().to_string();
            assert!(err.contains("no handler \"fail\""), "{err}");

            // Without any callbacks the call fails with a clear message.
            let run = start("chat", None).await;
            let err = run.result_request().send().promise.await.err().unwrap().to_string();
            assert!(err.contains("without callbacks") || err.contains("no callbacks"), "{err}");

            // Callbacks can be set after the start: this pipeline waits for a chunk first.
            log.borrow_mut().clear();
            let mut req = model.pipeline_request();
            req.get().set_name("chat-later");
            let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let mut set = run.set_callbacks_request();
            set.get().set_callbacks(client.clone());
            set.send().promise.await.unwrap();
            let sink = {
                let mut r = run.input_request();
                r.get().set_name("xs");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let mut push = sink.push_request();
            push.get().init_chunk().set_text("go");
            push.send().await.unwrap();
            sink.done_request().send().promise.await.unwrap();
            run.result_request().send().promise.await.unwrap();
            assert_eq!(*log.borrow(), ["log"]);
        })
        .await;
    std::fs::remove_dir_all(dir).unwrap();
}
