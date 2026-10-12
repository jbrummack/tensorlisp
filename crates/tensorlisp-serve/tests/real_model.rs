//! The server on a real model: T5Gemma 2 (270M) with the demo pipelines of ports/t5gemma2 plus
//! `generate-stream`, `generate-sampled` and `chat` (see docs/design/serve.md, "On a real model").
//! Ignored by default; needs a GGUF made with `tl pack` from ports/t5gemma2/t5gemma2.ss and the
//! pipelines appended:
//!
//!   TL_SERVE_MODEL=/path/t5gemma2-rpc-q8.gguf TL_SERVE_IMAGE=models/t5gemma2-270m/bee.jpg \
//!   [TL_SERVE_DEVICE=auto|cpu|gpu|native] [TL_SERVE_SOCKET=/tmp/tl.sock (an already running tl-serve)] \
//!   cargo test --release -p tensorlisp-serve --test real_model -- --ignored --nocapture
use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    time::{Duration, Instant},
};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tensorlisp::{Device, Model, PipelineOptions, RawInput};
use tensorlisp_serve::{
    Registry, serve_stream,
    store::Store,
    tensorlisp_capnp::{DType, callbacks, model, run, server, session, value},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

const PROMPTS: [&str; 3] = ["The capital of France is", "The capital of Germany is", "Two plus two equals"];
const WAIT: Duration = Duration::from_secs(120);

fn device() -> Device {
    match std::env::var("TL_SERVE_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("gpu") => Device::Gpu,
        Ok("native") => Device::Native,
        _ => Device::Auto,
    }
}

fn texts(run_outputs: &[(String, tensorlisp::Value)], name: &str) -> String {
    run_outputs.iter().find(|(n, _)| n == name).unwrap().1.as_text().unwrap().to_owned()
}

fn tokens(run_outputs: &[(String, tensorlisp::Value)]) -> Vec<i64> {
    run_outputs.iter().find(|(n, _)| n == "tokens").unwrap().1.as_array().unwrap().iter().map(|&v| v as i64).collect()
}

/// The client's sampler: the best candidate, or (with `skip_first`) the second best at step 0.
struct Sampler {
    skip_first: bool,
    asked: Rc<RefCell<u32>>,
}

impl callbacks::Server for Sampler {
    async fn sample(
        self: capnp::capability::Rc<Self>,
        params: callbacks::SampleParams,
        mut results: callbacks::SampleResults,
    ) -> Result<(), capnp::Error> {
        let r = params.get()?.get_request()?;
        let ids: Vec<i32> = r.get_candidate_ids()?.iter().collect();
        *self.asked.borrow_mut() += 1;
        results.get().set_token(if self.skip_first && r.get_step() == 0 { ids[1] } else { ids[0] });
        Ok(())
    }
}

/// The same choice made inside the runtime, to compare against.
struct LocalSampler(bool);

impl tensorlisp::Callbacks for LocalSampler {
    fn call(&self, _: &str, _: Vec<tensorlisp::Chunk>) -> Result<Option<tensorlisp::Chunk>, String> {
        Err("no calls".into())
    }

    fn sample(&self, r: tensorlisp::SampleRequest) -> Result<i32, String> {
        Ok(if self.0 && r.step == 0 { r.candidates[1].0 } else { r.candidates[0].0 })
    }
}

async fn pipeline(model: &model::Client, name: &str, args: &[(&str, &str)], session: Option<&session::Client>) -> run::Client {
    let mut req = model.pipeline_request();
    {
        let mut p = req.get();
        p.set_name(name);
        let mut list = p.reborrow().init_args(args.len() as u32);
        for (i, (k, v)) in args.iter().enumerate() {
            let mut a = list.reborrow().get(i as u32);
            a.set_name(*k);
            a.init_value().set_text(*v);
        }
        if let Some(s) = session {
            p.set_session(s.clone());
        }
    }
    req.send().promise.await.unwrap().get().unwrap().get_run().unwrap()
}

/// text and token ids of a finished run
async fn finished(run: &run::Client) -> (String, Vec<i64>) {
    let res = tokio::time::timeout(WAIT, run.result_request().send().promise).await.expect("the run finishes").unwrap();
    let outs = res.get().unwrap().get_outputs().unwrap();
    let mut text = String::new();
    let mut ids = Vec::new();
    for o in outs.iter() {
        match (o.get_name().unwrap().to_str().unwrap(), o.get_value().unwrap().which().unwrap()) {
            ("text", value::Which::Text(t)) => text = t.unwrap().to_str().unwrap().to_owned(),
            ("tokens", value::Which::Tensor(t)) => {
                let t = t.unwrap();
                ids = tensorlisp_serve::convert::decode(t.get_dtype().unwrap(), t.get_data().unwrap()).unwrap().iter().map(|&v| v as i64).collect()
            }
            _ => {}
        }
    }
    (text, ids)
}


#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn t5gemma2_over_rpc() {
    let Ok(model_path) = std::env::var("TL_SERVE_MODEL") else {
        eprintln!("TL_SERVE_MODEL is not set, skipping");
        return;
    };
    let model_path = PathBuf::from(model_path);
    let image_path = std::env::var("TL_SERVE_IMAGE").ok().map(PathBuf::from);

    // References: the same pipelines run directly on the runtime.
    eprintln!("loading the reference model ...");
    let local = Model::load(&model_path, device()).unwrap();
    let ask = |p: &str| vec![("prompt".to_string(), RawInput::Text(p.to_string()))];
    let reference: Vec<_> = PROMPTS
        .iter()
        .map(|p| {
            let t0 = Instant::now();
            let out = local.pipeline("generate", ask(p)).unwrap();
            eprintln!("local generate {p:?} -> {:?} ({:.0} ms)", texts(&out, "text"), t0.elapsed().as_secs_f64() * 1e3);
            (texts(&out, "text"), tokens(&out))
        })
        .collect();
    let reference_caption = image_path.as_ref().map(|path| {
        let image = image::load_from_memory(&std::fs::read(path).unwrap()).unwrap();
        let t0 = Instant::now();
        let out = local
            .pipeline(
                "caption",
                vec![
                    ("image".to_string(), RawInput::Image(image)),
                    ("prompt".to_string(), RawInput::Text("<start_of_image> in this image, there is".into())),
                ],
            )
            .unwrap();
        eprintln!("local caption -> {:?} ({:.0} ms, the vision tower included)", texts(&out, "text"), t0.elapsed().as_secs_f64() * 1e3);
        texts(&out, "text")
    });
    let sampled_reference: Vec<_> = [false, true]
        .iter()
        .map(|&skip| {
            let out = local
                .pipeline_with(
                    "generate-sampled",
                    ask(PROMPTS[0]),
                    PipelineOptions {
                        callbacks: Some(std::sync::Arc::new(LocalSampler(skip))),
                        emit: Some(std::sync::Arc::new(|_, _| true)),
                        ..Default::default()
                    },
                )
                .unwrap();
            tokens(&out)
        })
        .collect();
    assert_eq!(sampled_reference[0], reference[0].1, "a client that always takes the best candidate is greedy decoding");
    assert_ne!(sampled_reference[1], reference[0].1, "skipping the best candidate once changes the output");
    drop(local);

    let tasks = tokio::task::LocalSet::new();
    tasks
        .run_until(async {
            // The server: in this process, or an external `tl-serve` on TL_SERVE_SOCKET.
            let socket = match std::env::var("TL_SERVE_SOCKET") {
                Ok(path) => PathBuf::from(path),
                Err(_) => {
                    let dir = std::env::temp_dir().join(format!("tl-serve-real-{}", std::process::id()));
                    let _ = std::fs::remove_dir_all(&dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    let registry = Registry::new(Store::open(dir.join("store")).unwrap(), device());
                    let socket = dir.join("tl.sock");
                    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
                    tokio::task::spawn_local(async move {
                        loop {
                            let (stream, _) = listener.accept().await.unwrap();
                            serve_stream(registry.clone(), stream);
                        }
                    });
                    socket
                }
            };
            let stream = tokio::net::UnixStream::connect(&socket).await.unwrap();
            let (r, w) = tokio::io::split(stream);
            let network = twoparty::VatNetwork::new(r.compat(), w.compat_write(), Side::Client, Default::default());
            let mut rpc = RpcSystem::new(Box::new(network), None);
            let srv: server::Client = rpc.bootstrap(Side::Server);
            tokio::task::spawn_local(rpc);

            // Install by streaming the file.
            let t0 = Instant::now();
            let mut req = srv.install_request();
            req.get().init_options().set_name("t5gemma2");
            let upload = req.send().promise.await.unwrap().get().unwrap().get_upload().unwrap();
            {
                use std::io::Read;
                let mut file = std::fs::File::open(&model_path).unwrap();
                let mut buf = vec![0u8; 4 << 20];
                loop {
                    let n = file.read(&mut buf).unwrap();
                    if n == 0 {
                        break;
                    }
                    let mut push = upload.push_request();
                    push.get().set_chunk(&buf[..n]);
                    push.send().await.unwrap();
                }
            }
            let model = upload.finish_request().send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            eprintln!("installed and loaded {} MB in {:.1} s", std::fs::metadata(&model_path).unwrap().len() >> 20, t0.elapsed().as_secs_f64());

            // describe sees the real program.
            let d = model.describe_request().send().promise.await.unwrap();
            let info = d.get().unwrap().get_info().unwrap();
            let pipelines: Vec<String> =
                info.get_pipelines().unwrap().iter().map(|p| p.get_name().unwrap().to_str().unwrap().to_owned()).collect();
            let state_bytes: u64 = info.get_state().unwrap().iter().map(|s| s.get_bytes()).sum();
            eprintln!("pipelines {pipelines:?}, {} state tensors, {} MB of state", info.get_state().unwrap().len(), state_bytes >> 20);
            for name in ["generate", "caption", "generate-stream", "generate-sampled", "chat"] {
                assert!(pipelines.iter().any(|p| p == name), "{name} missing from {pipelines:?}");
            }

            // 1. generate over RPC equals the local result.
            for (p, (text, ids)) in PROMPTS.iter().zip(&reference) {
                let t0 = Instant::now();
                let run = pipeline(&model, "generate", &[("prompt", p)], None).await;
                let (got, got_ids) = finished(&run).await;
                eprintln!("rpc generate {p:?} -> {got:?} ({:.0} ms)", t0.elapsed().as_secs_f64() * 1e3);
                assert_eq!((&got, &got_ids), (text, ids));
            }

            // 2. caption with an encoded image argument.
            if let (Some(path), Some(want)) = (&image_path, &reference_caption) {
                let mut req = model.pipeline_request();
                {
                    let mut p = req.get();
                    p.set_name("caption");
                    let mut args = p.init_args(2);
                    let mut a = args.reborrow().get(0);
                    a.set_name("image");
                    let mut img = a.init_value().init_image();
                    let bytes = std::fs::read(path).unwrap();
                    img.reborrow().init_format().set_encoded("jpeg");
                    img.set_pixels(&bytes);
                    let mut b = args.reborrow().get(1);
                    b.set_name("prompt");
                    b.init_value().set_text("<start_of_image> in this image, there is");
                }
                let t0 = Instant::now();
                let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
                let (got, _) = finished(&run).await;
                eprintln!("rpc caption -> {got:?} ({:.0} ms)", t0.elapsed().as_secs_f64() * 1e3);
                assert_eq!(&got, want);
            }

            // 3. Two sessions, two prompts, running at the same time: each keeps its own KV cache.
            let new_session = || async { model.session_request().send().promise.await.unwrap().get().unwrap().get_session().unwrap() };
            let (s1, s2) = (new_session().await, new_session().await);
            let t0 = Instant::now();
            let run1 = pipeline(&model, "generate", &[("prompt", PROMPTS[0])], Some(&s1)).await;
            let run2 = pipeline(&model, "generate", &[("prompt", PROMPTS[1])], Some(&s2)).await;
            let (a, b) = futures::future::join(finished(&run1), finished(&run2)).await;
            eprintln!("two sessions at once: {:.0} ms total", t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(a.1, reference[0].1);
            assert_eq!(b.1, reference[1].1);
            // And again with the sessions swapped around (state persists across calls).
            let run1 = pipeline(&model, "generate", &[("prompt", PROMPTS[2])], Some(&s2)).await;
            let run2 = pipeline(&model, "generate", &[("prompt", PROMPTS[0])], Some(&s1)).await;
            let (a, b) = futures::future::join(finished(&run1), finished(&run2)).await;
            assert_eq!(a.1, reference[2].1);
            assert_eq!(b.1, reference[0].1);

            // For contrast: the same two prompts at once without sessions share the model's one KV
            // cache. Not asserted (the interleaving is up to the scheduler), only reported.
            let run1 = pipeline(&model, "generate", &[("prompt", PROMPTS[0])], None).await;
            let run2 = pipeline(&model, "generate", &[("prompt", PROMPTS[1])], None).await;
            let (a, b) = futures::future::join(finished(&run1), finished(&run2)).await;
            eprintln!(
                "without sessions: outputs {} / {} the solo results",
                if a.1 == reference[0].1 { "match" } else { "differ from" },
                if b.1 == reference[1].1 { "match" } else { "differ from" }
            );

            // 4. Tokens as they are produced.
            let t0 = Instant::now();
            let run = pipeline(&model, "generate-stream", &[("prompt", PROMPTS[1])], None).await;
            let mut source_req = run.output_request();
            source_req.get().set_name("id");
            let ids_source = source_req.send().promise.await.unwrap().get().unwrap().get_source().unwrap();
            let mut got = Vec::new();
            let mut first = None;
            loop {
                let pulled = tokio::time::timeout(WAIT, ids_source.pull_request().send().promise).await.expect("a token").unwrap();
                let pulled = pulled.get().unwrap();
                for c in pulled.get_chunks().unwrap().iter() {
                    let value::Which::Tensor(t) = c.which().unwrap() else { panic!() };
                    let t = t.unwrap();
                    got.extend(tensorlisp_serve::convert::decode(t.get_dtype().unwrap(), t.get_data().unwrap()).unwrap().iter().map(|&v| v as i64));
                    first.get_or_insert_with(|| t0.elapsed());
                }
                if pulled.get_eof() {
                    break;
                }
            }
            let total = t0.elapsed();
            eprintln!("streamed {} tokens: first after {:.0} ms, last after {:.0} ms", got.len(), first.unwrap().as_secs_f64() * 1e3, total.as_secs_f64() * 1e3);
            assert_eq!(got, reference[1].1[1..], "the streamed ids are the generated tokens (without <bos>)");
            assert!(first.unwrap() < total, "the first token arrived before the run ended");
            assert_eq!(finished(&run).await.1, reference[1].1);

            // 5. The client picks the tokens.
            for (skip, want) in [(false, &sampled_reference[0]), (true, &sampled_reference[1])] {
                let asked = Rc::new(RefCell::new(0));
                let client: callbacks::Client = capnp_rpc::new_client(Sampler { skip_first: skip, asked: asked.clone() });
                let mut req = model.pipeline_request();
                {
                    let mut p = req.get();
                    p.set_name("generate-sampled");
                    let mut args = p.reborrow().init_args(1);
                    let mut a = args.reborrow().get(0);
                    a.set_name("prompt");
                    a.init_value().set_text(PROMPTS[0]);
                    p.set_callbacks(client);
                }
                let t0 = Instant::now();
                let run = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
                let (text, ids) = finished(&run).await;
                let n = *asked.borrow();
                eprintln!("client sampler (skip first: {skip}) -> {text:?}; {n} round trips, {:.1} ms per token", t0.elapsed().as_secs_f64() * 1e3 / n as f64);
                assert_eq!(&ids, want);
                assert_eq!(n as usize, ids.len() - 1, "one question per generated token");
            }

            // 6. A conversation: prompts in, replies out, one at a time.
            let run = pipeline(&model, "chat", &[], None).await;
            let mut r = run.input_request();
            r.get().set_name("prompts");
            let prompts = r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap();
            let mut r = run.output_request();
            r.get().set_name("reply");
            let replies = r.send().promise.await.unwrap().get().unwrap().get_source().unwrap();
            for (p, (want, _)) in PROMPTS.iter().zip(&reference) {
                let t0 = Instant::now();
                let mut push = prompts.push_request();
                push.get().init_chunk().set_text(*p);
                push.send().await.unwrap();
                let pulled = tokio::time::timeout(WAIT, replies.pull_request().send().promise).await.expect("a reply").unwrap();
                let chunks = pulled.get().unwrap().get_chunks().unwrap();
                let value::Which::Text(t) = chunks.get(0).which().unwrap() else { panic!() };
                let got = t.unwrap().to_str().unwrap().to_owned();
                eprintln!("chat {p:?} -> {got:?} ({:.0} ms)", t0.elapsed().as_secs_f64() * 1e3);
                assert_eq!(&got, want);
            }
            prompts.done_request().send().promise.await.unwrap();
            let (_, _) = finished(&run).await;
            let _ = DType::F32;
        })
        .await;
}
