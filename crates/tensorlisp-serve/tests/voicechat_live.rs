//! VoiceChat as a live service (ports/voicechat, `listen` in stream.ss): 16 kHz audio is pushed in 80 ms pieces, paced to
//! real time, and the agent's speech comes back frame by frame. Ignored by default; needs the GGUF with the program packed
//! in (a q4 file with the TTS tables fits a 16 GiB machine, see the port's README) and a 16 kHz mono wav:
//!
//!   TL_VC_MODEL=/path/voicechat-q4-tts-live.gguf TL_VC_WAV=/path/user.wav [TL_VC_OUT=out.wav] [TL_VC_PIECE_MS=80] \
//!   cargo test --release -p tensorlisp-serve --test voicechat_live -- --ignored --nocapture
//!
//! With TL_VC_TOOLS=tools.json (a JSON array of tools) the `listen-tools` pipeline runs with a system prompt listing them,
//! and the tool calls come back to this test as `Callbacks.call("tool", [call])`, answered with TL_VC_REPLY (default
//! {"result": 20}).
use std::{path::PathBuf, time::{Duration, Instant}};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tensorlisp::Device;
use tensorlisp_serve::{
    Registry, serve_stream,
    store::Store,
    tensorlisp_capnp::{callbacks, model, run, server, sink, source, value},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

/// The client's side of a tool call: records the call and answers with a fixed response.
struct Tool {
    reply: String,
    calls: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
}

impl callbacks::Server for Tool {
    async fn call(
        self: capnp::capability::Rc<Self>,
        params: callbacks::CallParams,
        mut results: callbacks::CallResults,
    ) -> Result<(), capnp::Error> {
        let p = params.get()?;
        assert_eq!(p.get_name()?.to_str()?, "tool");
        let value::Which::Text(call) = p.get_args()?.get(0).which()? else { panic!("the call is text") };
        self.calls.borrow_mut().push(call?.to_str()?.to_owned());
        results.get().init_result().set_text(&self.reply);
        Ok(())
    }
}

async fn output(run: &run::Client, name: &str) -> source::Client {
    let mut r = run.output_request();
    r.get().set_name(name);
    r.send().promise.await.unwrap().get().unwrap().get_source().unwrap()
}

async fn push(sink: &sink::Client, samples: &[f32]) {
    let mut req = sink.push_request();
    {
        let mut a = req.get().init_chunk().init_audio();
        a.set_sample_rate(16000);
        a.set_channels(1);
        let mut list = a.init_samples(samples.len() as u32);
        for (i, &v) in samples.iter().enumerate() {
            list.set(i as u32, v);
        }
    }
    req.send().await.unwrap();
}

fn write_wav(path: &str, rate: u32, samples: &[f32]) {
    let mut b = Vec::new();
    let n = samples.len() as u32 * 2;
    b.extend(b"RIFF");
    b.extend((36 + n).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(rate.to_le_bytes());
    b.extend((rate * 2).to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend(n.to_le_bytes());
    for s in samples {
        b.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn voicechat_listens_and_speaks() {
    let (Ok(model_path), Ok(wav)) = (std::env::var("TL_VC_MODEL"), std::env::var("TL_VC_WAV")) else {
        eprintln!("TL_VC_MODEL and TL_VC_WAV are not set, skipping");
        return;
    };
    let audio = tensorlisp::autopro::audio::Audio::from_wav_bytes(&std::fs::read(&wav).unwrap()).unwrap();
    assert_eq!(audio.sample_rate, 16000);
    let piece_ms: usize = std::env::var("TL_VC_PIECE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(80);
    let piece = piece_ms * 16;
    eprintln!("input: {:.2} s in {piece_ms} ms pieces", audio.samples.len() as f64 / 16000.0);

    tokio::task::LocalSet::new()
        .run_until(async {
            let dir = std::env::temp_dir().join(format!("tl-serve-vc-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let registry = Registry::new(Store::open(dir.join("store")).unwrap(), Device::Gpu);
            let socket = dir.join("s.sock");
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

            let t0 = Instant::now();
            let mut req = srv.install_from_request();
            {
                let mut p = req.get();
                p.set_location(&model_path);
                p.init_options().set_name("voicechat");
            }
            let model: model::Client = req.send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            eprintln!("installed and loaded in {:.1} s", t0.elapsed().as_secs_f64());

            let tools = std::env::var("TL_VC_TOOLS").ok().map(|p| std::fs::read_to_string(p).unwrap());
            let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                if let Some(tools) = &tools {
                    p.set_name("listen-tools");
                    let mut args = p.reborrow().init_args(3);
                    for (i, (k, v)) in [("system", ""), ("tools", tools.as_str()), ("reply", "")].iter().enumerate() {
                        let mut a = args.reborrow().get(i as u32);
                        a.set_name(*k);
                        a.init_value().set_text(*v);
                    }
                    let reply = std::env::var("TL_VC_REPLY").unwrap_or_else(|_| "{\"result\": 20}".into());
                    p.set_callbacks(capnp_rpc::new_client(Tool { reply, calls: calls.clone() }));
                } else {
                    p.set_name("listen");
                }
            }
            let run: run::Client = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let sink = {
                let mut r = run.input_request();
                r.get().set_name("audio");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let speech = output(&run, "audio").await;
            let tokens = output(&run, "token").await;

            // the model needs a while to warm up the speech decoder before it reads the first piece
            tokio::time::sleep(Duration::from_secs(10)).await;
            let start = Instant::now();
            let collect_tokens = tokio::task::spawn_local(async move {
                let mut got = Vec::new();
                loop {
                    let pulled = tokens.pull_request().send().promise.await.unwrap();
                    let pulled = pulled.get().unwrap();
                    for c in pulled.get_chunks().unwrap().iter() {
                        let value::Which::Text(t) = c.which().unwrap() else { panic!() };
                        got.push((start.elapsed(), t.unwrap().to_str().unwrap().to_owned()));
                    }
                    if pulled.get_eof() {
                        return got;
                    }
                }
            });
            let collect_audio = tokio::task::spawn_local(async move {
                let mut frames = Vec::new();
                let mut samples = Vec::new();
                loop {
                    let pulled = speech.pull_request().send().promise.await.unwrap();
                    let pulled = pulled.get().unwrap();
                    for c in pulled.get_chunks().unwrap().iter() {
                        let value::Which::Tensor(t) = c.which().unwrap() else { panic!() };
                        let t = t.unwrap();
                        samples.extend(tensorlisp_serve::convert::decode(t.get_dtype().unwrap(), t.get_data().unwrap()).unwrap());
                        frames.push(start.elapsed());
                    }
                    if pulled.get_eof() {
                        return (frames, samples);
                    }
                }
            });
            for (i, p) in audio.samples.chunks(piece).enumerate() {
                tokio::time::sleep_until(tokio::time::Instant::from_std(start + Duration::from_secs_f64(((i * piece + p.len()) as f64) / 16000.0))).await;
                push(&sink, p).await;
            }
            let t_end = start.elapsed();
            sink.done_request().send().promise.await.unwrap();
            let (frames, samples) = collect_audio.await.unwrap();
            let tokens = collect_tokens.await.unwrap();
            run.result_request().send().promise.await.unwrap();
            let total = start.elapsed();

            let text: String = tokens.iter().map(|(_, t)| t.as_str()).collect();
            eprintln!("agent text: {text:?}");
            eprintln!("audio ended after {:.2} s; {} speech frames ({:.2} s of speech) done after {:.2} s", t_end.as_secs_f64(), frames.len(), samples.len() as f64 / 22050.0, total.as_secs_f64());
            // frame k should be heard once k * 80 ms of the conversation have passed (plus the first frame's own wait)
            let lag: Vec<f64> = frames.iter().enumerate().map(|(k, t)| t.as_secs_f64() - 0.08 * (k as f64 + 1.0)).collect();
            eprintln!("frame lag behind real time: first {:.2} s, last {:.2} s, worst {:.2} s", lag[0], lag[lag.len() - 1], lag.iter().cloned().fold(f64::MIN, f64::max));
            if let Some((t, w)) = tokens.first() {
                eprintln!("first agent word {w:?} reached the client at {:.2} s", t.as_secs_f64());
            }
            // 196 tokens of 1764 samples, sent as 1758 + 195 x 1764 + 6 (the codec trims 6 samples at each end)
            assert_eq!(samples.len() % 1764, 0);
            assert!(!text.is_empty(), "the agent said something");
            if tools.is_some() {
                eprintln!("tool calls reaching the client: {:?}", calls.borrow());
            }
            if let Ok(out) = std::env::var("TL_VC_OUT") {
                write_wav(&out, 22050, &samples);
                eprintln!("wrote {out}");
            }
        })
        .await;
}
