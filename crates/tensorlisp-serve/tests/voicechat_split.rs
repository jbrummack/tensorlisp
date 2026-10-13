//! VoiceChat split over two servers (ports/voicechat, `brain` and `voice` in stream.ss): one server holds the encoder and
//! the language model (no speech weights) and turns the user's audio into a stream of [text token, in-call] frames, the
//! other holds the speech decoder and the codec and turns those frames into audio. This test is the relay between them.
//! Ignored by default; needs both GGUFs with the program packed in (see the port's README):
//!
//!   TL_VC_BRAIN=brain.gguf TL_VC_VOICE=voice.gguf TL_VC_WAV=/path/user.wav [TL_VC_OUT=out.wav] [TL_VC_TOOLS=tools.json] \
//!   cargo test --release -p tensorlisp-serve --test voicechat_split -- --ignored --nocapture
use std::{time::{Duration, Instant}};

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


async fn connect(dir: &std::path::Path, name: &str) -> server::Client {
    let registry = Registry::new(Store::open(dir.join(format!("store-{name}"))).unwrap(), Device::Gpu);
    let socket = dir.join(format!("{name}.sock"));
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
    srv
}

async fn install(srv: &server::Client, path: &str, name: &str) -> model::Client {
    let t0 = Instant::now();
    let mut req = srv.install_from_request();
    {
        let mut p = req.get();
        p.set_location(path);
        p.init_options().set_name(name);
    }
    let m = req.send().promise.await.unwrap().get().unwrap().get_model().unwrap();
    eprintln!("{name}: installed and loaded in {:.1} s", t0.elapsed().as_secs_f64());
    m
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn voicechat_split_over_two_servers() {
    let (Ok(brain_path), Ok(voice_path), Ok(wav)) =
        (std::env::var("TL_VC_BRAIN"), std::env::var("TL_VC_VOICE"), std::env::var("TL_VC_WAV"))
    else {
        eprintln!("TL_VC_BRAIN, TL_VC_VOICE and TL_VC_WAV are not set, skipping");
        return;
    };
    let audio = tensorlisp::autopro::audio::Audio::from_wav_bytes(&std::fs::read(&wav).unwrap()).unwrap();
    assert_eq!(audio.sample_rate, 16000);
    let piece = 80 * 16;
    eprintln!("input: {:.2} s", audio.samples.len() as f64 / 16000.0);

    tokio::task::LocalSet::new()
        .run_until(async {
            let dir = std::env::temp_dir().join(format!("tl-serve-vcs-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let voice_srv = connect(&dir, "voice").await;
            let voice_model = install(&voice_srv, &voice_path, "voice").await;
            let brain_srv = connect(&dir, "brain").await;
            let brain_model = install(&brain_srv, &brain_path, "brain").await;

            let tools = std::env::var("TL_VC_TOOLS").ok().map(|p| std::fs::read_to_string(p).unwrap());
            let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

            let mut req = voice_model.pipeline_request();
            req.get().set_name("voice");
            let voice: run::Client = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let frames_in = {
                let mut r = voice.input_request();
                r.get().set_name("frames");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let speech = output(&voice, "audio").await;

            let mut req = brain_model.pipeline_request();
            {
                let mut p = req.get();
                p.set_name("brain");
                let mut args = p.reborrow().init_args(3);
                for (i, (k, v)) in [("system", ""), ("tools", tools.as_deref().unwrap_or("")), ("reply", "")].iter().enumerate() {
                    let mut a = args.reborrow().get(i as u32);
                    a.set_name(*k);
                    a.init_value().set_text(*v);
                }
                let reply = std::env::var("TL_VC_REPLY").unwrap_or_else(|_| "{\"result\": 20}".into());
                p.set_callbacks(capnp_rpc::new_client(Tool { reply, calls: calls.clone() }));
            }
            let brain: run::Client = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let sink = {
                let mut r = brain.input_request();
                r.get().set_name("audio");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let frames_out = output(&brain, "frame").await;
            let tokens = output(&brain, "token").await;

            tokio::time::sleep(Duration::from_secs(10)).await;
            let start = Instant::now();
            // the relay: every frame the brain emits goes to the voice server (8 bytes of payload each)
            let relay = tokio::task::spawn_local(async move {
                let mut n = 0usize;
                loop {
                    let pulled = frames_out.pull_request().send().promise.await.unwrap();
                    let pulled = pulled.get().unwrap();
                    for c in pulled.get_chunks().unwrap().iter() {
                        let value::Which::Tensor(t) = c.which().unwrap() else { panic!() };
                        let array = tensorlisp_serve::convert::read_tensor(t.unwrap()).unwrap();
                        let mut req = frames_in.push_request();
                        tensorlisp_serve::convert::write_tensor(
                            req.get().init_chunk().init_tensor(),
                            &array,
                            tensorlisp_serve::tensorlisp_capnp::DType::F32,
                        )
                        .unwrap();
                        req.send().await.unwrap();
                        n += 1;
                    }
                    if pulled.get_eof() {
                        frames_in.done_request().send().promise.await.unwrap();
                        return n;
                    }
                }
            });
            let collect_tokens = tokio::task::spawn_local(async move {
                let mut got = String::new();
                loop {
                    let pulled = tokens.pull_request().send().promise.await.unwrap();
                    let pulled = pulled.get().unwrap();
                    for c in pulled.get_chunks().unwrap().iter() {
                        let value::Which::Text(t) = c.which().unwrap() else { panic!() };
                        got.push_str(t.unwrap().to_str().unwrap());
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
            let relayed = relay.await.unwrap();
            let (frames, samples) = collect_audio.await.unwrap();
            let text = collect_tokens.await.unwrap();
            brain.result_request().send().promise.await.unwrap();
            voice.result_request().send().promise.await.unwrap();
            let total = start.elapsed();
            eprintln!("agent text: {text:?}");
            eprintln!("{relayed} frames relayed; audio ended after {:.2} s; {:.2} s of speech done after {:.2} s", t_end.as_secs_f64(), samples.len() as f64 / 22050.0, total.as_secs_f64());
            let lag: Vec<f64> = frames.iter().enumerate().map(|(k, t)| t.as_secs_f64() - 0.08 * (k as f64 + 1.0)).collect();
            eprintln!("frame lag behind real time: first {:.2} s, last {:.2} s", lag[0], lag[lag.len() - 1]);
            assert_eq!(samples.len() % 1764, 0);
            assert!(!text.is_empty(), "the agent said something");
            if tools.is_some() {
                eprintln!("tool calls reaching the client: {:?}", calls.borrow());
            }
            if let Ok(out) = std::env::var("TL_VC_OUT") {
                write_wav(&out, 22050, &samples);
                eprintln!("wrote {out}");
            }
            let _ = std::fs::remove_dir_all(&dir);
        })
        .await;
}

/// The voice server alone, fed 40 made-up frames: checks that it consumes a stream of frames and speaks.
#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn voice_alone() {
    let Ok(voice_path) = std::env::var("TL_VC_VOICE") else { return };
    tokio::task::LocalSet::new()
        .run_until(async {
            let dir = std::env::temp_dir().join(format!("tl-serve-vcv-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let srv = connect(&dir, "voice").await;
            let model = install(&srv, &voice_path, "voice").await;
            let mut req = model.pipeline_request();
            req.get().set_name("voice");
            let voice: run::Client = req.send().promise.await.unwrap().get().unwrap().get_run().unwrap();
            let frames_in = {
                let mut r = voice.input_request();
                r.get().set_name("frames");
                r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
            };
            let speech = output(&voice, "audio").await;
            for (i, id) in [1u32, 3000, 4000, 5000, 6000].iter().cycle().take(40).enumerate() {
                let array = ndarray::arr1(&[*id as f32, 0.0]).into_dyn();
                let mut req = frames_in.push_request();
                tensorlisp_serve::convert::write_tensor(req.get().init_chunk().init_tensor(), &array, tensorlisp_serve::tensorlisp_capnp::DType::F32).unwrap();
                req.send().await.unwrap();
            }
            eprintln!("CK alone pushing done");
            let r = tokio::time::timeout(Duration::from_secs(30), voice.result_request().send().promise).await;
            match r {
                Ok(Ok(_)) => eprintln!("CK alone result ok"),
                Ok(Err(e)) => eprintln!("CK alone result error: {e}"),
                Err(_) => eprintln!("CK alone result timed out"),
            }
            frames_in.done_request().send().promise.await.unwrap();
            eprintln!("CK alone sink done");
            let mut n = 0;
            loop {
                let pulled = speech.pull_request().send().promise.await.unwrap();
                let pulled = pulled.get().unwrap();
                n += pulled.get_chunks().unwrap().len();
                if pulled.get_eof() { break; }
            }
            eprintln!("CK alone audio chunks: {n}");
            let _ = std::fs::remove_dir_all(&dir);
            assert!(n > 20);
        })
        .await;
}
