//! Streaming audio into Gemma 4 E2B's conformer over RPC and timing the first token after the audio ends
//! (docs/design/serve.md, "Streaming audio into Gemma 4"). Ignored by default; needs the GGUF packed with
//! ports/gemma4 (see its README) and a 16 kHz mono wav:
//!
//!   TL_AUDIO_MODEL=/path/gemma4-audio.gguf TL_AUDIO_WAV=/path/speech.wav [TL_AUDIO_CHUNK_MS=100] \
//!   [TL_AUDIO_DEVICE=gpu] cargo test --release -p tensorlisp-serve --test real_audio -- --ignored --nocapture
use std::{path::PathBuf, time::{Duration, Instant}};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tensorlisp::Device;
use tensorlisp_serve::{
    Registry, serve_stream,
    store::Store,
    tensorlisp_capnp::{model, run, server, sink, value},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

const PRE: &str = "<|turn>user\nTranscribe this audio.<|audio>";
const POST: &str = "<audio|><turn|>\n<|turn>model\n";

fn device() -> Device {
    match std::env::var("TL_AUDIO_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("native") => Device::Native,
        _ => Device::Gpu,
    }
}

fn read_wav(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let audio = tensorlisp::autopro::audio::Audio::from_wav_bytes(&bytes).unwrap();
    assert_eq!(audio.sample_rate, 16000, "the front end takes 16 kHz audio");
    audio.samples
}

async fn start(model: &model::Client, pipeline: &str, strings: &[(&str, &str)]) -> run::Client {
    let mut req = model.pipeline_request();
    {
        let mut p = req.get();
        p.set_name(pipeline);
        let mut args = p.init_args(strings.len() as u32);
        for (i, (k, v)) in strings.iter().enumerate() {
            let mut a = args.reborrow().get(i as u32);
            a.set_name(*k);
            a.init_value().set_text(*v);
        }
    }
    req.send().promise.await.unwrap().get().unwrap().get_run().unwrap()
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

struct Utterance {
    text: String,
    /// from the end of the audio (the last push, before `done`) to the first token reaching the client
    first_token: Duration,
    /// to the last token
    answer: Duration,
    /// from the start of the call to the last push
    audio_time: Duration,
    /// (time since the audio ended, marker) as the pipeline reported them
    marks: Vec<(Duration, String)>,
}

/// Pushes the samples in `chunk`-sized pieces, paced to real time if `realtime`, then closes the stream.
async fn transcribe(model: &model::Client, samples: &[f32], chunk: usize, realtime: bool) -> Utterance {
    let run = start(model, "transcribe", &[("pre", PRE), ("post", POST)]).await;
    let sink = {
        let mut r = run.input_request();
        r.get().set_name("audio");
        r.send().promise.await.unwrap().get().unwrap().get_sink().unwrap()
    };
    let tokens = {
        let mut r = run.output_request();
        r.get().set_name("token");
        r.send().promise.await.unwrap().get().unwrap().get_source().unwrap()
    };
    let stages = {
        let mut r = run.output_request();
        r.get().set_name("stage");
        r.send().promise.await.unwrap().get().unwrap().get_source().unwrap()
    };
    // Let the pipeline do its start-up (prefilling the prompt) before the clock for the audio starts.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let t0 = Instant::now();
    for (i, piece) in samples.chunks(chunk).enumerate() {
        if realtime {
            // the piece is complete once its last sample has been spoken
            let due = t0 + Duration::from_secs_f64(((i * chunk + piece.len()) as f64) / 16000.0);
            tokio::time::sleep_until(tokio::time::Instant::from_std(due)).await;
        }
        push(&sink, piece).await;
    }
    let t_end = Instant::now();
    sink.done_request().send().promise.await.unwrap();
    // The pipeline's own markers, timed when they reach the client.
    let marks = tokio::task::spawn_local(async move {
        let mut marks = Vec::new();
        loop {
            let pulled = stages.pull_request().send().promise.await.unwrap();
            let pulled = pulled.get().unwrap();
            for c in pulled.get_chunks().unwrap().iter() {
                let value::Which::Text(t) = c.which().unwrap() else { panic!() };
                marks.push((t_end.elapsed(), t.unwrap().to_str().unwrap().to_owned()));
            }
            if pulled.get_eof() {
                return marks;
            }
        }
    });
    let mut text = String::new();
    let mut first = None;
    loop {
        let pulled = tokio::time::timeout(Duration::from_secs(120), tokens.pull_request().send().promise).await.expect("a token").unwrap();
        let pulled = pulled.get().unwrap();
        for c in pulled.get_chunks().unwrap().iter() {
            let value::Which::Text(t) = c.which().unwrap() else { panic!() };
            text.push_str(t.unwrap().to_str().unwrap());
            first.get_or_insert_with(|| t_end.elapsed());
        }
        if pulled.get_eof() {
            break;
        }
    }
    let answer = t_end.elapsed();
    run.result_request().send().promise.await.unwrap();
    Utterance { text, first_token: first.unwrap(), answer, audio_time: t_end - t0, marks: marks.await.unwrap() }
}

#[tokio::test(flavor = "current_thread")]
#[ignore]
async fn gemma4_streams_audio() {
    let (Ok(model_path), Ok(wav)) = (std::env::var("TL_AUDIO_MODEL"), std::env::var("TL_AUDIO_WAV")) else {
        eprintln!("TL_AUDIO_MODEL and TL_AUDIO_WAV are not set, skipping");
        return;
    };
    let samples = read_wav(&PathBuf::from(wav));
    let chunk_ms: usize = std::env::var("TL_AUDIO_CHUNK_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let chunk = chunk_ms * 16;
    eprintln!("audio: {:.2} s, pushed in {chunk_ms} ms pieces", samples.len() as f64 / 16000.0);

    tokio::task::LocalSet::new()
        .run_until(async {
            let dir = std::env::temp_dir().join(format!("tl-serve-audio-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let registry = Registry::new(Store::open(dir.join("store")).unwrap(), device());
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
                p.init_options().set_name("gemma4");
            }
            let model = req.send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            eprintln!("installed and loaded in {:.1} s", t0.elapsed().as_secs_f64());

            // The first utterance builds the graphs (cold); the rest are what a running service sees.
            let cold = transcribe(&model, &samples, chunk, false).await;
            eprintln!("cold: {:?}", cold.text);
            let mut want = cold.text.clone();
            for round in 1..=3 {
                for (label, realtime) in [("streamed in real time", true), ("all at once at the end", false)] {
                    let u = transcribe(&model, &samples, chunk, realtime).await;
                    eprintln!(
                        "{label:>24}: first token {:6.0} ms after the audio ended, answer done {:6.0} ms ({} chars; audio took {:.2} s to deliver)",
                        u.first_token.as_secs_f64() * 1e3,
                        u.answer.as_secs_f64() * 1e3,
                        u.text.len(),
                        u.audio_time.as_secs_f64()
                    );
                    let marks: Vec<String> = u.marks.iter().map(|(t, m)| format!("{m} at {:.0} ms", t.as_secs_f64() * 1e3)).collect();
                    eprintln!("{:>24}  {}", "", marks.join(", "));
                    assert_eq!(u.text, want, "round {round} ({label}): the transcript changed");
                }
                want = want.clone();
            }
            eprintln!("transcript: {want:?}");
        })
        .await;
}
