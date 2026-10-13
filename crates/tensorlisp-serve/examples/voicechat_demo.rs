//! A voice chat client for `tl-serve`: streams the microphone (or a wav file) to a server running the VoiceChat
//! port (ports/voicechat) over Cap'n Proto RPC and plays the agent's speech as it comes back.
//!
//!   tl-serve --listen tcp:127.0.0.1:7700 --store STORE --device gpu          # the model is packed with its program
//!   cargo run --release -p tensorlisp-serve --example voicechat_demo -- --model /path/voicechat.gguf
//!
//! `--model` is a path the *server* can read; it is installed once (content-addressed, so a second start is
//! quick) under `--name`. Without `--wav` the default microphone is captured and the default output device
//! plays the reply (use headphones: the model hears itself otherwise and keeps answering). With `--wav FILE`
//! (16 kHz mono) the file is streamed at real time instead, and `--out reply.wav` saves the agent's speech.
//! Ctrl-C ends the conversation. `--tools tools.json` runs `listen-tools` (see the port's README);
//! the calls are printed and answered with `--reply` (default `{"result": 20}`).
use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tensorlisp_serve::tensorlisp_capnp::{callbacks, model, run, server, sink, source, value};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

const IN_RATE: u32 = 16_000; // what the model reads
const OUT_RATE: u32 = 22_050; // what the codec writes
const PIECE: usize = 1280; // 80 ms of input, one frame of the model

#[derive(Parser)]
#[command(about = "Talk to a VoiceChat model served by tl-serve")]
struct Args {
    /// tl-serve address.
    #[arg(long, default_value = "127.0.0.1:7700")]
    server: String,
    /// The name the model is (or gets) installed under.
    #[arg(long, default_value = "voicechat")]
    name: String,
    /// A GGUF with the program packed in, as the server sees the path. Needed the first time only.
    #[arg(long)]
    model: Option<String>,
    /// Stream this 16 kHz mono wav instead of the microphone.
    #[arg(long)]
    wav: Option<String>,
    /// Write the agent's speech here (22.05 kHz wav) when the conversation ends.
    #[arg(long)]
    out: Option<String>,
    /// A JSON array of tools: run `listen-tools`.
    #[arg(long)]
    tools: Option<String>,
    /// What every tool call is answered with.
    #[arg(long, default_value = "{\"result\": 20}")]
    reply: String,
    /// Don't play the speech.
    #[arg(long)]
    no_play: bool,
    /// List the audio devices and exit.
    #[arg(long)]
    devices: bool,
    /// Input device (a part of its name).
    #[arg(long)]
    input: Option<String>,
    /// Output device (a part of its name).
    #[arg(long)]
    output: Option<String>,
}

/// The client's side of a tool call.
struct Tool {
    reply: String,
}

impl callbacks::Server for Tool {
    async fn call(
        self: capnp::capability::Rc<Self>,
        params: callbacks::CallParams,
        mut results: callbacks::CallResults,
    ) -> Result<(), capnp::Error> {
        let p = params.get()?;
        if let value::Which::Text(call) = p.get_args()?.get(0).which()? {
            println!("\n[tool call {}: {}]", p.get_name()?.to_str()?, call?.to_str()?);
        }
        results.get().init_result().set_text(&self.reply);
        Ok(())
    }
}

fn write_wav(path: &str, rate: u32, samples: &[f32]) -> Result<()> {
    let n = samples.len() as u32 * 2;
    let mut b = Vec::with_capacity(44 + n as usize);
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
    std::fs::write(path, b).with_context(|| format!("writing {path}"))
}

/// Linear-interpolating resampler that keeps its position between calls, so a stream can be fed in pieces.
struct Resampler {
    step: f64, // input samples per output sample
    pos: f64,  // position of the next output sample, relative to the start of `held`
    held: Vec<f32>,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        Resampler { step: from as f64 / to as f64, pos: 0.0, held: Vec::new() }
    }

    fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.held.extend_from_slice(input);
        // box-filter when downsampling by more than 2x would alias less, but speech at 16 kHz from 44.1/48 is fine linear
        while self.pos + 1.0 < self.held.len() as f64 {
            let i = self.pos as usize;
            let f = (self.pos - i as f64) as f32;
            out.push(self.held[i] * (1.0 - f) + self.held[i + 1] * f);
            self.pos += self.step;
        }
        let drop = (self.pos as usize).min(self.held.len());
        self.held.drain(..drop);
        self.pos -= drop as f64;
    }
}

fn pick<D: DeviceTrait>(devices: impl Iterator<Item = D>, want: &Option<String>) -> Option<D> {
    let want = want.as_ref()?.to_lowercase();
    devices.into_iter().find(|d| d.name().is_ok_and(|n| n.to_lowercase().contains(&want)))
}

/// The microphone: mono samples at 16 kHz arrive on the channel. The stream must stay alive.
fn open_mic(want: &Option<String>) -> Result<(cpal::Stream, tokio::sync::mpsc::UnboundedReceiver<Vec<f32>>)> {
    let host = cpal::default_host();
    let dev = pick(host.input_devices()?, want).or_else(|| host.default_input_device()).ok_or_else(|| anyhow!("no input device"))?;
    let cfg = dev.default_input_config()?;
    eprintln!("microphone: {} ({} Hz, {} ch, {:?})", dev.name()?, cfg.sample_rate().0, cfg.channels(), cfg.sample_format());
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let ch = cfg.channels() as usize;
    let mut rs = Resampler::new(cfg.sample_rate().0, IN_RATE);
    let mut feed = move |mono: Vec<f32>| {
        let mut out = Vec::new();
        rs.push(&mono, &mut out);
        if !out.is_empty() {
            let _ = tx.send(out);
        }
    };
    let err = |e| eprintln!("microphone: {e}");
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            &cfg.into(),
            move |d: &[f32], _| feed(d.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()),
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            &cfg.into(),
            move |d: &[i16], _| feed(d.chunks(ch).map(|f| f.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / ch as f32).collect()),
            err,
            None,
        )?,
        f => bail!("unsupported microphone sample format {f:?}"),
    };
    stream.play()?;
    Ok((stream, rx))
}

/// The speakers: samples pushed into the queue (at the device's rate) are played, silence when it runs dry.
fn open_speaker(want: &Option<String>) -> Result<(cpal::Stream, Arc<Mutex<VecDeque<f32>>>, u32)> {
    let host = cpal::default_host();
    let dev = pick(host.output_devices()?, want).or_else(|| host.default_output_device()).ok_or_else(|| anyhow!("no output device"))?;
    let cfg = dev.default_output_config()?;
    let rate = cfg.sample_rate().0;
    eprintln!("speakers:   {} ({} Hz, {} ch, {:?})", dev.name()?, rate, cfg.channels(), cfg.sample_format());
    let queue = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let ch = cfg.channels() as usize;
    let err = |e| eprintln!("speakers: {e}");
    let q = queue.clone();
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_output_stream(
            &cfg.into(),
            move |d: &mut [f32], _| {
                let mut q = q.lock().unwrap();
                for frame in d.chunks_mut(ch) {
                    frame.fill(q.pop_front().unwrap_or(0.0));
                }
            },
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_output_stream(
            &cfg.into(),
            move |d: &mut [i16], _| {
                let mut q = q.lock().unwrap();
                for frame in d.chunks_mut(ch) {
                    frame.fill((q.pop_front().unwrap_or(0.0).clamp(-1.0, 1.0) * 32767.0) as i16);
                }
            },
            err,
            None,
        )?,
        f => bail!("unsupported output sample format {f:?}"),
    };
    stream.play()?;
    Ok((stream, queue, rate))
}

async fn push(sink: &sink::Client, samples: &[f32]) -> Result<()> {
    let mut req = sink.push_request();
    {
        let mut a = req.get().init_chunk().init_audio();
        a.set_sample_rate(IN_RATE);
        a.set_channels(1);
        let mut list = a.init_samples(samples.len() as u32);
        for (i, &v) in samples.iter().enumerate() {
            list.set(i as u32, v);
        }
    }
    req.send().await.map_err(|e| anyhow!("pushing audio: {e}"))
}

async fn output(run: &run::Client, name: &str) -> Result<source::Client> {
    let mut r = run.output_request();
    r.get().set_name(name);
    Ok(r.send().promise.await?.get()?.get_source()?)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.devices {
        let host = cpal::default_host();
        for d in host.input_devices()? {
            println!("input:  {}", d.name()?);
        }
        for d in host.output_devices()? {
            println!("output: {}", d.name()?);
        }
        return Ok(());
    }
    let wav = match &args.wav {
        Some(p) => {
            let a = tensorlisp::autopro::audio::Audio::from_wav_bytes(&std::fs::read(p)?).map_err(|e| anyhow!("{p}: {e}"))?;
            if a.sample_rate != IN_RATE {
                bail!("{p}: the wav must be {IN_RATE} Hz mono, it is {} Hz", a.sample_rate);
            }
            Some(a.samples)
        }
        None => None,
    };

    tokio::task::LocalSet::new()
        .run_until(async move {
            let stream = tokio::net::TcpStream::connect(&args.server).await.with_context(|| format!("connecting to {}", args.server))?;
            stream.set_nodelay(true)?;
            let (r, w) = tokio::io::split(stream);
            let network = twoparty::VatNetwork::new(r.compat(), w.compat_write(), Side::Client, Default::default());
            let mut rpc = RpcSystem::new(Box::new(network), None);
            let srv: server::Client = rpc.bootstrap(Side::Server);
            tokio::task::spawn_local(rpc);

            // the model: already installed under this name, or install it from the path the server can read
            let t0 = Instant::now();
            let mut open = srv.open_request();
            open.get().set_name(&args.name);
            let model: model::Client = match open.send().promise.await {
                Ok(r) => r.get()?.get_model()?,
                Err(_) => {
                    let location = args.model.as_ref().ok_or_else(|| anyhow!("{:?} is not installed on the server: pass --model PATH", args.name))?;
                    eprintln!("installing {location} ...");
                    let mut req = srv.install_from_request();
                    {
                        let mut p = req.get();
                        p.set_location(location);
                        p.init_options().set_name(&args.name);
                    }
                    req.send().promise.await.map_err(|e| anyhow!("install: {e}"))?.get()?.get_model()?
                }
            };
            eprintln!("model ready in {:.1} s", t0.elapsed().as_secs_f64());

            let mut req = model.pipeline_request();
            {
                let mut p = req.get();
                if let Some(tools) = &args.tools {
                    p.set_name("listen-tools");
                    let text = std::fs::read_to_string(tools).with_context(|| format!("reading {tools}"))?;
                    let mut a = p.reborrow().init_args(3);
                    for (i, (k, v)) in [("system", ""), ("tools", text.as_str()), ("reply", "")].iter().enumerate() {
                        let mut n = a.reborrow().get(i as u32);
                        n.set_name(*k);
                        n.init_value().set_text(*v);
                    }
                    p.set_callbacks(capnp_rpc::new_client(Tool { reply: args.reply.clone() }));
                } else {
                    p.set_name("listen");
                }
            }
            let run: run::Client = req.send().promise.await.map_err(|e| anyhow!("starting the pipeline: {e}"))?.get()?.get_run()?;
            let sink = {
                let mut r = run.input_request();
                r.get().set_name("audio");
                r.send().promise.await?.get()?.get_sink()?
            };
            let speech = output(&run, "audio").await?;
            let tokens = output(&run, "token").await?;

            // output side: the agent's words are printed as they arrive, the speech is queued for the speakers
            let (_speaker, queue, dev_rate) = if args.no_play {
                (None, Arc::new(Mutex::new(VecDeque::new())), OUT_RATE)
            } else {
                let (s, q, r) = open_speaker(&args.output)?;
                (Some(s), q, r)
            };
            let play = !args.no_play;
            let received = Rc::new(RefCell::new(0usize)); // speech frames so far
            let reply = Rc::new(RefCell::new(Vec::<f32>::new()));
            let started = Instant::now();
            let words = tokio::task::spawn_local(async move {
                loop {
                    let Ok(pulled) = tokens.pull_request().send().promise.await else { return };
                    let Ok(pulled) = pulled.get() else { return };
                    for c in pulled.get_chunks().unwrap().iter() {
                        if let Ok(value::Which::Text(t)) = c.which() {
                            print!("{}", t.unwrap().to_str().unwrap());
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        }
                    }
                    if pulled.get_eof() {
                        return;
                    }
                }
            });
            let heard = {
                let (received, reply, queue) = (received.clone(), reply.clone(), queue.clone());
                tokio::task::spawn_local(async move {
                    let mut rs = Resampler::new(OUT_RATE, dev_rate);
                    loop {
                        let Ok(pulled) = speech.pull_request().send().promise.await else { return };
                        let Ok(pulled) = pulled.get() else { return };
                        for c in pulled.get_chunks().unwrap().iter() {
                            let Ok(value::Which::Tensor(t)) = c.which() else { continue };
                            let t = t.unwrap();
                            let Ok(samples) = tensorlisp_serve::convert::decode(t.get_dtype().unwrap(), t.get_data().unwrap()) else { continue };
                            *received.borrow_mut() += 1;
                            if play {
                                let mut out = Vec::new();
                                rs.push(&samples, &mut out);
                                queue.lock().unwrap().extend(out);
                            }
                            reply.borrow_mut().extend(samples);
                        }
                        if pulled.get_eof() {
                            return;
                        }
                    }
                })
            };

            // warm-up: the model compiles its graphs on the first frames, so a second of silence goes in first and
            // the real input starts when the matching speech frames are back
            let warm = 12;
            eprintln!("warming up ...");
            let t = Instant::now();
            for _ in 0..warm {
                push(&sink, &[0.0; PIECE]).await?;
            }
            while *received.borrow() < warm {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            eprintln!("warm in {:.1} s", t.elapsed().as_secs_f64());
            queue.lock().unwrap().clear();
            reply.borrow_mut().clear();

            // input side
            let start = Instant::now();
            let stop = tokio::signal::ctrl_c();
            tokio::pin!(stop);
            let mut _mic = None;
            if let Some(samples) = wav {
                eprintln!("streaming {:.1} s of audio (then 4 s of silence); Ctrl-C stops", samples.len() as f64 / IN_RATE as f64);
                let tail = vec![0.0f32; 4 * IN_RATE as usize];
                'feed: for (i, p) in samples.iter().chain(tail.iter()).copied().collect::<Vec<_>>().chunks(PIECE).enumerate() {
                    let due = start + Duration::from_secs_f64(((i * PIECE + p.len()) as f64) / IN_RATE as f64);
                    tokio::select! {
                        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(due)) => {}
                        _ = &mut stop => break 'feed,
                    }
                    push(&sink, p).await?;
                }
            } else {
                let (m, mut rx) = open_mic(&args.input)?;
                _mic = Some(m);
                eprintln!("listening: talk to the agent (headphones recommended), Ctrl-C ends");
                let mut pending: Vec<f32> = Vec::new();
                'live: loop {
                    tokio::select! {
                        got = rx.recv() => {
                            let Some(got) = got else { break 'live };
                            pending.extend(got);
                            while pending.len() >= PIECE {
                                let piece: Vec<f32> = pending.drain(..PIECE).collect();
                                push(&sink, &piece).await?;
                            }
                        }
                        _ = &mut stop => break 'live,
                    }
                }
            }
            drop(_mic);
            eprintln!("\nending the conversation ...");
            sink.done_request().send().promise.await?;
            let _ = heard.await;
            let _ = words.await;
            let _ = run.result_request().send().promise.await;
            let reply = reply.borrow();
            eprintln!("agent speech: {:.1} s over {:.1} s", reply.len() as f64 / OUT_RATE as f64, start.elapsed().as_secs_f64());
            if let Some(out) = &args.out {
                write_wav(out, OUT_RATE, &reply)?;
                eprintln!("wrote {out}");
            }
            Ok(())
        })
        .await
}
