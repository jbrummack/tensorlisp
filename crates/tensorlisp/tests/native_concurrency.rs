//! How much decoding several sequences per step helps T5Gemma 2 on the native CUDA
//! device: the single-sequence `decode-step` run once per sequence, a batched step over
//! dense per-sequence caches, and a batched step over a paged pool (vLLM's paged
//! attention kernels). Needs models/t5gemma2-270m/t5gemma2-270m-f16.gguf.
//!   cargo test --release -p tensorlisp --features native-cuda --test native_concurrency -- --ignored --nocapture
#![cfg(native_device)]

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, LoadOptions, Model, Program, RawInput, RunOptions};

const LAYERS: usize = 18;
const ROWS: usize = 1088; // dense cache rows per sequence: 64 decoder + 1024 memory
const SELF_ROWS: usize = 64;
const BLOCK: usize = 16;
const MAX_BLOCKS: usize = 68;
const STEPS: usize = 32;

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/t5gemma2-270m/t5gemma2-270m-f16.gguf");
    p.exists().then_some(p)
}

fn load(paged_max_context: usize) -> Option<Model> {
    let path = model_path()?;
    let ports = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ports/t5gemma2");
    let base = std::fs::read_to_string(ports.join("t5gemma2.ss")).unwrap();
    let extra = std::fs::read_to_string(ports.join("concurrency.ss")).unwrap();
    let line = "(define paged-max-context 1088)";
    assert!(extra.contains(line));
    let source = format!("{base}\n{}", extra.replace(line, &format!("(define paged-max-context {paged_max_context})")));
    let options = LoadOptions { program: Some(Program::Text(source)), ..Default::default() };
    Some(Model::load_with(path, Device::Native, options).unwrap())
}

fn text(model: &Model, pipeline: &str, prompt: &str, extra: Option<(&str, Vec<f32>)>) -> Vec<(String, Vec<f32>)> {
    let mut example = vec![("prompt".to_string(), RawInput::Text(prompt.into()))];
    if let Some((name, v)) = extra {
        example.push((name.into(), RawInput::Array(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap())));
    }
    model
        .pipeline(pipeline, example)
        .unwrap_or_else(|e| panic!("{pipeline}: {e}"))
        .into_iter()
        .filter_map(|(n, v)| v.as_array().map(|a| (n, a.iter().copied().collect())))
        .collect()
}

fn length(r: &[(String, Vec<f32>)]) -> usize {
    r.iter().find(|(n, _)| n == "length").unwrap().1[0] as usize
}

fn ints(rows: usize, cols: usize, f: impl Fn(usize, usize) -> usize) -> ArrayD<f32> {
    Array::from_shape_fn(IxDyn(&[rows, cols]), |ix| f(ix[0], ix[1]) as f32)
}

/// The prompt for sequence `s`: about `sentences` sentences, a different one per sequence.
fn prompt(s: usize, sentences: usize) -> String {
    const OPENERS: [&str; 4] = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
        "The author of Hamlet is",
    ];
    let filler = "The quick brown fox jumps over the lazy dog. ";
    format!("{}{}", filler.repeat(sentences), OPENERS[s % OPENERS.len()])
}

fn first(r: &RunOutputs) -> &ArrayD<f32> {
    &r.0
}
struct RunOutputs(ArrayD<f32>);

fn next(model: &Model, entry: &str, inputs: &[(&str, ArrayD<f32>)]) -> RunOutputs {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    let out = model.run_entry(entry, &views, &RunOptions::default()).unwrap();
    RunOutputs(out.outputs.into_iter().find(|(n, _)| n == "next").unwrap().1)
}

/// Fills the dense caches of `n` sequences; returns the prompt lengths.
fn fill_dense(model: &Model, prompts: &[String]) -> Vec<usize> {
    prompts.iter().enumerate().map(|(s, p)| length(&text(model, "fill-dense", p, Some(("slot", vec![s as f32]))))).collect()
}

fn blocks_for(ms: &[usize]) -> usize {
    (ms.iter().max().unwrap() + STEPS + 1).div_ceil(BLOCK)
}

fn fill_paged(model: &Model, prompts: &[String], nb: usize) -> Vec<usize> {
    prompts
        .iter()
        .enumerate()
        .map(|(s, p)| length(&text(model, "fill-paged", p, Some(("layout", vec![s as f32, nb as f32])))))
        .collect()
}

/// `STEPS` greedy steps of all sequences at once on dense caches; tokens per sequence and seconds.
fn decode_dense(model: &Model, ms: &[usize], bos: usize) -> (Vec<Vec<usize>>, f64) {
    let n = ms.len();
    let mut tokens = vec![vec![bos]; n];
    let step_inputs = |tokens: &Vec<Vec<usize>>, step: usize| {
        [
            ("token", ints(1, n, |_, s| *tokens[s].last().unwrap())),
            ("pos", ints(1, n, |_, _| step)),
            ("rows", ints(1, n, |_, s| s * ROWS + step)),
            ("mask", ints(n, ROWS, |s, j| (j <= step || (SELF_ROWS..SELF_ROWS + ms[s]).contains(&j)) as usize)),
        ]
    };
    next(model, "decode-dense", &step_inputs(&tokens, 0)); // compiles; rewrites the same K/V when repeated
    let t = Instant::now();
    for step in 0..STEPS {
        let out = next(model, "decode-dense", &step_inputs(&tokens, step));
        for (s, tk) in tokens.iter_mut().enumerate() {
            tk.push(first(&out).iter().nth(s).copied().unwrap() as usize);
        }
    }
    (tokens, t.elapsed().as_secs_f64())
}

fn decode_paged(model: &Model, ms: &[usize], nb: usize, bos: usize) -> (Vec<Vec<usize>>, f64) {
    let n = ms.len();
    let mut tokens = vec![vec![bos]; n];
    let step_inputs = |tokens: &Vec<Vec<usize>>, step: usize| {
        [
            ("token", ints(1, n, |_, s| *tokens[s].last().unwrap())),
            ("pos", ints(1, n, |_, _| step)),
            ("tables", ints(n, MAX_BLOCKS, |s, k| if k < nb { s * nb + k } else { 0 })),
            ("lens", ints(1, n, |_, s| ms[s] + step + 1)),
            ("slots", ints(1, 2 * n, |_, i| if i % 2 == 0 { (i / 2) * nb * BLOCK + ms[i / 2] + step } else { 0 })),
        ]
    };
    next(model, "decode-paged", &step_inputs(&tokens, 0));
    let t = Instant::now();
    for step in 0..STEPS {
        let out = next(model, "decode-paged", &step_inputs(&tokens, step));
        for (s, tk) in tokens.iter_mut().enumerate() {
            tk.push(first(&out).iter().nth(s).copied().unwrap() as usize);
        }
    }
    (tokens, t.elapsed().as_secs_f64())
}

/// The single-sequence decode-step (what `generate` uses), `STEPS` steps; seconds per step.
fn sequential_step_seconds(model: &Model, m: usize, bos: usize) -> f64 {
    let run = |step: usize| {
        let self_mask = Array::from_shape_fn(IxDyn(&[1, SELF_ROWS]), |ix| (ix[1] <= step) as usize as f32);
        let memory_mask = Array::from_shape_fn(IxDyn(&[1, 1024]), |ix| (ix[1] < m) as usize as f32);
        next(
            model,
            "decode-step",
            &[
                ("token", ints(1, 1, |_, _| bos)),
                ("pos", ints(1, 1, |_, _| step)),
                ("self-mask", self_mask),
                ("memory-mask", memory_mask),
            ],
        );
    };
    run(0);
    let t = Instant::now();
    for step in 0..STEPS {
        run(step);
    }
    t.elapsed().as_secs_f64() / STEPS as f64
}

fn reference(model: &Model, p: &str) -> Vec<usize> {
    let r = text(model, "generate", p, None);
    r.iter().find(|(n, _)| n == "tokens").unwrap().1.iter().map(|&t| t as usize).collect()
}

#[test]
#[ignore]
fn batched_decoding_matches_sequential() {
    let Some(model) = load(1088) else { return };
    let prompts: Vec<String> = [0usize, 1, 2, 3].iter().map(|&s| prompt(s, [0, 1, 3, 20][s])).collect();
    let want: Vec<Vec<usize>> = prompts.iter().map(|p| reference(&model, p)).collect();
    let bos = want[0][0];

    let ms = fill_dense(&model, &prompts);
    let (dense, _) = decode_dense(&model, &ms, bos);
    let nb = blocks_for(&ms);
    fill_paged(&model, &prompts, nb);
    let (paged, _) = decode_paged(&model, &ms, nb, bos);
    for s in 0..prompts.len() {
        let n = want[s].len();
        println!("seq {s} (prompt {} tokens)\n  sequential {:?}\n  dense      {:?}\n  paged      {:?}", ms[s], want[s], &dense[s][..n], &paged[s][..n]);
    }
    for s in 0..prompts.len() {
        let n = want[s].len();
        assert_eq!(&dense[s][..n], &want[s][..], "dense, sequence {s}");
        assert_eq!(&paged[s][..n], &want[s][..], "paged, sequence {s}");
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

fn sweep(label: &str, sentences: usize, paged_max_context: usize) {
    let Some(model) = load(paged_max_context) else { return };
    let mut rows = Vec::new();
    let bos = 2;
    let probe = length(&text(&model, "fill-dense", &prompt(0, sentences), Some(("slot", vec![0.0]))));
    let seq = sequential_step_seconds(&model, probe, bos);
    println!("\n{label}: prompt of {probe} tokens + {STEPS} generated, paged max context {paged_max_context}");
    println!("sequential decode-step: {:.2} ms/step = {:.0} tokens/s at any concurrency", seq * 1e3, 1.0 / seq);
    for n in [1usize, 2, 4, 8, 16, 32, 64] {
        let prompts: Vec<String> = (0..n).map(|s| prompt(s, sentences)).collect();
        let ms = fill_dense(&model, &prompts);
        let (_, dense) = decode_dense(&model, &ms, bos);
        let nb = blocks_for(&ms);
        fill_paged(&model, &prompts, nb);
        let (_, paged) = decode_paged(&model, &ms, nb, bos);
        let tokens = (n * STEPS) as f64;
        rows.push((n, tokens / dense, tokens / paged, dense / STEPS as f64, paged / STEPS as f64, nb));
    }
    println!("{:>4} {:>12} {:>12} {:>12} {:>9} {:>9}", "seqs", "sequential", "dense", "paged", "dense ms", "paged ms");
    for (n, d, p, dm, pm, _) in &rows {
        println!("{n:>4} {:>10.0}/s {d:>10.0}/s {p:>10.0}/s {:>9.2} {:>9.2}", 1.0 / seq, dm * 1e3, pm * 1e3);
    }
    let per_row = LAYERS * 2 * 256 * 2;
    println!(
        "cache per sequence: dense {:.1} MiB, paged {:.1} MiB",
        mib(ROWS * per_row),
        mib(rows[0].5 * BLOCK * per_row)
    );
}

#[test]
#[ignore]
fn concurrency_sweep_short_prompts() {
    sweep("short", 0, 64);
}

#[test]
#[ignore]
fn concurrency_sweep_long_prompts() {
    sweep("long", 95, 1088);
}
