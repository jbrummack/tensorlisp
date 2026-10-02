//! Gemma 4 E2B on the native CUDA device: greedy tokens against transformers, then how
//! throughput grows when several sequences decode per step (dense per-sequence caches
//! vs a paged pool). Needs models/gemma4-e2b/gemma4-e2b-q8_0.gguf (see ports/gemma4/README.md).
//!   cargo test --release -p tensorlisp --features native-cuda --test native_gemma4 -- --ignored --nocapture
#![cfg(native_device)]

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array, ArrayD, IxDyn};
use tensorlisp::{Device, LoadOptions, Model, Program, RawInput, RunOptions};

const BATCH_ROWS: usize = 640;
const BLOCK: usize = 16;
const TABLE: usize = 40;
const GENERATED: usize = 32;
const BYTES_PER_TOKEN: usize = (12 * 256 + 3 * 512) * 2 * 2;

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma4-e2b/gemma4-e2b-q8_0.gguf");
    p.exists().then_some(p)
}

fn load(paged_max_context: usize) -> Option<Model> {
    let path = model_path()?;
    let ports = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ports/gemma4");
    let base = std::fs::read_to_string(ports.join("gemma4.ss")).unwrap();
    let extra = std::fs::read_to_string(ports.join("concurrency.ss")).unwrap();
    let line = "(define paged-max-context 640)";
    assert!(extra.contains(line));
    let source = format!("{base}\n{}", extra.replace(line, &format!("(define paged-max-context {paged_max_context})")));
    let options = LoadOptions { program: Some(Program::Text(source)), ..Default::default() };
    Some(Model::load_with(path, Device::Native, options).unwrap())
}

fn arrays(model: &Model, pipeline: &str, prompt: &str, extra: Option<Vec<f32>>) -> Vec<(String, Vec<f32>)> {
    let mut example = vec![("prompt".to_string(), RawInput::Text(prompt.into()))];
    if let Some(v) = extra {
        let name = if pipeline == "fill-dense" { "slot" } else { "layout" };
        example.push((name.into(), RawInput::Array(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap())));
    }
    model
        .pipeline(pipeline, example)
        .unwrap_or_else(|e| panic!("{pipeline}: {e}"))
        .into_iter()
        .filter_map(|(n, v)| v.as_array().map(|a| (n, a.iter().copied().collect())))
        .collect()
}

fn get(r: &[(String, Vec<f32>)], name: &str) -> Vec<f32> {
    r.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("no output {name}")).1.clone()
}

fn ints(rows: usize, cols: usize, f: impl Fn(usize, usize) -> usize) -> ArrayD<f32> {
    Array::from_shape_fn(IxDyn(&[rows, cols]), |ix| f(ix[0], ix[1]) as f32)
}

fn next(model: &Model, entry: &str, inputs: &[(&str, ArrayD<f32>)]) -> Vec<usize> {
    let views: Vec<_> = inputs.iter().map(|(n, a)| (*n, a.view())).collect();
    let out = model.run_entry(entry, &views, &RunOptions::default()).unwrap();
    out.outputs.into_iter().find(|(n, _)| n == "next").unwrap().1.iter().map(|&v| v as usize).collect()
}

/// The prompt for sequence `s`: about `sentences` filler sentences, then a different question per sequence.
fn prompt(s: usize, sentences: usize) -> String {
    const OPENERS: [&str; 4] = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
        "The author of Hamlet is",
    ];
    format!("{}{}", "The quick brown fox jumps over the lazy dog. ".repeat(sentences), OPENERS[s % OPENERS.len()])
}

/// Prefills `prompts` into dense slots; (prompt lengths, first generated tokens).
fn fill_dense(model: &Model, prompts: &[String]) -> (Vec<usize>, Vec<usize>) {
    let r: Vec<_> = prompts.iter().enumerate().map(|(s, p)| arrays(model, "fill-dense", p, Some(vec![s as f32]))).collect();
    (r.iter().map(|r| get(r, "length")[0] as usize).collect(), r.iter().map(|r| get(r, "first")[0] as usize).collect())
}

fn blocks_for(ms: &[usize]) -> usize {
    (ms.iter().max().unwrap() + GENERATED).div_ceil(BLOCK)
}

fn fill_paged(model: &Model, prompts: &[String], nb: usize) -> (Vec<usize>, Vec<usize>) {
    let r: Vec<_> =
        prompts.iter().enumerate().map(|(s, p)| arrays(model, "fill-paged", p, Some(vec![s as f32, nb as f32]))).collect();
    (r.iter().map(|r| get(r, "length")[0] as usize).collect(), r.iter().map(|r| get(r, "first")[0] as usize).collect())
}

/// `GENERATED - 1` steps of all sequences at once; tokens per sequence (first included) and seconds.
fn decode_dense(model: &Model, ms: &[usize], first: &[usize]) -> (Vec<Vec<usize>>, f64) {
    let n = ms.len();
    let mut tokens: Vec<Vec<usize>> = first.iter().map(|&t| vec![t]).collect();
    let inputs = |tokens: &Vec<Vec<usize>>, step: usize| {
        [
            ("token", ints(1, n, |_, s| *tokens[s].last().unwrap())),
            ("pos", ints(1, n, |_, s| ms[s] + step)),
            ("rows", ints(1, n, |_, s| s * BATCH_ROWS + ms[s] + step)),
        ]
    };
    next(model, "decode-dense", &inputs(&tokens, 0)); // compiles; rewrites the same K/V when repeated
    let t = Instant::now();
    for step in 0..GENERATED - 1 {
        let out = next(model, "decode-dense", &inputs(&tokens, step));
        for (s, tk) in tokens.iter_mut().enumerate() {
            tk.push(out[s]);
        }
    }
    (tokens, t.elapsed().as_secs_f64())
}

fn decode_paged(model: &Model, ms: &[usize], first: &[usize], nb: usize) -> (Vec<Vec<usize>>, f64) {
    let n = ms.len();
    let mut tokens: Vec<Vec<usize>> = first.iter().map(|&t| vec![t]).collect();
    let inputs = |tokens: &Vec<Vec<usize>>, step: usize| {
        [
            ("token", ints(1, n, |_, s| *tokens[s].last().unwrap())),
            ("pos", ints(1, n, |_, s| ms[s] + step)),
            ("tables", ints(n, TABLE, |s, k| if k < nb { s * nb + k } else { 0 })),
            ("lens", ints(1, n, |_, s| ms[s] + step + 1)),
            ("slots", ints(1, 2 * n, |_, i| if i % 2 == 0 { (i / 2) * nb * BLOCK + ms[i / 2] + step } else { 0 })),
        ]
    };
    next(model, "decode-paged", &inputs(&tokens, 0));
    let t = Instant::now();
    for step in 0..GENERATED - 1 {
        let out = next(model, "decode-paged", &inputs(&tokens, step));
        for (s, tk) in tokens.iter_mut().enumerate() {
            tk.push(out[s]);
        }
    }
    (tokens, t.elapsed().as_secs_f64())
}

/// The single-sequence `step` with one token, `GENERATED` times; seconds per step.
fn sequential_step_seconds(model: &Model, m: usize) -> f64 {
    let run = |step: usize| {
        next(model, "step", &[("token", ints(1, 1, |_, _| 2)), ("pos", ints(1, 1, |_, _| m + step))]);
    };
    run(0);
    let t = Instant::now();
    for step in 0..GENERATED {
        run(step);
    }
    t.elapsed().as_secs_f64() / GENERATED as f64
}

fn reference(model: &Model, p: &str) -> Vec<usize> {
    get(&arrays(model, "generate", p, None), "tokens").iter().map(|&t| t as usize).collect()
}

#[test]
#[ignore]
fn greedy_tokens_match_transformers() {
    let Some(model) = load(640) else { return };
    // from ports/gemma4/reference.py (bf16 on CPU); the long prompt is 80 filler sentences + the question
    let short = [9079, 236761, 108, 818, 5279, 529, 7001, 563, 9079, 236761, 108, 818, 5279, 529, 7001, 563, 9079, 236761, 108, 818, 5279, 529, 7001, 563, 9079, 236761, 108, 818, 5279, 529, 7001, 563];
    let long = [9079, 236761, 669, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529, 7001, 563, 9079, 236761, 669, 5279];
    assert_eq!(reference(&model, &prompt(0, 0)), short);
    assert_eq!(reference(&model, &prompt(0, 80)), long);
}

#[test]
#[ignore]
fn batched_decoding_matches_sequential() {
    let Some(model) = load(640) else { return };
    let prompts: Vec<String> = [0usize, 1, 2, 3].iter().map(|&s| prompt(s, [0, 1, 3, 20][s])).collect();
    let want: Vec<Vec<usize>> = prompts.iter().map(|p| reference(&model, p)).collect();
    let (ms, first) = fill_dense(&model, &prompts);
    let (dense, _) = decode_dense(&model, &ms, &first);
    let nb = blocks_for(&ms);
    let (ms2, first2) = fill_paged(&model, &prompts, nb);
    assert_eq!(ms, ms2);
    let (paged, _) = decode_paged(&model, &ms, &first2, nb);
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
    let probe = fill_dense(&model, &[prompt(0, sentences)]).0[0];
    let seq = sequential_step_seconds(&model, probe);
    println!("\n{label}: prompt of {probe} tokens + {GENERATED} generated, paged max context {paged_max_context}");
    println!("sequential step: {:.2} ms = {:.0} tokens/s at any concurrency", seq * 1e3, 1.0 / seq);
    println!("{:>4} {:>12} {:>12} {:>12} {:>9} {:>9}", "seqs", "sequential", "dense", "paged", "dense ms", "paged ms");
    let mut nb0 = 0;
    for n in [1usize, 2, 4, 8, 16, 32, 64] {
        let prompts: Vec<String> = (0..n).map(|s| prompt(s, sentences)).collect();
        let (ms, first) = fill_dense(&model, &prompts);
        let (_, dense) = decode_dense(&model, &ms, &first);
        let nb = blocks_for(&ms);
        nb0 = nb;
        let (_, first) = fill_paged(&model, &prompts, nb);
        let (_, paged) = decode_paged(&model, &ms, &first, nb);
        let steps = (GENERATED - 1) as f64;
        let tokens = n as f64 * steps;
        println!(
            "{n:>4} {:>10.0}/s {:>10.0}/s {:>10.0}/s {:>9.2} {:>9.2}",
            1.0 / seq,
            tokens / dense,
            tokens / paged,
            dense / steps * 1e3,
            paged / steps * 1e3
        );
    }
    println!(
        "cache per sequence: dense {:.1} MiB, paged {:.1} MiB",
        mib(BATCH_ROWS * BYTES_PER_TOKEN),
        mib(nb0 * BLOCK * BYTES_PER_TOKEN)
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
    sweep("long (under the 512 window)", 40, 640);
}
