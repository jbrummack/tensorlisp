//! `tl train`: fine-tune a program's `(define-param ...)` parameters (LoRA adapters) on JSON lines.
//!
//! The program provides a pipeline (default `train-example`) that turns one example (its raw string
//! inputs, taken from the JSON object's fields of the same names) into the inputs of a training entry
//! (default `train`, which outputs a scalar `loss`); see ports/t5gemma2/train.ss and
//! docs/design/training.md. Examples share one padded shape, so the graph is built once.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use ndarray::ArrayD;
use serde_json::json;
use tensorlisp::{LoadOptions, Model, Program, RawInput, RawKind, TrainOptions, Value};

use crate::{
    ModelArgs,
    common::{adapter_format, program_text, read_assets, set_define},
};

#[derive(Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Training data: JSON lines, each an object with the pipeline's raw inputs as string fields.
    #[arg(long)]
    pub data: PathBuf,
    /// Directory the adapter is written to (PEFT's adapter_model.safetensors + adapter_config.json).
    #[arg(short, long)]
    pub output: PathBuf,
    /// Pipeline that turns an example into the entry's inputs.
    #[arg(long, default_value = "train-example")]
    pub pipeline: String,
    /// Entry that computes the loss.
    #[arg(long, default_value = "train")]
    pub entry: String,
    /// LoRA rank, written into the program's `(define lora-rank N)`.
    #[arg(long, default_value_t = 8)]
    pub rank: usize,
    /// LoRA alpha, written into the program's `(define lora-alpha X)`.
    #[arg(long, default_value_t = 16.0)]
    pub alpha: f32,
    /// Set a `(define NAME ...)` of the program, NAME=VALUE (repeatable), e.g. train-src-len=32.
    #[arg(long = "define")]
    pub defines: Vec<String>,
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,
    /// Stop after this many optimizer steps.
    #[arg(long)]
    pub max_steps: Option<usize>,
    /// Peak learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f32,
    /// Warm-up, as a fraction of all steps.
    #[arg(long, default_value_t = 0.03)]
    pub warmup: f32,
    /// Learning-rate decay after the warm-up: cosine, linear or constant.
    #[arg(long, default_value = "cosine")]
    pub schedule: String,
    /// Examples per optimizer step (gradient accumulation; the entry takes one example).
    #[arg(long, default_value_t = 1)]
    pub accum: usize,
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f32,
    /// Seed of the adapter initialization and of the shuffling.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Start from this adapter (a directory written by a previous run).
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Also save the adapter every this many optimizer steps.
    #[arg(long)]
    pub save_every: Option<usize>,
}

fn lr_at(args: &TrainArgs, step: usize, total: usize) -> f32 {
    let warm = ((args.warmup * total as f32).ceil() as usize).max(1);
    if args.warmup > 0.0 && step < warm {
        return args.lr * (step + 1) as f32 / warm as f32;
    }
    let progress = (step.saturating_sub(warm)) as f32 / (total.saturating_sub(warm)).max(1) as f32;
    match args.schedule.as_str() {
        "cosine" => args.lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()),
        "linear" => args.lr * (1.0 - progress),
        _ => args.lr,
    }
}

fn shuffle(order: &mut [usize], rng: &mut u64) {
    for i in (1..order.len()).rev() {
        *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        order.swap(i, ((z ^ (z >> 31)) % (i as u64 + 1)) as usize);
    }
}

pub fn run(args: &TrainArgs, json_out: bool) -> Result<i32> {
    if args.accum == 0 || args.epochs == 0 {
        bail!("--accum and --epochs must be at least 1");
    }
    let mut text = program_text(&args.model)?.context("give the program with --program (and train.ss with --append)")?;
    text = set_define(&text, "lora-rank", &args.rank.to_string())?;
    text = set_define(&text, "lora-alpha", &format!("{:?}", args.alpha))?;
    for d in &args.defines {
        let (name, value) = d.split_once('=').with_context(|| format!("--define wants NAME=VALUE, got {d:?}"))?;
        text = set_define(&text, name, value)?;
    }
    let options = LoadOptions { program: Some(Program::Text(text)), assets: read_assets(&args.model.assets)? };
    let model = Model::load_with(&args.model.model, args.model.device.0, options)
        .with_context(|| format!("loading {}", args.model.model.display()))?;
    model.init_params(args.seed)?;
    let format = adapter_format(&args.model, args.alpha);
    if let Some(dir) = &args.resume {
        let n = model.load_adapter(dir, &format)?;
        eprintln!("resumed {n} tensors from {}", dir.display());
    }

    // Tokenize the data through the program's pipeline.
    let pipelines = model.pipelines();
    let pipeline = pipelines
        .iter()
        .find(|p| p.name == args.pipeline)
        .with_context(|| format!("the program has no pipeline {:?}", args.pipeline))?;
    let entry = model.entry(Some(&args.entry))?.clone();
    let text = std::fs::read_to_string(&args.data).with_context(|| format!("reading {}", args.data.display()))?;
    let mut examples: Vec<Vec<(String, ArrayD<f32>)>> = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let object: serde_json::Value = serde_json::from_str(line).with_context(|| format!("{}:{}", args.data.display(), i + 1))?;
        let mut raw = Vec::new();
        for spec in &pipeline.raw_inputs {
            if spec.kind != RawKind::Text {
                bail!("pipeline input {:?} is {:?}; tl train feeds text fields only", spec.name, spec.kind);
            }
            let field = object[&spec.name].as_str().with_context(|| format!("{}:{}: no string field {:?}", args.data.display(), i + 1, spec.name))?;
            raw.push((spec.name.clone(), RawInput::Text(field.to_string())));
        }
        let results = model.pipeline(&args.pipeline, raw).with_context(|| format!("{}:{}", args.data.display(), i + 1))?;
        let mut inputs = Vec::new();
        for spec in &entry.inputs {
            let (_, value) = results
                .iter()
                .find(|(n, _)| *n == spec.name)
                .with_context(|| format!("the pipeline has no output {:?}, which entry {:?} takes", spec.name, args.entry))?;
            let Value::Array(a) = value else { bail!("pipeline output {:?} is not an array", spec.name) };
            // The entries declare a leading batch of 1.
            let a = if a.ndim() < spec.dims.as_ref().map_or(0, |d| d.len()) { a.clone().insert_axis(ndarray::Axis(0)) } else { a.clone() };
            inputs.push((spec.name.clone(), a));
        }
        if let Some(first) = examples.first() {
            for ((n, a), (_, f)) in inputs.iter().zip(first) {
                if a.shape() != f.shape() {
                    bail!("{}:{}: input {n} has shape {:?}, the first example's is {:?}; pad to one shape", args.data.display(), i + 1, a.shape(), f.shape());
                }
            }
        }
        examples.push(inputs);
    }
    if examples.is_empty() {
        bail!("{} has no examples", args.data.display());
    }
    let shapes: Vec<(&str, Vec<usize>)> = examples[0].iter().map(|(n, a)| (n.as_str(), a.shape().to_vec())).collect();

    let mut trainer = model.trainer(
        &shapes,
        TrainOptions { entry: args.entry.clone(), lr: args.lr, weight_decay: args.weight_decay, accum_steps: args.accum, ..Default::default() },
    )?;
    let n = examples.len();
    let total = (args.epochs * n / args.accum).max(1).min(args.max_steps.unwrap_or(usize::MAX));
    eprintln!(
        "{n} examples, {} epochs, {} optimizer steps of {} example(s), {} trainable tensors",
        args.epochs,
        total,
        args.accum,
        trainer.param_names().len()
    );

    let mut rng = args.seed;
    let mut step = 0;
    let (mut micro, mut running) = (0, 0f32);
    let started = Instant::now();
    'epochs: for epoch in 0..args.epochs {
        let mut order: Vec<usize> = (0..n).collect();
        shuffle(&mut order, &mut rng);
        for &i in &order {
            if micro + 1 == args.accum {
                trainer.set_lr(lr_at(args, step, total));
            }
            let views: Vec<_> = examples[i].iter().map(|(name, a)| (name.as_str(), a.view())).collect();
            let stats = trainer.step(&views)?;
            running += stats.loss;
            micro += 1;
            if stats.optimized {
                step += 1;
                let loss = running / micro as f32;
                let lr = lr_at(args, step - 1, total);
                let rate = step as f64 * args.accum as f64 / started.elapsed().as_secs_f64();
                if json_out {
                    println!("{}", json!({ "step": step, "epoch": epoch + 1, "loss": loss, "lr": lr, "examples_per_s": rate }));
                } else {
                    println!("step {step:>5}/{total}  epoch {}  loss {loss:.4}  lr {lr:.2e}  {rate:.2} examples/s", epoch + 1);
                }
                (micro, running) = (0, 0.0);
                if args.save_every.is_some_and(|k| step % k == 0) {
                    model.save_adapter(&args.output, &format)?;
                }
                if step >= total {
                    break 'epochs;
                }
            }
        }
    }
    model.save_adapter(&args.output, &format)?;
    eprintln!("saved the adapter to {}", args.output.display());
    Ok(0)
}
