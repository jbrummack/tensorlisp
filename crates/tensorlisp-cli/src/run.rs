use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use ndarray::{ArrayD, ArrayViewD};
use serde_json::json;
use tensorlisp::{Model, RunOptions, Taps};

use crate::{
    ModelArgs,
    common::{Stats, file_name, load_model, print_json, read_npy_args, shape_string, stats},
    npy,
};

#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Input as NAME=file.npy (repeatable).
    #[arg(short, long = "input")]
    pub inputs: Vec<String>,
    /// Write every output (and requested tap) to DIR/NAME.npy.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Also read back taps: "all" or comma-separated tap names.
    #[arg(long)]
    pub taps: Option<String>,
    /// Run this many times and report the mean time of the runs after the first.
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
}

#[derive(Args)]
pub struct CompareArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Input as NAME=file.npy (repeatable).
    #[arg(short, long = "input")]
    pub inputs: Vec<String>,
    /// Expected values of an output or tap, NAME=file.npy (repeatable).
    #[arg(short, long = "reference")]
    pub references: Vec<String>,
    /// Directory with expected values as NAME.npy for any outputs and taps.
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
    /// Absolute tolerance: |got - want| <= atol + rtol * |want|.
    #[arg(long, default_value_t = 1e-4)]
    pub atol: f32,
    /// Relative tolerance.
    #[arg(long, default_value_t = 1e-3)]
    pub rtol: f32,
}

fn parse_taps(taps: &Option<String>) -> Taps {
    match taps.as_deref() {
        None => Taps::None,
        Some("all") => Taps::All,
        Some(names) => Taps::Names(names.split(',').map(|n| n.trim().to_string()).collect()),
    }
}

fn views(inputs: &[(String, ArrayD<f32>)]) -> Vec<(&str, ArrayViewD<'_, f32>)> {
    inputs.iter().map(|(n, a)| (n.as_str(), a.view())).collect()
}

pub fn run(args: &RunArgs, json: bool) -> Result<i32> {
    let inputs = read_npy_args(&args.inputs)?;
    let start = Instant::now();
    let model = load_model(&args.model)?;
    let load_time = start.elapsed();
    let options = RunOptions { taps: parse_taps(&args.taps) };

    let mut times = Vec::new();
    let mut result = None;
    for _ in 0..args.repeat.max(1) {
        let start = Instant::now();
        result = Some(model.run_with(&views(&inputs), &options)?);
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let result = result.unwrap();
    let mean_after_first = (times.len() > 1).then(|| times[1..].iter().sum::<f64>() / (times.len() - 1) as f64);

    let mut written = Vec::new();
    if let Some(dir) = &args.output {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        for (name, array) in result.outputs.iter().chain(&result.taps) {
            let path = dir.join(format!("{}.npy", file_name(name)));
            npy::write(&path, array)?;
            written.push((name.clone(), path));
        }
    }
    let file_of = |name: &str| written.iter().find(|(n, _)| n == name).map(|(_, p)| p.clone());

    if json {
        let entry = |(name, array): &(String, ArrayD<f32>)| {
            json!({ "name": name, "shape": array.shape(), "stats": stats(array), "file": file_of(name) })
        };
        print_json(&json!({
            "device": model.device_name(),
            "load_ms": load_time.as_secs_f64() * 1e3,
            "first_run_ms": times[0],
            "mean_run_ms": mean_after_first,
            "outputs": result.outputs.iter().map(entry).collect::<Vec<_>>(),
            "taps": result.taps.iter().map(entry).collect::<Vec<_>>(),
        }))?;
        return Ok(0);
    }

    print!("device {}, load {:.1} ms, first run {:.1} ms", model.device_name(), load_time.as_secs_f64() * 1e3, times[0]);
    match mean_after_first {
        Some(mean) => println!(", mean of next {}: {mean:.2} ms", times.len() - 1),
        None => println!(),
    }
    let print_group = |title: &str, results: &[(String, ArrayD<f32>)]| {
        if results.is_empty() {
            return;
        }
        println!("{title}:");
        let width = results.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        for (name, array) in results {
            let Stats { min, max, mean, std, nan } = stats(array);
            let nan = if nan > 0 { format!(" nan {nan}") } else { String::new() };
            let file = file_of(name).map(|p| format!("  -> {}", p.display())).unwrap_or_default();
            println!(
                "  {name:width$}  {:16} min {min:<10.4} max {max:<10.4} mean {mean:<10.4} std {std:.4}{nan}{file}",
                shape_string(array.shape())
            );
        }
    };
    print_group("outputs", &result.outputs);
    print_group("taps", &result.taps);
    Ok(0)
}

#[derive(serde::Serialize)]
struct Comparison {
    kind: &'static str,
    name: String,
    pass: bool,
    shape: Vec<usize>,
    reference_shape: Vec<usize>,
    /// Values outside atol + rtol * |want| (NaN mismatches included).
    mismatches: usize,
    max_abs_error: f64,
    mean_abs_error: f64,
    cosine_similarity: f64,
    /// Where the largest error is (numpy index), with both values.
    worst: Option<(Vec<usize>, f32, f32)>,
    note: Option<String>,
}

fn compare_arrays(kind: &'static str, name: &str, got: &ArrayD<f32>, want: &ArrayD<f32>, atol: f32, rtol: f32) -> Comparison {
    let mut c = Comparison {
        kind,
        name: name.to_string(),
        pass: false,
        shape: got.shape().to_vec(),
        reference_shape: want.shape().to_vec(),
        mismatches: 0,
        max_abs_error: 0.0,
        mean_abs_error: 0.0,
        cosine_similarity: f64::NAN,
        worst: None,
        note: None,
    };
    if got.len() != want.len() {
        c.note = Some("shapes differ in number of elements; values not compared".into());
        c.mismatches = got.len().max(want.len());
        return c;
    }
    if got.shape() != want.shape() {
        c.note = Some("shapes differ but have the same number of elements; compared in memory order".into());
    }
    let (mut dot, mut norm_got, mut norm_want, mut sum_abs) = (0f64, 0f64, 0f64, 0f64);
    let mut worst_index = None;
    let got_std = got.as_standard_layout();
    let want_std = want.as_standard_layout();
    for (i, (&g, &w)) in got_std.iter().zip(want_std.iter()).enumerate() {
        if g.is_nan() || w.is_nan() {
            if g.is_nan() != w.is_nan() {
                c.mismatches += 1;
            }
            continue;
        }
        let err = (g - w).abs();
        if err > atol + rtol * w.abs() {
            c.mismatches += 1;
        }
        if err as f64 > c.max_abs_error || worst_index.is_none() {
            c.max_abs_error = c.max_abs_error.max(err as f64);
            worst_index = Some(i);
        }
        sum_abs += err as f64;
        dot += g as f64 * w as f64;
        norm_got += g as f64 * g as f64;
        norm_want += w as f64 * w as f64;
    }
    c.mean_abs_error = sum_abs / got.len().max(1) as f64;
    c.cosine_similarity = if norm_got == 0.0 && norm_want == 0.0 { 1.0 } else { dot / (norm_got.sqrt() * norm_want.sqrt()) };
    c.worst = worst_index.map(|i| {
        let mut rest = i;
        let index = got
            .shape()
            .iter()
            .rev()
            .map(|&d| {
                let v = rest % d;
                rest /= d;
                v
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        (index, got_std.iter().nth(i).copied().unwrap(), want_std.iter().nth(i).copied().unwrap())
    });
    c.pass = c.mismatches == 0 && c.note.as_deref().is_none_or(|n| !n.contains("not compared"));
    c
}

/// Output and tap names the program produces for these inputs.
fn result_names(model: &Model, inputs: &[(String, ArrayD<f32>)]) -> Result<(Vec<String>, Vec<String>)> {
    let shapes: Vec<(&str, Vec<usize>)> = inputs.iter().map(|(n, a)| (n.as_str(), a.shape().to_vec())).collect();
    let graph = model.graph(&shapes, &Taps::None)?;
    Ok((graph.outputs.into_iter().map(|(n, _)| n).collect(), graph.taps))
}

pub fn compare(args: &CompareArgs, json: bool) -> Result<i32> {
    let inputs = read_npy_args(&args.inputs)?;
    let model = load_model(&args.model)?;
    let (outputs, taps) = result_names(&model, &inputs)?;

    let mut references = read_npy_args(&args.references)?;
    if let Some(dir) = &args.reference_dir {
        for name in outputs.iter().chain(&taps) {
            let path = dir.join(format!("{}.npy", file_name(name)));
            if path.exists() && !references.iter().any(|(n, _)| n == name) {
                references.push((name.clone(), npy::read(&path)?));
            }
        }
    }
    if references.is_empty() {
        bail!(
            "no references: pass --reference NAME=file.npy or --reference-dir DIR with NAME.npy files\n  outputs: {}\n  taps: {}",
            outputs.join(", "),
            if taps.is_empty() { "(none; mark tensors with (tap \"name\" t))".into() } else { taps.join(", ") }
        );
    }
    for (name, _) in &references {
        if !outputs.contains(name) && !taps.contains(name) {
            bail!("{name:?} is neither an output ({}) nor a tap ({})", outputs.join(", "), taps.join(", "));
        }
    }

    let wanted_taps: Vec<String> = taps.iter().filter(|t| references.iter().any(|(n, _)| n == *t)).cloned().collect();
    let result = model.run_with(&views(&inputs), &RunOptions { taps: Taps::Names(wanted_taps) })?;

    // Taps in definition order first: the first failing one is closest to where the port diverges.
    let mut comparisons = Vec::new();
    for (kind, results) in [("tap", &result.taps), ("output", &result.outputs)] {
        for (name, got) in results {
            if let Some((_, want)) = references.iter().find(|(n, _)| n == name) {
                comparisons.push(compare_arrays(kind, name, got, want, args.atol, args.rtol));
            }
        }
    }
    let first_failure = comparisons.iter().find(|c| !c.pass).map(|c| format!("{} {}", c.kind, c.name));
    let all_pass = first_failure.is_none();

    if json {
        print_json(&json!({
            "pass": all_pass,
            "atol": args.atol,
            "rtol": args.rtol,
            "first_failure": first_failure,
            "comparisons": comparisons,
        }))?;
    } else {
        println!("compare (pass if |got - want| <= {} + {} * |want|):", args.atol, args.rtol);
        let width = comparisons.iter().map(|c| c.name.len()).max().unwrap_or(0);
        for c in &comparisons {
            let status = if c.pass { "ok  " } else { "FAIL" };
            let mut line = format!(
                "  {status} {:6} {:width$}  {:16} max_abs {:.3e}  mean_abs {:.3e}  cos {:.6}",
                c.kind,
                c.name,
                shape_string(&c.shape),
                c.max_abs_error,
                c.mean_abs_error,
                c.cosine_similarity
            );
            if !c.pass {
                if let Some((index, got, want)) = &c.worst {
                    line += &format!("  worst at {index:?}: got {got} want {want}");
                }
                line += &format!("  ({} of {} outside tolerance)", c.mismatches, c.shape.iter().product::<usize>());
            }
            println!("{line}");
            if c.shape != c.reference_shape {
                println!("       reference shape {}", shape_string(&c.reference_shape));
            }
            if let Some(note) = &c.note {
                println!("       note: {note}");
            }
        }
        match &first_failure {
            Some(f) => println!("first failure: {f}"),
            None => println!("all {} match", comparisons.len()),
        }
    }
    Ok(if all_pass { 0 } else { 2 })
}
