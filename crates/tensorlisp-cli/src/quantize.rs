use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::json;
use tensorlisp::{
    DType,
    gguf::{GgufFile, GgufWriter, TensorInfo},
};

use crate::common::{glob_match, human_bytes, print_json, shape_string};

#[derive(Args)]
pub struct QuantizeArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Target type, e.g. q8_0, q6_K, q4_K, q4_0 or f16.
    #[arg(short = 't', long = "type")]
    pub dtype: String,
    /// Only quantize tensors whose name matches this glob (repeatable).
    #[arg(long)]
    pub include: Vec<String>,
    /// Never quantize tensors whose name matches this glob (repeatable).
    #[arg(long)]
    pub exclude: Vec<String>,
    /// Keep tensors with fewer elements unchanged.
    #[arg(long, default_value_t = 1024)]
    pub min_elements: usize,
    /// Also quantize 1-D tensors (biases, norms); they are kept by default.
    #[arg(long)]
    pub include_1d: bool,
    /// Also convert tensors that are already quantized (loses precision twice).
    #[arg(long)]
    pub requantize: bool,
}

/// Why a tensor keeps its type, or None to convert it.
fn skip_reason(t: &TensorInfo, target: DType, args: &QuantizeArgs) -> Option<String> {
    let elements: usize = t.shape.iter().product();
    if t.dtype == target {
        return Some(format!("already {target}"));
    }
    if !args.include.is_empty() && !args.include.iter().any(|g| glob_match(g, &t.name)) {
        return Some("not included".into());
    }
    if let Some(g) = args.exclude.iter().find(|g| glob_match(g, &t.name)) {
        return Some(format!("excluded by {g}"));
    }
    if ["i8", "i16", "i32", "i64"].contains(&t.dtype.name()) {
        return Some("integer tensor".into());
    }
    if t.dtype.is_quantized() && !args.requantize {
        return Some("already quantized (use --requantize)".into());
    }
    if t.shape.len() < 2 && !args.include_1d {
        return Some("1-D (use --include-1d)".into());
    }
    if elements < args.min_elements {
        return Some(format!("fewer than {} elements", args.min_elements));
    }
    let row = t.shape.last().copied().unwrap_or(1);
    if row % target.block_size() != 0 {
        return Some(format!("row length {row} is not a multiple of the {target} block size {}", target.block_size()));
    }
    None
}

pub fn run(args: &QuantizeArgs, json: bool) -> Result<i32> {
    let target = DType::parse(&args.dtype)?;
    if !target.can_quantize_to() {
        let names: Vec<_> = DType::quantize_targets().filter(|t| t.can_quantize_to()).map(|t| t.name()).collect();
        bail!("can't quantize to {target}; supported: {}", names.join(", "));
    }
    if args.input == args.output {
        bail!("output must differ from input");
    }
    let file = GgufFile::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;
    let infos = file.tensor_infos();

    let mut writer = GgufWriter::new();
    writer.copy_metadata(&file);
    let mut report = Vec::new();
    for (i, t) in infos.iter().enumerate() {
        let reason = skip_reason(t, target, args);
        let stored = if reason.is_none() { target } else { t.dtype };
        let (file, input, source) = (&file, &args.input, t.dtype);
        let row = t.shape.last().copied().unwrap_or(1);
        writer.add_tensor_with(&t.name, stored, &t.shape, move || {
            let bytes = file.read_tensor_bytes(input, i as i64)?;
            if stored == source { Ok(bytes) } else { stored.from_f32(&source.to_f32(&bytes)?, row) }
        })?;
        let new_bytes = match reason {
            Some(_) => t.nbytes,
            None => stored.row_size(row) * (t.shape.iter().product::<usize>() / row.max(1)),
        };
        report.push((t, stored, new_bytes, reason));
    }
    writer.write(&args.output).with_context(|| format!("writing {}", args.output.display()))?;

    let before: usize = infos.iter().map(|t| t.nbytes).sum();
    let after: usize = report.iter().map(|r| r.2).sum();
    let converted = report.iter().filter(|r| r.3.is_none()).count();

    if json {
        print_json(&json!({
            "output": args.output,
            "type": target.name(),
            "tensor_bytes_before": before,
            "tensor_bytes_after": after,
            "tensors": report.iter().map(|(t, stored, bytes, reason)| json!({
                "name": t.name, "from": t.dtype.name(), "to": stored.name(), "shape": t.shape,
                "bytes_before": t.nbytes, "bytes_after": bytes, "kept_because": reason,
            })).collect::<Vec<_>>(),
        }))?;
    } else {
        let width = report.iter().map(|r| r.0.name.len()).max().unwrap_or(0);
        for (t, stored, bytes, reason) in &report {
            let what = match reason {
                None => format!("{:>5} -> {:5}", t.dtype.name(), stored.name()),
                Some(_) => format!("{:>5}    {:5}", t.dtype.name(), ""),
            };
            println!(
                "  {:width$}  {what} {:18} {:>10} -> {:>10}{}",
                t.name,
                shape_string(&t.shape),
                human_bytes(t.nbytes),
                human_bytes(*bytes),
                reason.as_ref().map(|r| format!("  kept: {r}")).unwrap_or_default()
            );
        }
        println!(
            "quantized {converted} of {} tensors to {target}: {} -> {} ({})",
            report.len(),
            human_bytes(before),
            human_bytes(after),
            args.output.display()
        );
    }
    Ok(0)
}
