//! `tl merge`: build one GGUF from the tensors of several (e.g. a llama.cpp text model and its mmproj),
//! renaming them to the names a program expects.
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use tensorlisp::{DType, gguf::{GgufFile, GgufWriter}};

use crate::common::{print_json, read_assets, read_program};

#[derive(Args)]
pub struct MergeArgs {
    /// GGUF files whose tensors go into the output (in order; a name already taken is an error).
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,
    #[arg(short, long)]
    pub output: PathBuf,
    /// Rename tensors, OLD=NEW. One `*` in OLD matches any non-empty run of characters and `*` in NEW pastes it,
    /// e.g. `blk.*.attn_q.weight=layers.*.self_attn.q_proj.weight` (repeatable; the first match applies).
    #[arg(long = "map")]
    pub maps: Vec<String>,
    /// Leave out tensors matching this pattern (same `*` syntax, matched against the original name; repeatable).
    #[arg(long = "drop")]
    pub drops: Vec<String>,
    /// Type of float tensors: keep, or f16 (bf16/f32 matrices become f16; 1-D tensors stay f32 and quantized ones are copied).
    #[arg(long, default_value = "keep")]
    pub dtype: String,
    /// Program to store in the file.
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Embed a file the program reads with (asset NAME), NAME=PATH (repeatable).
    #[arg(long = "asset")]
    pub assets: Vec<String>,
}

/// `Some(captured)` if `name` matches `pattern` (at most one `*`).
fn matches(pattern: &str, name: &str) -> Option<String> {
    match pattern.split_once('*') {
        None => (pattern == name).then(String::new),
        Some((prefix, suffix)) => {
            let middle = name.strip_prefix(prefix)?.strip_suffix(suffix)?;
            (!middle.is_empty()).then(|| middle.to_owned())
        }
    }
}

fn rename(name: &str, maps: &[(String, String)]) -> String {
    for (old, new) in maps {
        if let Some(capture) = matches(old, name) {
            return new.replacen('*', &capture, 1);
        }
    }
    name.to_owned()
}

pub fn run(args: &MergeArgs, json: bool) -> Result<i32> {
    let maps: Vec<(String, String)> = args
        .maps
        .iter()
        .map(|m| m.split_once('=').map(|(a, b)| (a.to_owned(), b.to_owned())).with_context(|| format!("--map expects OLD=NEW, got {m:?}")))
        .collect::<Result<_>>()?;
    let to_f16 = match args.dtype.as_str() {
        "keep" => false,
        "f16" => true,
        other => bail!("--dtype is keep or f16, not {other:?}"),
    };
    let files = args.inputs.iter().map(|p| GgufFile::open(p).with_context(|| format!("opening {}", p.display()))).collect::<Result<Vec<_>>>()?;
    let mut writer = GgufWriter::new();
    let (mut kept, mut dropped, mut converted) = (0, 0, 0);
    for (path, file) in args.inputs.iter().zip(&files) {
        for (i, info) in file.tensor_infos().into_iter().enumerate() {
            if args.drops.iter().any(|d| matches(d, &info.name).is_some()) {
                dropped += 1;
                continue;
            }
            let name = rename(&info.name, &maps);
            let convert = to_f16 && matches!(info.dtype, DType::F32 | DType::BF16) && info.shape.len() >= 2;
            let target = if convert { DType::F16 } else { info.dtype };
            let (source_type, shape) = (info.dtype, info.shape.clone());
            let row = *shape.last().unwrap_or(&1);
            converted += convert as usize;
            kept += 1;
            writer.add_tensor_with(&name, target, &shape, move || {
                let bytes = file.read_tensor_bytes(path, i as i64)?;
                if !convert {
                    return Ok(bytes);
                }
                let values = source_type.to_f32(&bytes)?;
                target.from_f32(&values, row)
            })?;
        }
    }
    let mut assets = read_assets(&args.assets)?;
    if let Some(p) = &args.program {
        let program = read_program(p)?;
        tensorlisp::program_entries(&program, assets.clone()).context("the program is invalid")?;
        writer.set_program(&program)?;
    }
    for (name, bytes) in std::mem::take(&mut assets) {
        writer.set_asset(&name, &bytes)?;
    }
    writer.write(&args.output).with_context(|| format!("writing {}", args.output.display()))?;
    if json {
        print_json(&serde_json::json!({ "output": args.output, "tensors": kept, "dropped": dropped, "converted": converted }))?;
    } else {
        println!("wrote {} tensors to {} ({dropped} dropped, {converted} converted to f16)", kept, args.output.display());
    }
    Ok(0)
}
