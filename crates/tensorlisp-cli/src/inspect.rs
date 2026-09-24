use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use tensorlisp::{
    Program,
    gguf::{GgufFile, MetaValue},
    program::{TL_BIN, TL_TXT},
};

use crate::common::{human_bytes, print_json, shape_string};

#[derive(Args)]
pub struct InspectArgs {
    pub file: PathBuf,
    /// Print only the program source (e.g. to edit it and pass it back with --program).
    #[arg(long)]
    pub program: bool,
    /// Show metadata arrays in full instead of their first 16 values.
    #[arg(long)]
    pub full: bool,
}

const ARRAY_PREVIEW: usize = 16;

fn meta_json(value: &MetaValue, full: bool) -> Value {
    match value {
        MetaValue::U8(v) => json!(v),
        MetaValue::I8(v) => json!(v),
        MetaValue::U16(v) => json!(v),
        MetaValue::I16(v) => json!(v),
        MetaValue::U32(v) => json!(v),
        MetaValue::I32(v) => json!(v),
        MetaValue::U64(v) => json!(v),
        MetaValue::I64(v) => json!(v),
        MetaValue::F32(v) => json!(v),
        MetaValue::F64(v) => json!(v),
        MetaValue::Bool(v) => json!(v),
        MetaValue::String(v) => json!(v),
        MetaValue::Array(items) => {
            let shown = if full { items.len() } else { items.len().min(ARRAY_PREVIEW) };
            json!({
                "len": items.len(),
                "values": items[..shown].iter().map(|v| meta_json(v, full)).collect::<Vec<_>>(),
            })
        }
    }
}

fn meta_type(value: &MetaValue) -> String {
    match value {
        MetaValue::U8(_) => "u8".into(),
        MetaValue::I8(_) => "i8".into(),
        MetaValue::U16(_) => "u16".into(),
        MetaValue::I16(_) => "i16".into(),
        MetaValue::U32(_) => "u32".into(),
        MetaValue::I32(_) => "i32".into(),
        MetaValue::U64(_) => "u64".into(),
        MetaValue::I64(_) => "i64".into(),
        MetaValue::F32(_) => "f32".into(),
        MetaValue::F64(_) => "f64".into(),
        MetaValue::Bool(_) => "bool".into(),
        MetaValue::String(_) => "str".into(),
        MetaValue::Array(items) => {
            format!("{}[{}]", items.first().map(meta_type).unwrap_or_else(|| "?".into()), items.len())
        }
    }
}

fn meta_text(value: &MetaValue, full: bool) -> String {
    match value {
        MetaValue::String(s) if s.len() > 80 && !full => format!("{:?}… ({} bytes)", &s[..s.floor_char_boundary(80)], s.len()),
        MetaValue::Array(items) => {
            let shown = if full { items.len() } else { items.len().min(ARRAY_PREVIEW) };
            let values: Vec<_> = items[..shown].iter().map(|v| meta_text(v, full)).collect();
            let more = if shown < items.len() { ", …" } else { "" };
            format!("[{}{more}]", values.join(", "))
        }
        other => meta_json(other, full).to_string(),
    }
}

pub fn run(args: &InspectArgs, json: bool) -> Result<i32> {
    let file = GgufFile::open(&args.file).with_context(|| format!("opening {}", args.file.display()))?;
    let program = file.program();

    if args.program {
        let Program::Text(text) = program?;
        if json {
            print_json(&json!({ "program": text }))?;
        } else {
            print!("{text}");
        }
        return Ok(0);
    }

    let metadata: Vec<_> = file.metadata().into_iter().filter(|(k, _)| k != TL_TXT && k != TL_BIN).collect();
    let tensors = file.tensor_infos();
    let total: usize = tensors.iter().map(|t| t.nbytes).sum();

    if json {
        let program = match &program {
            Ok(Program::Text(text)) => json!({ "format": "text", "text": text }),
            Err(e) => json!({ "error": e.to_string() }),
        };
        return print_json(&json!({
            "file": args.file,
            "gguf_version": file.version(),
            "program": program,
            "metadata": metadata.iter().map(|(k, v)| (k.clone(), meta_json(v, args.full))).collect::<serde_json::Map<_, _>>(),
            "tensors": tensors.iter().map(|t| json!({
                "name": t.name,
                "type": t.dtype.name(),
                "shape": t.shape,
                "bytes": t.nbytes,
                "offset": t.offset,
            })).collect::<Vec<_>>(),
            "tensor_bytes": total,
        }))
        .map(|_| 0);
    }

    println!(
        "{}: GGUF v{}, {} tensors ({}), {} metadata keys",
        args.file.display(),
        file.version(),
        tensors.len(),
        human_bytes(total),
        metadata.len()
    );
    match &program {
        Ok(Program::Text(text)) => println!(
            "program: text, {} lines ({}; print it with `tl inspect --program`)",
            text.lines().count(),
            human_bytes(text.len())
        ),
        Err(tensorlisp::Error::Program(e)) => println!("program: none ({e})"),
        Err(e) => println!("program: none ({e})"),
    }
    if !metadata.is_empty() {
        println!("\nmetadata:");
        let width = metadata.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (key, value) in &metadata {
            println!("  {key:width$}  {:10} {}", meta_type(value), meta_text(value, args.full));
        }
    }
    if !tensors.is_empty() {
        println!("\ntensors (shapes in numpy order):");
        let width = tensors.iter().map(|t| t.name.len()).max().unwrap_or(0);
        for t in &tensors {
            println!(
                "  {:width$}  {:6} {:22} {:>10}",
                t.name,
                t.dtype.name(),
                shape_string(&t.shape),
                human_bytes(t.nbytes)
            );
        }
    }
    Ok(0)
}
