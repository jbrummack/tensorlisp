use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Args;
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::json;
use tensorlisp::{DType, gguf::GgufFile, gguf::GgufWriter};

use crate::common::{glob_match, human_bytes, print_json, read_assets, read_program, shape_string, split_assignment};

#[derive(Args)]
pub struct ConvertArgs {
    /// .safetensors files, or directories containing them.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,
    #[arg(short, long)]
    pub output: PathBuf,
    /// Program to store in the file (it can also be added later with `tl pack`).
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Type of float tensors: keep, f32, f16 or bf16. With f16/bf16, 1-D
    /// tensors (biases, norms) stay f32. f64 becomes f32 with `keep`.
    #[arg(long, default_value = "keep")]
    pub dtype: String,
    /// Remove this prefix from tensor names (repeatable; the first match is removed).
    #[arg(long)]
    pub strip_prefix: Vec<String>,
    /// Rename a tensor, OLD=NEW, after stripping prefixes (repeatable).
    #[arg(long)]
    pub rename: Vec<String>,
    /// Replace a name prefix, OLD=NEW, after stripping prefixes (repeatable;
    /// the first match applies), e.g. to shorten names past ggml's 63 bytes.
    #[arg(long)]
    pub replace_prefix: Vec<String>,
    /// Add a constant to every value of float tensors whose (original) name
    /// matches, GLOB=VALUE (repeatable; the first match applies). E.g. Gemma's
    /// RMSNorm scales by (1 + w): `--offset '*norm.weight=1'` stores 1 + w.
    #[arg(long)]
    pub offset: Vec<String>,
    /// Only convert tensors whose (original) name matches this glob (repeatable).
    #[arg(long)]
    pub include: Vec<String>,
    /// Skip tensors whose (original) name matches this glob (repeatable).
    #[arg(long)]
    pub exclude: Vec<String>,
    /// Add a string metadata key, KEY=VALUE (repeatable).
    #[arg(long)]
    pub meta: Vec<String>,
    /// Embed a file the program reads with (asset NAME), NAME=PATH (repeatable).
    #[arg(long = "asset")]
    pub assets: Vec<String>,
}

#[derive(Args)]
pub struct PackArgs {
    /// GGUF file to add the program to.
    pub model: PathBuf,
    /// Program source file.
    #[arg(long)]
    pub program: PathBuf,
    /// Write to this file instead of replacing the model file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Embed a file the program reads with (asset NAME), NAME=PATH (repeatable).
    #[arg(long = "asset")]
    pub assets: Vec<String>,
}

fn safetensors_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut found: Vec<_> = std::fs::read_dir(input)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            found.sort();
            if found.is_empty() {
                bail!("no .safetensors files in {}", input.display());
            }
            files.extend(found);
        } else {
            files.push(input.clone());
        }
    }
    Ok(files)
}

fn to_f32(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        Dtype::F16 => bytes.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        Dtype::BF16 => bytes.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        Dtype::F64 => bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        _ => unreachable!("not a float type"),
    }
}

fn encode(target: DType, values: &[f32]) -> Vec<u8> {
    match target {
        DType::F16 => values.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect(),
        DType::BF16 => values.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes()).collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn int_to_i32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<u8>> {
    let values: Vec<i64> = match dtype {
        Dtype::I64 => bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect(),
        Dtype::U32 => bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as i64).collect(),
        Dtype::U16 => bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as i64).collect(),
        Dtype::U8 | Dtype::BOOL => bytes.iter().map(|&b| b as i64).collect(),
        _ => unreachable!("not handled as i32"),
    };
    values
        .into_iter()
        .map(|v| i32::try_from(v).map(i32::to_le_bytes).map_err(|_| anyhow::anyhow!("value {v} does not fit in i32")))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.concat())
}

enum Target {
    Keep,
    Float(DType),
}

/// The stored type for a safetensors dtype, or None if unsupported.
fn stored_type(dtype: Dtype, ndim: usize, target: &Target) -> Option<DType> {
    let float = |native: DType| match target {
        Target::Keep => Some(native),
        Target::Float(t) if ndim <= 1 && *t != DType::F32 => Some(DType::F32),
        Target::Float(t) => Some(*t),
    };
    match dtype {
        Dtype::F32 | Dtype::F64 => float(DType::F32),
        Dtype::F16 => float(DType::F16),
        Dtype::BF16 => float(DType::BF16),
        Dtype::I8 => DType::parse("i8").ok(),
        Dtype::I16 => DType::parse("i16").ok(),
        Dtype::I32 | Dtype::I64 | Dtype::U32 | Dtype::U16 | Dtype::U8 | Dtype::BOOL => DType::parse("i32").ok(),
        _ => None,
    }
}

fn convert_bytes(dtype: Dtype, stored: DType, bytes: &[u8]) -> Result<Vec<u8>> {
    let native = matches!(
        (dtype, stored.name()),
        (Dtype::F32, "f32") | (Dtype::F16, "f16") | (Dtype::BF16, "bf16") | (Dtype::I8, "i8") | (Dtype::I16, "i16") | (Dtype::I32, "i32")
    );
    if native {
        return Ok(bytes.to_vec());
    }
    match dtype {
        Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64 => Ok(encode(stored, &to_f32(dtype, bytes))),
        _ => int_to_i32(dtype, bytes),
    }
}

fn rename(name: &str, args: &ConvertArgs, renames: &[(&str, &str)], prefixes: &[(&str, &str)]) -> String {
    let stripped = args.strip_prefix.iter().find_map(|p| name.strip_prefix(p.as_str())).unwrap_or(name);
    let replaced = prefixes
        .iter()
        .find_map(|(old, new)| stripped.strip_prefix(old).map(|rest| format!("{new}{rest}")))
        .unwrap_or_else(|| stripped.to_string());
    renames.iter().find(|(old, _)| *old == replaced).map_or(replaced.clone(), |(_, new)| new.to_string())
}

pub fn run(args: &ConvertArgs, json: bool) -> Result<i32> {
    let target = match args.dtype.as_str() {
        "keep" => Target::Keep,
        "f32" => Target::Float(DType::F32),
        "f16" => Target::Float(DType::F16),
        "bf16" => Target::Float(DType::BF16),
        other => bail!("unknown --dtype {other:?} (expected keep, f32, f16 or bf16)"),
    };
    let renames: Vec<_> = args.rename.iter().map(|r| split_assignment(r)).collect::<Result<_>>()?;
    let prefixes: Vec<_> = args.replace_prefix.iter().map(|r| split_assignment(r)).collect::<Result<_>>()?;
    let offsets: Vec<(&str, f32)> = args
        .offset
        .iter()
        .map(|o| {
            let (glob, value) = split_assignment(o)?;
            Ok((glob, value.parse::<f32>().with_context(|| format!("--offset {o}: not a number"))?))
        })
        .collect::<Result<_>>()?;
    let program = args.program.as_ref().map(read_program).transpose()?;
    let assets = read_assets(&args.assets)?;
    if let Some(program) = &program {
        tensorlisp::program_entries(program, assets.clone()).context("the program is invalid")?;
    }

    let files = safetensors_files(&args.inputs)?;
    let maps: Vec<Mmap> = files
        .iter()
        .map(|f| {
            let file = File::open(f).with_context(|| format!("opening {}", f.display()))?;
            Ok(unsafe { Mmap::map(&file)? })
        })
        .collect::<Result<_>>()?;

    let mut writer = GgufWriter::new();
    if let Some(program) = &program {
        writer.set_program(program)?;
    }
    for meta in &args.meta {
        let (key, value) = split_assignment(meta)?;
        writer.set_str(key, value)?;
    }
    for (name, bytes) in &assets {
        writer.set_asset(name, bytes)?;
    }

    let mut report = Vec::new();
    let mut skipped = Vec::new();
    for (path, map) in files.iter().zip(&maps) {
        let st = SafeTensors::deserialize(map).with_context(|| format!("reading {}", path.display()))?;
        let mut tensors = st.tensors();
        tensors.sort_by_key(|(_, view)| view.data().as_ptr() as usize);
        for (name, view) in tensors {
            let included = args.include.is_empty() || args.include.iter().any(|g| glob_match(g, &name));
            if !included || args.exclude.iter().any(|g| glob_match(g, &name)) {
                skipped.push(name);
                continue;
            }
            let dtype = view.dtype();
            let shape = view.shape().to_vec();
            let stored = stored_type(dtype, shape.len(), &target)
                .with_context(|| format!("tensor {name}: unsupported safetensors dtype {dtype:?}"))?;
            let new_name = rename(&name, args, &renames, &prefixes);
            let data = view.data();
            let bytes_name = new_name.clone();
            let offset = offsets.iter().find(|(g, _)| glob_match(g, &name)).map(|(_, v)| *v);
            if offset.is_some() && !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64) {
                bail!("--offset matches {name}, which is not a float tensor");
            }
            writer
                .add_tensor_with(&new_name, stored, &shape, move || match offset {
                    Some(offset) => {
                        let values: Vec<f32> = to_f32(dtype, data).into_iter().map(|v| v + offset).collect();
                        Ok(encode(stored, &values))
                    }
                    None => convert_bytes(dtype, stored, data)
                        .map_err(|e| tensorlisp::Error::Input(format!("tensor {bytes_name}: {e}"))),
                })
                .with_context(|| format!("tensor {name}"))?;
            report.push((name, new_name, format!("{dtype:?}").to_lowercase(), stored, shape));
        }
    }
    writer.write(&args.output).with_context(|| format!("writing {}", args.output.display()))?;
    let size = std::fs::metadata(&args.output)?.len() as usize;

    if json {
        print_json(&json!({
            "output": args.output,
            "bytes": size,
            "program": program.is_some(),
            "tensors": report.iter().map(|(from, to, src, dst, shape)| json!({
                "source_name": from, "name": to, "source_type": src, "type": dst.name(), "shape": shape,
            })).collect::<Vec<_>>(),
            "skipped": skipped,
        }))?;
    } else {
        let width = report.iter().map(|r| r.1.len()).max().unwrap_or(0);
        for (from, to, src, dst, shape) in &report {
            let renamed = if from != to { format!("  (from {from})") } else { String::new() };
            println!("  {to:width$}  {src:>5} -> {:5} {}{renamed}", dst.name(), shape_string(shape));
        }
        println!(
            "wrote {} tensors to {} ({}){}{}",
            report.len(),
            args.output.display(),
            human_bytes(size),
            if program.is_some() { ", with program" } else { ", no program (add one with `tl pack`)" },
            if skipped.is_empty() { String::new() } else { format!(", skipped {}", skipped.len()) },
        );
    }
    Ok(0)
}

/// Rewrites `model` with its tensors and metadata plus `program` and `assets`.
pub fn repack(model: &Path, output: &Path, program: &tensorlisp::Program, assets: &[(String, Vec<u8>)]) -> Result<()> {
    let file = GgufFile::open(model).with_context(|| format!("opening {}", model.display()))?;
    let mut writer = GgufWriter::new();
    writer.copy_metadata(&file);
    writer.set_program(program)?;
    for (name, bytes) in assets {
        writer.set_asset(name, bytes)?;
    }
    for (i, info) in file.tensor_infos().into_iter().enumerate() {
        let file = &file;
        writer.add_tensor_with(&info.name, info.dtype, &info.shape, move || file.read_tensor_bytes(model, i as i64))?;
    }
    // Write next to the destination, then move into place, so the source can be the destination.
    let tmp = output.with_extension("gguf.tmp");
    writer.write(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, output)?;
    Ok(())
}

pub fn pack(args: &PackArgs, json: bool) -> Result<i32> {
    let program = read_program(&args.program)?;
    let new_assets = read_assets(&args.assets)?;
    // The program sees the file's assets plus the new ones.
    let mut all_assets = GgufFile::open(&args.model).with_context(|| format!("opening {}", args.model.display()))?.assets();
    for (name, bytes) in &new_assets {
        all_assets.retain(|(n, _)| n != name);
        all_assets.push((name.clone(), bytes.clone()));
    }
    let (entries, pipelines) = tensorlisp::program_entries(&program, all_assets).context("the program is invalid")?;
    let output = args.output.as_ref().unwrap_or(&args.model);
    repack(&args.model, output, &program, &new_assets)?;
    if json {
        print_json(&json!({
            "output": output,
            "inputs": entries[0].inputs.iter().map(|i| &i.name).collect::<Vec<_>>(),
            "entries": entries.iter().map(|e| json!({ "name": e.name, "inputs": e.inputs.iter().map(|i| &i.name).collect::<Vec<_>>() })).collect::<Vec<_>>(),
            "pipelines": pipelines.iter().map(|p| &p.name).collect::<Vec<_>>(),
        }))?;
    } else {
        println!("stored {} in {}", args.program.display(), output.display());
    }
    Ok(0)
}
