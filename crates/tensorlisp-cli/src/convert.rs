use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use half::{bf16, f16};
use safetensors::Dtype;
use serde_json::json;
use tensorlisp::{DType, gguf::GgufFile, gguf::GgufWriter};

use crate::source::{self, Source, TensorEntry};
use crate::common::{glob_match, human_bytes, print_json, read_assets, read_program, shape_string, split_assignment};

#[derive(Args)]
pub struct ConvertArgs {
    /// .safetensors files, directories containing them, https:// URLs, or
    /// `hf:ORG/REPO[@REV]` (all of a Hugging Face repo's .safetensors; uses
    /// HF_TOKEN). URLs are streamed with Range requests: each tensor is
    /// downloaded, converted and written in turn, so the source is never on disk.
    #[arg(required = true)]
    pub inputs: Vec<String>,
    #[arg(short, long)]
    pub output: PathBuf,
    /// Program to store in the file (it can also be added later with `tl pack`).
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Type of float tensors: keep, f32, f16, bf16 or a quantized type (q8_0,
    /// q4_0, q4_K, q6_K, ...). With anything but f32, 1-D tensors (biases,
    /// norms) stay f32. A quantized type applies to 2-D+ tensors with at least
    /// --min-elements values whose rows fit its blocks; other matrices become
    /// f16. f64 becomes f32 with `keep`.
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
    /// With a quantized --dtype, store tensors whose (original) name matches
    /// this glob as f16 instead, e.g. '*embed*' or 'lm_head*' (repeatable).
    #[arg(long)]
    pub no_quant: Vec<String>,
    /// With a quantized --dtype, matrices with fewer values stay f16.
    #[arg(long, default_value_t = 1024)]
    pub min_elements: usize,
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
    Quant(DType),
}

/// The stored type for a safetensors dtype, or None if unsupported. `quant_ok`
/// says whether this tensor may take a quantized target.
fn stored_type(dtype: Dtype, ndim: usize, target: &Target, quant_ok: bool) -> Option<DType> {
    let float = |native: DType| match target {
        Target::Keep => Some(native),
        Target::Float(t) if ndim <= 1 && *t != DType::F32 => Some(DType::F32),
        Target::Float(t) => Some(*t),
        Target::Quant(_) if ndim <= 1 => Some(DType::F32),
        Target::Quant(q) if quant_ok => Some(*q),
        Target::Quant(_) => Some(DType::F16),
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

fn elem_size(dtype: Dtype) -> Option<usize> {
    Some(match dtype {
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        Dtype::I8 | Dtype::U8 | Dtype::BOOL => 1,
        _ => return None,
    })
}

/// Converts whole rows of `row_elems` values (a quantized type needs them whole).
fn convert_rows(dtype: Dtype, stored: DType, row_elems: usize, offset: Option<f32>, bytes: &[u8]) -> Result<Vec<u8>> {
    let native = matches!(
        (dtype, stored.name()),
        (Dtype::F32, "f32") | (Dtype::F16, "f16") | (Dtype::BF16, "bf16") | (Dtype::I8, "i8") | (Dtype::I16, "i16") | (Dtype::I32, "i32")
    );
    if native && offset.is_none() {
        return Ok(bytes.to_vec());
    }
    match dtype {
        Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64 => {
            let mut values = to_f32(dtype, bytes);
            if let Some(offset) = offset {
                values.iter_mut().for_each(|v| *v += offset);
            }
            if stored.is_quantized() { Ok(stored.from_f32(&values, row_elems)?) } else { Ok(encode(stored, &values)) }
        }
        _ => int_to_i32(dtype, bytes),
    }
}

/// Reads one tensor from `source` and converts it, a few rows at a time, so the
/// working set stays near `CHUNK` bytes beyond the output however big the tensor is.
fn convert_tensor(source: &Source, t: &TensorEntry, stored: DType, offset: Option<f32>) -> Result<Vec<u8>> {
    const CHUNK: usize = 128 << 20;
    let size = elem_size(t.dtype).context("unsupported dtype")?;
    let row_elems = t.shape.last().copied().unwrap_or(1).max(1);
    let elems: usize = t.shape.iter().product();
    if (t.end - t.start) as usize != elems * size {
        bail!("{} bytes of data for {} values of {:?}", t.end - t.start, elems, t.dtype);
    }
    let rows = elems / row_elems;
    let rows_per_chunk = (CHUNK / (row_elems * size)).max(1);
    let mut out = Vec::with_capacity(stored.row_size(row_elems) * rows);
    let mut row = 0;
    while row < rows {
        let n = rows_per_chunk.min(rows - row);
        let (start, end) = (t.start + (row * row_elems * size) as u64, t.start + ((row + n) * row_elems * size) as u64);
        out.extend(convert_rows(t.dtype, stored, row_elems, offset, &source.read(start, end)?)?);
        row += n;
    }
    Ok(out)
}

fn source_is_remote(source: &Source) -> bool {
    matches!(source, Source::Remote { .. })
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
        other => match DType::parse(other) {
            Ok(q) if q.is_quantized() && q.can_quantize_to() => Target::Quant(q),
            _ => bail!("unknown --dtype {other:?} (expected keep, f32, f16, bf16 or a quantized type such as q8_0, q4_0, q4_K)"),
        },
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

    // The sources (and, for URLs, just their headers) are read before the writer
    // exists; tensor data is only read while the file is written.
    let sources = source::open_all(&args.inputs)?;

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
    for (_, source, entries) in &sources {
        for t in entries {
            let name = &t.name;
            let included = args.include.is_empty() || args.include.iter().any(|g| glob_match(g, name));
            if !included || args.exclude.iter().any(|g| glob_match(g, name)) {
                skipped.push(name.clone());
                continue;
            }
            let (dtype, shape) = (t.dtype, t.shape.clone());
            let is_float = matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::F64);
            let quant_ok = match &target {
                Target::Quant(q) => {
                    is_float
                        && shape.iter().product::<usize>() >= args.min_elements
                        && shape.last().is_some_and(|row| row % q.block_size() == 0)
                        && !args.no_quant.iter().any(|g| glob_match(g, name))
                }
                _ => false,
            };
            let stored = stored_type(dtype, shape.len(), &target, quant_ok)
                .with_context(|| format!("tensor {name}: unsupported safetensors dtype {dtype:?}"))?;
            if elem_size(dtype).is_some_and(|size| (t.end - t.start) as usize != shape.iter().product::<usize>() * size) {
                bail!("tensor {name}: {} bytes of data do not match shape {}", t.end - t.start, shape_string(&shape));
            }
            let new_name = rename(name, args, &renames, &prefixes);
            let offset = offsets.iter().find(|(g, _)| glob_match(g, name)).map(|(_, v)| *v);
            if offset.is_some() && !is_float {
                bail!("--offset matches {name}, which is not a float tensor");
            }
            let bytes_name = new_name.clone();
            writer
                .add_tensor_with(&new_name, stored, &shape, move || {
                    let bytes = convert_tensor(source, t, stored, offset)
                        .map_err(|e| tensorlisp::Error::Input(format!("tensor {bytes_name}: {e:#}")))?;
                    if source_is_remote(source) {
                        eprintln!("  {bytes_name}: {}", human_bytes(bytes.len()));
                    }
                    Ok(bytes)
                })
                .with_context(|| format!("tensor {name}"))?;
            report.push((name.clone(), new_name, format!("{dtype:?}").to_lowercase(), stored, shape));
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
