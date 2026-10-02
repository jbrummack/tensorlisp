use std::{path::PathBuf, str::FromStr};

use anyhow::{Context, Result, bail};
use ndarray::ArrayD;
use tensorlisp::{Device, InputSpec, LoadOptions, Model, Program, RawInput, RawKind};

use crate::{ModelArgs, npy};

#[derive(Debug, Clone, Copy)]
pub struct DeviceArg(pub Device);

impl FromStr for DeviceArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Ok(DeviceArg(match s {
            "auto" => Device::Auto,
            "cpu" => Device::Cpu,
            "gpu" => Device::Gpu,
            "native" => Device::Native,
            other => return Err(format!("unknown device {other:?} (expected auto, cpu, gpu or native)")),
        }))
    }
}

pub fn read_program(path: &PathBuf) -> Result<Program> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading program {}", path.display()))?;
    Ok(Program::Text(text))
}

/// `NAME=PATH` asset arguments, read into memory.
pub fn read_assets(args: &[String]) -> Result<Vec<(String, Vec<u8>)>> {
    args.iter()
        .map(|arg| {
            let (name, path) = split_assignment(arg)?;
            let bytes = std::fs::read(path).with_context(|| format!("reading asset {path}"))?;
            Ok((name.to_string(), bytes))
        })
        .collect()
}

pub fn load_model(args: &ModelArgs) -> Result<Model> {
    let options = LoadOptions {
        program: args.program.as_ref().map(read_program).transpose()?,
        assets: read_assets(&args.assets)?,
    };
    Model::load_with(&args.model, args.device.0, options).with_context(|| format!("loading {}", args.model.display()))
}

/// Raw examples from `NAME=VALUE` arguments: `@path` reads a file (decoded
/// by the input's kind), anything else is literal text. Repeating a name adds
/// examples to the batch.
pub fn read_raw_examples(model: &Model, args: &[String]) -> Result<Vec<Vec<(String, RawInput)>>> {
    let specs = model.raw_inputs().context("the program has no (preprocess ...) form; pass arrays with -i")?;
    read_raw_examples_for(&specs, args)
}

/// Like [`read_raw_examples`], for the given raw input specs (e.g. a pipeline's).
pub fn read_raw_examples_for(specs: &[tensorlisp::RawSpec], args: &[String]) -> Result<Vec<Vec<(String, RawInput)>>> {
    let mut columns: Vec<(String, Vec<RawInput>)> = specs.iter().map(|s| (s.name.clone(), Vec::new())).collect();
    for arg in args {
        let (name, value) = split_assignment(arg)?;
        let spec = specs.iter().find(|s| s.name == name).with_context(|| {
            format!("unknown raw input {name:?}, expected: {}", specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "))
        })?;
        let file = value.strip_prefix('@');
        let read = |p: &str| std::fs::read(p).with_context(|| format!("reading {p}"));
        let input = match (spec.kind, file) {
            (RawKind::Text, None) => RawInput::Text(value.to_string()),
            (RawKind::Text, Some(p)) => RawInput::Text(String::from_utf8(read(p)?).context("text file is not UTF-8")?),
            (RawKind::Image, Some(p)) => RawInput::Image(tensorlisp::autopro::image::decode(&read(p)?)?),
            (RawKind::Audio, Some(p)) => RawInput::Audio(tensorlisp::autopro::audio::Audio::from_wav_bytes(&read(p)?)?),
            (RawKind::Array, Some(p)) => RawInput::Array(npy::read(p.as_ref())?),
            (kind, None) => bail!("raw input {name:?} is {kind:?}; pass a file as {name}=@path"),
        };
        columns.iter_mut().find(|(n, _)| n == name).unwrap().1.push(input);
    }
    let batch = columns.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    if let Some((name, v)) = columns.iter().find(|(_, v)| v.len() != batch) {
        bail!("raw input {name:?} given {} times, others {batch} times", v.len());
    }
    let mut columns: Vec<(String, std::vec::IntoIter<RawInput>)> =
        columns.into_iter().map(|(n, v)| (n, v.into_iter())).collect();
    Ok((0..batch)
        .map(|_| columns.iter_mut().map(|(n, it)| (n.clone(), it.next().unwrap())).collect())
        .collect())
}

/// Splits `name=value`.
pub fn split_assignment(s: &str) -> Result<(&str, &str)> {
    s.split_once('=').with_context(|| format!("expected NAME=VALUE, got {s:?}"))
}

/// `NAME=file.npy` arguments, read as f32 arrays.
pub fn read_npy_args(args: &[String]) -> Result<Vec<(String, ArrayD<f32>)>> {
    args.iter()
        .map(|arg| {
            let (name, path) = split_assignment(arg)?;
            Ok((name.to_string(), npy::read(path.as_ref())?))
        })
        .collect()
}

/// Parses "1,3,224,224" (or "1x3x224x224").
pub fn parse_shape(s: &str) -> Result<Vec<usize>> {
    let dims: Vec<usize> = s
        .split([',', 'x'])
        .map(|d| d.trim().parse().with_context(|| format!("bad dimension {d:?} in shape {s:?}")))
        .collect::<Result<_>>()?;
    if dims.is_empty() {
        bail!("empty shape");
    }
    Ok(dims)
}

pub fn human_bytes(n: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 { format!("{n} B") } else { format!("{v:.1} {}", UNITS[unit]) }
}

pub fn shape_string(shape: &[usize]) -> String {
    format!("[{}]", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

/// Summary statistics of an array.
#[derive(serde::Serialize)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub nan: usize,
}

pub fn stats(a: &ArrayD<f32>) -> Stats {
    let (mut min, mut max, mut sum, mut sum_sq, mut nan, mut n) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64, 0, 0);
    for &v in a.iter() {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
        n += 1;
    }
    let mean = if n > 0 { sum / n as f64 } else { f64::NAN };
    let std = if n > 0 { (sum_sq / n as f64 - mean * mean).max(0.0).sqrt() } else { f64::NAN };
    Stats { min, max, mean, std, nan }
}

pub fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// A tap or output name as a file name.
pub fn file_name(name: &str) -> String {
    name.chars().map(|c| if c == '/' || c == '\\' { '_' } else { c }).collect()
}

/// Glob matching with `*` (any run of characters) and `?` (one character).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    // Positions after the last `*` in the pattern and the name it resumed at.
    let (mut pi, mut ni, mut star, mut resume) = (0, 0, None, 0);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            pi += 1;
            resume = ni;
        } else if let Some(s) = star {
            pi = s + 1;
            resume += 1;
            ni = resume;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("*", "a.b"));
        assert!(glob_match("*.weight", "blocks.0.attn.weight"));
        assert!(!glob_match("*.weight", "blocks.0.attn.bias"));
        assert!(glob_match("blocks.?.attn.*", "blocks.3.attn.qkv.weight"));
        assert!(!glob_match("blocks.?.attn.*", "blocks.12.attn.qkv.weight"));
        assert!(glob_match("*norm*", "layer_norm.bias"));
        assert!(glob_match("a*b*c", "axxbyyc"));
        assert!(!glob_match("a*b*c", "axxbyy"));
    }
}

/// The declared shape in numpy order, if every dimension is fixed.
fn declared_shape(spec: &InputSpec) -> Option<Vec<usize>> {
    let dims = spec.dims.as_ref()?;
    dims.iter().rev().map(|d| d.map(|d| d as usize)).collect()
}

fn declared_string(spec: &InputSpec) -> String {
    match &spec.dims {
        Some(dims) => format!(
            "[{}]",
            dims.iter().rev().map(|d| d.map_or("_".into(), |d| d.to_string())).collect::<Vec<_>>().join(", ")
        ),
        None => "any shape".into(),
    }
}

/// Input shapes (numpy order) from `NAME=1,3,224,224` / `NAME=file.npy`
/// arguments, falling back to the program's fixed declared shapes.
pub fn resolve_input_shapes(model: &Model, entry: Option<&str>, args: &[String]) -> Result<Vec<(String, Vec<usize>)>> {
    let entry = model.entry(entry)?;
    let mut shapes: Vec<(String, Vec<usize>)> = args
        .iter()
        .map(|arg| {
            let (name, value) = split_assignment(arg)?;
            let shape = if value.ends_with(".npy") { npy::read(value.as_ref())?.shape().to_vec() } else { parse_shape(value)? };
            Ok((name.to_string(), shape))
        })
        .collect::<Result<_>>()?;
    for spec in &entry.inputs {
        if shapes.iter().any(|(n, _)| *n == spec.name) {
            continue;
        }
        match declared_shape(spec) {
            Some(shape) => shapes.push((spec.name.clone(), shape)),
            None => bail!(
                "input {:?} ({}) needs a shape: --input {}=DIMS (numpy order)",
                spec.name,
                declared_string(spec),
                spec.name
            ),
        }
    }
    Ok(shapes)
}
