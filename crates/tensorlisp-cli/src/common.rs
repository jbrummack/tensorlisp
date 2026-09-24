use std::{path::PathBuf, str::FromStr};

use anyhow::{Context, Result, bail};
use ndarray::ArrayD;
use tensorlisp::{Device, InputSpec, Model, Program};

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
            other => return Err(format!("unknown device {other:?} (expected auto, cpu or gpu)")),
        }))
    }
}

pub fn read_program(path: &PathBuf) -> Result<Program> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading program {}", path.display()))?;
    Ok(Program::Text(text))
}

pub fn load_model(args: &ModelArgs) -> Result<Model> {
    let program = args.program.as_ref().map(read_program).transpose()?;
    Model::load_with(&args.model, args.device.0, program)
        .with_context(|| format!("loading {}", args.model.display()))
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
pub fn resolve_input_shapes(model: &Model, args: &[String]) -> Result<Vec<(String, Vec<usize>)>> {
    let mut shapes: Vec<(String, Vec<usize>)> = args
        .iter()
        .map(|arg| {
            let (name, value) = split_assignment(arg)?;
            let shape = if value.ends_with(".npy") { npy::read(value.as_ref())?.shape().to_vec() } else { parse_shape(value)? };
            Ok((name.to_string(), shape))
        })
        .collect::<Result<_>>()?;
    for spec in model.inputs() {
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
