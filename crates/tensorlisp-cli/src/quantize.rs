use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::json;
use tensorlisp::{
    DType, Device, GraphInfo, Model, Taps,
    gguf::{GgufFile, GgufWriter, TensorInfo},
};

use crate::common::{glob_match, human_bytes, print_json, read_program, resolve_input_shapes, shape_string};

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
    /// With a program, only weights the graph feeds into matmuls (or
    /// embedding lookups) are converted; the rest keep their type.
    #[arg(long, default_value_t = 1024)]
    pub min_elements: usize,
    /// Also quantize 1-D tensors (biases, norms); they are kept by default.
    #[arg(long)]
    pub include_1d: bool,
    /// Also convert tensors that are already quantized (loses precision twice).
    #[arg(long)]
    pub requantize: bool,
    /// Input shape for building the program's graphs, NAME=1,3,448,448 or
    /// NAME=file.npy (repeatable; fixed declared shapes can be omitted). With
    /// several entries, ENTRY:NAME=... sets one entry's input; an unprefixed
    /// NAME applies to every entry that has it.
    #[arg(short, long = "input")]
    pub inputs: Vec<String>,
    /// Analyze this program instead of the one stored in the file.
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Don't analyze the program's graph; choose tensors by name, rank and size only.
    #[arg(long)]
    pub no_graph: bool,
}

/// Ops that only reinterpret a tensor; uses of their result count as uses of their source.
const VIEW_OPS: &[&str] = &["RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"];
/// (op, source index) pairs that accept quantized weights.
const QUANTIZABLE_USES: &[(&str, usize)] = &[("MUL_MAT", 0), ("MUL_MAT_ID", 0), ("GET_ROWS", 0), ("IM2COL", 0)];

/// Every (op, source index) that consumes each weight, looking through views.
fn weight_uses(graph: &GraphInfo, weights: &HashSet<&str>) -> HashMap<String, Vec<(String, usize)>> {
    let mut root: Vec<Option<String>> = Vec::with_capacity(graph.nodes.len());
    let mut uses: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    for node in &graph.nodes {
        let resolve = |src: &str| match src.strip_prefix('%') {
            Some(i) => i.parse::<usize>().ok().and_then(|i| root.get(i).cloned().flatten()),
            None => weights.contains(src).then(|| src.to_string()),
        };
        let is_view = VIEW_OPS.contains(&node.op.as_str());
        for (j, src) in node.srcs.iter().enumerate() {
            if let Some(w) = resolve(src) {
                if !(is_view && j == 0) {
                    uses.entry(w).or_default().push((node.op.clone(), j));
                }
            }
        }
        root.push(if is_view { node.srcs.first().and_then(|s| resolve(s)) } else { None });
    }
    uses
}

/// Whether a shape argument (numpy order, or a .npy file) fits an input's declared dims.
fn shape_fits(spec: &tensorlisp::InputSpec, value: &str) -> bool {
    let Some(dims) = &spec.dims else { return true };
    let shape = if value.ends_with(".npy") {
        crate::npy::read(value.as_ref()).map(|a| a.shape().to_vec()).ok()
    } else {
        crate::common::parse_shape(value).ok()
    };
    shape.is_none_or(|shape| {
        shape.len() == dims.len() && shape.iter().rev().zip(dims).all(|(&n, d)| d.is_none_or(|d| d == n as i64))
    })
}

/// Builds the program's graph and returns how it uses each weight.
fn analyze(args: &QuantizeArgs, file: &GgufFile, names: &HashSet<&str>) -> Result<Option<HashMap<String, Vec<(String, usize)>>>> {
    if args.no_graph {
        return Ok(None);
    }
    let program = match &args.program {
        Some(path) => read_program(path)?,
        None => match file.program() {
            Ok(program) => program,
            Err(_) => {
                eprintln!("note: {} has no program; choosing tensors by name, rank and size only", args.input.display());
                return Ok(None);
            }
        },
    };
    let options = tensorlisp::LoadOptions { program: Some(program), assets: Vec::new() };
    let model = Model::load_with(&args.input, Device::Cpu, options).context("loading the program to analyze its graph")?;
    // Every entry's graph: a weight is used however any entry uses it.
    let mut uses: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    for entry in model.entries() {
        let args_for_entry: Vec<String> = args
            .inputs
            .iter()
            .filter_map(|arg| match arg.split_once('=').and_then(|(name, _)| name.split_once(':')) {
                Some((e, _)) if e == entry.name => Some(arg[e.len() + 1..].to_string()),
                Some(_) => None,
                // Unprefixed shapes apply to every entry with that input whose
                // declared dims accept them, unless ENTRY:NAME sets it for this one.
                None => arg
                    .split_once('=')
                    .is_some_and(|(name, value)| {
                        entry.inputs.iter().any(|i| i.name == name && shape_fits(i, value))
                            && !args.inputs.iter().any(|a| a.starts_with(&format!("{}:{name}=", entry.name)))
                    })
                    .then(|| arg.clone()),
            })
            .collect();
        let shapes = resolve_input_shapes(&model, Some(&entry.name), &args_for_entry).with_context(|| {
            format!(
                "entry {:?}: the graph is needed to see which weights feed matmuls (pass -i {}NAME=DIMS, or --no-graph)",
                entry.name,
                if model.entries().len() > 1 { format!("{}:", entry.name) } else { String::new() }
            )
        })?;
        let given: Vec<(&str, Vec<usize>)> = shapes.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
        let graph = model.graph_entry(&entry.name, &given, &Taps::None).with_context(|| format!("building entry {:?}", entry.name))?;
        for (weight, u) in weight_uses(&graph, names) {
            uses.entry(weight).or_default().extend(u);
        }
    }
    Ok(Some(uses))
}

/// Why a tensor keeps its type, or None to convert it.
fn skip_reason(
    t: &TensorInfo,
    target: DType,
    args: &QuantizeArgs,
    uses: Option<&HashMap<String, Vec<(String, usize)>>>,
) -> Option<String> {
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
    if let Some(uses) = uses {
        match uses.get(&t.name) {
            None => return Some("not used by the program".into()),
            Some(list) => {
                if let Some((op, j)) = list.iter().find(|(op, j)| !QUANTIZABLE_USES.contains(&(op.as_str(), *j))) {
                    return Some(format!("used by {op} (source {j}); only matmul/get-rows weights are quantized"));
                }
            }
        }
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
    let names: HashSet<&str> = infos.iter().map(|t| t.name.as_str()).collect();
    let uses = analyze(args, &file, &names)?;

    let mut writer = GgufWriter::new();
    writer.copy_metadata(&file);
    let mut report = Vec::new();
    for (i, t) in infos.iter().enumerate() {
        let reason = skip_reason(t, target, args, uses.as_ref());
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
