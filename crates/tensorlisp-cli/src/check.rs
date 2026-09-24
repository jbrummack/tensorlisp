use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::json;
use tensorlisp::{Device, InputSpec, Taps};

use crate::{
    ModelArgs,
    common::{DeviceArg, load_model, parse_shape, print_json, shape_string, split_assignment},
    npy,
};

#[derive(Args)]
pub struct CheckArgs {
    /// GGUF file with the weights (and usually the program).
    pub model: PathBuf,
    /// Check this program file instead of the one stored in the model.
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Input shape in numpy order, NAME=1,3,224,224, or NAME=file.npy to use
    /// the file's shape (repeatable). Inputs declared with fixed shapes can be omitted.
    #[arg(short, long = "input")]
    pub inputs: Vec<String>,
    /// Only show inputs, outputs and taps, not every node.
    #[arg(long)]
    pub summary: bool,
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

pub fn run(args: &CheckArgs, json: bool) -> Result<i32> {
    // Weights only need to be readable, so keep them on the CPU.
    let model = load_model(&ModelArgs {
        model: args.model.clone(),
        program: args.program.clone(),
        device: DeviceArg(Device::Cpu),
    })?;

    let mut shapes: Vec<(String, Vec<usize>)> = args
        .inputs
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

    let given: Vec<(&str, Vec<usize>)> = shapes.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
    let graph = model.graph(&given, &Taps::None).context("building the graph")?;

    if json {
        print_json(&json!({
            "inputs": shapes.iter().map(|(n, s)| json!({ "name": n, "shape": s })).collect::<Vec<_>>(),
            "nodes": graph.nodes.iter().enumerate().map(|(i, n)| json!({
                "index": i, "op": n.op, "name": n.name, "type": n.dtype.name(), "ne": n.ne, "srcs": n.srcs,
            })).collect::<Vec<_>>(),
            "outputs": graph.outputs.iter().map(|(n, s)| json!({ "name": n, "shape": s })).collect::<Vec<_>>(),
            "taps": graph.taps,
        }))?;
        return Ok(0);
    }

    println!("inputs (numpy order):");
    for (name, shape) in &shapes {
        println!("  {name}  {}", shape_string(shape));
    }
    if !args.summary {
        println!("\nnodes ({}, shapes in ggml order: innermost first):", graph.nodes.len());
        let op_width = graph.nodes.iter().map(|n| n.op.len()).max().unwrap_or(0);
        let index_width = graph.nodes.len().to_string().len() + 1;
        for (i, node) in graph.nodes.iter().enumerate() {
            let ne: Vec<usize> = node.ne.iter().map(|&d| d as usize).collect();
            let name = if node.name.is_empty() { String::new() } else { format!("  \"{}\"", node.name) };
            println!(
                "  {:>index_width$}  {:op_width$}  {:5} {:18} <- {}{name}",
                format!("%{i}"),
                node.op,
                node.dtype.name(),
                shape_string(&ne),
                node.srcs.join(", ")
            );
        }
    }
    println!("\noutputs (numpy order):");
    for (name, shape) in &graph.outputs {
        println!("  {name}  {}", shape_string(shape));
    }
    if !graph.taps.is_empty() {
        println!("\ntaps: {}", graph.taps.join(", "));
    }
    Ok(0)
}
