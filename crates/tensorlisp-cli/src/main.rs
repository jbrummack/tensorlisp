//! `tl`: tooling for porting networks to tensorlisp.
mod check;
mod common;
mod convert;
mod inspect;
mod npy;
mod quantize;
mod run;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use tensorlisp::log::{LogLevel, set_ggml_log_level};

#[derive(Parser)]
#[command(name = "tl", version, about = "Port, inspect, run and quantize tensorlisp models (GGUF + Scheme graph program)")]
struct Cli {
    /// Print machine-readable JSON on stdout (errors too, as {"error": ...}).
    #[arg(long, global = true)]
    json: bool,
    /// Show ggml's log output (-v: info, -vv: debug). Default: errors only.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show a GGUF file's metadata, tensors and tensorlisp program.
    Inspect(inspect::InspectArgs),
    /// Convert safetensors weights into a GGUF file, optionally with a program.
    Convert(convert::ConvertArgs),
    /// Store a program in an existing GGUF file (replacing any previous one).
    Pack(convert::PackArgs),
    /// Build the graph for given input shapes without running it, and list its nodes.
    Check(check::CheckArgs),
    /// Run a model on .npy inputs, or on raw inputs through its preprocess.
    Run(run::RunArgs),
    /// Run only the program's preprocess on raw inputs and show or save the arrays.
    Process(run::ProcessArgs),
    /// Run a model and compare outputs and taps with reference .npy files.
    Compare(run::CompareArgs),
    /// Quantize the weights of a GGUF file.
    Quantize(quantize::QuantizeArgs),
}

/// Options shared by commands that load a model.
#[derive(Args, Clone)]
pub struct ModelArgs {
    /// GGUF file with the weights (and usually the program).
    pub model: PathBuf,
    /// Run this program file instead of the one stored in the model.
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Where to run: auto (GPU if available), cpu or gpu.
    #[arg(long, default_value = "auto")]
    pub device: common::DeviceArg,
    /// Add or replace an asset the program reads with (asset NAME), NAME=PATH (repeatable).
    #[arg(long = "asset")]
    pub assets: Vec<String>,
}

fn main() {
    // Exit quietly when stdout is closed early (e.g. `tl inspect ... | head`).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    set_ggml_log_level(match cli.verbose {
        0 => LogLevel::Error,
        1 => LogLevel::Info,
        _ => LogLevel::Debug,
    });
    let result = match &cli.command {
        Command::Inspect(args) => inspect::run(args, cli.json),
        Command::Convert(args) => convert::run(args, cli.json),
        Command::Pack(args) => convert::pack(args, cli.json),
        Command::Check(args) => check::run(args, cli.json),
        Command::Run(args) => run::run(args, cli.json),
        Command::Process(args) => run::process(args, cli.json),
        Command::Compare(args) => run::compare(args, cli.json),
        Command::Quantize(args) => quantize::run(args, cli.json),
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            if cli.json {
                println!("{}", serde_json::json!({ "error": format!("{e:#}") }));
            } else {
                eprintln!("error: {e:#}");
            }
            std::process::exit(1);
        }
    }
}
