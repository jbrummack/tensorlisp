//! tensorlisp: network definitions as Scheme programs stored in GGUF,
//! executed by building a ggml cgraph.
//!
//! ```no_run
//! # use tensorlisp::{Device, Model};
//! let model = Model::load("mlp.gguf", Device::Auto)?;
//! let x = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 784]));
//! let out = model.run(&[("x", x.view())])?;
//! println!("{:?}", out["logits"].shape());
//! # Ok::<(), tensorlisp::Error>(())
//! ```

pub mod dtype;
pub mod error;
pub mod gguf;
mod guard;
mod host;
pub mod log;
pub mod model;
pub mod program;
mod scheme;

pub use dtype::DType;
pub use error::{Error, Result};
pub use model::{Device, GraphInfo, Inference, LoadOptions, Model, NodeInfo, RawInput, RunOptions, RunOutput, Value};
pub use program::Program;
pub use scheme::{EntrySpec, InputSpec, PipelineSpec, RawKind, RawSpec, StateSpec, Taps};

/// Evaluates `program` without weights and returns its entries (default
/// first) and pipelines; fails on syntax errors or a missing `(model ...)`.
pub fn program_entries(program: &Program, assets: Vec<(String, Vec<u8>)>) -> Result<(Vec<EntrySpec>, Vec<PipelineSpec>)> {
    let Program::Text(text) = program;
    let loaded = scheme::LoadedProgram::load(text, assets)?;
    Ok((loaded.entries.clone(), loaded.pipelines.clone()))
}

pub use autopro;
