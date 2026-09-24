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
pub mod log;
pub mod model;
pub mod program;
mod scheme;

pub use dtype::DType;
pub use error::{Error, Result};
pub use model::{Device, GraphInfo, Model, NodeInfo, RunOptions, RunOutput};
pub use program::Program;
pub use scheme::{InputSpec, Taps};

/// Evaluates `program` without weights and returns its declared inputs;
/// fails on syntax errors or a missing `(model ...)`.
pub fn program_inputs(program: &Program) -> Result<Vec<InputSpec>> {
    let Program::Text(text) = program;
    Ok(scheme::LoadedProgram::load(text)?.inputs.clone())
}
