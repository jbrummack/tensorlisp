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
#[cfg(native_device)]
mod native_ir;
#[cfg(target_os = "macos")]
#[path = "native_metal.rs"]
mod native;
#[cfg(all(feature = "native-cuda", not(target_os = "macos")))]
#[path = "native_cuda.rs"]
mod native;
pub mod log;
pub mod model;
pub mod program;
mod scheme;

pub use dtype::DType;
pub use error::{Error, Result};
pub use model::{
    AdapterFormat, Device, GraphInfo, Inference, LoadOptions, Model, NodeInfo, RawInput, RunOptions, RunOutput, StepStats, TrainOptions, Trainer, Value,
};
pub use program::Program;
pub use scheme::{EntrySpec, InputSpec, PipelineSpec, RawKind, RawSpec, StateSpec, Taps};

/// Evaluates `program` without weights and returns its entries (default
/// first) and pipelines; fails on syntax errors or a missing `(model ...)`.
pub fn program_entries(program: &Program, assets: Vec<(String, Vec<u8>)>) -> Result<(Vec<EntrySpec>, Vec<PipelineSpec>)> {
    let Program::Text(text) = program;
    let loaded = scheme::LoadedProgram::load(text, assets)?;
    Ok((loaded.entries.clone(), loaded.pipelines.clone()))
}

/// The simplest pass of tensorlisp's nanopass AOT compiler: rewrites
/// `(tl generic)`-vocabulary source (see `src/scheme/stdlib/generic.ss`)
/// into ggml-targeted source, a pure syntax-level rename with no semantic
/// transformation -- see `$tl-compile-generic-to-ggml` in `scheme/core.ss`
/// for what it actually rewrites and why. The result is ordinary tensorlisp
/// source: load and run it exactly like any other `Program::Text`.
pub fn compile_generic_to_ggml(src: &str) -> Result<String> {
    scheme::compile_generic_to_ggml(src)
}

/// A second nanopass target, CoreML MIL, via a real symbolic trace rather
/// than a rename: see `$tl-compile-generic-to-mil` in `scheme/core.ss` for
/// the shadow-op mechanism (each covered primitive both computes its real
/// reference value, so the model's own shape-dependent control flow stays
/// correct, and emits an equivalent MIL op node addressed by %name). The
/// result still needs to be *run* (real weights, real input) for the trace
/// to populate, and the compiled model body itself must call
/// `(%mil-render! ...)` somewhere (its tensors are only live while the
/// graph is being built); [`mil_last_render`] fetches the text afterwards.
pub fn compile_generic_to_mil(src: &str) -> Result<String> {
    scheme::compile_generic_to_mil(src)
}

/// The MIL program text a compiled-to-mil model's own `(%mil-render! ...)`
/// call last stored while its graph was built (during the most recent
/// `Model::run`/`Model::graph` on this process's Scheme thread). Call right
/// after that run.
pub fn mil_last_render() -> Result<String> {
    scheme::mil_last_render()
}

pub use autopro;
