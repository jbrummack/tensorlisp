//! Kernel launcher: compile native GPU kernels, bind buffers/scalars, dispatch.
//!
//! This is the layer *below* an op graph. It knows nothing about tensors or
//! shapes, only about libraries (source text), pipelines (entry point +
//! specialization constants), buffers and dispatches, so it can be driven from
//! Scheme (`(launch kernel grid threads args..)`) as well as from typed Rust
//! wrappers like [`metal::paged_attn`].
//!
//! Backends mirror each other: `metal` today, `cuda` per docs/design/kernel-launcher.md.

pub mod graph;
pub mod plan;
#[cfg(target_os = "macos")]
pub mod metal;

/// Element types the vendored kernels are instantiated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// fp8 e4m3, stored as bytes; only valid for KV caches (needs scales).
    F8E4M3,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 => 1,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no GPU device available")]
    NoDevice,
    #[error("shader compilation failed ({lib}): {msg}")]
    Compile { lib: String, msg: String },
    #[error("kernel `{entry}` not found / pipeline creation failed: {msg}")]
    Pipeline { entry: String, msg: String },
    #[error("buffer allocation of {0} bytes failed")]
    Alloc(usize),
    #[error("command buffer failed: {0}")]
    Execution(String),
    #[error("{0}")]
    Invalid(String),
    /// A graph node with an op (or op configuration) the backend can't lower yet.
    #[error("native Metal: {op} `{node}`: {why}")]
    Unsupported { op: String, node: String, why: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// A dispatch: threadgroups (CUDA: blocks) x threads per threadgroup, plus
/// dynamic threadgroup/shared memory in bytes (bound at slot 0 on Metal).
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub groups: [u32; 3],
    pub threads: [u32; 3],
    pub shared: u32,
}

impl Dims {
    pub fn new(groups: [u32; 3], threads: [u32; 3]) -> Self {
        Dims { groups, threads, shared: 0 }
    }
    pub fn shared(mut self, bytes: u32) -> Self {
        self.shared = bytes;
        self
    }
}

/// A compile-time specialization constant (Metal function constant; CUDA: a
/// template/`-D` value baked into the NVRTC name expression).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Const {
    Bool(bool),
    I16(i16),
    I32(i32),
}
