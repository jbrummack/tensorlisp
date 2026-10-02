//! The native CUDA device: tensorlisp's own executor instead of a ggml
//! backend (the CUDA counterpart of `native_metal.rs`).
//!
//! The Scheme program still builds a ggml graph (graph IR and shape
//! inference only); the graph is lowered to `tensorlisp_kernels`' IR, planned
//! into one arena and compiled to a recorded list of launches of our own
//! NVRTC-compiled kernels, replayed per run. Matrix multiplication is chosen
//! entirely on the Rust side (`cuda/lower.rs`).

use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

use ggml_sys::ffi::*;
use tensorlisp_kernels::{
    cuda::{
        Buffer, Device as KDevice,
        exec::{Executor, Program, Shared},
    },
    plan::{Leaves, align},
};

use crate::{
    error::{Error, Result},
    gguf::GgufFile,
    native_ir::{backend_err, lower_graph, read_results, Lowered, Results},
};

/// Weights, states and the compiled executor of one model.
pub(crate) struct Native {
    pub exec: Executor,
    weights: Vec<Buffer>,
    /// ggml tensor pointer (of the file's tensor context) -> index into `weight_loc`.
    weight_index: HashMap<usize, usize>,
    weight_loc: Vec<(usize, u64)>,
    pub state: Buffer,
    state_index: HashMap<usize, usize>,
    state_offsets: Vec<u64>,
}

impl Native {
    /// Creates the CUDA device and loads every weight tensor of `file` from `path`.
    pub fn load(file: &GgufFile, path: &Path) -> Result<Native> {
        let dev = Arc::new(KDevice::system_default().map_err(backend_err)?);
        let exec = Executor::new(dev.clone()).map_err(backend_err)?;

        // All weights share one buffer, each tensor at an aligned offset.
        let mut weight_index = HashMap::new();
        let mut weight_loc = Vec::new();
        let mut total = 0u64;
        let mut tensors = Vec::new();
        for i in 0..file.tensor_count() {
            let name = file.tensor_name(i);
            let t = file.tensor(name)?.ok_or_else(|| Error::Gguf(format!("tensor {name} missing from context")))?;
            let bytes = unsafe { ggml_nbytes(t) } as u64;
            weight_index.insert(t as usize, weight_loc.len());
            weight_loc.push((0usize, total));
            tensors.push((i, bytes));
            total += align(bytes.max(1));
        }
        let weights = dev.alloc(total.max(1) as usize).map_err(backend_err)?;

        let data = File::open(path)?;
        let mut staging = Vec::new();
        for ((i, bytes), &(_, off)) in tensors.iter().zip(&weight_loc) {
            staging.resize(*bytes as usize, 0u8);
            crate::gguf::read_exact_at(&data, &mut staging, file.tensor_file_offset(*i) as u64)?;
            weights.write_at(off as usize, &staging);
        }

        let state = dev.alloc(1).map_err(backend_err)?;
        Ok(Native {
            exec,
            weights: vec![weights],
            weight_index,
            weight_loc,
            state,
            state_index: HashMap::new(),
            state_offsets: Vec::new(),
        })
    }

    /// Allocates (zeroed) storage for the `(define-state ...)` tensors of `ctx`.
    pub fn alloc_states(&mut self, ctx: *mut ggml_context) -> Result<()> {
        let mut total = 0u64;
        let mut t = unsafe { ggml_get_first_tensor(ctx) };
        while !t.is_null() {
            self.state_index.insert(t as usize, self.state_offsets.len());
            self.state_offsets.push(total);
            total += align(unsafe { ggml_nbytes(t) } as u64);
            t = unsafe { ggml_get_next_tensor(ctx, t) };
        }
        let buf = self.exec.dev.alloc(total.max(1) as usize).map_err(backend_err)?;
        buf.zero();
        self.state = buf;
        Ok(())
    }

    pub fn reset_state(&self) {
        self.state.zero();
    }

    pub fn device_name(&self) -> String {
        format!("native {}", self.exec.dev.name())
    }
}

/// A compiled graph with what is needed to feed and read it.
pub(crate) struct NativeProgram {
    program: Program,
    outputs: Results,
    taps: Results,
}

impl NativeProgram {
    /// Lowers and compiles a built ggml graph. `inputs`/`outputs`/`taps` are the graph's.
    pub fn compile(
        native: &Native,
        graph: *mut ggml_cgraph,
        inputs: &[*mut ggml_tensor],
        outputs: &[(String, *mut ggml_tensor, usize)],
        taps: &[(String, *mut ggml_tensor, usize)],
    ) -> Result<NativeProgram> {
        let Lowered { ir, outputs, taps } =
            lower_graph(graph, inputs, outputs, taps, &native.weight_index, &native.state_index);
        let leaves = Leaves { weights: &native.weight_loc, states: &native.state_offsets };
        let program = native.exec.compile(ir, &leaves).map_err(backend_err)?;
        Ok(NativeProgram { program, outputs, taps })
    }

    pub fn set_input(&self, i: usize, data: &[u8]) -> Result<()> {
        self.program.set_input(i, data).map_err(backend_err)
    }

    pub fn run(&self, native: &Native) -> Result<()> {
        self.program.run(&Shared { weights: &native.weights, state: &native.state }).map_err(backend_err)
    }

    pub fn outputs(&self) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        read_results(&self.outputs, |id, n| self.program.read(id, n))
    }

    pub fn taps(&self) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        read_results(&self.taps, |id, n| self.program.read(id, n))
    }
}
