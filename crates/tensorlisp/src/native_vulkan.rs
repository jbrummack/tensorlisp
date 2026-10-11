//! The native Vulkan device: tensorlisp's own executor instead of a ggml
//! backend (the Vulkan counterpart of `native_cuda.rs` / `native_metal.rs`).
//!
//! The Scheme program still builds a ggml graph (shape inference and the op
//! vocabulary only). It is converted to `tensorlisp_kernels`' IR, planned into
//! one arena and lowered to launches of ggml's precompiled Vulkan shaders, which
//! the scheduler's `Engine` runs. Weights are `Space::Model` buffers, the
//! `(define-state ...)` tensors one `Space::Session` block, activations the
//! per-submission arena.

use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{Arc, Mutex},
};

use ggml_sys::ffi::*;
use tensorlisp_kernels::{
    plan::{Leaves, align},
    sched::{Backend, MemKind, ModelMem, Session},
    vulkan::{
        Vulkan,
        exec::{Executor, Program},
        tree::TreeKernels,
    },
};

use crate::{
    error::{Error, Result},
    gguf::GgufFile,
    native_ir::{Lowered, Results, backend_err, lower_graph, read_results},
};

/// Weight buffers hold up to this many bytes (a tensor larger than that gets a buffer of its own).
const WEIGHT_CHUNK: u64 = 1 << 30;

/// Weights, states and the executor of one model.
pub(crate) struct Native {
    exec: Executor,
    model: Arc<ModelMem<Vulkan>>,
    /// ggml tensor pointer (of the file's tensor context) -> index into `weight_loc`.
    weight_index: HashMap<usize, usize>,
    /// Per weight: (buffer, byte offset in it).
    weight_loc: Vec<(usize, u64)>,
    /// One zeroed block for every state tensor, replaced by a fresh one on reset.
    session: Mutex<Option<Arc<Session<Vulkan>>>>,
    state_bytes: u64,
    state_index: HashMap<usize, usize>,
    state_offsets: Vec<u64>,
}

impl Native {
    /// Opens the Vulkan device and loads every weight tensor of `file` from `path`.
    pub fn load(file: &GgufFile, path: &Path) -> Result<Native> {
        let vk = Vulkan::system_default().map_err(backend_err)?;
        let mut exec = Executor::new(vk).map_err(backend_err)?;
        // TL_NATIVE_TREE=1: nodes with a Tree generator run on kernels compiled by the Scheme
        // compiler (docs/design/tree.md), the rest on ggml's shaders; =only fails on the rest.
        if let Ok(mode) = std::env::var("TL_NATIVE_TREE") {
            if mode != "0" {
                exec.set_tree(TreeKernels::new(mode == "only", |expr| {
                    crate::scheme::tree_compile(expr).map_err(|e| tensorlisp_kernels::Error::Invalid(e.to_string()))
                }));
            }
        }
        let backend = exec.backend().clone();

        // Lay the tensors out in chunks.
        let mut weight_index = HashMap::new();
        let mut weight_loc = Vec::new();
        let mut tensors = Vec::new(); // (file index, bytes, chunk, offset)
        let mut chunk_sizes: Vec<u64> = vec![0];
        for i in 0..file.tensor_count() {
            let name = file.tensor_name(i);
            let t = file.tensor(name)?.ok_or_else(|| Error::Gguf(format!("tensor {name} missing from context")))?;
            let bytes = unsafe { ggml_nbytes(t) } as u64;
            let padded = align(bytes.max(1));
            let last = chunk_sizes.len() - 1;
            if chunk_sizes[last] > 0 && chunk_sizes[last] + padded > WEIGHT_CHUNK {
                chunk_sizes.push(0);
            }
            let chunk = chunk_sizes.len() - 1;
            let off = chunk_sizes[chunk];
            chunk_sizes[chunk] += padded;
            weight_index.insert(t as usize, weight_loc.len());
            weight_loc.push((chunk, off));
            tensors.push((i, bytes, chunk, off));
        }

        let buffers = chunk_sizes
            .iter()
            .map(|&size| backend.alloc(size.max(4), MemKind::Device).map_err(backend_err))
            .collect::<Result<Vec<_>>>()?;
        let data = File::open(path)?;
        let mut staging = Vec::new();
        for (i, bytes, chunk, off) in tensors {
            staging.resize(bytes as usize, 0u8);
            crate::gguf::read_exact_at(&data, &mut staging, file.tensor_file_offset(i) as u64)?;
            backend.write(&buffers[chunk], off, &staging).map_err(backend_err)?;
        }

        Ok(Native {
            exec,
            model: Arc::new(ModelMem { buffers }),
            weight_index,
            weight_loc,
            session: Mutex::new(None),
            state_bytes: 0,
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
        self.state_bytes = total;
        self.reset_state();
        Ok(())
    }

    pub fn reset_state(&self) {
        if self.state_bytes == 0 {
            return;
        }
        // A fresh session block is zeroed by the pool; the old one goes back to it.
        if let Ok(s) = self.exec.engine.new_session(&[self.state_bytes]) {
            *self.session.lock().unwrap() = Some(Arc::new(s));
        }
    }

    pub fn device_name(&self) -> String {
        format!("native {}", self.exec.backend().name())
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
        let session = native.session.lock().unwrap().clone();
        self.program.run(&native.model, session.as_ref()).map_err(backend_err)
    }

    pub fn outputs(&self) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        let t0 = std::time::Instant::now();
        let r = read_results(&self.outputs, |id, n| self.program.read(id, n));
        if std::env::var_os("TL_NATIVE_TIME").is_some() {
            eprintln!("outputs: read back in {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        r
    }

    pub fn taps(&self) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        read_results(&self.taps, |id, n| self.program.read(id, n))
    }
}
