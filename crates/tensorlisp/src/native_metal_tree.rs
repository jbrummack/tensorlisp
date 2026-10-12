//! The native Metal device on Tree kernels (feature `native-metal-tree`): the Vulkan executor's
//! lowering (`vulkan::lower`, every node a Tree kernel) with MSL blobs from the Tree compiler, run
//! through the scheduler's `Engine` on the Metal backend. The Scheme program still builds a ggml graph
//! (shape inference and the op vocabulary only), converted to `tensorlisp_kernels`' IR and planned into
//! one arena. Weights are `Space::Model` buffers, the
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
    metal::{backend::Metal, tree::executor},
    sched::{Access, Backend, Code, KernelBlob, MemKind, ModelMem, OffsetMode, Session, Signature, Target},
    vulkan::{
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

/// One session's state tensors, laid out like the default session's.
pub(crate) type StateBlock = Arc<Session<Metal>>;

/// Weights, states and the executor of one model.
pub(crate) struct Native {
    exec: Executor<Metal>,
    model: Arc<ModelMem<Metal>>,
    /// ggml tensor pointer (of the file's tensor context) -> index into `weight_loc`.
    weight_index: HashMap<usize, usize>,
    /// Per weight: (buffer, byte offset in it).
    weight_loc: Vec<(usize, u64)>,
    /// One zeroed block for every state tensor, replaced by a fresh one on reset.
    session: Mutex<Option<Arc<Session<Metal>>>>,
    state_bytes: u64,
    state_index: HashMap<usize, usize>,
    state_offsets: Vec<u64>,
}

impl Native {
    /// Opens the Metal device and loads every weight tensor of `file` from `path`.
    pub fn load(file: &GgufFile, path: &Path) -> Result<Native> {
        let metal = Metal::system_default().map_err(backend_err)?;
        let mut exec = executor(metal).map_err(backend_err)?;
        // Every node runs on a Tree kernel; one with no generator fails to compile by name.
        exec.set_tree(TreeKernels::from_blobs(true, |expr| {
            let reply = crate::scheme::tree_compile_msl(expr).map_err(|e| tensorlisp_kernels::Error::Invalid(e.to_string()))?;
            msl_blob(expr, reply).map_err(tensorlisp_kernels::Error::Invalid)
        }));
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

    /// A zeroed block of state for another session.
    pub fn new_state_block(&self) -> Result<StateBlock> {
        self.exec.engine.new_session(&[self.state_bytes.max(4)]).map(Arc::new).map_err(backend_err)
    }

    /// A copy of a session's state.
    pub fn fork_block(&self, block: &StateBlock) -> Result<StateBlock> {
        self.exec.engine.fork_session(block).map(Arc::new).map_err(backend_err)
    }

    pub fn reset_block(&self, block: &mut StateBlock) -> Result<()> {
        *block = self.new_state_block()?;
        Ok(())
    }

    pub fn device_name(&self) -> String {
        format!("native tree {}", self.exec.backend().name())
    }
}

/// A compiled graph with what is needed to feed and read it.
pub(crate) struct NativeProgram {
    program: Program<Metal>,
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
        if std::env::var_os("TL_NATIVE_TIME").is_some() {
            eprintln!("kernel cache: {:?}", crate::kernel_cache::kernel_cache_stats());
        }
        Ok(NativeProgram { program, outputs, taps })
    }

    pub fn set_input(&self, i: usize, data: &[u8]) -> Result<()> {
        self.program.set_input(i, data).map_err(backend_err)
    }

    pub fn run(&self, native: &Native, state: Option<&StateBlock>) -> Result<()> {
        let session = state.cloned().or_else(|| native.session.lock().unwrap().clone());
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

/// The kernel blob for what `compile-kernel-msl` returned: `(msl entry accesses push-bytes workgroup)`.
fn msl_blob(label: &str, reply: chez::Value) -> std::result::Result<KernelBlob, String> {
    use chez::Value;
    let Value::List(items) = reply else { return Err(format!("{label}: unexpected reply {reply:?}")) };
    let [Value::String(text), Value::String(name), accesses, Value::Int(push), wg] = &items[..] else {
        return Err(format!("{label}: unexpected reply {items:?}"));
    };
    let bindings = match accesses {
        Value::List(a) => a
            .iter()
            .map(|v| match v {
                Value::Symbol(s) if s == "ro" => Access::Read,
                Value::Symbol(s) if s == "wo" => Access::Write,
                _ => Access::ReadWrite,
            })
            .collect(),
        _ => vec![],
    };
    let workgroup = [0, 1, 2].map(|i| match wg {
        Value::List(w) => match w.get(i) {
            Some(Value::Int(n)) => *n as u32,
            _ => 1,
        },
        _ => 1,
    });
    Ok(KernelBlob {
        target: Target::Metal,
        code: Code::Msl(text.clone()),
        entry: name.clone(),
        label: format!("msl:{label}"),
        sig: Signature { bindings, push_bytes: *push as u32, workgroup, offsets: OffsetMode::Prefix },
        spec: vec![],
    })
}
