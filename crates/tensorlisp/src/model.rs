use std::{
    collections::HashMap,
    ffi::CStr,
    fs::File,
    mem::ManuallyDrop,
    ops::Deref,
    path::Path,
    ptr::null_mut,
    sync::{Arc, Mutex, Weak},
};

use ggml_sys::ffi::*;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::{
    error::{Error, Result},
    gguf::{GgufFile, ne_from_shape},
    guard,
    program::Program,
    dtype::DType,
    scheme::{EntrySpec, InputSpec, LoadedProgram, PipelineSpec, PostValue, RawKind, RawSpec, RawValue, ResultValue, StateSpec, Taps},
};

/// Options for [`Model::load_with`].
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Run this program instead of the file's (the file then needn't have one).
    pub program: Option<Program>,
    /// Assets in addition to (or replacing, by name) the file's `TL_ASSET.*`.
    pub assets: Vec<(String, Vec<u8>)>,
}

/// A raw input for a program's `(preprocess ...)`.
pub enum RawInput {
    Text(String),
    Image(image::DynamicImage),
    Audio(autopro::audio::Audio),
    Array(ArrayD<f32>),
}

impl RawInput {
    fn kind(&self) -> RawKind {
        match self {
            RawInput::Text(_) => RawKind::Text,
            RawInput::Image(_) => RawKind::Image,
            RawInput::Audio(_) => RawKind::Audio,
            RawInput::Array(_) => RawKind::Array,
        }
    }
}

/// Maximum number of nodes in one graph.
const GRAPH_SIZE: usize = 16384;

/// Where a model runs. Ops the chosen device can't run fall back to the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// The first GPU if there is one, otherwise the CPU.
    #[default]
    Auto,
    Cpu,
    Gpu,
}

/// A tensorlisp model: weights on the device plus the program that builds its graphs.
///
/// A program has one or more entries (`(model name (inputs ...) ...)`), all
/// sharing the weights. Each entry's graph for its current input shapes stays
/// allocated and is recomputed directly while the shapes stay the same; a
/// shape change rebuilds it (each entry has its own scheduler, since
/// resetting one invalidates every graph it allocated).
///
/// ggml assertions become [`Error::Ggml`] where they can be caught: while the
/// graph is built, and on the calling thread during allocation and compute.
/// An assertion during compute leaves the backend's threads in an unknown
/// state, so the model then refuses to run and leaks its ggml objects instead
/// of freeing memory those threads may still use. Load it again to recover.
/// Assertions on backend worker threads still abort the process.
pub struct Model {
    shared: Arc<Shared>,
}

pub struct Shared {
    /// Default entry first.
    entries: Vec<EntrySpec>,
    pipelines: Vec<PipelineSpec>,
    program_id: i64,
    inner: Mutex<Inner>,
}

impl Deref for Model {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.shared
    }
}

/// Loaded models by program id, for pipelines that run entries from Scheme.
static MODELS: Mutex<Vec<(i64, Weak<Shared>)>> = Mutex::new(Vec::new());

impl Drop for Shared {
    fn drop(&mut self) {
        MODELS.lock().unwrap().retain(|(id, _)| *id != self.program_id);
    }
}

/// Runs an entry of the model whose program has id `program` (called by
/// `(run ...)` in pipelines). A missing leading batch dimension of 1 is added.
pub(crate) fn run_program_entry(
    program: i64,
    entry: &str,
    inputs: Vec<(String, ArrayD<f32>)>,
) -> Result<Vec<(String, ArrayD<f32>)>> {
    let shared = MODELS
        .lock()
        .unwrap()
        .iter()
        .find(|(id, _)| *id == program)
        .and_then(|(_, weak)| weak.upgrade())
        .ok_or_else(|| Error::Program("the model of this pipeline is gone".into()))?;
    let model = Model { shared };
    let spec = model.entry(Some(entry))?;
    let inputs: Vec<(String, ArrayD<f32>)> = inputs
        .into_iter()
        .map(|(name, array)| {
            let declared = spec.inputs.iter().find(|s| s.name == name).and_then(|s| s.dims.as_ref()).map(|d| d.len());
            if declared == Some(array.ndim() + 1) {
                (name, array.insert_axis(ndarray::Axis(0)))
            } else {
                (name, array)
            }
        })
        .collect();
    let views: Vec<(&str, ArrayViewD<f32>)> = inputs.iter().map(|(n, a)| (n.as_str(), a.view())).collect();
    Ok(model.run_entry(entry, &views, &RunOptions::default())?.outputs)
}

/// Options for [`Model::run_with`].
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Intermediate tensors marked with `(tap name t)` to read back.
    pub taps: Taps,
}

/// Results of a postprocess or pipeline: arrays, or text (e.g. generated).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Array(ArrayD<f32>),
    Text(String),
}

impl Value {
    pub fn as_array(&self) -> Option<&ArrayD<f32>> {
        match self {
            Value::Array(a) => Some(a),
            Value::Text(_) => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(t) => Some(t),
            Value::Array(_) => None,
        }
    }
}

impl From<ResultValue> for Value {
    fn from(v: ResultValue) -> Value {
        match v {
            ResultValue::Array(a) => Value::Array(a),
            ResultValue::Text(t) => Value::Text(t),
        }
    }
}

/// Named results in the order the program declares them.
#[derive(Debug, Clone, Default)]
pub struct RunOutput {
    pub outputs: Vec<(String, ArrayD<f32>)>,
    pub taps: Vec<(String, ArrayD<f32>)>,
}

/// Result of [`Model::infer`]: the postprocessed results of each example,
/// plus what the model computed for the whole batch.
#[derive(Debug, Clone, Default)]
pub struct Inference {
    /// Per example, the `(results ...)` of the program's postprocess (in the
    /// order it lists them), or the example's slice of every output without one.
    pub examples: Vec<Vec<(String, Value)>>,
    pub run: RunOutput,
}

/// The graph a program builds for given input shapes, without computing it.
#[derive(Debug, Clone)]
pub struct GraphInfo {
    pub nodes: Vec<NodeInfo>,
    /// Name and ndarray-order shape of each output.
    pub outputs: Vec<(String, Vec<usize>)>,
    /// Every tap the program defines.
    pub taps: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// ggml op, e.g. "MUL_MAT" or "UNARY(RELU)".
    pub op: String,
    /// Set for inputs and taps; empty for nodes ggml named itself ("node_N").
    pub name: String,
    pub dtype: DType,
    /// ggml order (innermost first).
    pub ne: Vec<i64>,
    /// Byte strides, ggml order (innermost first), matching `ne`. Needed
    /// (with `view_offs`) to tell a plain reshape/broadcast apart from a
    /// strided slice when lowering a `VIEW`-family op to another backend.
    pub nb: Vec<usize>,
    /// Byte offset from `view_src`'s base, for `VIEW`/`RESHAPE`-family ops
    /// that alias another tensor's storage instead of computing a new one
    /// (0 for ops that compute their own contiguous output).
    pub view_offs: usize,
    /// Raw `ggml_tensor.op_params`, as the 16 `i32` slots ggml stores them
    /// in (`GGML_MAX_OP_PARAMS / sizeof(i32)`). Meaning is op-specific and
    /// matches ggml's own `ggml_set_op_params_i32`/`_f32` call sites for
    /// that op (e.g. `CONV_2D`: `[s0, s1, p0, p1, d0, d1]`; `ARANGE`: three
    /// `f32` bit patterns `[start, stop, step]`) -- see `vendor/ggml/src/ggml.c`.
    pub op_params: [i32; 16],
    /// Sources: "%3" for node 3, otherwise the tensor's name (weights, inputs).
    pub srcs: Vec<String>,
}

type GraphKey = (Vec<Vec<usize>>, Taps);

struct Graph {
    ctx: *mut ggml_context,
    graph: *mut ggml_cgraph,
    inputs: Vec<*mut ggml_tensor>,
    outputs: Vec<(String, *mut ggml_tensor, usize)>,
    taps: Vec<(String, *mut ggml_tensor, usize)>,
    tap_names: Vec<String>,
}

/// An entry's scheduler and allocated graph.
struct EntryGraph {
    sched: ggml_backend_sched_t,
    /// The allocated graph and what it was built for.
    graph: Option<(GraphKey, Graph)>,
}

struct Inner {
    program: LoadedProgram,
    /// Holds the weight tensors' metadata; ManuallyDrop so a poisoned model can leak it.
    file: ManuallyDrop<GgufFile>,
    /// Primary device first, CPU last.
    backends: Vec<ggml_backend_t>,
    weights: ggml_backend_buffer_t,
    /// `(define-state ...)` tensors (null without any) and their buffer on the primary device.
    states: *mut ggml_context,
    state_buffer: ggml_backend_buffer_t,
    /// Per entry, created on first use.
    graphs: HashMap<String, EntryGraph>,
    /// Set when an assertion fired during compute.
    poisoned: Option<String>,
}

// ggml objects aren't tied to a thread; the Mutex serializes all use.
unsafe impl Send for Inner {}

fn init_gpu() -> Option<ggml_backend_t> {
    [ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU, ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_IGPU]
        .into_iter()
        .map(|ty| unsafe { ggml_backend_dev_by_type(ty) })
        .find(|dev| !dev.is_null())
        .map(|dev| unsafe { ggml_backend_dev_init(dev, std::ptr::null()) })
        .filter(|backend| !backend.is_null())
}

fn init_backends(device: Device) -> Result<Vec<ggml_backend_t>> {
    let gpu = match device {
        Device::Cpu => None,
        Device::Auto => init_gpu(),
        Device::Gpu => Some(init_gpu().ok_or_else(|| Error::Backend("no GPU device available".into()))?),
    };
    let cpu = unsafe { ggml_backend_cpu_init() };
    if cpu.is_null() {
        return Err(Error::Backend("failed to initialize the CPU backend".into()));
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    unsafe { ggml_backend_cpu_set_n_threads(cpu, threads as i32) };
    Ok(gpu.into_iter().chain([cpu]).collect())
}

/// Copies every tensor's data from the file into the allocated weight tensors.
fn load_weights(file: &GgufFile, path: &Path) -> Result<()> {
    let data = File::open(path)?;
    let mut buf = Vec::new();
    for i in 0..file.tensor_count() {
        let name = file.tensor_name(i);
        let t = file.tensor(name)?.ok_or_else(|| Error::Gguf(format!("tensor {name} missing from context")))?;
        buf.resize(unsafe { ggml_nbytes(t) }, 0);
        crate::gguf::read_exact_at(&data, &mut buf, file.tensor_file_offset(i) as u64)?;
        guard::tensor_set(t, &buf)?;
    }
    Ok(())
}

impl Model {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Model> {
        Self::load_with(path, device, LoadOptions::default())
    }

    pub fn load_with(path: impl AsRef<Path>, device: Device, options: LoadOptions) -> Result<Model> {
        guard::install();
        let path = path.as_ref();
        let file = GgufFile::open(path)?;
        let Program::Text(text) = match options.program {
            Some(program) => program,
            None => file.program()?,
        };
        let mut assets = file.assets();
        for (name, bytes) in options.assets {
            assets.retain(|(n, _)| *n != name);
            assets.push((name, bytes));
        }
        let program = LoadedProgram::load(&text, assets)?;
        let backends = init_backends(device)?;

        // From here on, Inner's Drop frees what was created so far.
        let mut inner = Inner {
            program,
            file: ManuallyDrop::new(file),
            backends,
            weights: null_mut(),
            states: null_mut(),
            state_buffer: null_mut(),
            graphs: HashMap::new(),
            poisoned: None,
        };
        inner.weights = guard::alloc_ctx_tensors(inner.file.tensors, inner.backends[0])?;
        if inner.weights.is_null() && inner.file.tensor_count() > 0 {
            return Err(Error::Backend("failed to allocate the weight buffer".into()));
        }
        load_weights(&inner.file, path)?;
        inner.alloc_states()?;

        let shared = Arc::new(Shared {
            entries: inner.program.entries.clone(),
            pipelines: inner.program.pipelines.clone(),
            program_id: inner.program.id(),
            inner: Mutex::new(inner),
        });
        MODELS.lock().unwrap().push((shared.program_id, Arc::downgrade(&shared)));
        Ok(Model { shared })
    }

    /// Inputs of the default entry (`main`, or the first one defined), in declaration order.
    pub fn inputs(&self) -> &[InputSpec] {
        &self.entries[0].inputs
    }

    /// The program's entries, the default first.
    pub fn entries(&self) -> &[EntrySpec] {
        &self.entries
    }

    /// The program's pipelines (host code that runs entries, e.g. generation).
    pub fn pipelines(&self) -> &[PipelineSpec] {
        &self.pipelines
    }

    /// Zeroes every `(define-state ...)` tensor, as after loading.
    pub fn reset_state(&self) {
        let inner = self.inner.lock().unwrap();
        if !inner.state_buffer.is_null() {
            unsafe { ggml_backend_buffer_clear(inner.state_buffer, 0) };
        }
    }

    /// An entry by name, or the default one.
    pub fn entry(&self, name: Option<&str>) -> Result<&EntrySpec> {
        match name {
            None => Ok(&self.entries[0]),
            Some(name) => self.entries.iter().find(|e| e.name == name).ok_or_else(|| {
                let names: Vec<&str> = self.entries.iter().map(|e| e.name.as_str()).collect();
                Error::Input(format!("no entry {name:?}; entries: {}", names.join(", ")))
            }),
        }
    }

    /// Runs pipeline `name` on one example of its raw inputs.
    pub fn pipeline(&self, name: &str, example: Vec<(String, RawInput)>) -> Result<Vec<(String, Value)>> {
        let spec = self.pipelines.iter().find(|p| p.name == name).ok_or_else(|| {
            let names: Vec<&str> = self.pipelines.iter().map(|p| p.name.as_str()).collect();
            Error::Input(format!("no pipeline {name:?}; pipelines: {}", names.join(", ")))
        })?;
        let values = raw_values(&spec.raw_inputs, example)?;
        // Not under the lock: the pipeline runs entries, which take it.
        let results = LoadedProgram::run_pipeline(self.program_id, name, values.into_iter().map(|(_, v)| v).collect())?;
        Ok(results.into_iter().map(|(n, v)| (n, v.into())).collect())
    }

    /// Raw inputs of the program's `(preprocess ...)`, if it has one.
    pub fn raw_inputs(&self) -> Option<Vec<RawSpec>> {
        self.inner.lock().unwrap().program.raw_inputs.clone()
    }

    /// Arguments of the program's `(postprocess ...)`, if it has one: names
    /// of model outputs or raw inputs.
    pub fn postprocess_args(&self) -> Option<Vec<String>> {
        self.inner.lock().unwrap().program.post_args.clone()
    }

    /// Runs the program's `(preprocess ...)` on one example: one array per
    /// input of the default entry (without the batch dimension), in input order.
    pub fn preprocess(&self, example: Vec<(String, RawInput)>) -> Result<Vec<(String, ArrayD<f32>)>> {
        let values = self.raw_values(example)?;
        self.inner.lock().unwrap().program.preprocess(values.into_iter().map(|(_, v)| v).collect())
    }

    /// Checks an example against the preprocess's raw inputs and orders it like them.
    fn raw_values(&self, example: Vec<(String, RawInput)>) -> Result<Vec<(String, RawValue)>> {
        let specs = self
            .raw_inputs()
            .ok_or_else(|| Error::Program("the program has no (preprocess ...) form".into()))?;
        raw_values(&specs, example)
    }

    /// Runs the program's `(postprocess ...)` on batched model outputs (e.g.
    /// [`RunOutput::outputs`]), once per example: each output's first
    /// dimension is the batch. Fails if the postprocess needs raw inputs;
    /// use [`Model::infer`] for those.
    pub fn postprocess(&self, outputs: &[(String, ArrayD<f32>)]) -> Result<Vec<Vec<(String, Value)>>> {
        self.postprocess_batch(outputs, None)
    }

    /// Like [`Model::postprocess`], with the raw examples the outputs were
    /// computed from, for a postprocess that reads raw inputs.
    pub fn postprocess_with_raw(
        &self,
        outputs: &[(String, ArrayD<f32>)],
        examples: Vec<Vec<(String, RawInput)>>,
    ) -> Result<Vec<Vec<(String, Value)>>> {
        let raws = examples.into_iter().map(|e| self.raw_values(e)).collect::<Result<Vec<_>>>()?;
        self.postprocess_batch(outputs, Some(raws))
    }

    /// Preprocesses raw examples, runs the model on them as one batch and
    /// postprocesses each example. Without a postprocess, each example gets
    /// its slice of every output.
    pub fn infer(&self, examples: Vec<Vec<(String, RawInput)>>, options: &RunOptions) -> Result<Inference> {
        let post_args = self.postprocess_args();
        let mut kept = Vec::with_capacity(examples.len());
        let mut processed = Vec::with_capacity(examples.len());
        for example in examples {
            let values = self.raw_values(example)?;
            // Raw inputs the postprocess reads are kept for it.
            kept.push(
                values
                    .iter()
                    .filter(|(name, _)| post_args.as_ref().is_some_and(|args| args.contains(name)))
                    .map(|(name, v)| (name.clone(), v.clone()))
                    .collect::<Vec<_>>(),
            );
            let values = values.into_iter().map(|(_, v)| v).collect();
            processed.push(self.inner.lock().unwrap().program.preprocess(values)?);
        }
        let inputs = stack_examples(processed)?;
        let views: Vec<(&str, ArrayViewD<f32>)> = inputs.iter().map(|(n, a)| (n.as_str(), a.view())).collect();
        let run = self.run_with(&views, options)?;
        let examples = self.postprocess_batch(&run.outputs, Some(kept))?;
        Ok(Inference { examples, run })
    }

    fn postprocess_batch(
        &self,
        outputs: &[(String, ArrayD<f32>)],
        raws: Option<Vec<Vec<(String, RawValue)>>>,
    ) -> Result<Vec<Vec<(String, Value)>>> {
        let post_args = self.postprocess_args();
        let batch = match (&raws, outputs.first()) {
            (Some(raws), _) => raws.len(),
            (None, Some((_, a))) => *a.shape().first().ok_or_else(|| Error::Input("outputs are scalars; no batch dimension".into()))?,
            (None, None) => return Err(Error::Input("no outputs to postprocess".into())),
        };
        for (name, a) in outputs {
            if a.shape().first() != Some(&batch) {
                return Err(Error::Input(format!(
                    "postprocess splits outputs into {batch} examples along their first dimension, but {name:?} has shape {:?}",
                    a.shape()
                )));
            }
        }
        let example = |i: usize| -> Vec<(String, ArrayD<f32>)> {
            outputs.iter().map(|(n, a)| (n.clone(), a.index_axis(ndarray::Axis(0), i).to_owned())).collect()
        };
        let Some(args) = post_args else {
            return Ok((0..batch)
                .map(|i| example(i).into_iter().map(|(n, a)| (n, Value::Array(a))).collect())
                .collect());
        };
        let mut raws = raws.map(|r| r.into_iter());
        (0..batch)
            .map(|i| {
                let mut arrays = example(i);
                let mut raw = raws.as_mut().and_then(|r| r.next()).unwrap_or_default();
                let values = args
                    .iter()
                    .map(|name| {
                        if let Some(j) = arrays.iter().position(|(n, _)| n == name) {
                            return Ok(PostValue::Array(arrays.swap_remove(j).1));
                        }
                        if let Some(j) = raw.iter().position(|(n, _)| n == name) {
                            return Ok(PostValue::Raw(raw.swap_remove(j).1));
                        }
                        let is_raw = self.raw_inputs().is_some_and(|specs| specs.iter().any(|s| s.name == *name));
                        Err(Error::Input(if is_raw {
                            format!("postprocess needs raw input {name:?}; run the model on raw inputs (Model::infer)")
                        } else {
                            let outputs: Vec<&str> = outputs.iter().map(|(n, _)| n.as_str()).collect();
                            format!("postprocess argument {name:?} is neither an output ({}) nor a raw input", outputs.join(", "))
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let results = self.inner.lock().unwrap().program.postprocess(values)?;
                Ok(results.into_iter().map(|(n, v)| (n, v.into())).collect())
            })
            .collect()
    }

    /// Preprocesses every example and stacks them: one `[batch, ...]` array per model input.
    pub fn preprocess_batch(&self, examples: Vec<Vec<(String, RawInput)>>) -> Result<Vec<(String, ArrayD<f32>)>> {
        stack_examples(examples.into_iter().map(|e| self.preprocess(e)).collect::<Result<_>>()?)
    }

    /// Preprocesses raw examples, stacks them into a batch and runs the model.
    pub fn run_raw(&self, examples: Vec<Vec<(String, RawInput)>>, options: &RunOptions) -> Result<RunOutput> {
        let inputs = self.preprocess_batch(examples)?;
        let views: Vec<(&str, ArrayViewD<f32>)> = inputs.iter().map(|(n, a)| (n.as_str(), a.view())).collect();
        self.run_with(&views, options)
    }

    /// Name of the primary backend, e.g. "MTL0" or "CPU".
    pub fn device_name(&self) -> String {
        let inner = self.inner.lock().unwrap();
        unsafe { CStr::from_ptr(ggml_backend_name(inner.backends[0])) }.to_string_lossy().into_owned()
    }

    /// Runs the model on named inputs and returns its named outputs.
    /// Shapes are in ndarray order (outermost first).
    pub fn run(&self, inputs: &[(&str, ArrayViewD<f32>)]) -> Result<HashMap<String, ArrayD<f32>>> {
        Ok(self.run_with(inputs, &RunOptions::default())?.outputs.into_iter().collect())
    }

    /// Runs the default entry.
    pub fn run_with(&self, inputs: &[(&str, ArrayViewD<f32>)], options: &RunOptions) -> Result<RunOutput> {
        let name = self.entries[0].name.clone();
        self.run_entry(&name, inputs, options)
    }

    /// Runs entry `name` on named inputs.
    pub fn run_entry(&self, name: &str, inputs: &[(&str, ArrayViewD<f32>)], options: &RunOptions) -> Result<RunOutput> {
        let spec = self.entry(Some(name))?;
        let ordered = order_inputs(spec, inputs, |a| a.shape())?;
        let key: GraphKey = (ordered.iter().map(|a| a.shape().to_vec()).collect(), options.taps.clone());

        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        if let Some(reason) = &inner.poisoned {
            return Err(Error::Backend(format!(
                "the model is unusable after an earlier ggml assertion during compute ({reason}); load it again"
            )));
        }
        if inner.graphs.get(name).and_then(|g| g.graph.as_ref()).is_none_or(|(k, _)| *k != key) {
            inner.rebuild_graph(name, key)?;
        }
        let entry = &inner.graphs[name];
        let (_, graph) = entry.graph.as_ref().unwrap();

        for ((&t, array), spec) in graph.inputs.iter().zip(&ordered).zip(&spec.inputs) {
            let data = array.as_standard_layout();
            if spec.dtype == "i32" {
                guard::tensor_set(t, &to_i32_bytes(&spec.name, data.as_slice().unwrap())?)?;
            } else {
                let bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * size_of::<f32>()) };
                guard::tensor_set(t, bytes)?;
            }
        }
        match guard::sched_graph_compute(entry.sched, graph.graph) {
            Ok(ggml_status::GGML_STATUS_SUCCESS) => {}
            Ok(status) => {
                let msg = unsafe { CStr::from_ptr(ggml_status_to_string(status)) }.to_string_lossy();
                return Err(Error::Backend(format!("graph compute failed: {msg}")));
            }
            Err(e) => {
                inner.poisoned = Some(e.to_string());
                return Err(e);
            }
        }

        Ok(RunOutput { outputs: read_results(&graph.outputs)?, taps: read_results(&graph.taps)? })
    }

    /// Builds the default entry's graph for the given input shapes (ndarray
    /// order) without allocating or computing it, and describes its nodes.
    pub fn graph(&self, inputs: &[(&str, Vec<usize>)], taps: &Taps) -> Result<GraphInfo> {
        let name = self.entries[0].name.clone();
        self.graph_entry(&name, inputs, taps)
    }

    /// Like [`Model::graph`], for entry `name`.
    pub fn graph_entry(&self, name: &str, inputs: &[(&str, Vec<usize>)], taps: &Taps) -> Result<GraphInfo> {
        let spec = self.entry(Some(name))?;
        let ordered = order_inputs(spec, inputs, |s| s.as_slice())?;
        let shapes: Vec<Vec<usize>> = ordered.into_iter().cloned().collect();
        let mut inner = self.inner.lock().unwrap();
        let graph = inner.build_graph(name, &shapes, taps)?;
        let info = describe(&graph);
        unsafe { ggml_free(graph.ctx) };
        Ok(info)
    }
}

/// Matches named inputs to an entry's declared inputs and checks dtype and shape.
fn order_inputs<'a, T>(spec: &EntrySpec, given: &'a [(&str, T)], shape_of: impl Fn(&T) -> &[usize]) -> Result<Vec<&'a T>> {
    let inputs = &spec.inputs;
    let expected = || inputs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ");
    for (name, _) in given {
        if !inputs.iter().any(|s| s.name == *name) {
            return Err(Error::Input(format!("unknown input {name:?} of entry {:?}, expected: {}", spec.name, expected())));
        }
    }
    inputs
        .iter()
        .map(|spec| {
            let mut matches = given.iter().filter(|(n, _)| *n == spec.name);
            let (_, value) = matches
                .next()
                .ok_or_else(|| Error::Input(format!("missing input {:?}, expected: {}", spec.name, expected())))?;
            if matches.next().is_some() {
                return Err(Error::Input(format!("input {:?} given twice", spec.name)));
            }
            if spec.dtype != "f32" && spec.dtype != "i32" {
                return Err(Error::Input(format!("input {:?} has unsupported dtype {}", spec.name, spec.dtype)));
            }
            check_shape(spec, shape_of(value))?;
            Ok(value)
        })
        .collect()
}

/// Checks an example against raw input specs and orders it like them.
fn raw_values(specs: &[RawSpec], example: Vec<(String, RawInput)>) -> Result<Vec<(String, RawValue)>> {
    let mut example = example;

    let expected = || specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ");
    if let Some((name, _)) = example.iter().find(|(n, _)| !specs.iter().any(|s| s.name == *n)) {
        return Err(Error::Input(format!("unknown raw input {name:?}, expected: {}", expected())));
    }
    specs
        .iter()
        .map(|spec| {
            let i = example
                .iter()
                .position(|(n, _)| *n == spec.name)
                .ok_or_else(|| Error::Input(format!("missing raw input {:?}, expected: {}", spec.name, expected())))?;
            let (_, input) = example.swap_remove(i);
            if input.kind() != spec.kind {
                return Err(Error::Input(format!("raw input {:?} must be {:?}", spec.name, spec.kind)));
            }
            let value = match input {
                RawInput::Text(t) => RawValue::Text(t),
                RawInput::Image(i) => RawValue::Image(i.to_rgb8()),
                RawInput::Audio(a) => RawValue::Audio(a),
                RawInput::Array(a) => RawValue::Array(a),
            };
            Ok((spec.name.clone(), value))
        })
        .collect()
}

/// Stacks per-example arrays into `[batch, ...]` arrays.
fn stack_examples(processed: Vec<Vec<(String, ArrayD<f32>)>>) -> Result<Vec<(String, ArrayD<f32>)>> {
    let Some(first) = processed.first() else { return Err(Error::Input("no examples".into())) };
    (0..first.len())
        .map(|i| {
            let name = first[i].0.clone();
            let views: Vec<_> = processed.iter().map(|p| p[i].1.view()).collect();
            let stacked = ndarray::stack(ndarray::Axis(0), &views).map_err(|_| {
                Error::Input(format!("preprocessed {name} differs in shape between examples; pad to a fixed size"))
            })?;
            Ok((name, stacked))
        })
        .collect()
}

/// i32 inputs are passed as f32 arrays holding whole numbers.
fn to_i32_bytes(name: &str, values: &[f32]) -> Result<Vec<u8>> {
    values
        .iter()
        .map(|&v| {
            if v.fract() == 0.0 && v >= i32::MIN as f32 && v <= i32::MAX as f32 {
                Ok((v as i32).to_le_bytes())
            } else {
                Err(Error::Input(format!("input {name:?} is i32 but contains {v}")))
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(|b| b.concat())
}

/// Reads graph results back as ndarrays of their declared rank.
fn read_results(results: &[(String, *mut ggml_tensor, usize)]) -> Result<Vec<(String, ArrayD<f32>)>> {
    results
        .iter()
        .map(|(name, t, rank)| {
            let t = *t;
            let n = unsafe { ggml_nelements(t) } as usize;
            let mut data = vec![0f32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), n * size_of::<f32>()) };
            guard::tensor_get(t, bytes)?;
            let array = ArrayD::from_shape_vec(IxDyn(&result_shape(t, *rank)), data)
                .map_err(|e| Error::Backend(format!("{name}: {e}")))?;
            Ok((name.clone(), array))
        })
        .collect()
}

/// ndarray-order shape of a result tensor with the given rank.
fn result_shape(t: *const ggml_tensor, rank: usize) -> Vec<usize> {
    let ne = unsafe { (*t).ne };
    (0..rank).rev().map(|i| ne[i] as usize).collect()
}

fn tensor_name(t: *const ggml_tensor) -> String {
    unsafe { CStr::from_ptr(ggml_get_name(t)) }.to_string_lossy().into_owned()
}

fn describe(graph: &Graph) -> GraphInfo {
    unsafe {
        let n = ggml_graph_n_nodes(graph.graph);
        let nodes: Vec<*mut ggml_tensor> = (0..n).map(|i| ggml_graph_node(graph.graph, i)).collect();
        let index: HashMap<*mut ggml_tensor, usize> = nodes.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let nodes = nodes
            .iter()
            .map(|&t| NodeInfo {
                op: CStr::from_ptr(ggml_op_desc(t)).to_string_lossy().into_owned(),
                name: Some(tensor_name(t))
                    .filter(|n| !n.strip_prefix("node_").is_some_and(|i| i.parse::<usize>().is_ok()))
                    .unwrap_or_default(),
                dtype: DType((*t).type_),
                ne: { let ne = (*t).ne; ne[..ggml_n_dims(t) as usize].to_vec() },
                nb: { let nb = (*t).nb; nb[..ggml_n_dims(t) as usize].to_vec() },
                view_offs: (*t).view_offs,
                op_params: (*t).op_params,
                srcs: { (*t).src }
                    .iter()
                    .filter(|s| !s.is_null())
                    .map(|&s| match index.get(&s) {
                        Some(i) => format!("%{i}"),
                        None => tensor_name(s),
                    })
                    .collect(),
            })
            .collect();
        GraphInfo {
            nodes,
            outputs: graph.outputs.iter().map(|(name, t, rank)| (name.clone(), result_shape(*t, *rank))).collect(),
            taps: graph.tap_names.clone(),
        }
    }
}

fn check_shape(spec: &InputSpec, shape: &[usize]) -> Result<()> {
    let ne = ne_from_shape(shape)?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(Error::Input(format!("input {:?} has empty shape {shape:?}", spec.name)));
    }
    let Some(dims) = &spec.dims else { return Ok(()) };
    let fits = dims.len() == shape.len()
        && dims.iter().zip(ne).all(|(d, n)| d.is_none_or(|d| d == n));
    if fits {
        Ok(())
    } else {
        // Report the declaration in ndarray order, like the shape we got.
        let declared: Vec<String> =
            dims.iter().rev().map(|d| d.map_or("_".into(), |d| d.to_string())).collect();
        Err(Error::Input(format!(
            "input {:?} has shape {shape:?}, the program declares [{}]",
            spec.name,
            declared.join(", ")
        )))
    }
}

impl Inner {
    /// Creates the program's state tensors on the primary device, zeroed.
    fn alloc_states(&mut self) -> Result<()> {
        let specs: Vec<StateSpec> = self.program.states.clone();
        if specs.is_empty() {
            return Ok(());
        }
        let mem_size = unsafe { ggml_tensor_overhead() } * specs.len();
        self.states = unsafe { ggml_init(ggml_init_params { mem_size, mem_buffer: null_mut(), no_alloc: true }) };
        if self.states.is_null() {
            return Err(Error::Backend("failed to create the state context".into()));
        }
        for spec in &specs {
            let ty = if spec.dtype == "f16" { ggml_type::GGML_TYPE_F16 } else { ggml_type::GGML_TYPE_F32 };
            let t = unsafe { ggml_new_tensor(self.states, ty, spec.dims.len() as i32, spec.dims.as_ptr()) };
            let name = std::ffi::CString::new(spec.name.as_str()).map_err(|_| Error::Program("bad state name".into()))?;
            unsafe { ggml_set_name(t, name.as_ptr()) };
        }
        self.state_buffer = guard::alloc_ctx_tensors(self.states, self.backends[0])?;
        if self.state_buffer.is_null() {
            return Err(Error::Backend("failed to allocate the state buffer".into()));
        }
        unsafe { ggml_backend_buffer_clear(self.state_buffer, 0) };
        Ok(())
    }

    /// Replaces entry's current graph with a newly built and allocated one.
    fn rebuild_graph(&mut self, entry: &str, key: GraphKey) -> Result<()> {
        if !self.graphs.contains_key(entry) {
            let sched = unsafe {
                ggml_backend_sched_new(
                    self.backends.as_mut_ptr(),
                    null_mut(),
                    self.backends.len() as i32,
                    GRAPH_SIZE,
                    false,
                    true,
                )
            };
            if sched.is_null() {
                return Err(Error::Backend("failed to create the scheduler".into()));
            }
            self.graphs.insert(entry.to_string(), EntryGraph { sched, graph: None });
        }
        let sched = self.graphs[entry].sched;
        unsafe { ggml_backend_sched_reset(sched) };
        if let Some((_, old)) = self.graphs.get_mut(entry).unwrap().graph.take() {
            unsafe { ggml_free(old.ctx) };
        }
        let graph = self.build_graph(entry, &key.0, &key.1)?;
        let allocated = guard::sched_alloc_graph(sched, graph.graph);
        if !matches!(allocated, Ok(true)) {
            unsafe {
                ggml_backend_sched_reset(sched);
                ggml_free(graph.ctx);
            }
            allocated?;
            return Err(Error::Backend("failed to allocate the graph".into()));
        }
        self.graphs.get_mut(entry).unwrap().graph = Some((key, graph));
        Ok(())
    }

    fn build_graph(&mut self, entry: &str, shapes: &[Vec<usize>], taps: &Taps) -> Result<Graph> {
        let input_dims = shapes
            .iter()
            .map(|shape| Ok(ne_from_shape(shape)?[..shape.len()].to_vec()))
            .collect::<Result<Vec<_>>>()?;
        let mem_size = unsafe { ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false) };
        let ctx = unsafe { ggml_init(ggml_init_params { mem_size, mem_buffer: null_mut(), no_alloc: true }) };
        if ctx.is_null() {
            return Err(Error::Backend("failed to create the graph context".into()));
        }
        let device = if unsafe { ggml_backend_is_cpu(self.backends[0]) } { "cpu" } else { "gpu" };
        match self.program.build(entry, ctx, self.file.tensors, self.states, device, &input_dims, GRAPH_SIZE, taps) {
            Ok(built) => Ok(Graph {
                ctx,
                graph: built.graph,
                inputs: built.inputs,
                outputs: built.outputs,
                taps: built.taps,
                tap_names: built.tap_names,
            }),
            Err(e) => {
                unsafe { ggml_free(ctx) };
                Err(e)
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.poisoned.is_some() {
            // Backend threads may still be blocked in or reading from the
            // failed graph; freeing its memory or joining them could crash or hang.
            return;
        }
        unsafe {
            for entry in self.graphs.values() {
                if let Some((_, graph)) = &entry.graph {
                    ggml_free(graph.ctx);
                }
                ggml_backend_sched_free(entry.sched);
            }
            if !self.weights.is_null() {
                ggml_backend_buffer_free(self.weights);
            }
            if !self.state_buffer.is_null() {
                ggml_backend_buffer_free(self.state_buffer);
            }
            if !self.states.is_null() {
                ggml_free(self.states);
            }
            for &backend in &self.backends {
                ggml_backend_free(backend);
            }
            ManuallyDrop::drop(&mut self.file);
        }
    }
}
