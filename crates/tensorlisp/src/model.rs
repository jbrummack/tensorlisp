use std::{
    collections::HashMap,
    ffi::CStr,
    fs::File,
    mem::ManuallyDrop,
    os::unix::fs::FileExt,
    path::Path,
    ptr::null_mut,
    sync::Mutex,
};

use ggml_sys::ffi::*;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::{
    error::{Error, Result},
    gguf::{GgufFile, ne_from_shape},
    guard,
    program::Program,
    dtype::DType,
    scheme::{InputSpec, LoadedProgram, RawKind, RawSpec, RawValue, Taps},
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

/// A tensorlisp model: weights on the device plus the program that builds its graph.
///
/// The graph for the current input shapes stays allocated and is recomputed
/// directly while the shapes stay the same. A shape change rebuilds it, since
/// resetting the scheduler invalidates every previously allocated graph.
///
/// ggml assertions become [`Error::Ggml`] where they can be caught: while the
/// graph is built, and on the calling thread during allocation and compute.
/// An assertion during compute leaves the backend's threads in an unknown
/// state, so the model then refuses to run and leaks its ggml objects instead
/// of freeing memory those threads may still use. Load it again to recover.
/// Assertions on backend worker threads still abort the process.
pub struct Model {
    inputs: Vec<InputSpec>,
    inner: Mutex<Inner>,
}

/// Options for [`Model::run_with`].
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Intermediate tensors marked with `(tap name t)` to read back.
    pub taps: Taps,
}

/// Named results in the order the program declares them.
#[derive(Debug, Clone, Default)]
pub struct RunOutput {
    pub outputs: Vec<(String, ArrayD<f32>)>,
    pub taps: Vec<(String, ArrayD<f32>)>,
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

struct Inner {
    program: LoadedProgram,
    /// Holds the weight tensors' metadata; ManuallyDrop so a poisoned model can leak it.
    file: ManuallyDrop<GgufFile>,
    /// Primary device first, CPU last.
    backends: Vec<ggml_backend_t>,
    weights: ggml_backend_buffer_t,
    sched: ggml_backend_sched_t,
    /// The allocated graph and what it was built for.
    graph: Option<(GraphKey, Graph)>,
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
        data.read_exact_at(&mut buf, file.tensor_file_offset(i) as u64)?;
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
            sched: null_mut(),
            graph: None,
            poisoned: None,
        };
        inner.weights = guard::alloc_ctx_tensors(inner.file.tensors, inner.backends[0])?;
        if inner.weights.is_null() && inner.file.tensor_count() > 0 {
            return Err(Error::Backend("failed to allocate the weight buffer".into()));
        }
        load_weights(&inner.file, path)?;

        inner.sched = unsafe {
            ggml_backend_sched_new(
                inner.backends.as_mut_ptr(),
                null_mut(),
                inner.backends.len() as i32,
                GRAPH_SIZE,
                false,
                true,
            )
        };
        if inner.sched.is_null() {
            return Err(Error::Backend("failed to create the scheduler".into()));
        }
        Ok(Model { inputs: inner.program.inputs.clone(), inner: Mutex::new(inner) })
    }

    /// Inputs declared by the program, in declaration order.
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }

    /// Raw inputs of the program's `(preprocess ...)`, if it has one.
    pub fn raw_inputs(&self) -> Option<Vec<RawSpec>> {
        self.inner.lock().unwrap().program.raw_inputs.clone()
    }

    /// Runs the program's `(preprocess ...)` on one example: one array per
    /// model input (without the batch dimension), in input order.
    pub fn preprocess(&self, example: Vec<(String, RawInput)>) -> Result<Vec<(String, ArrayD<f32>)>> {
        let inner = self.inner.lock().unwrap();
        let specs = inner
            .program
            .raw_inputs
            .clone()
            .ok_or_else(|| Error::Program("the program has no (preprocess ...) form".into()))?;
        let mut example = example;
        let expected = || specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ");
        if let Some((name, _)) = example.iter().find(|(n, _)| !specs.iter().any(|s| s.name == *n)) {
            return Err(Error::Input(format!("unknown raw input {name:?}, expected: {}", expected())));
        }
        let values = specs
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
                Ok(match input {
                    RawInput::Text(t) => RawValue::Text(t),
                    RawInput::Image(i) => RawValue::Image(i.to_rgb8()),
                    RawInput::Audio(a) => RawValue::Audio(a),
                    RawInput::Array(a) => RawValue::Array(a),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        inner.program.preprocess(values)
    }

    /// Preprocesses every example and stacks them: one `[batch, ...]` array per model input.
    pub fn preprocess_batch(&self, examples: Vec<Vec<(String, RawInput)>>) -> Result<Vec<(String, ArrayD<f32>)>> {
        let processed: Vec<Vec<(String, ArrayD<f32>)>> =
            examples.into_iter().map(|e| self.preprocess(e)).collect::<Result<_>>()?;
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

    pub fn run_with(&self, inputs: &[(&str, ArrayViewD<f32>)], options: &RunOptions) -> Result<RunOutput> {
        let ordered = self.order_inputs(inputs, |a| a.shape())?;
        let key: GraphKey = (ordered.iter().map(|a| a.shape().to_vec()).collect(), options.taps.clone());

        let mut guard = self.inner.lock().unwrap();
        let inner = &mut *guard;
        if let Some(reason) = &inner.poisoned {
            return Err(Error::Backend(format!(
                "the model is unusable after an earlier ggml assertion during compute ({reason}); load it again"
            )));
        }
        if inner.graph.as_ref().is_none_or(|(k, _)| *k != key) {
            inner.rebuild_graph(key)?;
        }
        let (_, graph) = inner.graph.as_ref().unwrap();

        for ((&t, array), spec) in graph.inputs.iter().zip(&ordered).zip(&self.inputs) {
            let data = array.as_standard_layout();
            if spec.dtype == "i32" {
                guard::tensor_set(t, &to_i32_bytes(&spec.name, data.as_slice().unwrap())?)?;
            } else {
                let bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * size_of::<f32>()) };
                guard::tensor_set(t, bytes)?;
            }
        }
        match guard::sched_graph_compute(inner.sched, graph.graph) {
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

    /// Builds the graph for the given input shapes (ndarray order) without
    /// allocating or computing it, and describes its nodes.
    pub fn graph(&self, inputs: &[(&str, Vec<usize>)], taps: &Taps) -> Result<GraphInfo> {
        let ordered = self.order_inputs(inputs, |s| s.as_slice())?;
        let shapes: Vec<Vec<usize>> = ordered.into_iter().cloned().collect();
        let mut inner = self.inner.lock().unwrap();
        let graph = inner.build_graph(&shapes, taps)?;
        let info = describe(&graph);
        unsafe { ggml_free(graph.ctx) };
        Ok(info)
    }

    /// Matches named inputs to the declared inputs and checks dtype and shape.
    fn order_inputs<'a, T>(&self, given: &'a [(&str, T)], shape_of: impl Fn(&T) -> &[usize]) -> Result<Vec<&'a T>> {
        let expected = || self.inputs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ");
        for (name, _) in given {
            if !self.inputs.iter().any(|s| s.name == *name) {
                return Err(Error::Input(format!("unknown input {name:?}, expected: {}", expected())));
            }
        }
        self.inputs
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
    /// Replaces the current graph with a newly built and allocated one.
    fn rebuild_graph(&mut self, key: GraphKey) -> Result<()> {
        unsafe { ggml_backend_sched_reset(self.sched) };
        if let Some((_, old)) = self.graph.take() {
            unsafe { ggml_free(old.ctx) };
        }
        let graph = self.build_graph(&key.0, &key.1)?;
        let allocated = guard::sched_alloc_graph(self.sched, graph.graph);
        if !matches!(allocated, Ok(true)) {
            unsafe {
                ggml_backend_sched_reset(self.sched);
                ggml_free(graph.ctx);
            }
            allocated?;
            return Err(Error::Backend("failed to allocate the graph".into()));
        }
        self.graph = Some((key, graph));
        Ok(())
    }

    fn build_graph(&mut self, shapes: &[Vec<usize>], taps: &Taps) -> Result<Graph> {
        let input_dims = shapes
            .iter()
            .map(|shape| Ok(ne_from_shape(shape)?[..shape.len()].to_vec()))
            .collect::<Result<Vec<_>>>()?;
        let mem_size = unsafe { ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false) };
        let ctx = unsafe { ggml_init(ggml_init_params { mem_size, mem_buffer: null_mut(), no_alloc: true }) };
        if ctx.is_null() {
            return Err(Error::Backend("failed to create the graph context".into()));
        }
        match self.program.build(ctx, self.file.tensors, &input_dims, GRAPH_SIZE, taps) {
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
            if let Some((_, graph)) = &self.graph {
                ggml_free(graph.ctx);
            }
            if !self.sched.is_null() {
                ggml_backend_sched_free(self.sched);
            }
            if !self.weights.is_null() {
                ggml_backend_buffer_free(self.weights);
            }
            for &backend in &self.backends {
                ggml_backend_free(backend);
            }
            ManuallyDrop::drop(&mut self.file);
        }
    }
}
