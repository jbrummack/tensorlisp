//! Training of `(define-param ...)` parameters (LoRA adapters) with ggml's autodiff.
//!
//! A [`Trainer`] builds one entry of the program (its output `loss` must be a scalar) with
//! gradients, derives the backward graph with `ggml_build_backward_expand` and appends an
//! AdamW step per parameter, like `ggml_opt` does, but keeps the graphs static: they are
//! built and allocated once, so a step is "set inputs, compute". See docs/design/training.md.
//!
//! The gradient of every parameter lives in a persistent accumulator that stays readable
//! after a step ([`Trainer::grad`]); that is also what a native executor is checked against.

use std::{collections::HashMap, ffi::CString, ptr::null_mut};

use ggml_sys::ffi::*;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use super::{Graph, Inner, Model, order_inputs, to_i32_bytes};
use crate::{
    error::{Error, Result},
    guard,
    scheme::{StateSpec, Taps},
};

/// Maximum number of nodes of a training graph (forward, backward and optimizer).
const TRAIN_GRAPH_SIZE: usize = 65536;

/// AdamW hyperparameters and the gradient accumulation period.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    /// The entry that computes the loss (`(outputs [loss ...] ...)`, a scalar).
    pub entry: String,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay (AdamW), multiplied by the learning rate.
    pub weight_decay: f32,
    /// Micro-batches per optimizer step. The gradients of the micro-batches are averaged (the loss
    /// gradient starts at 1 / accum_steps); the loss a step reports is the micro-batch's own.
    pub accum_steps: usize,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions { entry: "train".into(), lr: 1e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, accum_steps: 1 }
    }
}

/// What one [`Trainer::step`] did.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    /// The micro-batch's loss.
    pub loss: f32,
    /// Whether this micro-batch ended an accumulation period and updated the parameters.
    pub optimized: bool,
}

struct Param {
    name: String,
    tensor: *mut ggml_tensor,
    grad: *mut ggml_tensor,
}

pub struct Trainer {
    model: Model,
    opts: TrainOptions,
    graph: Graph,
    gb_grad: *mut ggml_cgraph,
    gb_opt: *mut ggml_cgraph,
    allocated: *mut ggml_cgraph,
    sched: ggml_backend_sched_t,
    ctx_static: *mut ggml_context,
    ctx_cpu: *mut ggml_context,
    buf_static: ggml_backend_buffer_t,
    buf_cpu: ggml_backend_buffer_t,
    adamw: *mut ggml_tensor,
    loss: *mut ggml_tensor,
    /// The accumulator holding d loss / d loss, 1 / accum_steps during an accumulation period.
    loss_grad: *mut ggml_tensor,
    params: Vec<Param>,
    input_specs: Vec<crate::scheme::InputSpec>,
    /// Optimizer steps taken + 1 (AdamW's bias correction).
    iter: i64,
    /// Micro-batches done in the current accumulation period.
    micro: usize,
}

// ggml objects aren't tied to a thread; every use holds the model's lock.
unsafe impl Send for Trainer {}

/// `TL_TRAIN_TRACE=1` prints the steps of every micro-batch to stderr.
fn trace(what: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("TL_TRAIN_TRACE").is_some()) {
        eprintln!("[train {:>8.3}s] {what}", START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64());
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn c_name(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| Error::Program(format!("bad tensor name {name:?}")))
}

fn shape_of(t: *const ggml_tensor) -> Vec<usize> {
    let n = unsafe { ggml_n_dims(t) } as usize;
    let ne = unsafe { (*t).ne };
    (0..n).rev().map(|i| ne[i] as usize).collect()
}

fn read_f32(t: *const ggml_tensor) -> Result<ArrayD<f32>> {
    let n = unsafe { ggml_nelements(t) } as usize;
    let mut data = vec![0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), n * 4) };
    guard::tensor_get(t, bytes)?;
    ArrayD::from_shape_vec(IxDyn(&shape_of(t)), data).map_err(|e| Error::Backend(e.to_string()))
}

fn write_f32(t: *mut ggml_tensor, data: &[f32]) -> Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    guard::tensor_set(t, bytes)
}

impl Inner {
    /// The state tensor `name`.
    fn state_tensor(&self, name: &str) -> Result<*mut ggml_tensor> {
        if self.states.is_null() {
            return Err(Error::Input(format!("the program has no state {name:?}")));
        }
        let t = unsafe { ggml_get_tensor(self.states, c_name(name)?.as_ptr()) };
        if t.is_null() {
            return Err(Error::Input(format!("the program has no state {name:?}")));
        }
        Ok(t)
    }

    /// Fills every `(define-param ...)` tensor: zeros, or uniform in [-bound, bound] from `seed`.
    pub(super) fn init_params(&self, seed: u64) -> Result<()> {
        for spec in self.program.states.iter().filter(|s| s.trainable.is_some()) {
            let t = self.state_tensor(&spec.name)?;
            let n = unsafe { ggml_nelements(t) } as usize;
            let bound = spec.trainable.unwrap();
            let mut rng = seed ^ spec.name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    if bound == 0.0 {
                        0.0
                    } else {
                        let u = (splitmix(&mut rng) >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
                        (2.0 * u - 1.0) * bound
                    }
                })
                .collect();
            write_f32(t, &data)?;
        }
        Ok(())
    }
}

impl Model {
    /// The trainable parameters `(define-param ...)` of the program, with their ndarray-order shapes.
    pub fn params(&self) -> Vec<(String, Vec<usize>)> {
        let inner = self.inner.lock().unwrap();
        inner
            .program
            .states
            .iter()
            .filter(|s| s.trainable.is_some())
            .map(|s| (s.name.clone(), s.dims.iter().rev().map(|&d| d as usize).collect()))
            .collect()
    }

    /// Reads state (or parameter) `name` as an f32 array.
    pub fn state(&self, name: &str) -> Result<ArrayD<f32>> {
        let inner = self.inner.lock().unwrap();
        let t = inner.state_tensor(name)?;
        if unsafe { (*t).type_ } != ggml_type::GGML_TYPE_F32 {
            return Err(Error::Input(format!("state {name:?} is not f32")));
        }
        read_f32(t)
    }

    /// Overwrites f32 state (or parameter) `name`; the array must have its shape.
    pub fn set_state(&self, name: &str, value: &ArrayD<f32>) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let t = inner.state_tensor(name)?;
        if unsafe { (*t).type_ } != ggml_type::GGML_TYPE_F32 {
            return Err(Error::Input(format!("state {name:?} is not f32")));
        }
        if shape_of(t) != value.shape() {
            return Err(Error::Input(format!("state {name:?} has shape {:?}, got {:?}", shape_of(t), value.shape())));
        }
        write_f32(t, value.as_standard_layout().as_slice().unwrap())
    }

    /// Re-initializes the parameters with `seed` (`(define-param ...)` inits).
    pub fn init_params(&self, seed: u64) -> Result<()> {
        self.inner.lock().unwrap().init_params(seed)
    }

    /// Builds the training graphs of `opts.entry` for the given input shapes (ndarray order).
    pub fn trainer(&self, inputs: &[(&str, Vec<usize>)], opts: TrainOptions) -> Result<Trainer> {
        if opts.accum_steps == 0 {
            return Err(Error::Input("accum_steps must be at least 1".into()));
        }
        let spec = self.entry(Some(&opts.entry))?.clone();
        let ordered = order_inputs(&spec, inputs, |s| s.as_slice())?;
        let shapes: Vec<Vec<usize>> = ordered.into_iter().cloned().collect();

        let mut guard_ = self.inner.lock().unwrap();
        let inner = &mut *guard_;
        if let Some(reason) = &inner.poisoned {
            return Err(Error::Backend(format!("the model is unusable after an earlier ggml assertion ({reason})")));
        }
        #[cfg(native_device)]
        if inner.native.is_some() {
            return Err(Error::Backend("training is not supported on the native device yet".into()));
        }
        let param_specs: Vec<StateSpec> = inner.program.states.iter().filter(|s| s.trainable.is_some()).cloned().collect();
        if param_specs.is_empty() {
            return Err(Error::Program("the program defines no (define-param ...) to train".into()));
        }

        // Parameters are flagged before the graph is built: ggml only makes a flagged leaf a graph
        // node (which is what gets a gradient). The flag is removed again when the trainer is dropped.
        let mut params = Vec::new();
        for ps in &param_specs {
            let t = inner.state_tensor(&ps.name)?;
            unsafe { ggml_set_param(t) };
            params.push(Param { name: ps.name.clone(), tensor: t, grad: null_mut() });
        }
        let clear_flags = |params: &[Param]| {
            for p in params {
                unsafe { (*p.tensor).flags &= !(ggml_tensor_flag::GGML_TENSOR_FLAG_PARAM as i32) };
            }
        };
        let graph = match inner.build_graph_with(&opts.entry, &shapes, &Taps::None, true, TRAIN_GRAPH_SIZE) {
            Ok(g) => g,
            Err(e) => {
                clear_flags(&params);
                return Err(e);
            }
        };
        let ctx = graph.ctx;
        let loss = graph
            .outputs
            .iter()
            .find(|(name, _, _)| name == "loss")
            .map(|(_, t, _)| *t)
            .ok_or_else(|| Error::Program(format!("entry {:?} has no output named loss", opts.entry)))?;
        unsafe {
            if ggml_nelements(loss) != 1 || (*loss).type_ != ggml_type::GGML_TYPE_F32 {
                return Err(Error::Program("the loss must be one f32 value".into()));
            }
        }
        unsafe { ggml_set_loss(loss) };

        let gf = graph.graph;
        let n_nodes = unsafe { ggml_graph_n_nodes(gf) } as usize;
        let node = |g: *mut ggml_cgraph, i: usize| unsafe { ggml_graph_node(g, i as i32) };
        let is_param = |t: *mut ggml_tensor| unsafe { (*t).flags & ggml_tensor_flag::GGML_TENSOR_FLAG_PARAM as i32 != 0 };
        let is_loss = |t: *mut ggml_tensor| unsafe { (*t).flags & ggml_tensor_flag::GGML_TENSOR_FLAG_LOSS as i32 != 0 };
        let n_param_nodes = (0..n_nodes).filter(|&i| is_param(node(gf, i))).count();
        if n_param_nodes != params.len() {
            return Err(Error::Program(format!(
                "{} of {} parameters take part in the loss",
                n_param_nodes,
                params.len()
            )));
        }

        // Persistent tensors: the loss's gradient, a gradient accumulator and AdamW's m and v per parameter.
        let ctx_static = unsafe {
            ggml_init(ggml_init_params {
                mem_size: (1 + 3 * params.len()) * ggml_tensor_overhead(),
                mem_buffer: null_mut(),
                no_alloc: true,
            })
        };
        let mut grad_accs: Vec<*mut ggml_tensor> = vec![null_mut(); n_nodes];
        let mut ms: Vec<*mut ggml_tensor> = vec![null_mut(); n_nodes];
        let mut vs: Vec<*mut ggml_tensor> = vec![null_mut(); n_nodes];
        for i in 0..n_nodes {
            let n = node(gf, i);
            unsafe {
                if is_param(n) || is_loss(n) {
                    grad_accs[i] = ggml_new_tensor(ctx_static, ggml_type::GGML_TYPE_F32, GGML_MAX_DIMS as i32, (*n).ne.as_ptr());
                }
                if is_param(n) {
                    ms[i] = ggml_new_tensor(ctx_static, ggml_type::GGML_TYPE_F32, GGML_MAX_DIMS as i32, (*n).ne.as_ptr());
                    vs[i] = ggml_new_tensor(ctx_static, ggml_type::GGML_TYPE_F32, GGML_MAX_DIMS as i32, (*n).ne.as_ptr());
                }
            }
        }

        let (gb_grad, gb_opt, adamw, ctx_cpu) = unsafe {
            let gb_grad = ggml_graph_dup(ctx, gf, true);
            ggml_build_backward_expand(ctx, gb_grad, grad_accs.as_mut_ptr());
            let gb_opt = ggml_graph_dup(ctx, gb_grad, true);
            let ctx_cpu = ggml_init(ggml_init_params { mem_size: ggml_tensor_overhead(), mem_buffer: null_mut(), no_alloc: true });
            let adamw = ggml_new_tensor_1d(ctx_cpu, ggml_type::GGML_TYPE_F32, 7);
            ggml_set_input(adamw);
            ggml_set_name(adamw, c"AdamW_params".as_ptr());
            for i in (0..n_nodes).rev() {
                let n = node(gb_opt, i);
                let grad = ggml_graph_get_grad(gb_opt, n);
                if !grad.is_null() && is_param(n) {
                    let step = ggml_opt_step_adamw(ctx, n, grad, ms[i], vs[i], adamw);
                    ggml_build_forward_expand(gb_opt, step);
                }
            }
            (gb_grad, gb_opt, adamw, ctx_cpu)
        };
        // Which accumulator belongs to which parameter, by graph node.
        let mut loss_grad = null_mut();
        for i in 0..n_nodes {
            let n = node(gf, i);
            if is_loss(n) {
                loss_grad = grad_accs[i];
            }
            if is_param(n) {
                let p = params.iter_mut().find(|p| p.tensor == n).ok_or_else(|| Error::Backend("parameter node not found".into()))?;
                p.grad = grad_accs[i];
            }
        }

        let buf_static = unsafe { ggml_backend_alloc_ctx_tensors(ctx_static, inner.backends[0]) };
        let buf_cpu = unsafe { ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type()) };
        if buf_static.is_null() || buf_cpu.is_null() {
            return Err(Error::Backend("failed to allocate the training buffers".into()));
        }
        unsafe {
            ggml_backend_buffer_clear(buf_static, 0);
            ggml_graph_reset(gb_opt); // m, v to zero, the loss's gradient to 1
        }
        let sched = unsafe {
            ggml_backend_sched_new(inner.backends.as_mut_ptr(), null_mut(), inner.backends.len() as i32, TRAIN_GRAPH_SIZE, false, true)
        };
        if sched.is_null() {
            return Err(Error::Backend("failed to create the scheduler".into()));
        }
        drop(guard_);
        Ok(Trainer {
            model: self.clone(),
            opts,
            graph,
            gb_grad,
            gb_opt,
            allocated: null_mut(),
            sched,
            ctx_static,
            ctx_cpu,
            buf_static,
            buf_cpu,
            adamw,
            loss,
            loss_grad,
            params,
            input_specs: spec.inputs,
            iter: 1,
            micro: 0,
        })
    }
}

impl Trainer {
    pub fn set_lr(&mut self, lr: f32) {
        self.opts.lr = lr;
    }

    /// Names of the trained parameters.
    pub fn param_names(&self) -> Vec<String> {
        self.params.iter().map(|p| p.name.clone()).collect()
    }

    /// The gradient of parameter `name` from the last forward/backward (accumulated over the
    /// current period). Same shape as the parameter.
    pub fn grad(&self, name: &str) -> Result<ArrayD<f32>> {
        let p = self.params.iter().find(|p| p.name == name).ok_or_else(|| Error::Input(format!("no parameter {name:?}")))?;
        let _inner = self.model.inner.lock().unwrap();
        read_f32(p.grad)
    }

    fn alloc(&mut self, graph: *mut ggml_cgraph) -> Result<()> {
        if self.allocated == graph {
            return Ok(());
        }
        unsafe { ggml_backend_sched_reset(self.sched) };
        self.allocated = null_mut();
        match guard::sched_alloc_graph(self.sched, graph) {
            Ok(true) => {
                self.allocated = graph;
                Ok(())
            }
            Ok(false) => Err(Error::Backend("failed to allocate the training graph".into())),
            Err(e) => Err(e),
        }
    }

    fn set_inputs(&self, given: &[(&str, ArrayViewD<f32>)]) -> Result<()> {
        let spec = crate::scheme::EntrySpec { name: self.opts.entry.clone(), inputs: self.input_specs.clone() };
        let ordered = order_inputs(&spec, given, |a| a.shape())?;
        for ((&t, array), spec) in self.graph.inputs.iter().zip(&ordered).zip(&self.input_specs) {
            let data = array.as_standard_layout();
            if spec.dtype == "i32" {
                guard::tensor_set(t, &to_i32_bytes(&spec.name, data.as_slice().unwrap())?)?;
            } else {
                let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
                guard::tensor_set(t, bytes)?;
            }
        }
        Ok(())
    }

    fn compute(&mut self, graph: *mut ggml_cgraph) -> Result<f32> {
        let status = match guard::sched_graph_compute(self.sched, graph) {
            Ok(s) => s,
            Err(e) => {
                self.model.inner.lock().unwrap().poisoned = Some(e.to_string());
                return Err(e);
            }
        };
        if status != ggml_status::GGML_STATUS_SUCCESS {
            return Err(Error::Backend(format!("training graph compute failed: status {status:?}")));
        }
        let mut loss = 0f32;
        guard::tensor_get(self.loss, unsafe { std::slice::from_raw_parts_mut((&mut loss as *mut f32).cast::<u8>(), 4) })?;
        Ok(loss)
    }

    /// AdamW parameters that leave the weights and the moments untouched (alpha 0, beta 1, no bias
    /// correction), for the micro-batches of an accumulation period that don't end it. One graph
    /// serves every micro-batch: switching between a gradient-only and an optimizer graph makes
    /// ggml's scheduler re-reserve its buffers, which stalls on the CUDA backend.
    const SKIP_STEP: [f32; 7] = [0.0, 1.0, 1.0, 1e-8, 0.0, 1.0, 1.0];

    /// Forward and backward of one micro-batch without touching the parameters: returns the loss;
    /// the gradients are then readable with [`Trainer::grad`]. Resets the accumulation period.
    pub fn eval_grads(&mut self, inputs: &[(&str, ArrayViewD<f32>)]) -> Result<f32> {
        let model = self.model.clone();
        let _lock = model.inner.lock().unwrap();
        self.micro = 0;
        let g = self.gb_opt;
        self.alloc(g)?;
        unsafe { ggml_graph_reset(self.gb_grad) };
        write_f32(self.adamw, &Self::SKIP_STEP)?;
        self.set_inputs(inputs)?;
        self.compute(g)
    }

    /// One micro-batch: forward, backward, and, every `accum_steps`-th call, the AdamW update.
    pub fn step(&mut self, inputs: &[(&str, ArrayViewD<f32>)]) -> Result<StepStats> {
        let model = self.model.clone();
        let _lock = model.inner.lock().unwrap();
        let last = self.micro + 1 == self.opts.accum_steps;
        let g = self.gb_opt;
        trace(&format!("step micro {} last {last}: alloc", self.micro));
        self.alloc(g)?;
        trace("allocated");
        if self.micro == 0 {
            // Zero the accumulators (gb_grad has no optimizer nodes, so m and v stay) and start the
            // backward pass at 1 / accum_steps: the update uses the mean gradient.
            unsafe { ggml_graph_reset(self.gb_grad) };
            if self.opts.accum_steps > 1 {
                write_f32(self.loss_grad, &[1.0 / self.opts.accum_steps as f32])?;
            }
        }
        trace("reset done");
        self.set_inputs(inputs)?;
        trace("inputs set");
        if last {
            let o = &self.opts;
            let (b1h, b2h) = (1.0 / (1.0 - o.beta1.powi(self.iter as i32)), 1.0 / (1.0 - o.beta2.powi(self.iter as i32)));
            write_f32(self.adamw, &[o.lr, o.beta1, o.beta2, o.eps, o.weight_decay, b1h, b2h])?;
        } else {
            write_f32(self.adamw, &Self::SKIP_STEP)?;
        }
        trace("computing");
        let loss = self.compute(g)?;
        trace("computed");
        if last {
            self.iter += 1;
            self.micro = 0;
        } else {
            self.micro += 1;
        }
        Ok(StepStats { loss, optimized: last })
    }

    /// Current value of every parameter.
    pub fn params_values(&self) -> Result<HashMap<String, ArrayD<f32>>> {
        let _inner = self.model.inner.lock().unwrap();
        self.params.iter().map(|p| Ok((p.name.clone(), read_f32(p.tensor)?))).collect()
    }
}

impl Drop for Trainer {
    fn drop(&mut self) {
        let inner = self.model.inner.lock().unwrap();
        if inner.poisoned.is_some() {
            return;
        }
        unsafe {
            ggml_backend_sched_free(self.sched);
            ggml_backend_buffer_free(self.buf_static);
            ggml_backend_buffer_free(self.buf_cpu);
            ggml_free(self.ctx_static);
            ggml_free(self.ctx_cpu);
            ggml_free(self.graph.ctx);
            for p in &self.params {
                // Leave the parameter tensors usable by the inference graphs.
                (*p.tensor).flags &= !(ggml_tensor_flag::GGML_TENSOR_FLAG_PARAM as i32);
            }
        }
    }
}
