//! The Scheme side of the runtime.
//!
//! Chez is a single, process-global runtime tied to the thread that booted it,
//! so it lives on a dedicated thread and models send it jobs. Scheme is only
//! needed while a program is loaded and while a graph is built; computing a
//! built graph never touches it.
use std::{
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
};

use chez::{Scheme, Value};
use ggml_sys::ffi::{ggml_tensor, scheme as generated};

use crate::error::{Error, Result};

const CORE: &str = include_str!("core.ss");

/// Names exported to programs by `(tensorlisp)`, besides generated ops and constants.
const PUBLIC: &str = "tensor? shape strides dtype weight weight? model inputs outputs tap \
    preprocess model-inputs host? asset tokenizer tokenize \
    image-size image-resize image-resize-shortest image-resize-longest image-resize-multiple image-center-crop image->array \
    audio-rate audio-length audio-resample audio-pad audio->array log-mel whisper-features \
    array-shape array-affine array-reshape \
    postprocess results pipeline run output detokenize token-id define-state state effect device array-length array->list list->array array-slice array-transpose array-take array-argmax \
    image-letterbox boxes-convert nms detect boxes-scale boxes-clip boxes-unletterbox dbscan cluster-centroids";
/// Host entry points, only visible from Rust.
const HOST: &str = "$tl-load-program $tl-build $tl-unload $tl-abort-handler $tl-preprocess $tl-postprocess $tl-pipeline";

extern "C" fn tl_tensor_ne(t: *const ggml_tensor, i: i32) -> i64 {
    unsafe { (*t).ne[i as usize] }
}

extern "C" fn tl_tensor_nb(t: *const ggml_tensor, i: i32) -> usize {
    unsafe { (*t).nb[i as usize] }
}

extern "C" fn tl_tensor_type(t: *const ggml_tensor) -> i32 {
    unsafe { (*t).type_ as i32 }
}

fn library_source() -> String {
    let public = format!("{PUBLIC} {} {}", generated::OP_NAMES, generated::CONSTANT_NAMES);
    format!(
        "(library (tensorlisp runtime)\n\
           (export {public} {HOST})\n\
           (import (chezscheme))\n\
           {raw}\n{constants}\n{CORE}\n{ops})\n\
         (library (tensorlisp)\n\
           (export {public})\n\
           (import (tensorlisp runtime)))\n\
         (import (tensorlisp runtime))\n",
        raw = generated::RAW,
        constants = generated::CONSTANTS,
        ops = generated::OPS,
    )
}

fn boot() -> Result<Scheme> {
    let scheme = Scheme::new()?;
    unsafe {
        for (name, addr) in generated::symbols() {
            scheme.register_foreign(name, addr)?;
        }
        scheme.register_foreign("tl_tensor_ne", tl_tensor_ne as *const _)?;
        scheme.register_foreign("tl_tensor_type", tl_tensor_type as *const _)?;
        scheme.register_foreign("tl_tensor_nb", tl_tensor_nb as *const _)?;
        for (name, addr) in crate::host::symbols() {
            scheme.register_foreign(name, addr)?;
        }
    }
    scheme.eval(&library_source())?;

    // ggml assertions on this thread raise Scheme errors (see core.ss).
    crate::guard::install();
    let Value::Int(handler) = scheme.call("$tl-abort-handler", &[])? else {
        return Err(Error::Program("abort handler has no entry point".into()));
    };
    unsafe {
        let handler: unsafe extern "C" fn(*const std::ffi::c_char) = std::mem::transmute(handler as usize);
        ggml_sys::ffi::tl_guard_set_thread_handler(Some(handler));
    }
    Ok(scheme)
}

type Job = Box<dyn FnOnce(&Scheme) + Send>;

thread_local! {
    /// Set on the Scheme thread: the runtime, for calls made from inside a
    /// Scheme call (a pipeline running an entry builds its graph).
    static CURRENT: std::cell::Cell<*const Scheme> = const { std::cell::Cell::new(std::ptr::null()) };
}

struct SchemeThread {
    jobs: mpsc::Sender<Job>,
}

fn scheme_thread() -> Result<&'static SchemeThread> {
    static THREAD: OnceLock<Result<SchemeThread, String>> = OnceLock::new();
    THREAD
        .get_or_init(|| {
            let (jobs, rx) = mpsc::channel::<Job>();
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            thread::Builder::new()
                .name("tensorlisp-scheme".into())
                .stack_size(16 << 20)
                .spawn(move || match boot() {
                    Ok(scheme) => {
                        CURRENT.with(|c| c.set(&scheme));
                        ready_tx.send(Ok(())).unwrap();
                        for job in rx {
                            job(&scheme);
                        }
                    }
                    Err(e) => ready_tx.send(Err(e.to_string())).unwrap(),
                })
                .map_err(|e| e.to_string())?;
            ready_rx.recv().map_err(|e| e.to_string())??;
            Ok(SchemeThread { jobs })
        })
        .as_ref()
        .map_err(|e| Error::Scheme(chez::Error::Scheme(format!("failed to start Chez: {e}"))))
}

/// Runs `f` on the Scheme thread and waits for its result (directly, when
/// called on the Scheme thread).
fn with_scheme<R: Send + 'static>(f: impl FnOnce(&Scheme) -> R + Send + 'static) -> Result<R> {
    let current = CURRENT.with(|c| c.get());
    if !current.is_null() {
        // Only set on the Scheme thread, whose loop keeps the runtime alive.
        return Ok(f(unsafe { &*current }));
    }
    let (tx, rx) = mpsc::sync_channel(1);
    scheme_thread()?
        .jobs
        .send(Box::new(move |scheme| {
            let _ = tx.send(f(scheme));
        }))
        .map_err(|_| Error::Backend("the Scheme thread has stopped".into()))?;
    rx.recv().map_err(|_| Error::Backend("the Scheme thread has stopped".into()))
}

/// An input as declared by `(model (inputs ...))`.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSpec {
    pub name: String,
    pub dtype: String,
    /// ggml order; `None` entries accept any size.
    pub dims: Option<Vec<Option<i64>>>,
}

/// A graph built by the program, ready to be allocated.
pub(crate) struct BuiltGraph {
    pub graph: *mut ggml_sys::ffi::ggml_cgraph,
    pub inputs: Vec<*mut ggml_tensor>,
    /// Name, contiguous f32 tensor, rank.
    pub outputs: Vec<(String, *mut ggml_tensor, usize)>,
    /// Requested taps, like outputs.
    pub taps: Vec<(String, *mut ggml_tensor, usize)>,
    /// Every tap the program defines.
    pub tap_names: Vec<String>,
}

/// Which taps to read back.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum Taps {
    #[default]
    None,
    All,
    Names(Vec<String>),
}

/// A proper list; Scheme's empty list arrives as `Value::Nil`.
fn as_list(value: &Value) -> Option<&[Value]> {
    match value {
        Value::List(items) => Some(items),
        Value::Nil => Some(&[]),
        _ => None,
    }
}

fn bad_reply(what: &str, value: &Value) -> Error {
    Error::Program(format!("unexpected reply from {what}: {value:?}"))
}

/// Kind of a raw (pre-preprocessing) input declared by `(preprocess ...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawKind {
    Text,
    Image,
    Audio,
    Array,
}

/// A raw input declared by `(preprocess ([name kind] ...) ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSpec {
    pub name: String,
    pub kind: RawKind,
}

/// A raw input value for pre- or postprocessing.
#[derive(Clone)]
pub enum RawValue {
    Text(String),
    Image(image::RgbImage),
    Audio(autopro::audio::Audio),
    Array(ndarray::ArrayD<f32>),
}

/// A model entry: `(model name (inputs ...) ...)`; an unnamed model is `main`.
#[derive(Debug, Clone, PartialEq)]
pub struct EntrySpec {
    pub name: String,
    pub inputs: Vec<InputSpec>,
}

/// A pipeline: `(pipeline name ([raw kind] ...) ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineSpec {
    pub name: String,
    pub raw_inputs: Vec<RawSpec>,
}

/// A state tensor: `(define-state name type dim ...)`, dims in ggml order.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSpec {
    pub name: String,
    /// "f32" or "f16".
    pub dtype: String,
    pub dims: Vec<i64>,
}

/// A value from `(results ...)`.
pub(crate) enum ResultValue {
    Array(ndarray::ArrayD<f32>),
    Text(String),
}

/// A program evaluated on the Scheme thread, identified by `id`.
pub(crate) struct LoadedProgram {
    id: i64,
    /// Default entry first.
    pub entries: Vec<EntrySpec>,
    pub pipelines: Vec<PipelineSpec>,
    pub states: Vec<StateSpec>,
    pub raw_inputs: Option<Vec<RawSpec>>,
    /// Arguments of `(postprocess ...)`: names of outputs or raw inputs.
    pub post_args: Option<Vec<String>>,
}

/// An argument for `(postprocess ...)`.
pub(crate) enum PostValue {
    Array(ndarray::ArrayD<f32>),
    Raw(RawValue),
}

/// Parses `(results ...)` as exported by `%export-results`.
fn parse_results(who: &str, reply: &Value) -> Result<Vec<(String, ResultValue)>> {
    use crate::host::{clone_tensor, take_tensor};
    let bad = || bad_reply(who, reply);
    let number = |v: &Value| match v {
        Value::Float(f) => Ok(*f as f32),
        Value::Int(i) => Ok(*i as f32),
        _ => Err(bad()),
    };
    let entries = as_list(reply).ok_or_else(bad)?;
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let Some([Value::String(name), Value::String(kind), payload]) = as_list(entry) else {
                return Err(bad());
            };
            let value = match (kind.as_str(), payload) {
                ("array", Value::Int(h)) => {
                    // The same array may be listed twice: only the last use takes it.
                    let later = entries[i + 1..].iter().any(|e| {
                        matches!(as_list(e), Some([_, Value::String(k), Value::Int(o)]) if k == "array" && o == h)
                    });
                    let array = if later { clone_tensor(*h) } else { take_tensor(*h) };
                    ResultValue::Array(array.ok_or_else(|| Error::Program(format!("{who}: array for {name} is gone")))?)
                }
                ("scalar", v) => ResultValue::Array(ndarray::arr0(number(v)?).into_dyn()),
                ("vector", v) => ResultValue::Array(
                    ndarray::Array1::from(as_list(v).ok_or_else(bad)?.iter().map(number).collect::<Result<Vec<_>>>()?)
                        .into_dyn(),
                ),
                ("text", Value::String(t)) => ResultValue::Text(t.clone()),
                _ => return Err(bad()),
            };
            Ok((name.clone(), value))
        })
        .collect()
}

/// A raw value as a `$tl-preprocess` / `$tl-postprocess` argument: a string or (kind . host-id).
fn raw_argument(raw: RawValue) -> Value {
    use crate::host::{HostValue, insert};
    let (kind, value) = match raw {
        RawValue::Text(t) => return Value::String(t),
        RawValue::Image(i) => ("image", HostValue::Image(i)),
        RawValue::Audio(a) => ("audio", HostValue::Audio(a)),
        RawValue::Array(t) => ("tensor", HostValue::Tensor(t)),
    };
    Value::Pair(Box::new(Value::Symbol(kind.into())), Box::new(Value::Int(insert(value))))
}

impl LoadedProgram {
    /// Evaluates `text`; `assets` are available to it through `(asset name)`.
    pub fn load(text: &str, assets: Vec<(String, Vec<u8>)>) -> Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed) as i64;
        let text = text.to_owned();
        let reply = with_scheme(move |s| {
            crate::host::in_scope(crate::host::Scope::Model(id), || {
                let assets = Value::List(
                    assets
                        .into_iter()
                        .map(|(name, bytes)| {
                            let handle = crate::host::insert(crate::host::HostValue::Bytes(std::sync::Arc::new(bytes)));
                            Value::Pair(Box::new(Value::String(name)), Box::new(Value::Int(handle)))
                        })
                        .collect(),
                );
                s.call("$tl-load-program", &[Value::Int(id), Value::String(text), assets])
            })
        })?;
        let reply = match reply {
            Ok(reply) => reply,
            Err(e) => {
                crate::host::drop_model(id);
                return Err(e.into());
            }
        };
        let bad = || bad_reply("load", &reply);
        let Some([entries, raw, post, pipelines, states]) = as_list(&reply) else { return Err(bad()) };
        let parse_inputs = |specs: &Value| -> Result<Vec<InputSpec>> {
            as_list(specs)
                .ok_or_else(bad)?
                .iter()
                .map(|spec| match as_list(spec) {
                    Some([Value::String(name), Value::String(dtype), dims]) => Ok(InputSpec {
                        name: name.clone(),
                        dtype: dtype.clone(),
                        dims: match dims {
                            Value::Bool(false) => None,
                            other => Some(
                                as_list(other)
                                    .ok_or_else(bad)?
                                    .iter()
                                    .map(|d| if let Value::Int(n) = d { Some(*n) } else { None })
                                    .collect(),
                            ),
                        },
                    }),
                    _ => Err(bad()),
                })
                .collect::<Result<_>>()
        };
        let entries = as_list(entries)
            .ok_or_else(bad)?
            .iter()
            .map(|e| match as_list(e) {
                Some([Value::String(name), inputs]) => Ok(EntrySpec { name: name.clone(), inputs: parse_inputs(inputs)? }),
                _ => Err(bad()),
            })
            .collect::<Result<Vec<_>>>()?;
        let parse_raw = |specs: &Value| -> Result<Vec<RawSpec>> {
            as_list(specs)
                .ok_or_else(bad)?
                .iter()
                .map(|spec| match as_list(spec) {
                    Some([Value::String(name), Value::String(kind)]) => Ok(RawSpec {
                        name: name.clone(),
                        kind: match kind.as_str() {
                            "string" => RawKind::Text,
                            "image" => RawKind::Image,
                            "audio" => RawKind::Audio,
                            _ => RawKind::Array,
                        },
                    }),
                    _ => Err(bad()),
                })
                .collect()
        };
        let pipelines = as_list(pipelines)
            .ok_or_else(bad)?
            .iter()
            .map(|p| match as_list(p) {
                Some([Value::String(name), raw]) => Ok(PipelineSpec { name: name.clone(), raw_inputs: parse_raw(raw)? }),
                _ => Err(bad()),
            })
            .collect::<Result<Vec<_>>>()?;
        let raw_inputs = match raw {
            Value::Bool(false) => None,
            other => Some(parse_raw(other)?),
        };
        let post_args = match post {
            Value::Bool(false) => None,
            other => Some(
                as_list(other)
                    .ok_or_else(bad)?
                    .iter()
                    .map(|v| if let Value::String(n) = v { Ok(n.clone()) } else { Err(bad()) })
                    .collect::<Result<_>>()?,
            ),
        };
        let states = as_list(states)
            .ok_or_else(bad)?
            .iter()
            .map(|st| match as_list(st) {
                Some([Value::String(name), Value::String(dtype), dims]) => Ok(StateSpec {
                    name: name.clone(),
                    dtype: dtype.clone(),
                    dims: as_list(dims)
                        .ok_or_else(bad)?
                        .iter()
                        .map(|d| if let Value::Int(n) = d { Ok(*n) } else { Err(bad()) })
                        .collect::<Result<_>>()?,
                }),
                _ => Err(bad()),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LoadedProgram { id, entries, pipelines, states, raw_inputs, post_args })
    }

    /// Runs `(preprocess ...)` on one example (values in raw-input order).
    /// Returns one array per model input, in input order.
    pub fn preprocess(&self, raws: Vec<RawValue>) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        use crate::host::{Scope, clear_calls, in_scope, take_tensor};
        let id = self.id;
        with_scheme(move |s| {
            let result = in_scope(Scope::Call, || -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
                let args: Vec<Value> = raws.into_iter().map(raw_argument).collect();
                let reply = s.call("$tl-preprocess", &[Value::Int(id), Value::List(args)])?;
                let bad = || bad_reply("preprocess", &reply);
                as_list(&reply)
                    .ok_or_else(bad)?
                    .iter()
                    .map(|entry| match as_list(entry) {
                        Some([Value::String(name), Value::Int(array)]) => {
                            let array = take_tensor(*array)
                                .ok_or_else(|| Error::Program(format!("preprocess: array for {name} is gone")))?;
                            Ok((name.clone(), array))
                        }
                        _ => Err(bad()),
                    })
                    .collect()
            });
            clear_calls();
            result
        })?
    }

    pub fn id(&self) -> i64 {
        self.id
    }

    /// Runs `(postprocess ...)` on one example (values in argument order).
    /// Returns the named results in the order the program lists them.
    pub fn postprocess(&self, args: Vec<PostValue>) -> Result<Vec<(String, ResultValue)>> {
        use crate::host::{HostValue, Scope, clear_calls, in_scope, insert};
        let id = self.id;
        with_scheme(move |s| {
            let result = in_scope(Scope::Call, || {
                let args: Vec<Value> = args
                    .into_iter()
                    .map(|arg| match arg {
                        PostValue::Array(a) => Value::Pair(
                            Box::new(Value::Symbol("tensor".into())),
                            Box::new(Value::Int(insert(HostValue::Tensor(a)))),
                        ),
                        PostValue::Raw(raw) => raw_argument(raw),
                    })
                    .collect();
                let reply = s.call("$tl-postprocess", &[Value::Int(id), Value::List(args)])?;
                parse_results("postprocess", &reply)
            });
            clear_calls();
            result
        })?
    }

    /// Runs pipeline `name` of program `id` on its raw inputs (in declared
    /// order). Called without holding the model's lock: the pipeline runs
    /// entries, which take it.
    pub fn run_pipeline(id: i64, name: &str, raws: Vec<RawValue>) -> Result<Vec<(String, ResultValue)>> {
        use crate::host::{Scope, clear_calls, in_scope};
        let name = name.to_string();
        with_scheme(move |s| {
            let result = in_scope(Scope::Call, || {
                let args: Vec<Value> = raws.into_iter().map(raw_argument).collect();
                let reply = s.call("$tl-pipeline", &[Value::Int(id), Value::String(name), Value::List(args)])?;
                parse_results("pipeline", &reply)
            });
            clear_calls();
            result
        })?
    }

    /// Runs the model body in `ctx`, with inputs of the given ggml dims.
    pub fn build(
        &self,
        entry: &str,
        ctx: *mut ggml_sys::ffi::ggml_context,
        weights: *mut ggml_sys::ffi::ggml_context,
        states: *mut ggml_sys::ffi::ggml_context,
        device: &str,
        input_dims: &[Vec<i64>],
        graph_size: usize,
        taps: &Taps,
    ) -> Result<BuiltGraph> {
        let args = vec![
            Value::Int(self.id),
            Value::String(entry.to_string()),
            Value::Int(ctx as i64),
            Value::Int(weights as i64),
            Value::Int(states as i64),
            Value::String(device.to_string()),
            Value::List(
                input_dims.iter().map(|ne| Value::List(ne.iter().map(|&d| Value::Int(d)).collect())).collect(),
            ),
            Value::Int(graph_size as i64),
            match taps {
                Taps::None => Value::Nil,
                Taps::All => Value::Bool(true),
                Taps::Names(names) => Value::List(names.iter().map(|n| Value::String(n.clone())).collect()),
            },
        ];
        let reply = with_scheme(move |s| s.call("$tl-build", &args))??;
        let bad = || bad_reply("build", &reply);
        let Some([Value::Int(graph), inputs, outputs, taps, tap_names]) = as_list(&reply) else {
            return Err(bad());
        };
        let (Some(inputs), Some(outputs), Some(taps), Some(tap_names)) =
            (as_list(inputs), as_list(outputs), as_list(taps), as_list(tap_names))
        else {
            return Err(bad());
        };
        let inputs = inputs
            .iter()
            .map(|v| if let Value::Int(p) = v { Ok(*p as *mut ggml_tensor) } else { Err(bad()) })
            .collect::<Result<_>>()?;
        let results = |values: &[Value]| {
            values
                .iter()
                .map(|out| match out {
                    Value::List(fields) => match fields.as_slice() {
                        [Value::String(name), Value::Int(t), Value::Int(rank)] => {
                            Ok((name.clone(), *t as *mut ggml_tensor, *rank as usize))
                        }
                        _ => Err(bad()),
                    },
                    _ => Err(bad()),
                })
                .collect::<Result<Vec<_>>>()
        };
        let tap_names = tap_names
            .iter()
            .map(|v| if let Value::String(n) = v { Ok(n.clone()) } else { Err(bad()) })
            .collect::<Result<_>>()?;
        Ok(BuiltGraph {
            graph: *graph as *mut _,
            inputs,
            outputs: results(outputs)?,
            taps: results(taps)?,
            tap_names,
        })
    }
}

impl Drop for LoadedProgram {
    fn drop(&mut self) {
        let id = self.id;
        let _ = with_scheme(move |s| s.call("$tl-unload", &[Value::Int(id)]));
        crate::host::drop_model(id);
    }
}
