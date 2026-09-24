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
    array-shape array-affine array-reshape";
/// Host entry points, only visible from Rust.
const HOST: &str = "$tl-load-program $tl-build $tl-unload $tl-abort-handler $tl-preprocess";

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

/// Runs `f` on the Scheme thread and waits for its result.
fn with_scheme<R: Send + 'static>(f: impl FnOnce(&Scheme) -> R + Send + 'static) -> Result<R> {
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

/// A raw input value for preprocessing.
pub enum RawValue {
    Text(String),
    Image(image::RgbImage),
    Audio(autopro::audio::Audio),
    Array(ndarray::ArrayD<f32>),
}

/// A program evaluated on the Scheme thread, identified by `id`.
pub(crate) struct LoadedProgram {
    id: i64,
    pub inputs: Vec<InputSpec>,
    pub raw_inputs: Option<Vec<RawSpec>>,
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
        let Some([specs, raw]) = as_list(&reply) else { return Err(bad()) };
        let Some(specs) = as_list(specs) else { return Err(bad()) };
        let inputs = specs
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
            .collect::<Result<_>>()?;
        let raw_inputs = match raw {
            Value::Bool(false) => None,
            other => Some(
                as_list(other)
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
                    .collect::<Result<_>>()?,
            ),
        };
        Ok(LoadedProgram { id, inputs, raw_inputs })
    }

    /// Runs `(preprocess ...)` on one example (values in raw-input order).
    /// Returns one array per model input, in input order.
    pub fn preprocess(&self, raws: Vec<RawValue>) -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
        use crate::host::{HostValue, Scope, clear_calls, in_scope, insert, take_tensor};
        let id = self.id;
        with_scheme(move |s| {
            let result = in_scope(Scope::Call, || -> Result<Vec<(String, ndarray::ArrayD<f32>)>> {
                let args: Vec<Value> = raws
                    .into_iter()
                    .map(|raw| {
                        let (kind, value) = match raw {
                            RawValue::Text(t) => return Value::String(t),
                            RawValue::Image(i) => ("image", HostValue::Image(i)),
                            RawValue::Audio(a) => ("audio", HostValue::Audio(a)),
                            RawValue::Array(t) => ("tensor", HostValue::Tensor(t)),
                        };
                        Value::Pair(Box::new(Value::Symbol(kind.into())), Box::new(Value::Int(insert(value))))
                    })
                    .collect();
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

    /// Runs the model body in `ctx`, with inputs of the given ggml dims.
    pub fn build(
        &self,
        ctx: *mut ggml_sys::ffi::ggml_context,
        weights: *mut ggml_sys::ffi::ggml_context,
        input_dims: &[Vec<i64>],
        graph_size: usize,
        taps: &Taps,
    ) -> Result<BuiltGraph> {
        let args = vec![
            Value::Int(self.id),
            Value::Int(ctx as i64),
            Value::Int(weights as i64),
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
