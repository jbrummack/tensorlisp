//! The native Metal device: tensorlisp's own executor instead of a ggml
//! backend.
//!
//! The Scheme program still builds a ggml graph (ggml is the graph IR and
//! shape inference), but nothing is ever allocated or computed by ggml: the
//! graph is lowered to `tensorlisp_kernels`' IR, planned into one arena,
//! compiled to a recorded list of Metal dispatches (matrix multiplication
//! chosen by the Scheme policy in `scheme/native.ss`) and replayed per run.

use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{Arc, Mutex},
};

use chez::Value;
use ggml_sys::ffi::*;
use tensorlisp_kernels::{
    Const,
    metal::{
        Buffer, Device as KDevice,
        exec::{Executor, Program, Shared},
        lower::{MatmulLowering, MatmulQuery, Spec, SpecArg, TensorFacts, Which},
        kargs::{Field, pack_checked},
    },
    plan::{Leaves, align},
};

use crate::{
    error::{Error, Result},
    gguf::GgufFile,
    native_ir::{backend_err, lower_graph, read_results, Lowered, Results},
    scheme,
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
    matmul: ScmMatmul,
}

impl Native {
    /// Creates the Metal device and loads every weight tensor of `file` from `path`.
    pub fn load(file: &GgufFile, path: &Path) -> Result<Native> {
        let dev = Arc::new(KDevice::system_default().map_err(backend_err)?);
        let max = dev.max_buffer_len() as u64;
        let exec = Executor::new(dev.clone()).map_err(backend_err)?;

        // Lay the weights out in as few buffers as the device allows.
        let n = file.tensor_count();
        let mut weight_index = HashMap::new();
        let mut weight_loc = Vec::new();
        let mut sizes: Vec<u64> = vec![0];
        let mut tensors = Vec::new();
        for i in 0..n {
            let name = file.tensor_name(i);
            let t = file.tensor(name)?.ok_or_else(|| Error::Gguf(format!("tensor {name} missing from context")))?;
            let bytes = unsafe { ggml_nbytes(t) } as u64;
            let mut buf = sizes.len() - 1;
            if sizes[buf] > 0 && sizes[buf] + align(bytes) > max {
                sizes.push(0);
                buf += 1;
            }
            let off = sizes[buf];
            sizes[buf] += align(bytes.max(1));
            weight_index.insert(t as usize, weight_loc.len());
            weight_loc.push((buf, off));
            tensors.push((i, t, bytes));
        }
        let weights = sizes.iter().map(|&s| dev.alloc(s as usize)).collect::<std::result::Result<Vec<_>, _>>().map_err(backend_err)?;

        let data = File::open(path)?;
        for ((i, _, bytes), &(buf, off)) in tensors.iter().zip(&weight_loc) {
            // SAFETY: the range was allocated above; shared storage is CPU-visible.
            let dst = unsafe { std::slice::from_raw_parts_mut(weights[buf].contents().add(off as usize), *bytes as usize) };
            crate::gguf::read_exact_at(&data, dst, file.tensor_file_offset(*i) as u64)?;
        }

        let state = dev.alloc(1).map_err(backend_err)?;
        Ok(Native {
            exec,
            weights,
            weight_index,
            weight_loc,
            state,
            state_index: HashMap::new(),
            state_offsets: Vec::new(),
            matmul: ScmMatmul::default(),
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
        let buf = self.exec.dev.alloc(total as usize).map_err(backend_err)?;
        buf.write(&vec![0u8; total as usize]);
        self.state = buf;
        Ok(())
    }

    pub fn reset_state(&self) {
        self.state.write(&vec![0u8; self.state.len()]);
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
        let program = native.exec.compile(ir, &leaves, &native.matmul).map_err(backend_err)?;
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

// ---------------------------------------------------------------------------
// The matmul policy lives in Scheme (scheme/native.ss).
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ScmMatmul {
    cache: Mutex<HashMap<String, Vec<Spec>>>,
}

fn tensor_value(t: &TensorFacts) -> Value {
    Value::List(vec![
        Value::String(t.ty.name().to_string()),
        Value::List(t.ne.iter().map(|&n| Value::Int(n)).collect()),
        Value::List(t.nb.iter().map(|&n| Value::Int(n as i64)).collect()),
    ])
}

impl MatmulLowering for ScmMatmul {
    fn lower(&self, q: &MatmulQuery) -> tensorlisp_kernels::Result<Vec<Spec>> {
        let key = format!("{:?}", (&q.src0, &q.src1, &q.dst, q.props.simdgroup_mm));
        if let Some(hit) = self.cache.lock().unwrap().get(&key) {
            return Ok(hit.clone());
        }
        let args = vec![
            tensor_value(&q.src0),
            tensor_value(&q.src1),
            tensor_value(&q.dst),
            Value::List(vec![Value::Bool(q.props.simdgroup_mm), Value::Int(q.props.max_threadgroup_memory as i64)]),
        ];
        let reply = scheme::native_lower_mul_mat(args)
            .map_err(|e| tensorlisp_kernels::Error::Invalid(format!("matmul policy: {e}")))?;
        let specs = parse_specs(&reply).map_err(tensorlisp_kernels::Error::Invalid)?;
        self.cache.lock().unwrap().insert(key, specs.clone());
        Ok(specs)
    }
}

fn items(v: &Value) -> std::result::Result<&[Value], String> {
    match v {
        Value::List(l) => Ok(l),
        Value::Nil => Ok(&[]),
        other => Err(format!("expected a list, got {other:?}")),
    }
}

fn int(v: &Value) -> std::result::Result<i64, String> {
    match v {
        Value::Int(i) => Ok(*i),
        other => Err(format!("expected an integer, got {other:?}")),
    }
}

fn sym(v: &Value) -> std::result::Result<&str, String> {
    match v {
        Value::Symbol(s) | Value::String(s) => Ok(s),
        other => Err(format!("expected a symbol, got {other:?}")),
    }
}

fn triple(v: &Value) -> std::result::Result<[u32; 3], String> {
    let l = items(v)?;
    if l.len() != 3 {
        return Err(format!("expected 3 numbers, got {l:?}"));
    }
    Ok([int(&l[0])? as u32, int(&l[1])? as u32, int(&l[2])? as u32])
}

fn parse_specs(reply: &Value) -> std::result::Result<Vec<Spec>, String> {
    items(reply)?
        .iter()
        .map(|d| {
            let d = items(d)?;
            if d.len() != 6 {
                return Err(format!("a dispatch has 6 parts, got {}", d.len()));
            }
            let kernel = sym(&d[0])?.to_string();
            let consts = items(&d[1])?
                .iter()
                .map(|c| {
                    let c = items(c)?;
                    let idx = int(&c[0])? as u32;
                    Ok((
                        idx,
                        match (sym(&c[1])?, &c[2]) {
                            ("bool", Value::Bool(b)) => Const::Bool(*b),
                            ("i16", v) => Const::I16(int(v)? as i16),
                            ("i32", v) => Const::I32(int(v)? as i32),
                            (k, v) => return Err(format!("bad function constant ({k} {v:?})")),
                        },
                    ))
                })
                .collect::<std::result::Result<Vec<_>, String>>()?;
            let args = items(&d[2])?
                .iter()
                .map(|a| {
                    let a = items(a)?;
                    let slot = int(&a[0])? as u32;
                    match sym(&a[1])? {
                        "tensor" => Ok(SpecArg::Tensor {
                            slot,
                            which: match sym(&a[2])? {
                                "src0" => Which::Src0,
                                "src1" => Which::Src1,
                                "dst" => Which::Dst,
                                other => return Err(format!("unknown tensor argument {other}")),
                            },
                        }),
                        "bytes" => {
                            let name = sym(&a[2])?;
                            let fields = a[3..]
                                .iter()
                                .map(|f| {
                                    let f = items(f)?;
                                    let v = &f[1];
                                    Ok(match (sym(&f[0])?, v) {
                                        ("bool", Value::Bool(b)) => Field::Bool(*b),
                                        ("i16", v) => Field::I16(int(v)? as i16),
                                        ("i32", v) => Field::I32(int(v)? as i32),
                                        ("u32", v) => Field::U32(int(v)? as u32),
                                        ("i64", v) => Field::I64(int(v)?),
                                        ("u64", v) => Field::U64(int(v)? as u64),
                                        ("f32", Value::Float(x)) => Field::F32(*x as f32),
                                        ("f32", Value::Int(x)) => Field::F32(*x as f32),
                                        (k, v) => return Err(format!("bad kernel-argument field ({k} {v:?})")),
                                    })
                                })
                                .collect::<std::result::Result<Vec<_>, String>>()?;
                            let data = pack_checked(name, &fields).map_err(|e| e.to_string())?;
                            Ok(SpecArg::Bytes { slot, data })
                        }
                        other => Err(format!("unknown argument kind {other}")),
                    }
                })
                .collect::<std::result::Result<Vec<_>, String>>()?;
            Ok(Spec { kernel, consts, args, grid: triple(&d[3])?, threads: triple(&d[4])?, smem: int(&d[5])? as u32 })
        })
        .collect()
}
