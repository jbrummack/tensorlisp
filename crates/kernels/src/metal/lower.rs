//! Lowers graph nodes to Metal dispatches: a port of the host side of
//! ggml-metal (`ggml-metal-ops.cpp` and the pipeline getters of
//! `ggml-metal-device.cpp`) onto the IR, reusing ggml's kernels and argument
//! structs unchanged. What ggml decides per op at encode time (which kernel
//! variant, specialization constants, threadgroup shapes) is decided here, once,
//! at plan time. No fusion: each node is its own dispatch.
//!
//! Matrix multiplication is the exception: its policy is supplied by a
//! [`MatmulLowering`], implemented in Scheme (see `tensorlisp`).

use std::sync::Arc;

use super::kargs::{bytes_of, ffi::*};
use super::{Device, Kernel, Pipeline};
use crate::graph::{Graph, Tensor, TensorId, Ty};
use crate::plan::{Layout, Place};
use crate::{Const, Dims, Error, Result};

/// One kernel launch with its arguments resolved to places.
pub struct Dispatch {
    pub pipeline: Pipeline,
    pub args: Vec<(u32, DArg)>,
    pub dims: Dims,
    /// For error messages and debugging: the node's op and name.
    pub label: String,
}

pub enum DArg {
    Buf(Place),
    Bytes(Vec<u8>),
}

/// What the lowering knows about the GPU.
#[derive(Clone, Copy, Debug)]
pub struct DeviceProps {
    /// simdgroup matrix multiply (Apple7+ / M1+).
    pub simdgroup_mm: bool,
    pub max_threadgroup_memory: u32,
}

// ---------------------------------------------------------------------------
// Data description of a lowering, for policies written outside Rust (Scheme).
// ---------------------------------------------------------------------------

/// A tensor argument of a [`Spec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Src0,
    Src1,
    Dst,
}

#[derive(Clone)]
pub enum SpecArg {
    Tensor { slot: u32, which: Which },
    Bytes { slot: u32, data: Vec<u8> },
}

/// A dispatch described by value.
#[derive(Clone)]
pub struct Spec {
    pub kernel: String,
    pub consts: Vec<(u32, Const)>,
    pub args: Vec<SpecArg>,
    pub grid: [u32; 3],
    pub threads: [u32; 3],
    pub smem: u32,
}

/// The facts a matmul policy decides from.
#[derive(Clone, Debug)]
pub struct TensorFacts {
    pub ty: Ty,
    pub ne: [i64; 4],
    pub nb: [u64; 4],
}

#[derive(Clone, Debug)]
pub struct MatmulQuery {
    pub src0: TensorFacts,
    pub src1: TensorFacts,
    pub dst: TensorFacts,
    pub props: DeviceProps,
}

pub trait MatmulLowering: Send + Sync {
    fn lower(&self, q: &MatmulQuery) -> Result<Vec<Spec>>;
}

// ---------------------------------------------------------------------------

pub struct Ctx<'a> {
    pub dev: &'a Device,
    pub lib: Arc<str>,
    /// Our own kernels (shaders/extra.metal).
    pub extra: Arc<str>,
    pub graph: &'a Graph,
    pub layout: &'a Layout,
    pub matmul: &'a dyn MatmulLowering,
    pub props: DeviceProps,
}

fn ne32(t: &Tensor) -> [i32; 4] {
    t.ne.map(|n| n as i32)
}

fn pad(x: u64, n: u64) -> u64 {
    x.div_ceil(n) * n
}

fn unsupported<T>(t: &Tensor, why: &str) -> Result<T> {
    Err(Error::Unsupported { op: t.op.clone(), node: t.name.clone(), why: why.to_string() })
}

impl Ctx<'_> {
    fn t(&self, id: TensorId) -> &Tensor {
        &self.graph.tensors[id]
    }

    fn buf(&self, id: TensorId) -> DArg {
        DArg::Buf(self.layout.place[id])
    }

    fn src(&self, node: &Tensor, i: usize) -> Result<TensorId> {
        node.src(i).ok_or_else(|| Error::Invalid(format!("{} `{}` is missing source {i}", node.op, node.name)))
    }

    fn pipe(&self, entry: &str, consts: Vec<(u32, Const)>) -> Result<Pipeline> {
        self.dev.pipeline(&Kernel { library: self.lib.clone(), entry: entry.to_string(), consts })
    }

    fn pipe_extra(&self, entry: &str) -> Result<Pipeline> {
        self.dev.pipeline(&Kernel { library: self.extra.clone(), entry: entry.to_string(), consts: vec![] })
    }

    fn dispatch(&self, node: TensorId, pipeline: Pipeline, args: Vec<(u32, DArg)>, dims: Dims) -> Dispatch {
        let t = self.t(node);
        Dispatch { pipeline, args, dims, label: format!("{} {}", t.op, t.name) }
    }

    fn spec_dispatch(&self, node: TensorId, spec: Spec) -> Result<Dispatch> {
        let t = self.t(node);
        let pipeline = self.pipe(&spec.kernel, spec.consts)?;
        let args = spec
            .args
            .into_iter()
            .map(|a| {
                Ok(match a {
                    SpecArg::Tensor { slot, which } => {
                        let id = match which {
                            Which::Src0 => self.src(t, 0)?,
                            Which::Src1 => self.src(t, 1)?,
                            Which::Dst => node,
                        };
                        (slot, self.buf(id))
                    }
                    SpecArg::Bytes { slot, data } => (slot, DArg::Bytes(data)),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dims = Dims::new(spec.grid, spec.threads).shared(spec.smem);
        Ok(self.dispatch(node, pipeline, args, dims))
    }

    /// Lowers one node; view ops and empty tensors produce nothing.
    pub fn lower(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let t = self.t(id);
        if t.is_view_op() || t.nelements() == 0 {
            return Ok(vec![]);
        }
        match t.op.as_str() {
            "CONCAT" => self.concat(id),
            "ADD" | "SUB" | "MUL" | "DIV" => self.bin(id),
            "REPEAT" => self.repeat(id),
            "SCALE" | "FILL" | "CLAMP" | "LEAKY_RELU" | "SQR" | "SQRT" | "SIN" | "COS" | "LOG" | "UNARY" => self.unary(id),
            "SUM_ROWS" | "MEAN" => self.sum_rows(id),
            "GET_ROWS" => self.get_rows(id),
            "SOFT_MAX" => self.soft_max(id),
            "DUP" | "CPY" | "CONT" => self.cpy(id),
            "NORM" | "RMS_NORM" => self.norm(id),
            "IM2COL" => self.im2col(id),
            "ARANGE" => self.arange(id),
            "UPSCALE" => self.upscale(id),
            "ROPE" => self.rope(id),
            "POOL_2D" => self.pool_2d(id),
            "SET_ROWS" => self.set_rows(id),
            "ARGMAX" => self.argmax(id),
            "CONV_2D_DW" => self.conv_2d_dw(id),
            "PAD" | "PAD_REFLECT_1D" => self.pad(id),
            "MUL_MAT" => self.mul_mat(id),
            "FLASH_ATTN_EXT" => self.flash_attn_ext(id),
            _ => unsupported(t, "op not implemented in the native Metal backend"),
        }
    }

    // ------------------------------------------------------------------ concat

    fn concat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, b) = (self.t(s0), self.t(s1));
        let (ne0, ne1, ne) = (ne32(a), ne32(b), ne32(op));
        let args = ggml_metal_kargs_concat {
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne10: ne1[0], ne11: ne1[1], ne12: ne1[2], ne13: ne1[3],
            nb10: b.nb[0], nb11: b.nb[1], nb12: b.nb[2], nb13: b.nb[3],
            ne0: ne[0], ne1: ne[1], ne2: ne[2], ne3: ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            dim: op.op_params[0],
        };
        let pipeline = self.pipe(&format!("kernel_concat_{}", op.ty.name()), vec![])?;
        let nth = 256.min(ne[0]);
        let mut nrptg = 1;
        if nth < 256 {
            nrptg = ((256 + nth - 1) / nth).min(ne[1]);
            if nrptg * nth > 256 {
                nrptg = 256 / nth;
            }
        }
        let nw0 = (ne[1] + nrptg - 1) / nrptg;
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(s1)), (3, self.buf(id))],
            Dims::new([nw0 as u32, ne[2] as u32, ne[3] as u32], [nth as u32, nrptg as u32, 1]),
        )])
    }

    // ------------------------------------------------------------------ repeat

    fn repeat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        let (ne0, ne) = (ne32(a), ne32(op));
        let pipeline = self.pipe(&format!("kernel_repeat_{}", a.ty.name()), vec![])?;
        let args = ggml_metal_kargs_repeat {
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: ne[0], ne1: ne[1], ne2: ne[2], ne3: ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
        };
        let nth = (pipeline.max_threads as i32).min(ne[0]);
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([ne[1] as u32, ne[2] as u32, ne[3] as u32], [nth as u32, 1, 1]),
        )])
    }

    // ------------------------------------------------------------------- unary

    fn unary(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if !a.is_contiguous_rows() {
            return unsupported(op, "source rows must be contiguous");
        }
        let op_num = match op.op.as_str() {
            "SCALE" => OP_UNARY_NUM_SCALE,
            "FILL" => OP_UNARY_NUM_FILL,
            "CLAMP" => OP_UNARY_NUM_CLAMP,
            "SQR" => OP_UNARY_NUM_SQR,
            "SQRT" => OP_UNARY_NUM_SQRT,
            "SIN" => OP_UNARY_NUM_SIN,
            "COS" => OP_UNARY_NUM_COS,
            "LOG" => OP_UNARY_NUM_LOG,
            "LEAKY_RELU" => OP_UNARY_NUM_LEAKY_RELU,
            "UNARY" => match op.op_params[0] {
                0 => OP_UNARY_NUM_ABS,
                1 => OP_UNARY_NUM_SGN,
                2 => OP_UNARY_NUM_NEG,
                3 => OP_UNARY_NUM_STEP,
                4 => OP_UNARY_NUM_TANH,
                5 => OP_UNARY_NUM_ELU,
                6 => OP_UNARY_NUM_RELU,
                7 => OP_UNARY_NUM_SIGMOID,
                8 => OP_UNARY_NUM_GELU,
                9 => OP_UNARY_NUM_GELU_QUICK,
                10 => OP_UNARY_NUM_SILU,
                11 => OP_UNARY_NUM_HARDSWISH,
                12 => OP_UNARY_NUM_HARDSIGMOID,
                13 => OP_UNARY_NUM_EXP,
                14 => OP_UNARY_NUM_EXPM1,
                15 => OP_UNARY_NUM_SOFTPLUS,
                16 => OP_UNARY_NUM_GELU_ERF,
                17 => OP_UNARY_NUM_XIELU,
                18 => OP_UNARY_NUM_FLOOR,
                19 => OP_UNARY_NUM_CEIL,
                20 => OP_UNARY_NUM_ROUND,
                21 => OP_UNARY_NUM_TRUNC,
                n => return unsupported(op, &format!("unary op {n}")),
            },
            _ => unreachable!(),
        };
        let (ne0, ne) = (ne32(a), ne32(op));
        let mut args = ggml_metal_kargs_unary {
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: ne[0], ne1: ne[1], ne2: ne[2], ne3: ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            ..Default::default()
        };
        match op.op.as_str() {
            "LEAKY_RELU" => args.slope = op.op_param_f32(0),
            "SCALE" => {
                args.scale = op.op_param_f32(0);
                args.bias = op.op_param_f32(1);
            }
            "FILL" => args.val = op.op_param_f32(0),
            "CLAMP" => {
                args.min = op.op_param_f32(0);
                args.max = op.op_param_f32(1);
            }
            "UNARY" if op.op_params[0] == 17 => {
                args.slope = op.op_param_f32(1);
                args.scale = op.op_param_f32(2);
                args.bias = op.op_param_f32(3);
                args.val = op.op_param_f32(4);
            }
            _ => {}
        }
        let is_c4 = a.ne[0] % 4 == 0;
        let is_cnt = a.is_contiguous() && op.nelements() < 32768;
        let name = format!("kernel_unary_{}_{}{}", a.ty.name(), op.ty.name(), if is_c4 { "_4" } else { "" });
        let pipeline = self.pipe(
            &name,
            vec![(FC_UNARY, Const::I16(op_num as i16)), (FC_UNARY + 1, Const::Bool(is_cnt))],
        )?;
        if is_c4 {
            args.ne00 /= 4;
            args.ne0 /= 4;
        }
        let dims = if is_cnt {
            let n = if is_c4 { op.nelements() / 4 } else { op.nelements() };
            Dims::new([n as u32, 1, 1], [1, 1, 1])
        } else {
            let nth_max = 256.min(pipeline.max_threads as i32);
            let nth = args.ne00.min(nth_max);
            let nk0 = (args.ne00 + nth - 1) / nth;
            Dims::new([(nk0 * ne0[1]) as u32, ne0[2] as u32, ne0[3] as u32], [nth as u32, 1, 1])
        };
        Ok(vec![self.dispatch(id, pipeline, vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))], dims)])
    }

    // --------------------------------------------------------------------- bin

    fn bin(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, b) = (self.t(s0), self.t(s1));
        if !a.is_contiguous_rows() || !b.is_contiguous_rows() {
            return unsupported(op, "operand rows must be contiguous");
        }
        let op_num = match op.op.as_str() {
            "ADD" => 0,
            "SUB" => 1,
            "MUL" => 2,
            _ => 3,
        };
        let (ne0, ne1, ne) = (ne32(a), ne32(b), ne32(op));
        let Place { buf: b1, off: off1 } = self.layout.place[s1];
        let mut args = ggml_metal_kargs_bin {
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne10: ne1[0], ne11: ne1[1], ne12: ne1[2], ne13: ne1[3],
            nb10: b.nb[0], nb11: b.nb[1], nb12: b.nb[2], nb13: b.nb[3],
            ne0: ne[0], ne1: ne[1], ne2: ne[2], ne3: ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            offs: 0,
            o1: [off1, 0, 0, 0, 0, 0, 0, 0],
        };
        let is_c4 = a.ne[0] % 4 == 0 && b.ne[0] % 4 == 0;
        let is_cb = a.ne[0] != b.ne[0];
        let is_rb = a.is_contiguous() && b.is_contiguous() && b.nrows() == 1 && op.nelements() < 65536;
        let name = format!(
            "kernel_bin_fuse_{}_{}_{}{}",
            a.ty.name(),
            b.ty.name(),
            op.ty.name(),
            if is_c4 { "_4" } else { "" }
        );
        let pipeline = self.pipe(
            &name,
            vec![
                (FC_BIN, Const::I16(op_num)),
                (FC_BIN + 1, Const::I16(1)),
                (FC_BIN + 2, Const::Bool(is_rb)),
                (FC_BIN + 3, Const::Bool(is_cb)),
            ],
        )?;
        if is_c4 {
            args.ne00 /= 4;
            args.ne10 /= 4;
            args.ne0 /= 4;
        }
        let dims = if is_rb {
            Dims::new([args.ne0 as u32, op.nrows() as u32, 1], [1, 1, 1])
        } else {
            let nth_max = 256.min(pipeline.max_threads as i32);
            let mut nth = 1;
            while 2 * nth < args.ne0 && nth < nth_max {
                nth *= 2;
            }
            Dims::new([ne0[1] as u32, ne0[2] as u32, ne0[3] as u32], [nth as u32, 1, 1])
        };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![
                (0, DArg::Bytes(bytes_of(&args))),
                (1, self.buf(s0)),
                // src1 is bound at the start of its buffer; its offset travels in `o1`.
                (2, DArg::Buf(Place { buf: b1, off: 0 })),
                (3, self.buf(id)),
            ],
            dims,
        )])
    }

    // ---------------------------------------------------------------- sum_rows

    fn sum_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if !a.is_contiguous_rows() {
            return unsupported(op, "source rows must be contiguous");
        }
        let op_num = if op.op == "SUM_ROWS" { OP_SUM_ROWS_NUM_SUM_ROWS } else { OP_SUM_ROWS_NUM_MEAN };
        let is_c4 = a.ne[0] % 4 == 0;
        let name = format!("kernel_sum_rows_{}_{}{}", a.ty.name(), op.ty.name(), if is_c4 { "_4" } else { "" });
        let pipeline = self.pipe(&name, vec![(FC_SUM_ROWS, Const::I16(op_num as i16))])?;
        let mut args = ggml_metal_kargs_sum_rows {
            ne00: a.ne[0], ne01: a.ne[1], ne02: a.ne[2], ne03: a.ne[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: op.ne[0], ne1: op.ne[1], ne2: op.ne[2], ne3: op.ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
        };
        if is_c4 {
            args.ne00 /= 4;
            args.ne0 /= 4;
        }
        let max = pipeline.max_threads as i64;
        let mut nth = 32i64;
        while nth < args.ne00 && nth < max {
            nth *= 2;
        }
        nth = nth.min(max).min(args.ne00);
        let smem = 32 * 4 * if is_c4 { 4 } else { 1 };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([a.ne[1] as u32, a.ne[2] as u32, a.ne[3] as u32], [nth as u32, 1, 1]).shared(smem),
        )])
    }

    // ---------------------------------------------------------------- get_rows

    fn get_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, idx) = (self.t(s0), self.t(s1));
        let pipeline = self.pipe(&format!("kernel_get_rows_{}", a.ty.name()), vec![])?;
        let ne00 = a.ne[0] as i32;
        let args = ggml_metal_kargs_get_rows {
            ne00t: if a.ty.is_quantized() { ne00 / 16 } else { ne00 },
            ne00,
            nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne10: idx.ne[0] as i32,
            nb10: idx.nb[0], nb11: idx.nb[1], nb12: idx.nb[2],
            nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
        };
        let nth = args.ne00t.min(pipeline.max_threads as i32);
        let nw0 = (args.ne00t + nth - 1) / nth;
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(s1)), (3, self.buf(id))],
            Dims::new([(nw0 * args.ne10) as u32, idx.ne[1] as u32, idx.ne[2] as u32], [nth as u32, 1, 1]),
        )])
    }

    // --------------------------------------------------------------- soft_max

    fn soft_max(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        let mask = op.src(1);
        let sinks = op.src(2);
        if sinks.is_some() {
            return unsupported(op, "sinks are not supported by the soft_max lowering");
        }
        let (scale, max_bias) = (op.op_param_f32(0), op.op_param_f32(1));
        let n_head = a.ne[2] as u32;
        let n_head_log2 = 1u32 << (n_head as f32).log2().floor() as u32;
        let m0 = 2f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
        let m = mask.map(|m| self.t(m));
        let args = ggml_metal_kargs_soft_max {
            ne00: a.ne[0] as i32, ne01: a.ne[1] as i32, ne02: a.ne[2] as i32,
            nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne11: m.map_or(0, |m| m.ne[1] as i32),
            ne12: m.map_or(0, |m| m.ne[2] as i32),
            ne13: m.map_or(0, |m| m.ne[3] as i32),
            nb11: m.map_or(0, |m| m.nb[1]),
            nb12: m.map_or(0, |m| m.nb[2]),
            nb13: m.map_or(0, |m| m.nb[3]),
            nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            scale, max_bias, m0, m1,
            n_head_log2: n_head_log2 as i32,
        };
        let t1 = m.map_or(Ty::F32, |m| m.ty);
        if t1 != Ty::F32 && t1 != Ty::F16 {
            return unsupported(op, "mask must be f16 or f32");
        }
        let name = format!("kernel_soft_max_{}{}", t1.name(), if a.ne[0] % 4 == 0 { "_4" } else { "" });
        let pipeline = self.pipe(&name, vec![])?;
        let (ne00, ne01, ne02, ne03) = (a.ne[0], a.ne[1], a.ne[2], a.ne[3]);
        let mut nth = 32i64;
        if ne00 % 4 == 0 {
            while nth < ne00 / 4 && nth * ne01 * ne02 * ne03 < 256 {
                nth *= 2;
            }
        } else {
            while nth < ne00 && nth * ne01 * ne02 * ne03 < 256 {
                nth *= 2;
            }
        }
        let mask_buf = mask.map_or(self.buf(s0), |m| self.buf(m));
        let sinks_buf = self.buf(s0);
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, mask_buf), (3, sinks_buf), (4, self.buf(id))],
            Dims::new([ne01 as u32, ne02 as u32, ne03 as u32], [nth as u32, 1, 1]).shared(32 * 4),
        )])
    }

    // --------------------------------------------------------------------- cpy

    fn cpy(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        let pipeline = self.pipe(&format!("kernel_cpy_{}_{}", a.ty.name(), op.ty.name()), vec![])?;
        if a.ne[0] % a.ty.blck() != 0 {
            return unsupported(op, "row length must be a multiple of the source block size");
        }
        let mut nk0 = a.ne[0];
        if a.ty.is_quantized() {
            nk0 = a.ne[0] / 16;
        } else if op.ty.is_quantized() {
            nk0 = a.ne[0] / op.ty.blck();
        }
        let mut nth = (nk0 * a.ne[1]).min(256);
        let mut nrptg = 1;
        if a.ty.blck() == 1 && op.ty.blck() == 1 && nth > nk0 {
            nrptg = (nth + nk0 - 1) / nk0;
            nth = nk0;
            if nrptg * nth > 256 {
                nrptg -= 1;
            }
        }
        nth = nth.min(nk0);
        let args = ggml_metal_kargs_cpy {
            nk0,
            ne00: a.ne[0], ne01: a.ne[1], ne02: a.ne[2], ne03: a.ne[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: op.ne[0], ne1: op.ne[1], ne2: op.ne[2], ne3: op.ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
        };
        let nw0 = if nrptg == 1 { (nk0 + nth - 1) / nth } else { 1 };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([(nw0 * ((a.ne[1] + nrptg - 1) / nrptg)) as u32, a.ne[2] as u32, a.ne[3] as u32], [nth as u32, nrptg as u32, 1]),
        )])
    }

    // -------------------------------------------------------------------- norm

    fn norm(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if !a.is_contiguous_rows() {
            return unsupported(op, "source rows must be contiguous");
        }
        let ne00 = a.ne[0] as i32;
        let ne00_t = if ne00 % 4 == 0 { ne00 / 4 } else { ne00 };
        let args = ggml_metal_kargs_norm {
            ne00,
            ne00_t,
            nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            eps: op.op_param_f32(0),
            nef1: [a.ne[1] as i32, 0, 0],
            nef2: [a.ne[2] as i32, 0, 0],
            nef3: [a.ne[3] as i32, 0, 0],
            nbf1: [a.nb[1], 0, 0],
            nbf2: [a.nb[2], 0, 0],
            nbf3: [a.nb[3], 0, 0],
        };
        let base = if op.op == "NORM" { "kernel_norm_f32" } else { "kernel_rms_norm_f32" };
        let pipeline = self.pipe(&format!("{base}{}", if op.ne[0] % 4 == 0 { "_4" } else { "" }), vec![])?;
        let max = pipeline.max_threads as i32;
        let mut nth = 32;
        while nth < ne00_t && nth < max {
            nth *= 2;
        }
        nth = nth.min(max).min((ne00_t + 31) / 32 * 32);
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![
                (0, DArg::Bytes(bytes_of(&args))),
                (1, self.buf(s0)),
                (2, self.buf(s0)),
                (3, self.buf(s0)),
                (4, self.buf(id)),
            ],
            Dims::new([a.ne[1] as u32, a.ne[2] as u32, a.ne[3] as u32], [nth as u32, 1, 1]).shared(32 * 4),
        )])
    }

    // ------------------------------------------------------------------ im2col

    fn im2col(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (kern, inp) = (self.t(s0), self.t(s1));
        if !inp.is_contiguous() || inp.ty != Ty::F32 || !(op.ty == Ty::F16 || op.ty == Ty::F32) {
            return unsupported(op, "im2col needs a contiguous f32 input and f16/f32 output");
        }
        let p = &op.op_params;
        let (s0_, s1_, p0, p1, d0, d1) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let is_2d = p[6] == 1;
        let d = |i: usize| if is_2d { inp.ne[i] } else { inp.ne[i - 1] };
        let n = d(3) as i32;
        let ic = d(2) as i32;
        let ih = if is_2d { inp.ne[1] as i32 } else { 1 };
        let iw = inp.ne[0] as i32;
        let kh = if is_2d { kern.ne[1] as i32 } else { 1 };
        let kw = kern.ne[0] as i32;
        let oh = if is_2d { op.ne[2] as i32 } else { 1 };
        let ow = op.ne[1] as i32;
        let chw = ic * kh * kw;
        let ofs = |i: usize| if is_2d { inp.nb[i] / 4 } else { inp.nb[i - 1] / 4 };
        let args = ggml_metal_kargs_im2col {
            ofs0: ofs(3), ofs1: ofs(2),
            IW: iw, IH: ih, CHW: chw,
            s0: s0_, s1: s1_, p0, p1, d0, d1,
            N: n, KH: kh, KW: kw, KHW: kh * kw,
        };
        let base = if (kh * kw) as i64 <= 1024 { "kernel_im2col" } else { "kernel_im2col_ext" };
        let pipeline = self.pipe(&format!("{base}_{}", op.ty.name()), vec![])?;
        let max = pipeline.max_threads as i64;
        let (grid, threads) = if (kh * kw) as i64 <= max {
            let ntptg0 = (max / (kh * kw) as i64).min(n as i64);
            ([ic as u32, oh as u32, ow as u32], [ntptg0 as u32, kh as u32, kw as u32])
        } else {
            let n_threads = max.min(n as i64);
            let quotient = n as i64 / n_threads + i64::from(n as i64 % n_threads > 0);
            ([(quotient * chw as i64) as u32, oh as u32, ow as u32], [n_threads as u32, 1, 1])
        };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s1)), (2, self.buf(id))],
            Dims::new(grid, threads),
        )])
    }

    // ----------------------------------------------------------------- upscale

    fn upscale(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        let flags = op.op_params[0];
        let (mode, antialias) = (flags & 0xFF, flags & (1 << 9) != 0);
        let mut sf = [0, 1, 2, 3].map(|i| op.ne[i] as f32 / a.ne[i] as f32);
        let mut poffs = 0.5f32;
        if flags & (1 << 8) != 0 {
            poffs = 0.0;
            for i in 0..2 {
                if op.ne[i] > 1 && a.ne[i] > 1 {
                    sf[i] = (op.ne[i] - 1) as f32 / (a.ne[i] - 1) as f32;
                }
            }
        }
        let args = ggml_metal_kargs_upscale {
            ne00: a.ne[0], ne01: a.ne[1], ne02: a.ne[2], ne03: a.ne[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: op.ne[0], ne1: op.ne[1], ne2: op.ne[2], ne3: op.ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            sf0: sf[0], sf1: sf[1], sf2: sf[2], sf3: sf[3],
            poffs,
        };
        let kind = match mode {
            1 => "bilinear",
            2 => "bicubic",
            _ => "nearest",
        };
        let pipeline = self.pipe(&format!("kernel_upscale_{kind}_{}", a.ty.name()), vec![(FC_UPSCALE, Const::Bool(antialias))])?;
        let nth = (pipeline.max_threads as i64).min(op.ne[0]);
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([op.ne[1] as u32, op.ne[2] as u32, op.ne[3] as u32], [nth as u32, 1, 1]),
        )])
    }

    // -------------------------------------------------------------------- rope

    fn rope(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, pos) = (self.t(s0), self.t(s1));
        let s2 = op.src(2);
        if pos.ne[0] % a.ne[2] != 0 || pos.ne[0] < a.ne[2] {
            return unsupported(op, "needs one or more position ids per token");
        }
        let p = &op.op_params;
        let mode = p[2];
        let (is_neox, is_mrope, is_imrope, is_vision) = (mode & 2 != 0, mode & 8 != 0, mode == 40, mode == 24);
        let kind = if is_neox {
            "neox"
        } else if (is_mrope || is_imrope) && !is_vision {
            "multi"
        } else if is_vision {
            "vision"
        } else {
            "norm"
        };
        if matches!(kind, "multi" | "vision") && pos.ne[0] * 4 < a.ne[2] {
            return unsupported(op, "multi-section rope needs 4 positions per token");
        }
        let (ne0, ne) = (ne32(a), ne32(op));
        let args = ggml_metal_kargs_rope {
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: ne[0], ne1: ne[1], ne2: ne[2], ne3: ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            n_past: p[0],
            n_dims: p[1],
            n_ctx_orig: p[4],
            freq_base: op.op_param_f32(5),
            freq_scale: op.op_param_f32(6),
            ext_factor: op.op_param_f32(7),
            attn_factor: op.op_param_f32(8),
            beta_fast: op.op_param_f32(9),
            beta_slow: op.op_param_f32(10),
            sect_0: p[11], sect_1: p[12], sect_2: p[13], sect_3: p[14],
            src2: s2.is_some(),
        };
        let pipeline = self.pipe(
            &format!("kernel_rope_{kind}_{}", a.ty.name()),
            vec![(FC_ROPE, Const::Bool(is_imrope)), (FC_ROPE + 1, Const::Bool(false))],
        )?;
        let nth = 1024.min(ne0[0]);
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![
                (0, DArg::Bytes(bytes_of(&args))),
                (1, self.buf(s0)),
                (2, self.buf(s1)),
                (3, s2.map_or(self.buf(s0), |f| self.buf(f))),
                (4, self.buf(id)),
            ],
            Dims::new([ne0[1] as u32, ne0[2] as u32, ne0[3] as u32], [nth as u32, 1, 1]),
        )])
    }

    // ----------------------------------------------------------------- pool_2d

    fn pool_2d(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if !a.is_contiguous() || a.ty != Ty::F32 || op.ty != Ty::F32 {
            return unsupported(op, "pooling needs a contiguous f32 input");
        }
        let p = &op.op_params;
        let kind = match p[0] {
            0 => "max",
            1 => "avg",
            _ => return unsupported(op, "unknown pooling op"),
        };
        let np = op.ne[3] * op.ne[2] * op.ne[1] * op.ne[0];
        let args = ggml_metal_kargs_pool_2d {
            k0: p[1], k1: p[2], s0: p[3], s1: p[4], p0: p[5], p1: p[6],
            IH: a.ne[1], IW: a.ne[0], OH: op.ne[1], OW: op.ne[0], np,
        };
        let pipeline = self.pipe(&format!("kernel_pool_2d_{kind}_f32"), vec![])?;
        let nth = (pipeline.max_threads as i64).min(np);
        let ntg = (np + nth - 1) / nth;
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([ntg as u32, 1, 1], [nth as u32, 1, 1]),
        )])
    }

    // --------------------------------------------------------------- set_rows

    fn set_rows(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (a, idx) = (self.t(s0), self.t(s1));
        let pipeline =
            self.pipe(&format!("kernel_set_rows_{}_{}_{}", a.ty.name(), idx.ty.name(), op.ty.name()), vec![])?;
        let nk0 = (op.ne[0] / op.ty.blck()) as i32;
        let max = pipeline.max_threads as i32;
        let mut nth = 32;
        while nth < nk0 && nth < max {
            nth *= 2;
        }
        let mut nrptg = 1;
        if nth > nk0 {
            nrptg = (nth + nk0 - 1) / nk0;
            nth = nk0;
            if nrptg * nth > max {
                nrptg -= 1;
            }
        }
        nth = nth.min(nk0);
        let args = ggml_metal_kargs_set_rows {
            nk0,
            ne01: a.ne[1] as i32,
            nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne11: idx.ne[1] as i32, ne12: idx.ne[2] as i32,
            nb10: idx.nb[0], nb11: idx.nb[1], nb12: idx.nb[2],
            nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
        };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(s1)), (3, self.buf(id))],
            Dims::new([((a.ne[1] + nrptg as i64 - 1) / nrptg as i64) as u32, a.ne[2] as u32, a.ne[3] as u32], [nth as u32, nrptg as u32, 1]),
        )])
    }

    // ------------------------------------------------------------------ argmax

    fn argmax(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        if a.ty != Ty::F32 || !a.is_contiguous_n(1) || a.nb[0] != 4 {
            return unsupported(op, "argmax needs f32 rows");
        }
        let args = ggml_metal_kargs_argmax { ne00: a.ne[0], nb01: a.nb[1] };
        let pipeline = self.pipe("kernel_argmax_f32", vec![])?;
        let mut nth = 32i64;
        while nth < a.ne[0] && nth * a.ne[1] * a.ne[2] * a.ne[3] < 256 {
            nth *= 2;
        }
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([a.nrows() as u32, 1, 1], [nth as u32, 1, 1]).shared(32 * 8),
        )])
    }

    // -------------------------------------------------------------- conv_2d_dw

    fn conv_2d_dw(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let (k, x) = (self.t(s0), self.t(s1));
        if x.ty != Ty::F32 || op.ty != Ty::F32 || !(k.ty == Ty::F16 || k.ty == Ty::F32) {
            return unsupported(op, "depthwise convolution needs f32 data and f16/f32 weights");
        }
        let p = &op.op_params;
        let args = ggml_metal_kargs_conv_2d_dw {
            nb00: k.nb[0],
            nb01: k.nb[1],
            // ggml passes the weights' nb03 here.
            nb02: k.nb[3],
            nb10: x.nb[0], nb11: x.nb[1], nb12: x.nb[2], nb13: x.nb[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            IW: x.ne[0] as i32, IH: x.ne[1] as i32,
            KW: k.ne[0] as i32, KH: k.ne[1] as i32,
            C: x.ne[2] as i32,
            OW: op.ne[0] as i32, OH: op.ne[1] as i32,
            N: x.ne[3] as i32,
            s0: p[0], s1: p[1], p0: p[2], p1: p[3], d0: p[4], d1: p[5],
        };
        let tiled = x.nb[2] < x.nb[0];
        let name = format!("kernel_conv_2d_dw{}_{}_{}", if tiled { "_tiled" } else { "" }, k.ty.name(), x.ty.name());
        let pipeline = self.pipe(&name, vec![])?;
        let nth = (pipeline.max_threads as i32).min(256).max(1);
        let (ow, oh, c, n) = (op.ne[0] as i32, op.ne[1] as i32, x.ne[2] as i32, x.ne[3] as i32);
        let tg_x = if tiled { (c + nth - 1) / nth } else { (ow + nth - 1) / nth };
        let tg_z = if tiled { ow * n } else { c * n };
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(s0)), (2, self.buf(s1)), (3, self.buf(id))],
            Dims::new([tg_x as u32, oh as u32, tg_z as u32], [nth as u32, 1, 1]),
        )])
    }

    // --------------------------------------------------------------------- pad

    fn pad(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let s0 = self.src(op, 0)?;
        let a = self.t(s0);
        let ne = ne32(op);
        if op.op == "PAD" {
            let p = &op.op_params;
            if p[8] != 0 {
                return unsupported(op, "circular padding");
            }
            if a.ty != op.ty || !(a.ty == Ty::F32 || a.ty == Ty::F16) {
                return unsupported(op, "padding needs f32 or f16");
            }
            // ggml's kernel only appends; leading padding goes through ours.
            let lp = [p[0], p[2], p[4], p[6]];
            let pipeline = self.pipe_extra(&format!("kernel_pad_ext_{}", a.ty.name()))?;
            let mut bytes = Vec::new();
            for v in a.ne.iter().chain(&a.nb.map(|n| n as i64)).chain(&op.ne).chain(&op.nb.map(|n| n as i64)) {
                bytes.extend(v.to_ne_bytes());
            }
            for v in lp {
                bytes.extend(v.to_ne_bytes());
            }
            let nth = ne[0].min(64.min(pipeline.max_threads as i32));
            let nk0 = (ne[0] + 1023) / 1024;
            return Ok(vec![self.dispatch(
                id,
                pipeline,
                vec![(0, DArg::Bytes(bytes)), (1, self.buf(s0)), (2, self.buf(id))],
                Dims::new([(nk0 * ne[1]) as u32, ne[2] as u32, ne[3] as u32], [nth as u32, 1, 1]),
            )]);
        }
        let name = format!("kernel_pad_reflect_1d_{}", a.ty.name());
        let pipeline = self.pipe(&name, vec![])?;
        let bytes = bytes_of(&ggml_metal_kargs_pad_reflect_1d {
            ne00: a.ne[0], ne01: a.ne[1], ne02: a.ne[2], ne03: a.ne[3],
            nb00: a.nb[0], nb01: a.nb[1], nb02: a.nb[2], nb03: a.nb[3],
            ne0: op.ne[0], ne1: op.ne[1], ne2: op.ne[2], ne3: op.ne[3],
            nb0: op.nb[0], nb1: op.nb[1], nb2: op.nb[2], nb3: op.nb[3],
            p0: op.op_params[0], p1: op.op_params[1],
        });
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes)), (1, self.buf(s0)), (2, self.buf(id))],
            Dims::new([ne[1] as u32, ne[2] as u32, ne[3] as u32], [1024.min(ne[0]) as u32, 1, 1]),
        )])
    }

    // ------------------------------------------------------------------ arange

    fn arange(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let args = ggml_metal_kargs_arange { ne0: op.ne[0], start: op.op_param_f32(0), step: op.op_param_f32(2) };
        let pipeline = self.pipe(&format!("kernel_arange_{}", op.ty.name()), vec![])?;
        Ok(vec![self.dispatch(
            id,
            pipeline,
            vec![(0, DArg::Bytes(bytes_of(&args))), (1, self.buf(id))],
            Dims::new([1, 1, 1], [1024.min(op.ne[0]) as u32, 1, 1]),
        )])
    }

    // --------------------------------------------------------------------- mul_mat

    fn mul_mat(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (s0, s1) = (self.src(op, 0)?, self.src(op, 1)?);
        let facts = |t: &Tensor| TensorFacts { ty: t.ty, ne: t.ne, nb: t.nb };
        let query = MatmulQuery { src0: facts(self.t(s0)), src1: facts(self.t(s1)), dst: facts(op), props: self.props };
        self.matmul
            .lower(&query)?
            .into_iter()
            .map(|spec| self.spec_dispatch(id, spec))
            .collect()
    }

    // ----------------------------------------------------------- flash attention

    fn flash_attn_ext(&self, id: TensorId) -> Result<Vec<Dispatch>> {
        let op = self.t(id);
        let (q, k, v) = (self.src(op, 0)?, self.src(op, 1)?, self.src(op, 2)?);
        let (mask, sinks) = (op.src(3), op.src(4));
        let (tq, tk, tv) = (self.t(q), self.t(k), self.t(v));
        if tq.ne[0] % 4 != 0 || tq.ty != Ty::F32 || tk.ty != tv.ty {
            return unsupported(op, "needs f32 queries with a head size divisible by 4 and K/V of one type");
        }
        let (ne00, ne01, ne02, ne03) = (tq.ne[0], tq.ne[1], tq.ne[2], tq.ne[3]);
        let (ne11, ne12, ne13) = (tk.ne[1], tk.ne[2], tk.ne[3]);
        let ne20 = tv.ne[0];
        let tm = mask.map(|m| self.t(m));
        if tm.is_some_and(|m| m.ty != Ty::F16) {
            return unsupported(op, "mask must be f16");
        }
        let (mut scale, max_bias, logit_softcap) = (op.op_param_f32(0), op.op_param_f32(1), op.op_param_f32(2));
        if logit_softcap != 0.0 {
            scale /= logit_softcap;
        }
        let has_mask = mask.is_some();
        let has_sinks = sinks.is_some();
        let has_bias = max_bias != 0.0;
        let has_scap = logit_softcap != 0.0;
        let n_head = ne02 as u32;
        let n_head_log2 = 1u32 << (n_head as f32).log2().floor() as u32;
        let m0 = 2f32.powf(-max_bias / n_head_log2 as f32);
        let m1 = 2f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);

        let place = self.layout.scratch[self.pos(id)?]
            .ok_or_else(|| Error::Invalid("flash attention scratch was not planned".into()))?;
        let scratch = |off: u64| DArg::Buf(Place { buf: crate::plan::BufId::Arena, off: place + off });
        let (pad_bytes, blk_bytes, _) = fa_scratch_sizes(self.graph, id);
        let (bid_blk, bid_tmp) = (scratch(pad_bytes), scratch(pad_bytes + blk_bytes));

        let mask_or_q = |slot: u32| (slot, mask.map_or(self.buf(q), |m| self.buf(m)));
        let sinks_or_q = |slot: u32| (slot, sinks.map_or(self.buf(q), |s| self.buf(s)));
        let (nb11, nb12, nb13) = (tk.nb[1], tk.nb[2], tk.nb[3]);
        let (nb21, nb22, nb23) = (tv.nb[1], tv.nb[2], tv.nb[3]);
        let m_ne = tm.map_or([0i64; 4], |m| m.ne);
        let m_nb = tm.map_or([0u64; 4], |m| m.nb);

        let use_vec = fa_use_vec(tq);
        let mut out = Vec::new();

        let pad_args = |ncpsg: i64| {
            let _ = ncpsg;
            ggml_metal_kargs_flash_attn_ext_pad {
                ne11: ne11 as i32, ne_12_2: ne12 as i32, ne_12_3: ne13 as i32,
                nb11, nb12, nb13, nb21, nb22, nb23,
                ne31: m_ne[1] as i32, ne32: m_ne[2] as i32, ne33: m_ne[3] as i32,
                nb31: m_nb[1], nb32: m_nb[2], nb33: m_nb[3],
            }
        };
        let main_args = || ggml_metal_kargs_flash_attn_ext {
            ne01: ne01 as i32, ne02: ne02 as i32, ne03: ne03 as i32,
            nb01: tq.nb[1], nb02: tq.nb[2], nb03: tq.nb[3],
            ne11: ne11 as i32, ne_12_2: ne12 as i32, ne_12_3: ne13 as i32,
            ns10: (nb11 / tk.nb[0]) as i32, nb11, nb12, nb13,
            ns20: (nb21 / tv.nb[0]) as i32, nb21, nb22, nb23,
            ne31: m_ne[1] as i32, ne32: m_ne[2] as i32, ne33: m_ne[3] as i32,
            nb31: m_nb[1], nb32: m_nb[2], nb33: m_nb[3],
            ne1: op.ne[1] as i32, ne2: op.ne[2] as i32, ne3: op.ne[3] as i32,
            scale, max_bias, m0, m1,
            n_head_log2: n_head_log2 as i32,
            logit_softcap,
        };
        let (dk, dv) = (tk.ne[0], tv.ne[0]);
        let ns10 = (nb11 / tk.nb[0]) as i32;
        let ns20 = (nb21 / tv.nb[0]) as i32;

        if !use_vec {
            let nqptg = OP_FLASH_ATTN_EXT_NQPSG as i64;
            let ncpsg = OP_FLASH_ATTN_EXT_NCPSG as i64;
            let has_kvpad = ne11 % ncpsg != 0;
            if has_kvpad {
                let pipeline = self.pipe(
                    "kernel_flash_attn_ext_pad",
                    vec![(FC_FLASH_ATTN_EXT_PAD, Const::Bool(has_mask)), (FC_FLASH_ATTN_EXT_PAD + 25, Const::I32(ncpsg as i32))],
                )?;
                out.push(self.dispatch(
                    id,
                    pipeline,
                    vec![
                        (0, DArg::Bytes(bytes_of(&pad_args(ncpsg)))),
                        (1, self.buf(k)),
                        (2, self.buf(v)),
                        mask_or_q(3),
                        (4, scratch(0)),
                    ],
                    Dims::new([ncpsg as u32, ne12.max(m_ne[2]) as u32, ne13.max(m_ne[3]) as u32], [32, 1, 1]),
                ));
            }
            if has_mask {
                let m = tm.unwrap();
                let args = ggml_metal_kargs_flash_attn_ext_blk {
                    ne01: ne01 as i32, ne30: m.ne[0] as i32, ne31: m.ne[1] as i32, ne32: m.ne[2] as i32, ne33: m.ne[3] as i32,
                    nb31: m.nb[1], nb32: m.nb[2], nb33: m.nb[3],
                };
                let pipeline = self.pipe(
                    "kernel_flash_attn_ext_blk",
                    vec![
                        (FC_FLASH_ATTN_EXT_BLK + 24, Const::I32(nqptg as i32)),
                        (FC_FLASH_ATTN_EXT_BLK + 25, Const::I32(ncpsg as i32)),
                    ],
                )?;
                let nblk1 = (ne01 + nqptg - 1) / nqptg;
                let nblk0 = (m.ne[0] + ncpsg - 1) / ncpsg;
                out.push(self.dispatch(
                    id,
                    pipeline,
                    vec![(0, DArg::Bytes(bytes_of(&args))), mask_or_q(1), (2, bid_blk_clone(&bid_blk))],
                    Dims::new([nblk0 as u32, nblk1 as u32, (m.ne[2] * m.ne[3]) as u32], [32, 1, 1]),
                ));
            }
            let is_q = i64::from(tk.ty.is_quantized());
            let nsg: i64 = if ne00 >= 512 { 8 } else { 4 };
            let smem = pad(
                ((nqptg * (ne00 + 2 * pad(ne20 as u64, 64) as i64 + 2 * (2 * ncpsg)) + is_q * (16 * 32 * nsg)) * 2) as u64,
                16,
            );
            let bc_mask = tm.is_some_and(|m| m.ne[1] % 8 != 0);
            let name = format!("kernel_flash_attn_ext_{}_dk{dk}_dv{dv}", tk.ty.name());
            let pipeline = self.pipe(
                &name,
                vec![
                    (FC_FLASH_ATTN_EXT, Const::Bool(has_mask)),
                    (FC_FLASH_ATTN_EXT + 1, Const::Bool(has_sinks)),
                    (FC_FLASH_ATTN_EXT + 2, Const::Bool(has_bias)),
                    (FC_FLASH_ATTN_EXT + 3, Const::Bool(has_scap)),
                    (FC_FLASH_ATTN_EXT + 4, Const::Bool(has_kvpad)),
                    (FC_FLASH_ATTN_EXT + 10, Const::Bool(bc_mask)),
                    (FC_FLASH_ATTN_EXT + 20, Const::I32(ns10)),
                    (FC_FLASH_ATTN_EXT + 21, Const::I32(ns20)),
                    (FC_FLASH_ATTN_EXT + 22, Const::I32(nsg as i32)),
                ],
            )?;
            out.push(self.dispatch(
                id,
                pipeline,
                vec![
                    (0, DArg::Bytes(bytes_of(&main_args()))),
                    (1, self.buf(q)),
                    (2, self.buf(k)),
                    (3, self.buf(v)),
                    mask_or_q(4),
                    sinks_or_q(5),
                    (6, scratch(0)),
                    (7, bid_blk),
                    (8, self.buf(id)),
                ],
                Dims::new([((ne01 + nqptg - 1) / nqptg) as u32, ne02 as u32, ne03 as u32], [32, nsg as u32, 1])
                    .shared(smem as u32),
            ));
            return Ok(out);
        }

        // Vector kernel (few queries): one query per threadgroup, split over the KV cache.
        let nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG as i64;
        let ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG as i64;
        let has_kvpad = ne11 % ncpsg != 0;
        if has_kvpad {
            let pipeline = self.pipe(
                "kernel_flash_attn_ext_pad",
                vec![(FC_FLASH_ATTN_EXT_PAD, Const::Bool(has_mask)), (FC_FLASH_ATTN_EXT_PAD + 25, Const::I32(ncpsg as i32))],
            )?;
            out.push(self.dispatch(
                id,
                pipeline,
                vec![
                    (0, DArg::Bytes(bytes_of(&pad_args(ncpsg)))),
                    (1, self.buf(k)),
                    (2, self.buf(v)),
                    mask_or_q(3),
                    (4, scratch(0)),
                ],
                Dims::new([ncpsg as u32, ne12.max(m_ne[2]) as u32, ne13.max(m_ne[3]) as u32], [32, 1, 1]),
            ));
        }
        if tk.ne[0] < ne20 {
            return unsupported(op, "K head size must be at least V's");
        }
        let (nwg, mut nsg) = (32i64, 1i64);
        while 2 * nwg * nsg * ncpsg < ne11 && nsg < 4 {
            nsg *= 2;
        }
        let smem = pad(((pad(ne00 as u64, 128) as i64 + 4 * ncpsg + 2 * pad(ne20 as u64, 128) as i64) * nsg * 2) as u64, 16);
        if smem as u32 > self.props.max_threadgroup_memory {
            return unsupported(op, "flash attention needs more threadgroup memory than the device has");
        }
        let name = format!("kernel_flash_attn_ext_vec_{}_dk{dk}_dv{dv}", tk.ty.name());
        let pipeline = self.pipe(
            &name,
            vec![
                (FC_FLASH_ATTN_EXT_VEC, Const::Bool(has_mask)),
                (FC_FLASH_ATTN_EXT_VEC + 1, Const::Bool(has_sinks)),
                (FC_FLASH_ATTN_EXT_VEC + 2, Const::Bool(has_bias)),
                (FC_FLASH_ATTN_EXT_VEC + 3, Const::Bool(has_scap)),
                (FC_FLASH_ATTN_EXT_VEC + 4, Const::Bool(has_kvpad)),
                (FC_FLASH_ATTN_EXT_VEC + 20, Const::I32(ns10)),
                (FC_FLASH_ATTN_EXT_VEC + 21, Const::I32(ns20)),
                (FC_FLASH_ATTN_EXT_VEC + 22, Const::I32(nsg as i32)),
                (FC_FLASH_ATTN_EXT_VEC + 23, Const::I32(nwg as i32)),
            ],
        )?;
        if (nsg * 32) as usize > pipeline.max_threads {
            return unsupported(op, "flash attention threadgroup is too large for this pipeline");
        }
        let vec_args = {
            let a = main_args();
            ggml_metal_kargs_flash_attn_ext_vec {
                ne01: a.ne01, ne02: a.ne02, ne03: a.ne03, nb01: a.nb01, nb02: a.nb02, nb03: a.nb03,
                ne11: a.ne11, ne_12_2: a.ne_12_2, ne_12_3: a.ne_12_3,
                ns10: a.ns10, nb11: a.nb11, nb12: a.nb12, nb13: a.nb13,
                ns20: a.ns20, nb21: a.nb21, nb22: a.nb22, nb23: a.nb23,
                ne31: a.ne31, ne32: a.ne32, ne33: a.ne33, nb31: a.nb31, nb32: a.nb32, nb33: a.nb33,
                ne1: a.ne1, ne2: a.ne2, ne3: a.ne3,
                scale: a.scale, max_bias: a.max_bias, m0: a.m0, m1: a.m1,
                n_head_log2: a.n_head_log2, logit_softcap: a.logit_softcap,
            }
        };
        out.push(self.dispatch(
            id,
            pipeline,
            vec![
                (0, DArg::Bytes(bytes_of(&vec_args))),
                (1, self.buf(q)),
                (2, self.buf(k)),
                (3, self.buf(v)),
                mask_or_q(4),
                sinks_or_q(5),
                (6, scratch(0)),
                (7, bid_tmp_clone(&bid_tmp)),
            ],
            Dims::new([((ne01 + nqptg - 1) / nqptg) as u32, ne02 as u32, (ne03 * nwg) as u32], [32, nsg as u32, 1])
                .shared(smem as u32),
        ));
        let nrows = (op.ne[1] * op.ne[2] * op.ne[3]) as i32;
        let reduce = self.pipe(
            "kernel_flash_attn_ext_vec_reduce",
            vec![(FC_FLASH_ATTN_EXT_VEC_REDUCE, Const::I32(ne20 as i32)), (FC_FLASH_ATTN_EXT_VEC_REDUCE + 1, Const::I32(nwg as i32))],
        )?;
        let rargs = ggml_metal_kargs_flash_attn_ext_vec_reduce { nrows };
        out.push(self.dispatch(
            id,
            reduce,
            vec![(0, DArg::Bytes(bytes_of(&rargs))), (1, bid_tmp), (2, self.buf(id))],
            Dims::new([nrows as u32, 1, 1], [(32 * nwg) as u32, 1, 1]),
        ));
        Ok(out)
    }

    fn pos(&self, id: TensorId) -> Result<usize> {
        self.graph.nodes.iter().position(|&n| n == id).ok_or_else(|| Error::Invalid("node is not in the graph".into()))
    }
}

fn bid_blk_clone(a: &DArg) -> DArg {
    match a {
        DArg::Buf(p) => DArg::Buf(*p),
        DArg::Bytes(b) => DArg::Bytes(b.clone()),
    }
}

fn bid_tmp_clone(a: &DArg) -> DArg {
    bid_blk_clone(a)
}

fn fa_use_vec(q: &Tensor) -> bool {
    q.ne[1] < 20 && q.ne[0] % 32 == 0
}

/// Bytes the flash-attention lowering needs beyond its output: the padded KV
/// tail, the mask block map, and the per-workgroup partial results.
fn fa_scratch_sizes(graph: &Graph, id: TensorId) -> (u64, u64, u64) {
    let op = &graph.tensors[id];
    let q = &graph.tensors[op.src(0).unwrap()];
    let k = &graph.tensors[op.src(1).unwrap()];
    let v = &graph.tensors[op.src(2).unwrap()];
    let mask = op.src(3).map(|m| &graph.tensors[m]);
    let ncpsg = OP_FLASH_ATTN_EXT_NCPSG as u64;
    // Reserved for the larger (non-vec) kernel regardless of which runs.
    let mask_elems = mask.map_or(0, |m| (m.ne[1] * m.ne[2] * m.ne[3]) as u64);
    let pad_bytes = ncpsg
        * (k.nb[1] * k.ne[2] as u64 * k.ne[3] as u64 + v.nb[1] * v.ne[2] as u64 * v.ne[3] as u64 + 2 * mask_elems);
    let blk_bytes = match mask {
        None => 0,
        Some(m) => {
            let use_vec = fa_use_vec(q);
            let nqptg = if use_vec { OP_FLASH_ATTN_EXT_VEC_NQPSG } else { OP_FLASH_ATTN_EXT_NQPSG } as i64;
            let ncpsg = if use_vec { OP_FLASH_ATTN_EXT_VEC_NCPSG } else { OP_FLASH_ATTN_EXT_NCPSG } as i64;
            let ne1 = (q.ne[1] + nqptg - 1) / nqptg;
            let ne0 = (m.ne[0] + ncpsg - 1) / ncpsg;
            pad((ne0 * ne1 * m.ne[2] * m.ne[3]) as u64, 32)
        }
    };
    let nwg = 32u64;
    let ne01_max = (q.ne[1] as u64).min(32);
    let tmp_bytes = 4 * (ne01_max * q.ne[2] as u64 * q.ne[3] as u64 * nwg * (v.ne[0] as u64 + 2));
    (pad_bytes, blk_bytes, tmp_bytes)
}

/// Arena scratch a node needs in addition to its output.
pub fn scratch_bytes(graph: &Graph, id: TensorId) -> u64 {
    if graph.tensors[id].op == "FLASH_ATTN_EXT" {
        let (a, b, c) = fa_scratch_sizes(graph, id);
        a + b + c
    } else {
        0
    }
}
