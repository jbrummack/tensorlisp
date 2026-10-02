//! Compiles a [`Graph`] to a recorded dispatch list and replays it.
//!
//! `Executor::compile` does everything that depends on shapes once: memory
//! planning, kernel selection, pipeline creation, argument packing. `Program::run`
//! only encodes the list into one command buffer, commits it and waits.

use std::sync::Arc;

use super::lower::{scratch_bytes, Ctx, DArg, Dispatch, MatmulLowering};
use super::{Arg, BufRef, Buffer, Device};
use crate::graph::{Graph, TensorId};
use crate::plan::{plan, BufId, Layout, Leaves, Place};
use crate::{Error, Result};

const EXTRA_METAL: &str = include_str!("shaders/extra.metal");
const GGML_METAL: &str = include_str!(concat!(env!("OUT_DIR"), "/ggml-metal-merged.metal"));

pub struct Executor {
    pub dev: Arc<Device>,
    lib: Arc<str>,
    extra: Arc<str>,
}

impl Executor {
    /// Compiles ggml's Metal library for `dev` (a few seconds, once per Device).
    pub fn new(dev: Arc<Device>) -> Result<Self> {
        // The merged source has ggml-common.h inlined, which ggml selects with this macro.
        let mut macros = vec![("GGML_METAL_EMBED_LIBRARY", "1")];
        if dev.has_bfloat() {
            macros.push(("GGML_METAL_HAS_BF16", "1"));
        }
        let lib = dev.library_with("ggml-metal", &macros, || GGML_METAL.to_string())?;
        let extra = dev.library("tl-extra", || EXTRA_METAL.to_string())?;
        Ok(Executor { dev, lib, extra })
    }

    pub fn compile(&self, graph: Graph, leaves: &Leaves, matmul: &dyn MatmulLowering) -> Result<Program> {
        let layout = plan(&graph, leaves, &|id| scratch_bytes(&graph, id));
        let ctx = Ctx {
            dev: &self.dev,
            lib: self.lib.clone(),
            extra: self.extra.clone(),
            graph: &graph,
            layout: &layout,
            matmul,
            props: self.dev.props(),
        };
        let mut dispatches = Vec::new();
        // Lower everything before failing, so one run reports every missing op.
        let mut missing: Vec<(String, String)> = Vec::new();
        for &id in &graph.nodes {
            match ctx.lower(id) {
                Ok(d) => dispatches.extend(d),
                Err(Error::Unsupported { op, node, why }) => missing.push((op, format!("{node}: {why}"))),
                Err(e) => return Err(e),
            }
        }
        if !missing.is_empty() {
            let mut counts: Vec<(String, usize, String)> = Vec::new();
            for (op, detail) in missing {
                match counts.iter_mut().find(|(o, _, _)| *o == op) {
                    Some(c) => c.1 += 1,
                    None => counts.push((op, 1, detail)),
                }
            }
            let list: Vec<String> = counts.iter().map(|(op, n, d)| format!("{op} x{n} (e.g. {d})")).collect();
            return Err(Error::Invalid(format!("native Metal: not implemented: {}", list.join("; "))));
        }
        let arena = self.dev.alloc(layout.arena_size as usize)?;
        let io = self.dev.alloc(layout.io_size as usize)?;
        Ok(Program { dev: self.dev.clone(), dispatches, layout, arena, io })
    }
}

/// A compiled graph: dispatches plus the buffers it owns (arena, inputs).
pub struct Program {
    dev: Arc<Device>,
    dispatches: Vec<Dispatch>,
    pub layout: Layout,
    arena: Buffer,
    io: Buffer,
}

/// The buffers shared between programs.
pub struct Shared<'a> {
    pub weights: &'a [Buffer],
    pub state: &'a Buffer,
}

impl Program {
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.len()
    }

    /// Dispatch labels in order, for debugging.
    pub fn labels(&self) -> Vec<&str> {
        self.dispatches.iter().map(|d| d.label.as_str()).collect()
    }

    /// Writes graph input `i` (bytes in the tensor's layout).
    pub fn set_input(&self, i: usize, data: &[u8]) -> Result<()> {
        let (off, bytes) = *self.layout.inputs.get(i).ok_or_else(|| Error::Invalid(format!("no graph input {i}")))?;
        if data.len() as u64 != bytes {
            return Err(Error::Invalid(format!("input {i} takes {bytes} bytes, got {}", data.len())));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.io.contents().add(off as usize), data.len()) };
        Ok(())
    }

    fn resolve<'a>(&'a self, place: Place, shared: &'a Shared) -> BufRef<'a> {
        let buf = match place.buf {
            BufId::Weights(i) => &shared.weights[i],
            BufId::State => shared.state,
            BufId::Io => &self.io,
            BufId::Arena => &self.arena,
        };
        BufRef { buf, offset: place.off as usize }
    }

    /// Runs the graph and waits for it.
    pub fn run(&self, shared: &Shared) -> Result<()> {
        let mut args: Vec<(u32, Arg)> = Vec::with_capacity(10);
        for d in &self.dispatches {
            args.clear();
            for (slot, a) in &d.args {
                args.push((
                    *slot,
                    match a {
                        DArg::Buf(p) => Arg::Buf(self.resolve(*p, shared)),
                        DArg::Bytes(b) => Arg::Bytes(b),
                    },
                ));
            }
            self.dev.launch_pipeline(&d.pipeline, &args, d.dims)?;
        }
        self.dev.sync().map_err(|e| match e {
            Error::Execution(m) => Error::Execution(format!("{m} (graph of {} dispatches)", self.dispatches.len())),
            e => e,
        })
    }

    /// Copies out `bytes` of tensor `id` (after [`Program::run`]); only for tensors in the arena or io buffer.
    pub fn read(&self, id: TensorId, bytes: usize) -> Result<Vec<u8>> {
        let p = self.layout.place[id];
        let buf = match p.buf {
            BufId::Arena => &self.arena,
            BufId::Io => &self.io,
            other => return Err(Error::Invalid(format!("tensor {id} lives in {other:?}, which Program cannot read"))),
        };
        if p.off as usize + bytes > buf.len() {
            return Err(Error::Invalid(format!("tensor {id} reads past its buffer")));
        }
        Ok(buf.read_range::<u8>(p.off as usize, bytes))
    }
}
