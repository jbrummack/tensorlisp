//! Compiles a [`Graph`] to a recorded list of CUDA launches and replays it
//! (the CUDA analogue of `metal::exec`).

use std::sync::Arc;

use super::lower::{core_library, scratch_bytes, Ctx, Custom, Dispatch};
use super::paged_attn::{
    paged_attention_prepared, reshape_and_cache_with, PagedAttention, ReshapeAndCache,
};
use super::{Arg, BufRef, Buffer, Device};
use crate::graph::{Graph, TensorId};
use crate::plan::{plan, BufId, Layout, Leaves, Place};
use crate::{Error, Result};

pub struct Executor {
    pub dev: Arc<Device>,
    core: Arc<str>,
}

impl Executor {
    /// Compiles the core kernel library for `dev` (once per Device).
    pub fn new(dev: Arc<Device>) -> Result<Self> {
        let core = core_library(&dev)?;
        Ok(Executor { dev, core })
    }

    pub fn compile(&self, graph: Graph, leaves: &Leaves) -> Result<Program> {
        let sm_count = self.dev.sm_count();
        let layout = plan(&graph, leaves, &|id| scratch_bytes(&graph, id, sm_count));
        let ctx = Ctx {
            dev: &self.dev,
            core: self.core.clone(),
            graph: &graph,
            layout: &layout,
            max_shared: self.dev.max_shared_memory(),
            sm_count,
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
            return Err(Error::Invalid(format!("native CUDA: not implemented: {}", list.join("; "))));
        }
        let arena = self.dev.alloc(layout.arena_size.max(1) as usize)?;
        let io = self.dev.alloc(layout.io_size.max(1) as usize)?;
        Ok(Program { dev: self.dev.clone(), dispatches, layout, arena, io })
    }
}

/// A compiled graph: launches plus the buffers it owns (arena, inputs).
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

    /// Launch labels in order, for debugging.
    pub fn labels(&self) -> Vec<&str> {
        self.dispatches.iter().map(|d| d.label.as_str()).collect()
    }

    /// Writes graph input `i` (bytes in the tensor's layout).
    pub fn set_input(&self, i: usize, data: &[u8]) -> Result<()> {
        let (off, bytes) = *self.layout.inputs.get(i).ok_or_else(|| Error::Invalid(format!("no graph input {i}")))?;
        if data.len() as u64 != bytes {
            return Err(Error::Invalid(format!("input {i} takes {bytes} bytes, got {}", data.len())));
        }
        self.io.write_at(off as usize, data);
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

    fn launch(&self, d: &Dispatch, shared: &Shared) -> Result<()> {
        match &d.custom {
            None => {
                let mut args = [Arg::NullPtr; 6];
                args[0] = Arg::Bytes(&d.op.0);
                for (i, s) in d.srcs.iter().enumerate() {
                    if let Some(p) = s {
                        args[1 + i] = Arg::Buf(self.resolve(*p, shared));
                    }
                }
                args[5] = Arg::Buf(self.resolve(d.dst, shared));
                self.dev.launch_func(d.func.expect("a kernel dispatch has a function"), &args, d.dims)
            }
            Some(Custom::PagedAttention(l)) => {
                let g = &l.geometry;
                let p = PagedAttention {
                    dtype: g.dtype,
                    cache_dtype: g.cache_dtype,
                    num_seqs: g.num_seqs,
                    num_heads: g.num_heads,
                    num_kv_heads: g.num_kv_heads,
                    head_size: g.head_size,
                    block_size: g.block_size,
                    max_context_len: g.max_context_len,
                    max_num_blocks_per_seq: l.max_blocks,
                    scale: l.scale,
                    softcapping: 1.0,
                    q_stride: g.num_heads * g.head_size,
                    kv_block_stride: l.kv_block_stride,
                    kv_head_stride: l.kv_head_stride,
                    out: self.resolve(l.out, shared),
                    q: self.resolve(l.q, shared),
                    k_cache: self.resolve(l.k_cache, shared),
                    v_cache: self.resolve(l.v_cache, shared),
                    block_tables: self.resolve(l.block_tables, shared),
                    context_lens: self.resolve(l.context_lens, shared),
                    kv_scales: None,
                    alibi_slopes: None,
                    sinks: None,
                };
                paged_attention_prepared(&self.dev, &p, &l.prepared, Some(&l.scratch))
            }
            Some(Custom::CacheWrite(l)) => {
                let width = l.kv_heads * l.head_size;
                let p = ReshapeAndCache {
                    dtype: l.dtype,
                    cache_dtype: l.dtype,
                    num_tokens: l.tokens,
                    num_heads: l.kv_heads,
                    head_size: l.head_size,
                    block_size: l.block_size,
                    key_stride: width,
                    value_stride: width,
                    key: self.resolve(l.key, shared),
                    value: self.resolve(l.value, shared),
                    key_cache: self.resolve(l.k_cache, shared),
                    value_cache: self.resolve(l.v_cache, shared),
                    slot_mapping: self.resolve(l.slots, shared),
                    kv_scales: None,
                };
                reshape_and_cache_with(&self.dev, &p, l.func)
            }
        }
    }

    /// Enqueues the whole graph `n` times and waits once (for benchmarks).
    pub fn run_n(&self, shared: &Shared, n: usize) -> Result<()> {
        for _ in 0..n {
            for d in &self.dispatches {
                self.launch(d, shared)?;
            }
        }
        self.dev.sync()
    }

    /// Runs the graph and waits for it.
    pub fn run(&self, shared: &Shared) -> Result<()> {
        let profile = std::env::var_os("TL_NATIVE_PROFILE").is_some();
        let wall = std::time::Instant::now();
        let mut times: Vec<(&str, f64)> = Vec::new();
        for d in &self.dispatches {
            let t0 = std::time::Instant::now();
            self.launch(d, shared).map_err(|e| match e {
                Error::Execution(m) => Error::Execution(format!("launching `{}`: {m}", d.label)),
                e => e,
            })?;
            if profile {
                self.dev.sync()?;
                times.push((d.label.split(' ').next().unwrap_or(""), t0.elapsed().as_secs_f64() * 1e3));
            }
        }
        if profile {
            let mut by_op: Vec<(&str, f64, usize)> = Vec::new();
            for (op, ms) in &times {
                match by_op.iter_mut().find(|e| e.0 == *op) {
                    Some(e) => {
                        e.1 += ms;
                        e.2 += 1;
                    }
                    None => by_op.push((op, *ms, 1)),
                }
            }
            by_op.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = by_op.iter().map(|e| e.1).sum();
            let list: Vec<String> = by_op.iter().take(6).map(|(o, ms, n)| format!("{o} {ms:.2}ms/{n}")).collect();
            eprintln!("profile: {} launches {total:.2} ms (sync each): {}", times.len(), list.join(", "));
        }
        let done = self.dev.sync().map_err(|e| match e {
            Error::Execution(m) => Error::Execution(format!("{m} (graph of {} dispatches)", self.dispatches.len())),
            e => e,
        });
        if std::env::var_os("TL_NATIVE_TIME").is_some() {
            eprintln!("run: {} launches {:.2} ms", self.dispatches.len(), wall.elapsed().as_secs_f64() * 1e3);
        }
        done
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
