//! The native Vulkan executor: a [`Graph`] (the IR a Scheme program builds)
//! planned into one arena, lowered to launches of ggml's Vulkan shaders and
//! run through the scheduler's [`Engine`].
//!
//! Memory spaces map onto the scheduler's: weights are `Space::Model(buffer)`,
//! state tensors one `Space::Session(0)` block, and inputs plus the activation
//! arena live in the submission's `Space::Step` arena (inputs first, the planned
//! arena after them).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::ggml::SpvLibrary;
use super::lower::{scratch_bytes, Ctx, Shaders};
use super::tree::{TreeKernels, TreeStats};
use super::{Caps, Vulkan};
use crate::graph::{Graph, TensorId};
use crate::plan::{align, plan_spread, BufId, Layout, Leaves};
use crate::sched::{Backend, CompiledPlan, Engine, ModelMem, Session, Ticket};
use crate::{Error, Result};

pub struct Executor {
    pub engine: Arc<Engine<Vulkan>>,
    shaders: Shaders,
    pub caps: Caps,
    tree: Option<TreeKernels>,
    /// The Tree compiler for the paged-attention nodes, which have no ggml shader (kept apart from
    /// `tree`, whose presence switches fusion and lowering of every other node).
    paged: Option<TreeKernels>,
}

impl Executor {
    /// Wraps `vk`, loading ggml's precompiled shader library (see
    /// `scripts/build-ggml-vulkan-shaders.sh`; `TL_GGML_SPV` overrides the directory).
    pub fn new(vk: Vulkan) -> Result<Executor> {
        let dir = SpvLibrary::default_dir();
        let lib = SpvLibrary::open(&dir)
            .filter(|l| l.contains("add_f32_f32_f32"))
            .ok_or_else(|| Error::Invalid(format!("no precompiled ggml Vulkan shaders in {}: run scripts/build-ggml-vulkan-shaders.sh", dir.display())))?;
        let caps = vk.caps();
        Ok(Executor { engine: Arc::new(Engine::new(Arc::new(vk))), shaders: Shaders::new(lib), caps, tree: None, paged: None })
    }

    /// Lowers nodes with a Tree generator to generated kernels (ggml's shaders stay the fallback
    /// unless `tree.only`). Fusion is off then: its pseudo-ops are ggml shaders.
    pub fn set_tree(&mut self, tree: TreeKernels) {
        self.tree = Some(tree);
    }

    /// Lets `CUSTOM` nodes (paged attention, cache write) compile their Tree kernels.
    pub fn set_paged_kernels(&mut self, tree: TreeKernels) {
        self.paged = Some(tree);
    }

    pub fn backend(&self) -> &Arc<Vulkan> {
        self.engine.backend()
    }

    /// Lowers every node, then compiles the launches into a plan. Missing ops are
    /// all reported in one error.
    pub fn compile(&self, graph: Graph, leaves: &Leaves) -> Result<Program> {
        let caps = self.caps;
        let mut graph = graph;
        if std::env::var_os("TL_NATIVE_TRACE").is_some() {
            let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
            for &id in &graph.nodes {
                *counts.entry(graph.tensors[id].op.as_str()).or_default() += 1;
            }
            eprintln!("native vulkan: graph ops {counts:?}");
        }
        // TL_NATIVE_NOFUSE: "1" disables fusion, or a comma-separated list of pseudo-op names.
        let nofuse = std::env::var("TL_NATIVE_NOFUSE").unwrap_or_default();
        if nofuse != "1" {
            // With Tree kernels only the pseudo-ops that have one are formed (norm + weight).
            let disabled = if self.tree.is_some() { format!("{nofuse},RMS_NORM_MUL_ROPE") } else { nofuse.clone() };
            let n = super::fuse::fuse(&mut graph, &disabled);
            if std::env::var_os("TL_NATIVE_TRACE").is_some() {
                eprintln!("native vulkan: fused {n:?}");
            }
        }
        // Tree kernels: repeated nodes are computed once and elementwise regions become one kernel each
        // (TL_TREE_NOFUSE=1 turns both off).
        if self.tree.is_some() && std::env::var_os("TL_TREE_NOFUSE").is_none() {
            let dup = super::fuse_tree::cse(&mut graph);
            let rc = super::fuse_tree::row_chains(&mut graph);
            let nr = super::fuse_tree::norm_rope(&mut graph);
            let gu = if std::env::var_os("TL_TREE_GU").is_some() {
                super::fuse_tree::gate_up(&mut graph, &|g, id| super::lower::q8_dot_ok(g, &caps, id), caps.subgroup_size == 32)
            } else {
                0
            };
            let ew = super::fuse_tree::elementwise(&mut graph);
            let q8 = super::fuse_tree::mark_q8_outputs(&mut graph, &|g, id| super::lower::q8_dot_ok(g, &caps, id));
            if std::env::var_os("TL_NATIVE_TRACE").is_some() {
                eprintln!("native vulkan: {dup} repeated nodes removed, {rc} row chains, {nr} norm+rope chains, {gu} gate/up pairs, {ew} elementwise regions, {q8} quantized producers");
            }
        }
        let graph = graph;
        // TL_NATIVE_DUMP=<file>: the node list after fusion (op, type, shape, sources) for reading.
        if let Ok(path) = std::env::var("TL_NATIVE_DUMP") {
            let mut out = format!("=== program, {} nodes
", graph.nodes.len());
            for &id in &graph.nodes {
                let t = &graph.tensors[id];
                let srcs: Vec<String> = t.src.iter().flatten().map(|&s| {
                    let x = &graph.tensors[s];
                    format!("{}#{}{:?}", x.op, s, &x.ne[..])
                }).collect();
                out.push_str(&format!("#{id} {} {:?} {:?} <- {}\n", t.op, t.ty, &t.ne[..], srcs.join(" ")));
            }
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = f.write_all(out.as_bytes());
                }
            }
        }
        let tree_on = self.tree.is_some();
        let layout = plan_spread(&graph, leaves, &|id| scratch_bytes(&graph, id, &caps, tree_on), 64 << 20);
        let arena_base = align(layout.io_size);
        let step_bytes = (arena_base + layout.arena_size).max(4);
        let pos: HashMap<TensorId, usize> = graph.nodes.iter().enumerate().map(|(p, &id)| (id, p)).collect();
        let stats = TreeStats::default();
        let ctx = Ctx { shaders: &self.shaders, caps, graph: &graph, layout: &layout, arena_base, pos: &pos, tree: self.tree.as_ref(), paged: self.paged.as_ref(), stats: &stats };

        let mut launches = Vec::new();
        let mut missing: Vec<(String, String)> = Vec::new();
        for &id in &graph.nodes {
            match ctx.lower(id) {
                Ok(l) => launches.extend(l),
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
            return Err(Error::Invalid(format!("native Vulkan: not implemented: {}", list.join("; "))));
        }
        if self.tree.is_some() && std::env::var_os("TL_NATIVE_TRACE").is_some() {
            eprintln!("native vulkan: {} nodes on Tree kernels, ggml shaders for {:?}", stats.tree_nodes.get(), stats.fallbacks.borrow());
        }
        let plan = self.engine.compile(&launches, step_bytes)?;
        if std::env::var_os("TL_NATIVE_TRACE").is_some() {
            let sizes = plan.group_sizes();
            eprintln!("native vulkan: {} launches in {} groups (largest {}), arena {} bytes", plan.launch_count(), sizes.len(), sizes.iter().max().unwrap_or(&0), step_bytes);
        }
        let n_inputs = layout.inputs.len();
        // The results the caller will read (outputs and taps) are copied to the host with the run.
        let mut reads: Vec<(u64, u64)> = Vec::new();
        let mut budget = 32u64 << 20;
        for &k in &graph.keep {
            let (r, _) = graph.root(k);
            let bytes = graph.tensors[r].nbytes();
            let p = layout.place[r];
            let off = match p.buf {
                BufId::Arena => arena_base + p.off,
                BufId::Io => p.off,
                _ => continue,
            };
            if bytes > 0 && bytes <= budget && !reads.contains(&(off, bytes)) {
                budget -= bytes;
                reads.push((off, bytes));
            }
        }
        Ok(Program {
            reads,
            engine: self.engine.clone(),
            plan,
            layout,
            arena_base,
            inputs: Mutex::new(vec![None; n_inputs]),
            last: Mutex::new(None),
        })
    }
}

/// A compiled graph: the plan plus how to feed it and read it back.
pub struct Program {
    engine: Arc<Engine<Vulkan>>,
    /// Step-arena ranges copied to the host along with every run (the graph's outputs and taps).
    reads: Vec<(u64, u64)>,
    plan: CompiledPlan<Vulkan>,
    pub layout: Layout,
    arena_base: u64,
    inputs: Mutex<Vec<Option<Vec<u8>>>>,
    /// The finished submission; its arena holds the results until the next run.
    last: Mutex<Option<Ticket<Vulkan>>>,
}

impl Program {
    pub fn launch_count(&self) -> usize {
        self.plan.launch_count()
    }

    /// Stages graph input `i` (bytes in the tensor's layout) for the next [`Program::run`].
    pub fn set_input(&self, i: usize, data: &[u8]) -> Result<()> {
        let (_, bytes) = *self.layout.inputs.get(i).ok_or_else(|| Error::Invalid(format!("no graph input {i}")))?;
        if data.len() as u64 != bytes {
            return Err(Error::Invalid(format!("input {i} takes {bytes} bytes, got {}", data.len())));
        }
        self.inputs.lock().unwrap()[i] = Some(data.to_vec());
        Ok(())
    }

    /// Submits the plan with the staged inputs and waits for it.
    pub fn run(&self, model: &Arc<ModelMem<Vulkan>>, session: Option<&Arc<Session<Vulkan>>>) -> Result<()> {
        let staged = self.inputs.lock().unwrap();
        let mut writes: Vec<(u64, &[u8])> = Vec::new();
        for (i, data) in staged.iter().enumerate() {
            if let Some(d) = data {
                writes.push((self.layout.inputs[i].0, d.as_slice()));
            }
        }
        // Drop the previous results first, so its arena can be reused.
        *self.last.lock().unwrap() = None;
        let t0 = std::time::Instant::now();
        let ticket = self.engine.submit_reading(&self.plan, model, session, &writes, &self.reads)?;
        let t1 = std::time::Instant::now();
        ticket.wait()?;
        if std::env::var_os("TL_NATIVE_TIME").is_some() {
            eprintln!(
                "run: {} launches, record+submit {:.2} ms, device wait {:.2} ms",
                self.plan.launch_count(),
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        *self.last.lock().unwrap() = Some(ticket);
        Ok(())
    }

    /// Copies out `bytes` of tensor `id` after [`Program::run`]; only inputs and activations.
    pub fn read(&self, id: TensorId, bytes: usize) -> Result<Vec<u8>> {
        let p = self.layout.place[id];
        let off = match p.buf {
            BufId::Arena => self.arena_base + p.off,
            BufId::Io => p.off,
            other => return Err(Error::Invalid(format!("tensor {id} lives in {other:?}, which Program cannot read"))),
        };
        let last = self.last.lock().unwrap();
        let ticket = last.as_ref().ok_or_else(|| Error::Invalid("the program has not run".into()))?;
        ticket.read(off, bytes)
    }

    pub fn backend_name(&self) -> String {
        self.engine.backend().name()
    }
}
