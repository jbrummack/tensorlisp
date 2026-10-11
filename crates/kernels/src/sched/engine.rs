//! The engine: compiled plans, sessions, step arenas and asynchronous
//! submission over any [`Backend`].
//!
//! Mechanism only. Which ticket runs next, batching across sessions and
//! chunked prefill are policy and live above (the server's scheduler loop).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::backend::{Backend, Bound, Launch, MemKind};
use super::blob::KernelBlob;
use super::hazard::{group, level_group, Footprint, Range, Space};
use super::pool::{Block, BlockPool, PoolStats};
use crate::{Error, Result};

fn invalid<T>(msg: String) -> Result<T> {
    Err(Error::Invalid(msg))
}

/// Read-only buffers shared by every session of a model (`Space::Model(i)`).
pub struct ModelMem<B: Backend> {
    pub buffers: Vec<B::Buffer>,
}

impl<B: Backend> ModelMem<B> {
    /// Uploads each slice into its own device buffer.
    pub fn upload(backend: &B, parts: &[&[u8]]) -> Result<Self> {
        let mut buffers = Vec::new();
        for p in parts {
            let buf = backend.alloc(p.len().max(4) as u64, MemKind::Device)?;
            backend.write(&buf, 0, p)?;
            buffers.push(buf);
        }
        Ok(ModelMem { buffers })
    }
}

/// Zeroed state owned by one session (`Space::Session(i)`); its blocks go
/// back to the pool when the session is dropped.
pub struct Session<B: Backend> {
    blocks: Vec<Block<B>>,
}

impl<B: Backend> Session<B> {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// One launch of a plan, before pipelines are resolved.
#[derive(Clone)]
pub struct LaunchDesc {
    pub kernel: Arc<KernelBlob>,
    /// One range per signature binding.
    pub bindings: Vec<Range>,
    pub push: Vec<u8>,
    pub grid: [u32; 3],
}

struct PlannedLaunch {
    pipeline: usize,
    bindings: Vec<Range>,
    push: Vec<u8>,
    grid: [u32; 3],
}

/// A plan with its pipelines: compiled once per (entry, shape class).
pub struct CompiledPlan<B: Backend> {
    pipelines: Vec<Arc<B::Pipeline>>,
    launches: Vec<PlannedLaunch>,
    groups: Vec<Vec<usize>>,
    step_bytes: u64,
    /// Extent each space must have, for the check at submit.
    needs: HashMap<Space, u64>,
}

impl<B: Backend> CompiledPlan<B> {
    pub fn step_bytes(&self) -> u64 {
        self.step_bytes
    }

    /// Launches per barrier-separated group.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups.iter().map(Vec::len).collect()
    }

    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }
}

pub struct Engine<B: Backend> {
    backend: Arc<B>,
    pipelines: Mutex<HashMap<u64, Arc<B::Pipeline>>>,
    sessions: BlockPool<B>,
    steps: BlockPool<B>,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Engine {
            sessions: BlockPool::new(backend.clone(), MemKind::Device, 2, true),
            // An arena bigger than needed serves a smaller plan just as well.
            steps: BlockPool::new(backend.clone(), MemKind::Step, u64::MAX / (1 << 20), false),
            pipelines: Mutex::new(HashMap::new()),
            backend,
        }
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn session_stats(&self) -> PoolStats {
        self.sessions.stats()
    }

    pub fn step_stats(&self) -> PoolStats {
        self.steps.stats()
    }

    /// Uploads each slice into its own device buffer (`Space::Model(i)`).
    pub fn upload_model(&self, parts: &[&[u8]]) -> Result<Arc<ModelMem<B>>> {
        Ok(Arc::new(ModelMem::upload(self.backend.as_ref(), parts)?))
    }

    /// A session with one zeroed block per entry of `sizes` (`Space::Session(i)`).
    pub fn new_session(&self, sizes: &[u64]) -> Result<Session<B>> {
        let blocks = sizes.iter().map(|&s| self.sessions.take(s)).collect::<Result<Vec<_>>>()?;
        Ok(Session { blocks })
    }

    /// Copies `len` bytes at `off` out of the session's block `i`. The caller
    /// must have waited for any ticket that writes the session.
    pub fn read_session(&self, session: &Session<B>, i: usize, off: u64, len: usize) -> Result<Vec<u8>> {
        let block = session.blocks.get(i).ok_or_else(|| Error::Invalid(format!("session has no block {i}")))?;
        let mut out = vec![0u8; len];
        self.backend.read(block.buffer(), off, &mut out)?;
        Ok(out)
    }

    fn pipeline(&self, blob: &KernelBlob) -> Result<Arc<B::Pipeline>> {
        let id = blob.id();
        if let Some(p) = self.pipelines.lock().unwrap().get(&id) {
            return Ok(p.clone());
        }
        let p = Arc::new(self.backend.pipeline(blob)?);
        Ok(self.pipelines.lock().unwrap().entry(id).or_insert(p).clone())
    }

    /// Validates the launches, resolves their pipelines and cuts them into
    /// groups by hazard analysis. `step_bytes` is the extent of `Space::Step`
    /// (inputs, outputs and the planned arena).
    pub fn compile(&self, launches: &[LaunchDesc], step_bytes: u64) -> Result<CompiledPlan<B>> {
        let limits = self.backend.limits();
        let mut pipelines: Vec<Arc<B::Pipeline>> = Vec::new();
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut planned = Vec::with_capacity(launches.len());
        let mut footprints = Vec::with_capacity(launches.len());
        let mut needs: HashMap<Space, u64> = HashMap::new();

        for (n, l) in launches.iter().enumerate() {
            let sig = &l.kernel.sig;
            if l.kernel.target != B::TARGET {
                return invalid(format!("launch {n} ({}): a {:?} kernel cannot run on a {:?} backend", l.kernel.entry, l.kernel.target, B::TARGET));
            }
            if l.bindings.len() != sig.bindings.len() {
                return invalid(format!("launch {n} ({}): {} bindings for a kernel with {}", l.kernel.entry, l.bindings.len(), sig.bindings.len()));
            }
            if l.push.len() != sig.push_bytes as usize {
                return invalid(format!("launch {n} ({}): {} push bytes for a kernel taking {}", l.kernel.entry, l.push.len(), sig.push_bytes));
            }
            if l.grid.iter().any(|&g| g == 0) || l.grid.iter().zip(limits.max_groups).any(|(&g, m)| g > m) {
                return invalid(format!("launch {n} ({}): grid {:?} is empty or over the device limit {:?}", l.kernel.entry, l.grid, limits.max_groups));
            }
            let mut fp = Footprint::default();
            for (r, access) in l.bindings.iter().zip(&sig.bindings) {
                if r.space == Space::Step && r.off + r.len > step_bytes {
                    return invalid(format!("launch {n} ({}): step range {}+{} is past the arena ({step_bytes} bytes)", l.kernel.entry, r.off, r.len));
                }
                let e = needs.entry(r.space).or_insert(0);
                *e = (*e).max(r.off + r.len);
                if access.reads() {
                    fp.reads.push(*r);
                }
                if access.writes() {
                    fp.writes.push(*r);
                }
            }
            let id = l.kernel.id();
            let pipeline = match index.get(&id) {
                Some(&i) => i,
                None => {
                    pipelines.push(self.pipeline(&l.kernel)?);
                    index.insert(id, pipelines.len() - 1);
                    pipelines.len() - 1
                }
            };
            planned.push(PlannedLaunch { pipeline, bindings: l.bindings.clone(), push: l.push.clone(), grid: l.grid });
            footprints.push(fp);
        }
        let groups = if std::env::var_os("TL_SCHED_PROGRAM_ORDER").is_some() { group(&footprints) } else { level_group(&footprints) };
        Ok(CompiledPlan { pipelines, launches: planned, groups, step_bytes, needs })
    }

    /// Takes an arena, writes `inputs` (step offset, bytes) into it, records
    /// and submits the plan, and returns at once. The ticket keeps everything
    /// the device touches alive.
    pub fn submit(
        &self,
        plan: &CompiledPlan<B>,
        model: &Arc<ModelMem<B>>,
        session: Option<&Arc<Session<B>>>,
        inputs: &[(u64, &[u8])],
    ) -> Result<Ticket<B>> {
        self.submit_reading(plan, model, session, inputs, &[])
    }

    /// [`Engine::submit`] that also asks for the step-arena ranges `reads` (offset, length) to be copied
    /// to the host along the way: [`Ticket::read`] of them then costs no device round trip.
    pub fn submit_reading(
        &self,
        plan: &CompiledPlan<B>,
        model: &Arc<ModelMem<B>>,
        session: Option<&Arc<Session<B>>>,
        inputs: &[(u64, &[u8])],
        reads: &[(u64, u64)],
    ) -> Result<Ticket<B>> {
        for (space, &need) in &plan.needs {
            let have = match *space {
                Space::Step => plan.step_bytes,
                Space::Model(i) => model.buffers.get(i as usize).map(|b| self.backend.buffer_len(b)).unwrap_or(0),
                Space::Session(i) => match session {
                    Some(s) => s.blocks.get(i as usize).map(|b| self.backend.buffer_len(b.buffer())).unwrap_or(0),
                    None => return invalid("the plan uses session state but no session was given".into()),
                },
            };
            if have < need {
                return invalid(format!("{space:?} has {have} bytes, the plan needs {need}"));
            }
        }
        let arena = self.steps.take(plan.step_bytes.max(4))?;
        for (off, data) in inputs {
            if off + data.len() as u64 > plan.step_bytes {
                return invalid(format!("input at {off} ({} bytes) is past the arena ({} bytes)", data.len(), plan.step_bytes));
            }
            self.backend.write(arena.buffer(), *off, data)?;
        }

        let resolve = |r: &Range| -> Bound<'_, B> {
            let buffer = match r.space {
                Space::Model(i) => &model.buffers[i as usize],
                Space::Session(i) => session.expect("checked above").blocks[i as usize].buffer(),
                Space::Step => arena.buffer(),
            };
            Bound { buffer, offset: r.off, len: r.len }
        };
        let groups: Vec<Vec<Launch<'_, B>>> = plan
            .groups
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&i| {
                        let l = &plan.launches[i];
                        Launch {
                            pipeline: plan.pipelines[l.pipeline].as_ref(),
                            bindings: l.bindings.iter().map(&resolve).collect(),
                            push: &l.push,
                            grid: l.grid,
                        }
                    })
                    .collect()
            })
            .collect();
        let reads: Vec<(&B::Buffer, u64, u64)> = reads.iter().filter(|(off, len)| off + len <= plan.step_bytes).map(|&(off, len)| (arena.buffer(), off, len)).collect();
        let pending = self.backend.submit_reading(&groups, &reads)?;
        drop(groups);
        drop(reads);
        Ok(Ticket {
            backend: self.backend.clone(),
            pending: Some(pending),
            arena: Some(arena),
            _model: model.clone(),
            _session: session.cloned(),
            _pipelines: plan.pipelines.clone(),
        })
    }
}

/// A submission in flight.
pub struct Ticket<B: Backend> {
    backend: Arc<B>,
    pending: Option<B::Pending>,
    arena: Option<Block<B>>,
    _model: Arc<ModelMem<B>>,
    _session: Option<Arc<Session<B>>>,
    _pipelines: Vec<Arc<B::Pipeline>>,
}

impl<B: Backend> Ticket<B> {
    /// True once the device has finished.
    pub fn poll(&self) -> Result<bool> {
        match &self.pending {
            Some(p) => self.backend.poll(p),
            None => Ok(true),
        }
    }

    pub fn wait(&self) -> Result<()> {
        match &self.pending {
            Some(p) => self.backend.wait(p),
            None => Ok(()),
        }
    }

    /// Waits, then copies `len` bytes at `off` of the step arena.
    pub fn read(&self, off: u64, len: usize) -> Result<Vec<u8>> {
        self.wait()?;
        let mut out = vec![0u8; len];
        let arena = self.arena.as_ref().expect("a live ticket holds its arena").buffer();
        if let Some(p) = &self.pending {
            if self.backend.prefetched(p, arena, off, &mut out) {
                return Ok(out);
            }
        }
        self.backend.read(arena, off, &mut out)?;
        Ok(out)
    }
}

impl<B: Backend> Drop for Ticket<B> {
    fn drop(&mut self) {
        // The device may still be using the arena and the session's blocks.
        let _ = self.wait();
        self.pending = None;
        self.arena = None;
    }
}
