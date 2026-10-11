//! The CUDA implementation of [`crate::sched::Backend`].
//!
//! Kernels are NVRTC-compiled CUDA C blobs ([`Code::Cuda`]) or PTX text ([`Code::Ptx`], JIT-compiled
//! by the driver). Calling
//! convention (the CUDA counterpart of the Vulkan one in docs/design/scheduler.md):
//!
//! * the kernel is `extern "C" __global__`;
//! * one pointer parameter per signature binding, in order (byte offsets are
//!   applied by the backend, so [`OffsetMode`] does not matter here);
//! * then, when `push_bytes > 0`, ONE by-value struct parameter holding the
//!   kernel's parameters, laid out exactly as the bytes the lowering passes;
//! * the launch grid is the number of blocks and `sig.workgroup` the block
//!   shape;
//! * `blob.spec[i]` becomes `#define TL_SPEC_<i> <value>u` ahead of the source,
//!   the CUDA analogue of a Vulkan specialization constant.
//!
//! Everything runs on the device's single in-order stream, so copies, launches
//! and the completion event of a submission are ordered by the stream itself:
//! hazard groups need no barrier (they are what a multi-stream backend would
//! use to overlap work). Host copies are the stream-ordered async variants; a
//! read synchronizes the stream.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::sys::{self, CUdeviceptr, CUevent, CUgraphExec, CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
use super::{check, Arg, BufRef, Buffer, Device, Func, Kernel};
use crate::sched::backend::{Backend, Launch, Limits, MemKind};
use crate::sched::blob::{Code, KernelBlob, Target};
use crate::{Dims, Error, Result};

/// Per-kernel GPU time (`TL_CUDA_PROFILE=1`): launches are timed with events one by one, so a run is
/// slower; the table (label, launches, total ms) is printed when the last handle to the backend goes.
#[derive(Default)]
struct Profile {
    on: bool,
    kernels: Mutex<HashMap<String, (u64, f64)>>,
}

impl Drop for Profile {
    fn drop(&mut self) {
        let kernels = self.kernels.lock().unwrap();
        if !self.on || kernels.is_empty() {
            return;
        }
        let mut rows: Vec<_> = kernels.iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        let total: f64 = rows.iter().map(|r| r.1 .1).sum();
        eprintln!("cuda profile: {total:.2} ms over {} kernels", rows.len());
        for (label, (n, ms)) in rows.iter().take(30) {
            eprintln!("  {ms:9.3} ms {n:6} x {:8.1} us  {label}", ms * 1e3 / *n as f64);
        }
    }
}

/// CUDA graphs of repeated submissions. A decode step launches the same few hundred kernels with the
/// same buffers, push constants and grids every time; once a submission equals the previous one it is
/// captured from the stream into a graph, and later equal submissions are one `cuGraphLaunch` instead of
/// hundreds of `cuLaunchKernel` calls (`TL_CUDA_GRAPHS=0` turns it off).
struct Graphs {
    dev: Arc<Device>,
    on: bool,
    last: Mutex<Vec<u64>>,
    execs: Mutex<Vec<(Vec<u64>, CUgraphExec)>>,
}

/// Graphs kept (the oldest goes first).
const GRAPHS: usize = 8;

impl Drop for Graphs {
    fn drop(&mut self) {
        let execs = self.execs.get_mut().unwrap();
        if execs.is_empty() || self.dev.make_current().is_err() {
            return;
        }
        if let Some(drv) = sys::driver() {
            for (_, e) in execs.drain(..) {
                unsafe { (drv.cuGraphExecDestroy)(e) };
            }
        }
    }
}

// SAFETY: a CUgraphExec is a driver handle used under the device's context.
unsafe impl Send for Graphs {}
unsafe impl Sync for Graphs {}

/// Everything a submission's kernels depend on, flattened: equal keys launch the same work.
fn graph_key(groups: &[Vec<Launch<'_, Cuda>>]) -> Vec<u64> {
    let mut key = Vec::new();
    for l in groups.iter().flatten() {
        key.push(l.pipeline.func.0 as u64);
        for b in &l.bindings {
            key.push(b.buffer.ptr + b.offset);
        }
        key.extend(l.push.chunks(8).map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        }));
        key.push(l.grid[0] as u64 | (l.grid[1] as u64) << 32);
        key.push(l.grid[2] as u64 | u64::MAX << 32);
    }
    key
}

#[derive(Clone)]
pub struct Cuda {
    dev: Arc<Device>,
    profile: Arc<Profile>,
    graphs: Arc<Graphs>,
}

impl Cuda {
    pub fn system_default() -> Result<Cuda> {
        Ok(Cuda::from_device(Arc::new(Device::system_default()?)))
    }

    /// Wraps an existing device (e.g. the one a native executor already owns).
    pub fn from_device(dev: Arc<Device>) -> Cuda {
        let on = std::env::var_os("TL_CUDA_PROFILE").is_some_and(|v| v != "0");
        let graphs = Graphs {
            dev: dev.clone(),
            on: !on && std::env::var("TL_CUDA_GRAPHS").map_or(true, |v| v != "0"),
            last: Mutex::new(Vec::new()),
            execs: Mutex::new(Vec::new()),
        };
        Cuda { dev, profile: Arc::new(Profile { on, kernels: Mutex::new(HashMap::new()) }), graphs: Arc::new(graphs) }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }
}

pub struct Pipeline {
    label: String,
    func: Func,
    threads: [u32; 3],
    bindings: usize,
    push_bytes: usize,
}

/// A recorded completion event; dropping it waits for the work before it.
pub struct Pending {
    dev: Arc<Device>,
    event: CUevent,
}

// SAFETY: a CUevent is a driver handle; every use sets the context current.
unsafe impl Send for Pending {}

impl Drop for Pending {
    fn drop(&mut self) {
        if self.dev.make_current().is_ok() {
            if let Some(drv) = sys::driver() {
                unsafe {
                    (drv.cuEventSynchronize)(self.event);
                    (drv.cuEventDestroy_v2)(self.event);
                }
            }
        }
    }
}

fn driver() -> Result<&'static sys::Driver> {
    sys::driver().ok_or(Error::NoDevice)
}

impl Cuda {
    /// Launches every kernel of `groups` on the stream, one after the other.
    fn launch_all(&self, groups: &[Vec<Launch<'_, Cuda>>]) -> Result<()> {
        for l in groups.iter().flatten() {
            let p = l.pipeline;
            if l.bindings.len() != p.bindings || l.push.len() != p.push_bytes {
                return Err(Error::Invalid(format!(
                    "launch has {} bindings and {} push bytes, its kernel takes {} and {}",
                    l.bindings.len(),
                    l.push.len(),
                    p.bindings,
                    p.push_bytes
                )));
            }
            let mut args: Vec<Arg<'_>> = l.bindings.iter().map(|b| Arg::Buf(BufRef { buf: b.buffer, offset: b.offset as usize })).collect();
            if p.push_bytes > 0 {
                args.push(Arg::Bytes(l.push));
            }
            if self.profile.on {
                self.dev.make_current()?;
                let drv = driver()?;
                let (mut a, mut b): (CUevent, CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
                check(unsafe { (drv.cuEventCreate)(&mut a, 0) })?;
                check(unsafe { (drv.cuEventCreate)(&mut b, 0) })?;
                check(unsafe { (drv.cuEventRecord)(a, self.dev.stream) })?;
                self.dev.launch_func(p.func, &args, Dims::new(l.grid, p.threads))?;
                check(unsafe { (drv.cuEventRecord)(b, self.dev.stream) })?;
                check(unsafe { (drv.cuEventSynchronize)(b) })?;
                let mut ms = 0f32;
                check(unsafe { (drv.cuEventElapsedTime)(&mut ms, a, b) })?;
                unsafe {
                    (drv.cuEventDestroy_v2)(a);
                    (drv.cuEventDestroy_v2)(b);
                }
                let mut k = self.profile.kernels.lock().unwrap();
                let e = k.entry(p.label.clone()).or_default();
                e.0 += 1;
                e.1 += ms as f64;
                continue;
            }
            self.dev.launch_func(p.func, &args, Dims::new(l.grid, p.threads))?;
        }
        Ok(())
    }

    /// `launch_all`, or the graph of an identical earlier submission: the first repeat of a submission is
    /// captured while it launches, later ones replay the graph.
    fn launch_or_replay(&self, groups: &[Vec<Launch<'_, Cuda>>]) -> Result<()> {
        let g = &self.graphs;
        if !g.on {
            return self.launch_all(groups);
        }
        let key = graph_key(groups);
        self.dev.make_current()?;
        let drv = driver()?;
        let cached = g.execs.lock().unwrap().iter().find(|(k, _)| *k == key).map(|(_, e)| *e);
        if let Some(exec) = cached {
            return check(unsafe { (drv.cuGraphLaunch)(exec, self.dev.stream) });
        }
        let repeat = {
            let mut last = g.last.lock().unwrap();
            let same = *last == key;
            *last = key.clone();
            same
        };
        if !repeat {
            return self.launch_all(groups);
        }
        // CU_STREAM_CAPTURE_MODE_RELAXED: other threads' driver calls do not invalidate the capture.
        check(unsafe { (drv.cuStreamBeginCapture_v2)(self.dev.stream, 2) })?;
        let launched = self.launch_all(groups);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let ended = check(unsafe { (drv.cuStreamEndCapture)(self.dev.stream, &mut graph) });
        launched?;
        ended?;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let made = check(unsafe { (drv.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) });
        unsafe { (drv.cuGraphDestroy)(graph) };
        made?;
        {
            let mut execs = g.execs.lock().unwrap();
            execs.push((key, exec));
            if execs.len() > GRAPHS {
                let (_, old) = execs.remove(0);
                unsafe { (drv.cuGraphExecDestroy)(old) };
            }
        }
        // The capture recorded the kernels without running them.
        check(unsafe { (drv.cuGraphLaunch)(exec, self.dev.stream) })
    }
}

impl Backend for Cuda {
    type Buffer = Buffer;
    type Pipeline = Pipeline;
    type Pending = Pending;

    const TARGET: Target = Target::Cuda;

    fn name(&self) -> String {
        format!("cuda {}", self.dev.name())
    }

    fn limits(&self) -> Limits {
        let attr = |a: i32, default: u32| {
            let Some(drv) = sys::driver() else { return default };
            let mut v = 0;
            let ret = unsafe { (drv.cuDeviceGetAttribute)(&mut v, a, self.dev.dev) };
            if ret == CUDA_SUCCESS && v > 0 { v as u32 } else { default }
        };
        Limits {
            max_buffer_bytes: u64::MAX / 2,
            // Device pointers address bytes: nothing to round.
            storage_offset_align: 1,
            // The parameter space is 4 KiB; leave one slot for the push struct.
            max_bindings: 31,
            max_push_bytes: 4096,
            max_workgroup_invocations: attr(sys::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1024),
            max_groups: [
                attr(sys::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, i32::MAX as u32),
                attr(sys::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, 65535),
                attr(sys::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, 65535),
            ],
        }
    }

    fn alloc(&self, bytes: u64, _kind: MemKind) -> Result<Buffer> {
        // Pinned or managed memory for MemKind::Step is a later optimization.
        self.dev.alloc(bytes as usize)
    }

    fn buffer_len(&self, buf: &Buffer) -> u64 {
        buf.len() as u64
    }

    fn write(&self, buf: &Buffer, off: u64, data: &[u8]) -> Result<()> {
        if off + data.len() as u64 > buf.len() as u64 {
            return Err(Error::Invalid(format!("write of {} bytes at {off} is past a buffer of {}", data.len(), buf.len())));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.dev.make_current()?;
        // Async on the in-order stream: ordered before any later launch, and the
        // driver stages pageable memory before returning, so `data` may go away.
        check(unsafe {
            (driver()?.cuMemcpyHtoDAsync_v2)(buf.ptr + off as CUdeviceptr, data.as_ptr() as *const c_void, data.len(), self.dev.stream)
        })
    }

    fn read(&self, buf: &Buffer, off: u64, out: &mut [u8]) -> Result<()> {
        if off + out.len() as u64 > buf.len() as u64 {
            return Err(Error::Invalid(format!("read of {} bytes at {off} is past a buffer of {}", out.len(), buf.len())));
        }
        if out.is_empty() {
            return Ok(());
        }
        self.dev.make_current()?;
        let drv = driver()?;
        check(unsafe { (drv.cuMemcpyDtoHAsync_v2)(out.as_mut_ptr() as *mut c_void, buf.ptr + off as CUdeviceptr, out.len(), self.dev.stream) })?;
        check(unsafe { (drv.cuStreamSynchronize)(self.dev.stream) })
    }

    fn zero(&self, buf: &Buffer, off: u64, len: u64) -> Result<()> {
        if off + len > buf.len() as u64 {
            return Err(Error::Invalid(format!("zero of {len} bytes at {off} is past a buffer of {}", buf.len())));
        }
        if len == 0 {
            return Ok(());
        }
        self.dev.make_current()?;
        check(unsafe { (driver()?.cuMemsetD8Async)(buf.ptr + off as CUdeviceptr, 0, len as usize, self.dev.stream) })
    }

    fn pipeline(&self, blob: &KernelBlob) -> Result<Pipeline> {
        if blob.sig.bindings.len() + usize::from(blob.sig.push_bytes > 0) > 32 {
            return Err(Error::Invalid(format!("`{}` has too many parameters", blob.entry)));
        }
        let key = format!("blob-{:016x}", blob.id());
        let library = match &blob.code {
            Code::Cuda(source) => self.dev.library(&key, &[], || {
                let mut text = String::new();
                for (i, v) in blob.spec.iter().enumerate() {
                    text.push_str(&format!("#define TL_SPEC_{i} {v}u
"));
                }
                text.push_str(source);
                text
            })?,
            // Generated PTX (the Tree compiler): the driver JIT-compiles it, no NVRTC needed.
            Code::Ptx(ptx) => self.dev.library_ptx(&key, ptx)?,
            _ => return Err(Error::Invalid(format!("the CUDA backend takes CUDA C or PTX blobs, `{}` is {:?}", blob.entry, blob.target))),
        };
        let func = self.dev.resolve(&Kernel { library, entry: blob.entry.clone() })?;
        Ok(Pipeline { label: blob.label.clone(), func, threads: blob.sig.workgroup, bindings: blob.sig.bindings.len(), push_bytes: blob.sig.push_bytes as usize })
    }

    fn submit(&self, groups: &[Vec<Launch<'_, Self>>]) -> Result<Pending> {
        // One in-order stream: program order is the schedule, groups are only a hint.
        self.launch_or_replay(groups)?;
        self.dev.make_current()?;
        let drv = driver()?;
        let mut event: CUevent = std::ptr::null_mut();
        check(unsafe { (drv.cuEventCreate)(&mut event, sys::CU_EVENT_DISABLE_TIMING) })?;
        check(unsafe { (drv.cuEventRecord)(event, self.dev.stream) })?;
        Ok(Pending { dev: self.dev.clone(), event })
    }

    fn poll(&self, pending: &Pending) -> Result<bool> {
        self.dev.make_current()?;
        match unsafe { (driver()?.cuEventQuery)(pending.event) } {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            e => Err(Error::Execution(sys::cu_error_string(e))),
        }
    }

    fn wait(&self, pending: &Pending) -> Result<()> {
        self.dev.make_current()?;
        check(unsafe { (driver()?.cuEventSynchronize)(pending.event) })
    }
}
