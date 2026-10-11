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

use std::ffi::c_void;
use std::sync::Arc;

use super::sys::{self, CUdeviceptr, CUevent, CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
use super::{check, Arg, BufRef, Buffer, Device, Func, Kernel};
use crate::sched::backend::{Backend, Launch, Limits, MemKind};
use crate::sched::blob::{Code, KernelBlob, Target};
use crate::{Dims, Error, Result};

#[derive(Clone)]
pub struct Cuda {
    dev: Arc<Device>,
}

impl Cuda {
    pub fn system_default() -> Result<Cuda> {
        Ok(Cuda { dev: Arc::new(Device::system_default()?) })
    }

    /// Wraps an existing device (e.g. the one a native executor already owns).
    pub fn from_device(dev: Arc<Device>) -> Cuda {
        Cuda { dev }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }
}

pub struct Pipeline {
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
        Ok(Pipeline { func, threads: blob.sig.workgroup, bindings: blob.sig.bindings.len(), push_bytes: blob.sig.push_bytes as usize })
    }

    fn submit(&self, groups: &[Vec<Launch<'_, Self>>]) -> Result<Pending> {
        // One in-order stream: program order is the schedule, groups are only a hint.
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
            self.dev.launch_func(p.func, &args, Dims::new(l.grid, p.threads))?;
        }
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
