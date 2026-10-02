//! Metal backend: runtime-compiled libraries, cached pipelines, one open
//! command buffer per [`Device`] that [`Device::sync`] commits and waits on.
//!
//! Dispatches inside the encoder are serial (`MTLDispatchTypeSerial`), so a
//! kernel that reads what the previous launch wrote needs no explicit barrier.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSDictionary, NSObject, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType,
    MTLDevice, MTLFunctionConstantValues, MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::{Const, Dims, Error, Result};

pub mod exec;
pub mod kargs;
pub mod lower;
pub mod paged_attn;
pub mod shaders;
#[cfg(feature = "ggml")]
pub mod ggml;

type RawBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type RawPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type RawLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;

/// A device allocation. Shared storage (unified memory): [`Buffer::contents`]
/// is directly readable/writable by the CPU once the GPU work touching it has
/// been [`Device::sync`]ed.
#[derive(Clone)]
pub struct Buffer {
    raw: RawBuffer,
}

// SAFETY: MTLBuffer is documented thread-safe.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn len(&self) -> usize {
        self.raw.length()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wraps an `id<MTLBuffer>` owned by someone else (e.g. ggml). Retains it.
    ///
    /// # Safety
    /// `ptr` must be a valid `id<MTLBuffer>`.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Option<Buffer> {
        unsafe { Retained::retain(ptr as *mut ProtocolObject<dyn MTLBuffer>) }.map(|raw| Buffer { raw })
    }

    /// CPU pointer to the start of the buffer. Only valid for shared storage;
    /// null for private buffers (e.g. most ggml tensors).
    pub fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }

    /// Typed copy of the buffer contents (after [`Device::sync`]).
    pub fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        assert!(count * size_of::<T>() <= self.len());
        let mut out = Vec::<T>::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(self.contents() as *const T, out.as_mut_ptr(), count);
            out.set_len(count);
        }
        out
    }

    /// `count` elements starting `byte_offset` bytes in.
    pub fn read_range<T: Copy>(&self, byte_offset: usize, count: usize) -> Vec<T> {
        assert!(byte_offset + count * size_of::<T>() <= self.len());
        let mut out = Vec::<T>::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(self.contents().add(byte_offset) as *const T, out.as_mut_ptr(), count);
            out.set_len(count);
        }
        out
    }

    pub fn write<T: Copy>(&self, data: &[T]) {
        assert!(std::mem::size_of_val(data) <= self.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.contents() as *mut T, data.len()) }
    }

    /// This buffer starting `bytes` in, as a kernel argument.
    pub fn at(&self, bytes: usize) -> BufRef<'_> {
        BufRef { buf: self, offset: bytes }
    }
}

impl<'a> From<&'a Buffer> for BufRef<'a> {
    fn from(buf: &'a Buffer) -> Self {
        BufRef { buf, offset: 0 }
    }
}

/// A buffer plus byte offset (what ggml/candle call a tensor's storage view).
#[derive(Clone, Copy)]
pub struct BufRef<'a> {
    pub buf: &'a Buffer,
    pub offset: usize,
}

/// A kernel argument, bound at an explicit index (optional buffers leave gaps).
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Buf(BufRef<'a>),
    I32(i32),
    U32(u32),
    I64(i64),
    F32(f32),
    /// Arbitrary bytes (a kernel-argument struct), copied at encode time.
    Bytes(&'a [u8]),
}

impl<'a> From<&'a Buffer> for Arg<'a> {
    fn from(b: &'a Buffer) -> Self {
        Arg::Buf(b.into())
    }
}
impl<'a> From<BufRef<'a>> for Arg<'a> {
    fn from(b: BufRef<'a>) -> Self {
        Arg::Buf(b)
    }
}

/// A compiled compute pipeline.
#[derive(Clone)]
pub struct Pipeline {
    pso: RawPipeline,
    /// `maxTotalThreadsPerThreadgroup`: kernels with many registers get fewer than the device's 1024.
    pub max_threads: usize,
}

// SAFETY: MTLComputePipelineState is immutable and thread-safe.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

/// Which pipeline to run: an entry point of a registered library, specialized
/// by function constants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub library: Arc<str>,
    pub entry: String,
    pub consts: Vec<(u32, Const)>,
}

struct Encoding {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

#[derive(Default)]
struct Stream {
    open: Option<Encoding>,
}

// SAFETY: the stream is only touched under the Device's mutex.
unsafe impl Send for Stream {}

pub struct Device {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: Mutex<HashMap<Arc<str>, RawLibrary>>,
    pipelines: Mutex<HashMap<Kernel, Pipeline>>,
    stream: Mutex<Stream>,
}

// SAFETY: MTLDevice/MTLCommandQueue/MTLLibrary/MTLComputePipelineState are thread-safe.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// The system default GPU with its own command queue.
    pub fn system_default() -> Result<Device> {
        let dev = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let queue = dev.newCommandQueue().ok_or(Error::NoDevice)?;
        Ok(Self::new(dev, queue))
    }

    /// Adopts an existing device/queue (retained), e.g. ggml's, so launches
    /// are ordered with its command buffers and buffers can be shared.
    ///
    /// # Safety
    /// Both pointers must be valid `id<MTLDevice>` / `id<MTLCommandQueue>`.
    pub unsafe fn from_raw(device: *mut c_void, queue: *mut c_void) -> Result<Device> {
        let dev = unsafe { Retained::retain(device as *mut ProtocolObject<dyn MTLDevice>) }.ok_or(Error::NoDevice)?;
        let queue =
            unsafe { Retained::retain(queue as *mut ProtocolObject<dyn MTLCommandQueue>) }.ok_or(Error::NoDevice)?;
        Ok(Self::new(dev, queue))
    }

    fn new(dev: Retained<ProtocolObject<dyn MTLDevice>>, queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Device {
            dev,
            queue,
            libraries: Default::default(),
            pipelines: Default::default(),
            stream: Default::default(),
        }
    }

    /// Capabilities the ggml kernels branch on.
    pub fn props(&self) -> lower::DeviceProps {
        lower::DeviceProps {
            simdgroup_mm: self.dev.supportsFamily(MTLGPUFamily::Apple7),
            max_threadgroup_memory: self.dev.maxThreadgroupMemoryLength() as u32,
        }
    }

    /// Largest single buffer the device can allocate.
    pub fn max_buffer_len(&self) -> usize {
        self.dev.maxBufferLength()
    }

    /// Whether the GPU does bfloat natively (ggml's `GGML_METAL_HAS_BF16`).
    pub fn has_bfloat(&self) -> bool {
        self.dev.supportsFamily(MTLGPUFamily::Apple6)
    }

    pub fn name(&self) -> String {
        self.dev.name().to_string()
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let raw = self
            .dev
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Alloc(bytes))?;
        Ok(Buffer { raw })
    }

    /// Allocates and fills a buffer from a host slice.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer> {
        let buf = self.alloc(std::mem::size_of_val(data))?;
        buf.write(data);
        Ok(buf)
    }

    /// Registers (and later compiles, once) a library under `key`. `source`
    /// only runs if the key is new, so callers can build text lazily.
    pub fn library(&self, key: &str, source: impl FnOnce() -> String) -> Result<Arc<str>> {
        self.library_with(key, &[], source)
    }

    /// Like [`Device::library`], with preprocessor macros (`-DNAME=VALUE`).
    pub fn library_with(&self, key: &str, macros: &[(&str, &str)], source: impl FnOnce() -> String) -> Result<Arc<str>> {
        let mut libs = self.libraries.lock().unwrap();
        if let Some((k, _)) = libs.get_key_value(key) {
            return Ok(k.clone());
        }
        let text = source();
        let options = (!macros.is_empty()).then(|| {
            let options = MTLCompileOptions::new();
            let keys: Vec<Retained<NSString>> = macros.iter().map(|(k, _)| NSString::from_str(k)).collect();
            let vals: Vec<Retained<NSObject>> = macros.iter().map(|(_, v)| NSString::from_str(v).into_super()).collect();
            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let dict: Retained<NSDictionary<NSString, NSObject>> = NSDictionary::from_retained_objects(&key_refs, &vals);
            unsafe { options.setPreprocessorMacros(Some(&dict)) };
            options
        });
        let lib = self
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(&text), options.as_deref())
            .map_err(|e| Error::Compile { lib: key.to_string(), msg: e.localizedDescription().to_string() })?;
        let key: Arc<str> = Arc::from(key);
        libs.insert(key.clone(), lib);
        Ok(key)
    }

    /// Compiles (or fetches) the pipeline for `k`.
    pub fn pipeline(&self, k: &Kernel) -> Result<Pipeline> {
        if let Some(p) = self.pipelines.lock().unwrap().get(k) {
            return Ok(p.clone());
        }
        let lib = self
            .libraries
            .lock()
            .unwrap()
            .get(&k.library)
            .cloned()
            .ok_or_else(|| Error::Invalid(format!("library `{}` is not registered", k.library)))?;
        let err = |msg: String| Error::Pipeline { entry: k.entry.clone(), msg };

        let name = NSString::from_str(&k.entry);
        let func = if k.consts.is_empty() {
            lib.newFunctionWithName(&name).ok_or_else(|| err("no such function".into()))?
        } else {
            let cv = MTLFunctionConstantValues::new();
            for (idx, c) in &k.consts {
                // SAFETY: the pointed-to value outlives the call; Metal copies it.
                unsafe {
                    match c {
                        Const::Bool(b) => {
                            let v = *b;
                            cv.setConstantValue_type_atIndex(
                                NonNull::from(&v).cast(),
                                MTLDataType::Bool,
                                *idx as usize,
                            );
                        }
                        Const::I16(i) => {
                            cv.setConstantValue_type_atIndex(
                                NonNull::from(i).cast(),
                                MTLDataType::Short,
                                *idx as usize,
                            );
                        }
                        Const::I32(i) => {
                            cv.setConstantValue_type_atIndex(
                                NonNull::from(i).cast(),
                                MTLDataType::Int,
                                *idx as usize,
                            );
                        }
                    }
                }
            }
            lib.newFunctionWithName_constantValues_error(&name, &cv).map_err(|e| err(e.localizedDescription().to_string()))?
        };
        let pso = self
            .dev
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| err(e.localizedDescription().to_string()))?;
        let pipeline = Pipeline { max_threads: pso.maxTotalThreadsPerThreadgroup(), pso };
        self.pipelines.lock().unwrap().insert(k.clone(), pipeline.clone());
        Ok(pipeline)
    }

    /// Encodes one dispatch. `args` are `(slot, value)`. Nothing runs until
    /// [`Device::sync`] (or [`Device::flush`]).
    pub fn launch(&self, kernel: &Kernel, args: &[(u32, Arg)], dims: Dims) -> Result<()> {
        let pipeline = self.pipeline(kernel)?;
        self.launch_pipeline(&pipeline, args, dims)
    }

    /// [`Device::launch`] with an already compiled pipeline (no lookup).
    pub fn launch_pipeline(&self, pipeline: &Pipeline, args: &[(u32, Arg)], dims: Dims) -> Result<()> {
        let pso = &pipeline.pso;
        let mut stream = self.stream.lock().unwrap();
        if stream.open.is_none() {
            let cmd = self.queue.commandBuffer().ok_or_else(|| Error::Execution("no command buffer".into()))?;
            let enc = cmd.computeCommandEncoder().ok_or_else(|| Error::Execution("no compute encoder".into()))?;
            stream.open = Some(Encoding { cmd, enc });
        }
        let enc = &stream.open.as_ref().unwrap().enc;
        enc.setComputePipelineState(pso);
        // SAFETY: slots/lengths come from the typed args below; Metal copies bytes.
        unsafe {
            for (slot, arg) in args {
                let slot = *slot as usize;
                match arg {
                    Arg::Buf(b) => enc.setBuffer_offset_atIndex(Some(&b.buf.raw), b.offset, slot),
                    Arg::I32(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 4, slot),
                    Arg::U32(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 4, slot),
                    Arg::I64(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 8, slot),
                    Arg::F32(v) => enc.setBytes_length_atIndex(NonNull::from(v).cast(), 4, slot),
                    Arg::Bytes(b) => {
                        enc.setBytes_length_atIndex(NonNull::new(b.as_ptr() as *mut c_void).unwrap(), b.len(), slot)
                    }
                }
            }
            if dims.shared > 0 {
                enc.setThreadgroupMemoryLength_atIndex(dims.shared as usize, 0);
            }
        }
        let size = |v: [u32; 3]| MTLSize { width: v[0] as usize, height: v[1] as usize, depth: v[2] as usize };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(dims.groups), size(dims.threads));
        Ok(())
    }

    /// Commits everything encoded so far without waiting. Returns the command
    /// buffer's completion handle for [`Pending::wait`].
    pub fn flush(&self) -> Option<Pending> {
        let open = self.stream.lock().unwrap().open.take()?;
        open.enc.endEncoding();
        open.cmd.commit();
        Some(Pending(open.cmd))
    }

    /// Commits and waits for all encoded work.
    pub fn sync(&self) -> Result<()> {
        match self.flush() {
            Some(p) => p.wait(),
            None => Ok(()),
        }
    }
}

/// A committed command buffer.
pub struct Pending(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

unsafe impl Send for Pending {}

impl Pending {
    pub fn wait(self) -> Result<()> {
        self.0.waitUntilCompleted();
        if self.0.status() == MTLCommandBufferStatus::Error {
            let msg = self.0.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default();
            return Err(Error::Execution(msg));
        }
        Ok(())
    }
}
