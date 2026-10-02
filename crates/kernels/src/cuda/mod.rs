//! CUDA backend: Driver API device/buffers, NVRTC-compiled libraries, one
//! stream per [`Device`]. See `docs/design/kernel-launcher.md`.
//!
//! Unlike Metal there is no explicit "open encoder": `cuLaunchKernel` enqueues
//! directly onto the device's stream, so [`Device::launch`] needs no
//! begin/end bookkeeping, just [`Device::sync`] to wait on everything queued.
//!
//! Function constants have no CUDA equivalent (templates are resolved at
//! compile time), so unlike [`crate::metal`]'s `Kernel`, there is no `consts`
//! field: a specialization is just a different NVRTC name expression, part of
//! [`Kernel::entry`] itself.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex};

use sys::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CUDA_SUCCESS, NVRTC_SUCCESS,
};

pub mod exec;
pub mod lower;
pub mod paged_attn;
pub mod shaders;
pub mod sys;

use crate::{Dims, Error, Result};

const CU_STREAM_NON_BLOCKING: u32 = 0x1;

/// A device allocation (plain `cuMemAlloc`: device-only memory, not host
/// mapped). [`Buffer::read`]/[`Buffer::write`] go through a blocking memcpy.
pub struct Buffer {
    ptr: CUdeviceptr,
    len: usize,
    ctx: CUcontext,
}

// SAFETY: the driver serializes access to a context internally; every call
// site sets the context current first.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(drv) = sys::driver() {
            unsafe {
                (drv.cuCtxSetCurrent)(self.ctx);
                (drv.cuMemFree_v2)(self.ptr);
            }
        }
    }
}

impl Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn set_current(&self) {
        if let Some(drv) = sys::driver() {
            unsafe { (drv.cuCtxSetCurrent)(self.ctx) };
        }
    }

    /// Blocking host -> device copy.
    pub fn write<T: Copy>(&self, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.len, "write of {bytes} bytes overflows a {}-byte buffer", self.len);
        self.set_current();
        let drv = sys::driver().expect("Buffer outlived the driver");
        let ret = unsafe { (drv.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr() as *const c_void, bytes) };
        assert_eq!(ret, CUDA_SUCCESS, "cuMemcpyHtoD: {}", sys::cu_error_string(ret));
    }

    /// Blocking host -> device copy to `byte_offset` bytes in.
    pub fn write_at(&self, byte_offset: usize, data: &[u8]) {
        assert!(byte_offset + data.len() <= self.len, "write past the end of a {}-byte buffer", self.len);
        if data.is_empty() {
            return;
        }
        self.set_current();
        let drv = sys::driver().expect("Buffer outlived the driver");
        let ret = unsafe {
            (drv.cuMemcpyHtoD_v2)(self.ptr + byte_offset as CUdeviceptr, data.as_ptr() as *const c_void, data.len())
        };
        assert_eq!(ret, CUDA_SUCCESS, "cuMemcpyHtoD: {}", sys::cu_error_string(ret));
    }

    /// Fills the whole buffer with zero bytes.
    pub fn zero(&self) {
        self.set_current();
        let drv = sys::driver().expect("Buffer outlived the driver");
        let ret = unsafe { (drv.cuMemsetD8_v2)(self.ptr, 0, self.len) };
        assert_eq!(ret, CUDA_SUCCESS, "cuMemsetD8: {}", sys::cu_error_string(ret));
    }

    /// Blocking device -> host copy of `count` elements from the start.
    pub fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        self.read_range(0, count)
    }

    /// `count` elements starting `byte_offset` bytes in.
    pub fn read_range<T: Copy>(&self, byte_offset: usize, count: usize) -> Vec<T> {
        let bytes = count * size_of::<T>();
        assert!(byte_offset + bytes <= self.len, "read past the end of a {}-byte buffer", self.len);
        self.set_current();
        let drv = sys::driver().expect("Buffer outlived the driver");
        let mut out = Vec::<T>::with_capacity(count);
        let ret = unsafe {
            let r = (drv.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, self.ptr + byte_offset as CUdeviceptr, bytes);
            out.set_len(count);
            r
        };
        assert_eq!(ret, CUDA_SUCCESS, "cuMemcpyDtoH: {}", sys::cu_error_string(ret));
        out
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

/// A kernel argument. Unlike Metal's `(slot, Arg)` pairs, CUDA parameters are
/// strictly positional: pass one `Arg` per parameter in signature order, and
/// use `NullPtr` for an optional pointer argument that is absent.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Buf(BufRef<'a>),
    NullPtr,
    I32(i32),
    U32(u32),
    I64(i64),
    F32(f32),
    /// Arbitrary bytes (a kernel-argument struct), passed in place by pointer.
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
impl<'a> From<Option<BufRef<'a>>> for Arg<'a> {
    fn from(b: Option<BufRef<'a>>) -> Self {
        match b {
            Some(b) => Arg::Buf(b),
            None => Arg::NullPtr,
        }
    }
}

/// Which pipeline to run: an entry of a registered library. For a template
/// kernel this is the full C++ name expression (e.g.
/// `vllm::paged_attention_v1_kernel<uint16_t,uint16_t,...>`), resolved
/// through the library's NVRTC-lowered-name table; for an `extern "C"`
/// kernel it is the plain symbol name, looked up directly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub library: Arc<str>,
    pub entry: String,
}

/// A resolved kernel entry point: launching through it skips the by-name lookup.
#[derive(Clone, Copy)]
pub struct Func(CUfunction);

// SAFETY: a CUfunction is an immutable handle into a module that lives as long as the Device.
unsafe impl Send for Func {}
unsafe impl Sync for Func {}

struct CompiledLibrary {
    module: CUmodule,
    /// Name expression -> mangled symbol, for the templates registered when
    /// this library was compiled (see [`Device::library`]).
    lowered: HashMap<String, CString>,
}

// SAFETY: a CUmodule is immutable once loaded; lookups only read `lowered`.
unsafe impl Send for CompiledLibrary {}
unsafe impl Sync for CompiledLibrary {}

pub struct Device {
    #[allow(dead_code)]
    dev: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    /// `"86"`-style SM arch for `--gpu-architecture=compute_86`.
    arch: String,
    /// The CUDA Toolkit's `include/` (`CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT`), if
    /// set: NVRTC's built-in headers don't cover `cuda_bf16.h` well enough in
    /// practice, so library sources that need it get `-I` to the real one.
    include_dir: Option<std::path::PathBuf>,
    libraries: Mutex<HashMap<Arc<str>, Arc<CompiledLibrary>>>,
    pipelines: Mutex<HashMap<(Arc<str>, String), CUfunction>>,
}

// SAFETY: every entry point sets the context current on the calling thread
// first; the driver itself serializes work submitted to one stream.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

fn check(ret: CUresult) -> Result<()> {
    if ret == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Execution(sys::cu_error_string(ret)))
    }
}

impl Device {
    /// The first CUDA device, with its own primary context and stream.
    /// `Err(NoDevice)` if no NVIDIA driver is installed or no GPU is visible
    /// (not a hard error: callers treat it like Metal's `system_default`).
    pub fn system_default() -> Result<Device> {
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        unsafe {
            let mut count = 0;
            check((drv.cuDeviceGetCount)(&mut count))?;
            if count == 0 {
                return Err(Error::NoDevice);
            }
            let mut dev: CUdevice = 0;
            check((drv.cuDeviceGet)(&mut dev, 0))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check((drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev))?;
            check((drv.cuCtxSetCurrent)(ctx))?;
            let mut stream: CUstream = std::ptr::null_mut();
            check((drv.cuStreamCreate)(&mut stream, CU_STREAM_NON_BLOCKING))?;
            let mut major = 0;
            let mut minor = 0;
            check((drv.cuDeviceGetAttribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))?;
            check((drv.cuDeviceGetAttribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))?;
            let include_dir = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
                .into_iter()
                .find_map(std::env::var_os)
                .map(|root| std::path::PathBuf::from(root).join("include"));
            Ok(Device {
                dev,
                ctx,
                stream,
                arch: format!("{major}{minor}"),
                include_dir,
                libraries: Default::default(),
                pipelines: Default::default(),
            })
        }
    }

    fn make_current(&self) -> Result<()> {
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        check(unsafe { (drv.cuCtxSetCurrent)(self.ctx) })
    }

    pub fn name(&self) -> String {
        let drv = sys::driver().expect("Device outlived the driver");
        let mut buf = [0u8; 256];
        let ret = unsafe { (drv.cuDeviceGetName)(buf.as_mut_ptr() as *mut i8, buf.len() as i32, self.dev) };
        if ret != CUDA_SUCCESS {
            return format!("cuda:{}", self.dev);
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        self.make_current()?;
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        let mut ptr: CUdeviceptr = 0;
        let ret = unsafe { (drv.cuMemAlloc_v2)(&mut ptr, bytes.max(1)) };
        if ret != CUDA_SUCCESS {
            return Err(Error::Alloc(bytes));
        }
        Ok(Buffer { ptr, len: bytes, ctx: self.ctx })
    }

    /// Allocates and fills a buffer from a host slice.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer> {
        let buf = self.alloc(std::mem::size_of_val(data))?;
        buf.write(data);
        Ok(buf)
    }

    /// Registers (and compiles, once) a library under `key`: NVRTC-compiles
    /// `source()`'s text, registering a name expression per entry in
    /// `templates` first so their mangled names can be recovered afterwards
    /// (`nvrtcAddNameExpression` must run before `nvrtcCompileProgram`). An
    /// `extern "C"` kernel needs no entry in `templates`: its symbol is its
    /// plain name already. `source` only runs if `key` is new.
    pub fn library(&self, key: &str, templates: &[&str], source: impl FnOnce() -> String) -> Result<Arc<str>> {
        {
            let libs = self.libraries.lock().unwrap();
            if let Some((k, _)) = libs.get_key_value(key) {
                return Ok(k.clone());
            }
        }
        self.make_current()?;
        let nv = sys::nvrtc().ok_or_else(|| {
            Error::Compile { lib: key.to_string(), msg: "no CUDA Toolkit install found (NVRTC not loaded)".into() }
        })?;
        let text = source();
        let src = CString::new(text).map_err(|e| Error::Compile { lib: key.to_string(), msg: e.to_string() })?;
        let prog_name = CString::new(key).unwrap();

        let mut prog: sys::nvrtcProgram = std::ptr::null_mut();
        let ret = unsafe {
            (nv.nvrtcCreateProgram)(&mut prog, src.as_ptr(), prog_name.as_ptr(), 0, std::ptr::null(), std::ptr::null())
        };
        if ret != NVRTC_SUCCESS {
            return Err(Error::Compile { lib: key.to_string(), msg: sys::nvrtc_error_string(nv, ret) });
        }
        // Keep the CStrings alive until after nvrtcCompileProgram reads them.
        let expr_cstrs: Vec<CString> = templates.iter().map(|t| CString::new(*t).unwrap()).collect();
        for c in &expr_cstrs {
            unsafe { (nv.nvrtcAddNameExpression)(prog, c.as_ptr()) };
        }

        let arch_opt = CString::new(format!("--gpu-architecture=compute_{}", self.arch)).unwrap();
        let std_opt = CString::new("--std=c++17").unwrap();
        let include_opt = self.include_dir.as_ref().map(|d| CString::new(format!("-I{}", d.display())).unwrap());
        let mut opts = vec![arch_opt.as_ptr(), std_opt.as_ptr()];
        if let Some(i) = &include_opt {
            opts.push(i.as_ptr());
        }
        let ret = unsafe { (nv.nvrtcCompileProgram)(prog, opts.len() as i32, opts.as_ptr()) };
        if ret != NVRTC_SUCCESS {
            let mut log_size = 0;
            unsafe { (nv.nvrtcGetProgramLogSize)(prog, &mut log_size) };
            let mut log = vec![0u8; log_size];
            if log_size > 0 {
                unsafe { (nv.nvrtcGetProgramLog)(prog, log.as_mut_ptr() as *mut i8) };
            }
            unsafe { (nv.nvrtcDestroyProgram)(&mut prog) };
            let msg = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
            return Err(Error::Compile { lib: key.to_string(), msg: if msg.is_empty() { sys::nvrtc_error_string(nv, ret) } else { msg } });
        }

        let mut lowered = HashMap::new();
        for (expr, c) in templates.iter().zip(&expr_cstrs) {
            let mut name: *const i8 = std::ptr::null();
            let ret = unsafe { (nv.nvrtcGetLoweredName)(prog, c.as_ptr(), &mut name) };
            if ret != NVRTC_SUCCESS || name.is_null() {
                unsafe { (nv.nvrtcDestroyProgram)(&mut prog) };
                return Err(Error::Compile { lib: key.to_string(), msg: format!("no lowered name for `{expr}`") });
            }
            let mangled = unsafe { std::ffi::CStr::from_ptr(name) }.to_owned();
            lowered.insert(expr.to_string(), mangled);
        }

        let mut ptx_size = 0;
        unsafe { (nv.nvrtcGetPTXSize)(prog, &mut ptx_size) };
        let mut ptx = vec![0u8; ptx_size];
        unsafe { (nv.nvrtcGetPTX)(prog, ptx.as_mut_ptr() as *mut i8) };
        unsafe { (nv.nvrtcDestroyProgram)(&mut prog) };

        let drv = sys::driver().ok_or(Error::NoDevice)?;
        let mut module: CUmodule = std::ptr::null_mut();
        let mut log_buf = vec![0u8; 8192];
        let mut opt_keys = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut opt_vals: [*mut c_void; 2] = [log_buf.as_mut_ptr() as *mut c_void, log_buf.len() as *mut c_void];
        let ret = unsafe {
            (drv.cuModuleLoadDataEx)(
                &mut module,
                ptx.as_ptr() as *const c_void,
                opt_keys.len() as u32,
                opt_keys.as_mut_ptr(),
                opt_vals.as_mut_ptr(),
            )
        };
        if ret != CUDA_SUCCESS {
            let end = log_buf.iter().position(|&b| b == 0).unwrap_or(0);
            let log = String::from_utf8_lossy(&log_buf[..end]);
            let msg = if log.is_empty() { sys::cu_error_string(ret) } else { log.into_owned() };
            return Err(Error::Compile { lib: key.to_string(), msg });
        }

        let key: Arc<str> = Arc::from(key);
        self.libraries.lock().unwrap().insert(key.clone(), Arc::new(CompiledLibrary { module, lowered }));
        Ok(key)
    }

    fn function(&self, k: &Kernel) -> Result<CUfunction> {
        if let Some(f) = self.pipelines.lock().unwrap().get(&(k.library.clone(), k.entry.clone())) {
            return Ok(*f);
        }
        let lib = self
            .libraries
            .lock()
            .unwrap()
            .get(&k.library)
            .cloned()
            .ok_or_else(|| Error::Invalid(format!("library `{}` is not registered", k.library)))?;
        let name = match lib.lowered.get(&k.entry) {
            Some(mangled) => mangled.clone(),
            None => CString::new(k.entry.as_str()).map_err(|e| Error::Invalid(e.to_string()))?,
        };
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        let mut func: CUfunction = std::ptr::null_mut();
        let ret = unsafe { (drv.cuModuleGetFunction)(&mut func, lib.module, name.as_ptr()) };
        if ret != CUDA_SUCCESS {
            return Err(Error::Pipeline { entry: k.entry.clone(), msg: sys::cu_error_string(ret) });
        }
        self.pipelines.lock().unwrap().insert((k.library.clone(), k.entry.clone()), func);
        Ok(func)
    }

    /// Looks `kernel` up once; pair with [`Device::launch_func`] for replay.
    pub fn resolve(&self, kernel: &Kernel) -> Result<Func> {
        self.make_current()?;
        self.function(kernel).map(Func)
    }

    /// Enqueues one dispatch on the device's stream. `args` are positional,
    /// matching the kernel's parameter list exactly (see [`Arg`]).
    pub fn launch(&self, kernel: &Kernel, args: &[Arg], dims: Dims) -> Result<()> {
        self.make_current()?;
        let func = Func(self.function(kernel)?);
        self.launch_func(func, args, dims).map_err(|e| match e {
            Error::Execution(m) => Error::Execution(format!("launching `{}`: {m}", kernel.entry)),
            e => e,
        })
    }

    /// [`Device::launch`] for an already resolved kernel; no allocation, no lookup.
    pub fn launch_func(&self, func: Func, args: &[Arg], dims: Dims) -> Result<()> {
        const MAX_ARGS: usize = 32;
        if args.len() > MAX_ARGS {
            return Err(Error::Invalid(format!("a launch takes at most {MAX_ARGS} arguments, got {}", args.len())));
        }
        self.make_current()?;
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        let func = func.0;

        if dims.shared > 48 * 1024 {
            unsafe {
                (drv.cuFuncSetAttribute)(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dims.shared as i32)
            };
        }

        // `kernel_params[i]` points into `storage[i]` (or at a `Bytes` argument
        // itself); both outlive the `cuLaunchKernel` call below.
        let mut storage = [[0u8; 8]; MAX_ARGS];
        let mut kernel_params = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (i, arg) in args.iter().enumerate() {
            match arg {
                Arg::Buf(b) => storage[i] = (b.buf.ptr + b.offset as CUdeviceptr).to_ne_bytes(),
                Arg::NullPtr => {}
                Arg::I32(v) => storage[i][..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::U32(v) => storage[i][..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::I64(v) => storage[i] = v.to_ne_bytes(),
                Arg::F32(v) => storage[i][..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::Bytes(b) => {
                    kernel_params[i] = b.as_ptr() as *mut c_void;
                    continue;
                }
            }
            kernel_params[i] = storage[i].as_mut_ptr() as *mut c_void;
        }

        let ret = unsafe {
            (drv.cuLaunchKernel)(
                func,
                dims.groups[0],
                dims.groups[1],
                dims.groups[2],
                dims.threads[0],
                dims.threads[1],
                dims.threads[2],
                dims.shared,
                self.stream,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        check(ret)
    }

    /// Largest dynamic shared memory a block may request (opt-in limit).
    pub fn max_shared_memory(&self) -> u32 {
        let drv = sys::driver().expect("Device outlived the driver");
        let mut v = 0;
        let ret = unsafe { (drv.cuDeviceGetAttribute)(&mut v, sys::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self.dev) };
        if ret == CUDA_SUCCESS { v as u32 } else { 48 * 1024 }
    }

    /// Number of streaming multiprocessors.
    pub fn sm_count(&self) -> u32 {
        let drv = sys::driver().expect("Device outlived the driver");
        let mut v = 0;
        let ret = unsafe { (drv.cuDeviceGetAttribute)(&mut v, sys::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev) };
        if ret == CUDA_SUCCESS { v.max(1) as u32 } else { 16 }
    }

    /// Waits for everything enqueued on the device's stream.
    pub fn sync(&self) -> Result<()> {
        self.make_current()?;
        let drv = sys::driver().ok_or(Error::NoDevice)?;
        check(unsafe { (drv.cuStreamSynchronize)(self.stream) })
    }
}
