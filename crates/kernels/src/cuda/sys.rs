//! Minimal CUDA Driver API + NVRTC bindings, `dlopen`'d at runtime via
//! `libloading`. Nothing here is linked at build time, so the crate builds
//! (and this module loads, returning [`Error::NoDevice`] on first use) on a
//! machine without a CUDA install: only `nvcuda`/`libcuda.so.1` (installed by
//! every NVIDIA display driver) and `nvrtc*`/`libnvrtc.so` (installed by the
//! CUDA Toolkit) are needed, both looked for lazily.
//!
//! Hand-written rather than via `cudarc`/`rust-cuda` to keep the surface to
//! exactly what [`super`] uses (~20 driver calls, ~8 NVRTC calls), matching
//! the scope of the Metal module next to it.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::os::raw::c_ulonglong;
use std::sync::OnceLock;

use libloading::Library;

pub type CUresult = c_int;
pub type nvrtcResult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;
pub const NVRTC_SUCCESS: nvrtcResult = 0;

pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
/// Always 64-bit: the driver API's pointer type regardless of host pointer width.
pub type CUdeviceptr = c_ulonglong;

macro_rules! driver_api {
    ($( $name:ident : fn($($arg:ident : $ty:ty),* $(,)?) -> $ret:ty ),* $(,)?) => {
        pub struct Driver {
            _lib: Library,
            $( pub $name: unsafe extern "C" fn($($arg: $ty),*) -> $ret, )*
        }

        impl Driver {
            unsafe fn load(lib: Library) -> Result<Driver, libloading::Error> {
                unsafe {
                    $( let $name = *lib.get::<unsafe extern "C" fn($($arg: $ty),*) -> $ret>(
                        concat!(stringify!($name), "\0").as_bytes(),
                    )?; )*
                    Ok(Driver { _lib: lib, $($name,)* })
                }
            }
        }
    };
}

// Symbol names are the versioned ones the driver actually exports for any
// call whose ABI changed historically (CUdeviceptr going 64-bit, etc.); the
// unversioned names in cuda.h are header macros that resolve to these.
driver_api! {
    cuInit: fn(flags: c_uint) -> CUresult,
    cuDeviceGetCount: fn(count: *mut c_int) -> CUresult,
    cuDeviceGet: fn(dev: *mut CUdevice, ordinal: c_int) -> CUresult,
    cuDeviceGetName: fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    cuDeviceGetAttribute: fn(value: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult,
    cuDevicePrimaryCtxRetain: fn(ctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    cuCtxSetCurrent: fn(ctx: CUcontext) -> CUresult,
    cuMemAlloc_v2: fn(ptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemFree_v2: fn(ptr: CUdeviceptr) -> CUresult,
    cuMemsetD8_v2: fn(dst: CUdeviceptr, value: u8, n: usize) -> CUresult,
    cuMemcpyHtoD_v2: fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult,
    cuMemcpyDtoH_v2: fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult,
    cuModuleLoadDataEx: fn(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: c_uint,
        options: *mut c_int,
        option_values: *mut *mut c_void,
    ) -> CUresult,
    cuModuleGetFunction: fn(func: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult,
    cuModuleUnload: fn(module: CUmodule) -> CUresult,
    cuFuncSetAttribute: fn(func: CUfunction, attrib: c_int, value: c_int) -> CUresult,
    cuStreamCreate: fn(stream: *mut CUstream, flags: c_uint) -> CUresult,
    cuStreamSynchronize: fn(stream: CUstream) -> CUresult,
    cuStreamDestroy_v2: fn(stream: CUstream) -> CUresult,
    cuGetErrorString: fn(err: CUresult, out: *mut *const c_char) -> CUresult,
    cuLaunchKernel: fn(
        f: CUfunction,
        grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
        block_x: c_uint, block_y: c_uint, block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
}

/// `CU_JIT_*` enum values `cuModuleLoadDataEx` takes as `options`.
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}`.
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;

macro_rules! nvrtc_api {
    ($( $name:ident : fn($($arg:ident : $ty:ty),* $(,)?) -> $ret:ty ),* $(,)?) => {
        pub struct Nvrtc {
            _lib: Library,
            $( pub $name: unsafe extern "C" fn($($arg: $ty),*) -> $ret, )*
        }

        impl Nvrtc {
            unsafe fn load(lib: Library) -> Result<Nvrtc, libloading::Error> {
                unsafe {
                    $( let $name = *lib.get::<unsafe extern "C" fn($($arg: $ty),*) -> $ret>(
                        concat!(stringify!($name), "\0").as_bytes(),
                    )?; )*
                    Ok(Nvrtc { _lib: lib, $($name,)* })
                }
            }
        }
    };
}

pub type nvrtcProgram = *mut c_void;

nvrtc_api! {
    nvrtcCreateProgram: fn(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> nvrtcResult,
    nvrtcDestroyProgram: fn(prog: *mut nvrtcProgram) -> nvrtcResult,
    nvrtcAddNameExpression: fn(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult,
    nvrtcCompileProgram: fn(prog: nvrtcProgram, num_opts: c_int, opts: *const *const c_char) -> nvrtcResult,
    nvrtcGetProgramLogSize: fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult,
    nvrtcGetProgramLog: fn(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult,
    nvrtcGetPTXSize: fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult,
    nvrtcGetPTX: fn(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult,
    nvrtcGetLoweredName: fn(prog: nvrtcProgram, name_expr: *const c_char, lowered: *mut *const c_char) -> nvrtcResult,
    nvrtcGetErrorString: fn(result: nvrtcResult) -> *const c_char,
}

/// Loads a library trying each candidate name in turn (CUDA's SONAME /
/// DLL name is version-suffixed on some platforms).
fn load_first(names: &[&str]) -> Option<Library> {
    names.iter().find_map(|n| unsafe { Library::new(n) }.ok())
}

static DRIVER: OnceLock<Option<Driver>> = OnceLock::new();
static NVRTC: OnceLock<Option<Nvrtc>> = OnceLock::new();

/// The Driver API, loaded and `cuInit`-ed on first use. `None` if
/// `nvcuda`/`libcuda` isn't installed (no NVIDIA driver) or `cuInit` fails
/// (no GPU visible, e.g. a driver with no supported card).
pub fn driver() -> Option<&'static Driver> {
    DRIVER
        .get_or_init(|| {
            let lib = load_first(&["nvcuda.dll", "libcuda.so.1", "libcuda.so"])?;
            let drv = unsafe { Driver::load(lib) }.ok()?;
            (unsafe { (drv.cuInit)(0) } == CUDA_SUCCESS).then_some(drv)
        })
        .as_ref()
}

/// NVRTC, loaded on first use. `None` if no CUDA Toolkit install is found;
/// the driver can still be used (e.g. with a precompiled PTX/cubin) without it.
pub fn nvrtc() -> Option<&'static Nvrtc> {
    NVRTC
        .get_or_init(|| {
            // Windows names the DLL after NVRTC's own (major-version-stable since
            // CUDA 11.2) version, not the toolkit's; try the versions this repo has
            // seen before falling back to the unversioned SONAME on Linux.
            let lib = load_first(&[
                "nvrtc64_120_0.dll",
                "nvrtc64_112_0.dll",
                "libnvrtc.so",
                "libnvrtc.so.12",
                "libnvrtc.so.11.2",
            ])?;
            unsafe { Nvrtc::load(lib) }.ok()
        })
        .as_ref()
}

/// `cuGetErrorString` as a Rust string, for an error code from a driver call.
pub fn cu_error_string(err: CUresult) -> String {
    let drv = match driver() {
        Some(d) => d,
        None => return format!("CUDA error {err}"),
    };
    unsafe {
        let mut ptr: *const c_char = std::ptr::null();
        if (drv.cuGetErrorString)(err, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {err}")
        }
    }
}

/// `nvrtcGetErrorString` as a Rust string.
pub fn nvrtc_error_string(nv: &Nvrtc, err: nvrtcResult) -> String {
    unsafe {
        let ptr = (nv.nvrtcGetErrorString)(err);
        if ptr.is_null() {
            format!("NVRTC error {err}")
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}
