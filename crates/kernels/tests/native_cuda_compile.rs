//! Every native CUDA kernel library compiles under NVRTC.
#![cfg(feature = "cuda")]

use tensorlisp_kernels::cuda::{lower::compile_all_libraries, Device};

#[test]
fn all_native_libraries_compile() {
    let Ok(dev) = Device::system_default() else { return };
    compile_all_libraries(&dev).unwrap_or_else(|e| panic!("{e}"));
}
