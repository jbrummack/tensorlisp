//! Interop with ggml's Metal backend: same `MTLDevice`, same command queue,
//! and ggml tensors usable as kernel arguments (their `MTLBuffer` + offset).
//!
//! Ordering: ggml and this crate commit command buffers to one serial queue,
//! so a launch followed by `Device::sync` is ordered with respect to any ggml
//! graph computed before or after it. Sharing the queue also keeps ggml's
//! `MTLResidencySet` (attached to it) covering buffers we touch.

use ggml_sys::ffi;

use super::{Buffer, Device};
use crate::{Error, Result};

/// A [`Device`] on ggml's Metal device and queue. Requires that ggml-sys was
/// built with Metal (`ggml_sys::HAS_METAL`).
pub fn device() -> Result<Device> {
    if !ggml_sys::HAS_METAL {
        return Err(Error::NoDevice);
    }
    // SAFETY: both handles are live for the process (ggml's device registry is static).
    unsafe { Device::from_raw(ffi::tl_ggml_metal_device(), ffi::tl_ggml_metal_queue()) }
}

/// The Metal buffer holding `t` and the byte offset of its data in it; `None`
/// if the tensor isn't allocated in a Metal buffer. Pass `buffer.at(offset)`
/// as a kernel argument.
///
/// # Safety
/// `t` must point to a live `ggml_tensor`, and the buffer's storage is only
/// valid while ggml keeps that tensor allocated.
pub unsafe fn tensor_buffer(t: *const ffi::ggml_tensor) -> Option<(Buffer, usize)> {
    let mut offset = 0usize;
    let raw = unsafe { ffi::tl_ggml_metal_tensor_buffer(t, &mut offset) };
    unsafe { Buffer::from_raw(raw) }.map(|b| (b, offset))
}
