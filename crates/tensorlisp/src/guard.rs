//! Safe wrappers over ggml-sys's guarded calls (csrc/guard.h): a ggml
//! assertion inside them becomes [`Error::Ggml`] instead of aborting.
use std::{ffi::CStr, os::raw::c_char, sync::Once};

use ggml_sys::ffi::*;

use crate::error::{Error, Result};

/// Registers the abort callback. Idempotent.
pub fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe { tl_guard_install() });
}

fn guarded(call: impl FnOnce(*mut c_char, usize) -> bool) -> Result<()> {
    install();
    let mut err = [0 as c_char; 2048];
    if call(err.as_mut_ptr(), err.len()) {
        Ok(())
    } else {
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
        Err(Error::Ggml(msg))
    }
}

pub(crate) fn alloc_ctx_tensors(ctx: *mut ggml_context, backend: ggml_backend_t) -> Result<ggml_backend_buffer_t> {
    let mut buffer = std::ptr::null_mut();
    guarded(|err, len| unsafe { tl_try_backend_alloc_ctx_tensors(ctx, backend, &mut buffer, err, len) })?;
    Ok(buffer)
}

pub(crate) fn tensor_set(t: *mut ggml_tensor, data: &[u8]) -> Result<()> {
    guarded(|err, len| unsafe { tl_try_backend_tensor_set(t, data.as_ptr().cast(), 0, data.len(), err, len) })
}

pub(crate) fn tensor_get(t: *const ggml_tensor, data: &mut [u8]) -> Result<()> {
    guarded(|err, len| unsafe { tl_try_backend_tensor_get(t, data.as_mut_ptr().cast(), 0, data.len(), err, len) })
}

/// Ok(false) when the scheduler could not allocate the graph.
pub(crate) fn sched_alloc_graph(sched: ggml_backend_sched_t, graph: *mut ggml_cgraph) -> Result<bool> {
    let mut ok = false;
    guarded(|err, len| unsafe { tl_try_sched_alloc_graph(sched, graph, &mut ok, err, len) })?;
    Ok(ok)
}

pub(crate) fn sched_graph_compute(sched: ggml_backend_sched_t, graph: *mut ggml_cgraph) -> Result<ggml_status> {
    let mut status = ggml_status::GGML_STATUS_SUCCESS;
    guarded(|err, len| unsafe { tl_try_sched_graph_compute(sched, graph, &mut status, err, len) })?;
    Ok(status)
}
