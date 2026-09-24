//! Filtering ggml's log output (it logs to stderr, including backend init chatter).
use std::{
    ffi::{CStr, c_char, c_void},
    sync::atomic::{AtomicU32, Ordering},
};

use ggml_sys::ffi::{ggml_log_level, ggml_log_set};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Off = 5,
}

static MIN_LEVEL: AtomicU32 = AtomicU32::new(LogLevel::Info as u32);
/// Level of the last message, which GGML_LOG_LEVEL_CONT messages continue.
static LAST_LEVEL: AtomicU32 = AtomicU32::new(LogLevel::Info as u32);

unsafe extern "C" fn log_callback(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
    let level = match level {
        ggml_log_level::GGML_LOG_LEVEL_CONT => LAST_LEVEL.load(Ordering::Relaxed),
        ggml_log_level::GGML_LOG_LEVEL_NONE => LogLevel::Info as u32,
        other => other as u32,
    };
    LAST_LEVEL.store(level, Ordering::Relaxed);
    if level >= MIN_LEVEL.load(Ordering::Relaxed) {
        eprint!("{}", unsafe { CStr::from_ptr(text) }.to_string_lossy());
    }
}

/// Only ggml messages at `level` or above are printed (to stderr).
pub fn set_ggml_log_level(level: LogLevel) {
    MIN_LEVEL.store(level as u32, Ordering::Relaxed);
    unsafe { ggml_log_set(Some(log_callback), std::ptr::null_mut()) };
}
