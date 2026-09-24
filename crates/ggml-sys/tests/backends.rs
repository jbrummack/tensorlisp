use std::ffi::CStr;

use ggml_sys::ffi::*;

#[test]
fn backends_register_and_init() {
    let names: Vec<String> = unsafe {
        (0..ggml_backend_dev_count())
            .map(|i| CStr::from_ptr(ggml_backend_dev_name(ggml_backend_dev_get(i))).to_string_lossy().into_owned())
            .collect()
    };
    assert!(names.iter().any(|n| n == "CPU"), "{names:?}");
    if ggml_sys::HAS_METAL {
        assert!(names.iter().any(|n| n.starts_with("MTL")), "{names:?}");
        let gpu = unsafe { ggml_backend_dev_by_type(ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU) };
        assert!(!gpu.is_null());
        let backend = unsafe { ggml_backend_dev_init(gpu, std::ptr::null()) };
        assert!(!backend.is_null());
        unsafe { ggml_backend_free(backend) };
    }
}
