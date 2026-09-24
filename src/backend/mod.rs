use std::ffi::CStr;
pub mod dyntype;
pub mod gguf;
pub mod hashcons;
pub mod list;
pub mod loader;
pub mod stupid_lisp;
use crate::backend::ffi::ggml_tensor;

include!("bindings.rs");

pub struct Tensor(*mut ggml_tensor);
impl Tensor {
    fn get(&self) -> &ggml_tensor {
        unsafe { self.0.as_ref_unchecked() }
    }
    pub fn name(&self) -> &str {
        let tensor = self.get();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(tensor.name.as_ptr() as *const u8, tensor.name.len())
        };
        let s = CStr::from_bytes_until_nul(bytes).map(|cs| cs.to_str().unwrap_or(""));
        s.unwrap_or("")
    }
}
