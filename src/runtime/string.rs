use std::ffi::CStr;

pub struct TString([u8; 64]);
impl TString {
    pub fn to_c(&self) -> Option<&CStr> {
        CStr::from_bytes_until_nul(&self.0).ok()
    }
}
