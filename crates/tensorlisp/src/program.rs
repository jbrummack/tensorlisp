//! Where a tensorlisp program lives inside a GGUF file.

/// UTF-8 source text of the program (GGUF string).
pub const TL_TXT: &str = "TL_TXT";
/// Binary program (GGUF u8 array). Reserved, not supported yet.
pub const TL_BIN: &str = "TL_BIN";
/// Program format version (GGUF u32).
pub const TL_VER: &str = "TL_VER";

/// The format version this runtime reads and writes.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum Program {
    Text(String),
}
