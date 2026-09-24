//! ggml tensor types: names, sizes, conversion to f32 and quantization.
use std::ffi::CStr;

use ggml_sys::ffi::{
    ggml_blck_size, ggml_get_type_traits, ggml_is_quantized, ggml_quantize_chunk, ggml_quantize_requires_imatrix,
    ggml_row_size, ggml_type, ggml_type_name,
};

use crate::error::{Error, Result};

/// A ggml tensor type such as `f32`, `f16` or `q4_K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DType(pub ggml_type);

use ggml_type::*;

const ALL: &[ggml_type] = &[
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F64,
    GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ1_M,
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
];

/// Types ggml_quantize_chunk can produce without an importance matrix.
const QUANTIZE_TARGETS: &[ggml_type] = &[
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_MXFP4,
];

impl DType {
    pub const F32: DType = DType(GGML_TYPE_F32);
    pub const F16: DType = DType(GGML_TYPE_F16);
    pub const BF16: DType = DType(GGML_TYPE_BF16);

    pub fn all() -> impl Iterator<Item = DType> {
        ALL.iter().map(|&t| DType(t))
    }

    /// Types that `quantize` accepts as a target.
    pub fn quantize_targets() -> impl Iterator<Item = DType> {
        QUANTIZE_TARGETS.iter().map(|&t| DType(t))
    }

    pub fn name(self) -> &'static str {
        unsafe { CStr::from_ptr(ggml_type_name(self.0)) }.to_str().unwrap_or("?")
    }

    /// Parses a ggml type name, ignoring case (`q4_k` is `q4_K`).
    pub fn parse(name: &str) -> Result<DType> {
        Self::all().find(|t| t.name().eq_ignore_ascii_case(name)).ok_or_else(|| {
            let names: Vec<_> = Self::all().map(|t| t.name()).collect();
            Error::Input(format!("unknown type {name:?}, expected one of: {}", names.join(", ")))
        })
    }

    /// Number of values per block (1 for plain types).
    pub fn block_size(self) -> usize {
        unsafe { ggml_blck_size(self.0) as usize }
    }

    /// Bytes for `n` values, which must be a multiple of the block size.
    pub fn row_size(self, n: usize) -> usize {
        unsafe { ggml_row_size(self.0, n as i64) }
    }

    pub fn is_quantized(self) -> bool {
        unsafe { ggml_is_quantized(self.0) }
    }

    pub fn can_quantize_to(self) -> bool {
        QUANTIZE_TARGETS.contains(&self.0) && !unsafe { ggml_quantize_requires_imatrix(self.0) }
    }

    /// Decodes `bytes` of this type into f32.
    pub fn to_f32(self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self == DType::F32 {
            return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
        }
        let block = self.block_size();
        let block_bytes = self.row_size(block);
        if bytes.len() % block_bytes != 0 {
            return Err(Error::Input(format!("{} data of {} bytes is not whole blocks", self.name(), bytes.len())));
        }
        let n = bytes.len() / block_bytes * block;
        let to_float = unsafe { (*ggml_get_type_traits(self.0)).to_float }
            .ok_or_else(|| Error::Input(format!("ggml can't convert {} to f32", self.name())))?;
        let mut out = vec![0f32; n];
        unsafe { to_float(bytes.as_ptr().cast(), out.as_mut_ptr(), n as i64) };
        Ok(out)
    }

    /// Encodes rows of `n_per_row` f32 values as this type.
    pub fn from_f32(self, data: &[f32], n_per_row: usize) -> Result<Vec<u8>> {
        if !self.can_quantize_to() {
            let names: Vec<_> = Self::quantize_targets().filter(|t| t.can_quantize_to()).map(|t| t.name()).collect();
            return Err(Error::Input(format!("can't convert to {}, supported: {}", self.name(), names.join(", "))));
        }
        if n_per_row == 0 || n_per_row % self.block_size() != 0 || data.len() % n_per_row != 0 {
            return Err(Error::Input(format!(
                "row size {n_per_row} is not a multiple of the {} block size {}",
                self.name(),
                self.block_size()
            )));
        }
        let rows = data.len() / n_per_row;
        let mut out = vec![0u8; rows * self.row_size(n_per_row)];
        unsafe {
            ggml_quantize_chunk(
                self.0,
                data.as_ptr(),
                out.as_mut_ptr().cast(),
                0,
                rows as i64,
                n_per_row as i64,
                std::ptr::null(),
            )
        };
        Ok(out)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
