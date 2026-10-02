//! ggml's `ggml_metal_kargs_*` structs (generated from ggml-metal-impl.h) and a
//! packer for building one from typed values, for lowerings that don't go
//! through the Rust struct (the Scheme matmul policy).

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ggml_metal_kargs.rs"));
}

use crate::{Error, Result};

/// A struct's bytes as passed to `setBytes`.
pub fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    // SAFETY: plain-old-data repr(C) structs from bindgen; padding bytes are
    // whatever the zeroed/initialized struct had, which the kernels never read.
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) }.to_vec()
}

/// A scalar of a kernel-argument struct field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Field {
    Bool(bool),
    I16(i16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
}

impl Field {
    fn size(self) -> usize {
        match self {
            Field::Bool(_) => 1,
            Field::I16(_) => 2,
            Field::I32(_) | Field::U32(_) | Field::F32(_) => 4,
            Field::I64(_) | Field::U64(_) => 8,
        }
    }
}

/// Packs fields like a C struct: each at its natural alignment, total size
/// padded to the largest alignment.
pub fn pack(fields: &[Field]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut max_align = 1;
    for f in fields {
        let size = f.size();
        max_align = max_align.max(size);
        while out.len() % size != 0 {
            out.push(0);
        }
        match *f {
            Field::Bool(v) => out.push(v as u8),
            Field::I16(v) => out.extend(v.to_ne_bytes()),
            Field::I32(v) => out.extend(v.to_ne_bytes()),
            Field::U32(v) => out.extend(v.to_ne_bytes()),
            Field::I64(v) => out.extend(v.to_ne_bytes()),
            Field::U64(v) => out.extend(v.to_ne_bytes()),
            Field::F32(v) => out.extend(v.to_ne_bytes()),
        }
    }
    while out.len() % max_align != 0 {
        out.push(0);
    }
    out
}

/// `sizeof(ggml_metal_kargs_<name>)` for the structs lowerings may pack by hand.
pub fn struct_size(name: &str) -> Option<usize> {
    use ffi::*;
    Some(match name {
        "mul_mm" => size_of::<ggml_metal_kargs_mul_mm>(),
        "mul_mv" => size_of::<ggml_metal_kargs_mul_mv>(),
        "mul_mv_ext" => size_of::<ggml_metal_kargs_mul_mv_ext>(),
        _ => return None,
    })
}

/// Packs `fields` for struct `name` and checks the size against ggml's definition.
pub fn pack_checked(name: &str, fields: &[Field]) -> Result<Vec<u8>> {
    let bytes = pack(fields);
    match struct_size(name) {
        Some(n) if n == bytes.len() => Ok(bytes),
        Some(n) => Err(Error::Invalid(format!(
            "kernel args for `{name}` pack to {} bytes, ggml_metal_kargs_{name} is {n}",
            bytes.len()
        ))),
        None => Err(Error::Invalid(format!("unknown kernel argument struct `{name}`"))),
    }
}
