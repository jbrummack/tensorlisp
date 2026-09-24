//! Safe wrappers over ggml's gguf API for reading models and writing them.
//!
//! Shapes on the Rust side are in ndarray/numpy order (outermost first); ggml
//! stores them reversed (`ne[0]` is the innermost dimension). The memory layout
//! of a contiguous array is the same either way.
use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr::null_mut,
};

use ggml_sys::ffi::*;
use ndarray::ArrayViewD;

use crate::{
    dtype::DType,
    error::{Error, Result},
    program::{FORMAT_VERSION, Program, TL_ASSET_PREFIX, TL_BIN, TL_TXT, TL_VER},
};

/// ggml tensors have at most four dimensions.
pub const MAX_DIMS: usize = 4;

pub(crate) fn c_string(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::Input(format!("string contains a NUL byte: {s:?}")))
}

fn path_c_string(path: &Path) -> Result<CString> {
    let s = path
        .to_str()
        .ok_or_else(|| Error::Input(format!("path is not UTF-8: {}", path.display())))?;
    c_string(s)
}

/// Converts an ndarray shape to ggml `ne` (reversed, padded with 1s).
pub(crate) fn ne_from_shape(shape: &[usize]) -> Result<[i64; MAX_DIMS]> {
    if shape.len() > MAX_DIMS {
        return Err(Error::Input(format!(
            "shape {shape:?} has {} dimensions, ggml supports at most {MAX_DIMS}",
            shape.len()
        )));
    }
    let mut ne = [1i64; MAX_DIMS];
    for (i, &d) in shape.iter().rev().enumerate() {
        ne[i] = d as i64;
    }
    Ok(ne)
}

/// A GGUF file opened with tensor metadata only; no tensor data is loaded.
pub struct GgufFile {
    pub(crate) gguf: *mut gguf_context,
    /// Holds one unallocated tensor per GGUF tensor, named like it.
    pub(crate) tensors: *mut ggml_context,
}

// The contexts are only read after opening and are not tied to a thread.
unsafe impl Send for GgufFile {}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no such file: {}", path.display()),
            )));
        }
        let c_path = path_c_string(path)?;
        let mut tensors = null_mut();
        let params = gguf_init_params { no_alloc: true, ctx: &mut tensors };
        let gguf = unsafe { gguf_init_from_file(c_path.as_ptr(), params) };
        if gguf.is_null() {
            return Err(Error::Gguf(format!("failed to read {}", path.display())));
        }
        Ok(GgufFile { gguf, tensors })
    }

    fn key_id(&self, key: &str) -> Result<Option<i64>> {
        let key = c_string(key)?;
        let id = unsafe { gguf_find_key(self.gguf, key.as_ptr()) };
        Ok((id >= 0).then_some(id))
    }

    fn typed_key(&self, key: &str, expected: gguf_type) -> Result<Option<i64>> {
        let Some(id) = self.key_id(key)? else { return Ok(None) };
        let actual = unsafe { gguf_get_kv_type(self.gguf, id) };
        if actual != expected {
            return Err(Error::Gguf(format!("{key} is {actual:?}, expected {expected:?}")));
        }
        Ok(Some(id))
    }

    pub fn get_str(&self, key: &str) -> Result<Option<String>> {
        let Some(id) = self.typed_key(key, gguf_type::GGUF_TYPE_STRING)? else { return Ok(None) };
        let s = unsafe { CStr::from_ptr(gguf_get_val_str(self.gguf, id)) };
        s.to_str()
            .map(|s| Some(s.to_owned()))
            .map_err(|_| Error::Gguf(format!("{key} is not valid UTF-8")))
    }

    pub fn get_u32(&self, key: &str) -> Result<Option<u32>> {
        let Some(id) = self.typed_key(key, gguf_type::GGUF_TYPE_UINT32)? else { return Ok(None) };
        Ok(Some(unsafe { gguf_get_val_u32(self.gguf, id) }))
    }

    /// The tensorlisp program stored in the TL_* keys.
    pub fn program(&self) -> Result<Program> {
        match self.get_u32(TL_VER)? {
            Some(FORMAT_VERSION) => {}
            Some(v) => {
                return Err(Error::Program(format!(
                    "{TL_VER} is {v}, this runtime supports version {FORMAT_VERSION}"
                )));
            }
            None => return Err(Error::Program(format!("not a tensorlisp file: {TL_VER} is missing"))),
        }
        if let Some(text) = self.get_str(TL_TXT)? {
            return Ok(Program::Text(text));
        }
        if self.key_id(TL_BIN)?.is_some() {
            return Err(Error::Program(format!("binary programs ({TL_BIN}) are not supported yet")));
        }
        Err(Error::Program(format!("no program: neither {TL_TXT} nor {TL_BIN} is set")))
    }

    /// GGUF format version.
    pub fn version(&self) -> u32 {
        unsafe { gguf_get_version(self.gguf) }
    }

    pub fn tensor_count(&self) -> i64 {
        unsafe { gguf_get_n_tensors(self.gguf) }
    }

    pub fn tensor_name(&self, i: i64) -> &str {
        unsafe { CStr::from_ptr(gguf_get_tensor_name(self.gguf, i)) }.to_str().unwrap_or("")
    }

    /// Absolute file offset of tensor `i`'s data.
    pub fn tensor_file_offset(&self, i: i64) -> usize {
        unsafe { gguf_get_data_offset(self.gguf) + gguf_get_tensor_offset(self.gguf, i) }
    }

    /// All metadata key-value pairs, in file order.
    pub fn metadata(&self) -> Vec<(String, MetaValue)> {
        let n = unsafe { gguf_get_n_kv(self.gguf) };
        (0..n)
            .map(|id| {
                let key = unsafe { CStr::from_ptr(gguf_get_key(self.gguf, id)) }.to_string_lossy().into_owned();
                (key, self.value(id))
            })
            .collect()
    }

    fn value(&self, id: i64) -> MetaValue {
        use gguf_type::*;
        let ctx = self.gguf;
        unsafe {
            match gguf_get_kv_type(ctx, id) {
                GGUF_TYPE_ARRAY => {
                    let n = gguf_get_arr_n(ctx, id);
                    let ty = gguf_get_arr_type(ctx, id);
                    if ty == GGUF_TYPE_STRING {
                        return MetaValue::Array(
                            (0..n)
                                .map(|i| {
                                    MetaValue::String(
                                        CStr::from_ptr(gguf_get_arr_str(ctx, id, i)).to_string_lossy().into_owned(),
                                    )
                                })
                                .collect(),
                        );
                    }
                    let data = gguf_get_arr_data(ctx, id);
                    MetaValue::Array((0..n).map(|i| read_scalar(ty, data, i)).collect())
                }
                GGUF_TYPE_STRING => {
                    MetaValue::String(CStr::from_ptr(gguf_get_val_str(ctx, id)).to_string_lossy().into_owned())
                }
                ty => read_scalar(ty, gguf_get_val_data(ctx, id), 0),
            }
        }
    }

    /// Files embedded under `TL_ASSET.<name>`: (name, bytes).
    pub fn assets(&self) -> Vec<(String, Vec<u8>)> {
        let n = unsafe { gguf_get_n_kv(self.gguf) };
        (0..n)
            .filter_map(|id| unsafe {
                let key = CStr::from_ptr(gguf_get_key(self.gguf, id)).to_string_lossy();
                let name = key.strip_prefix(TL_ASSET_PREFIX)?.to_string();
                if gguf_get_kv_type(self.gguf, id) != gguf_type::GGUF_TYPE_ARRAY
                    || gguf_get_arr_type(self.gguf, id) != gguf_type::GGUF_TYPE_UINT8
                {
                    return None;
                }
                let len = gguf_get_arr_n(self.gguf, id);
                let data = std::slice::from_raw_parts(gguf_get_arr_data(self.gguf, id) as *const u8, len);
                Some((name, data.to_vec()))
            })
            .collect()
    }

    /// Name, type, shape and location of every tensor, in file order.
    pub fn tensor_infos(&self) -> Vec<TensorInfo> {
        (0..self.tensor_count())
            .map(|i| {
                let name = self.tensor_name(i).to_owned();
                let dtype = DType(unsafe { gguf_get_tensor_type(self.gguf, i) });
                let shape = match self.tensor(&name) {
                    Ok(Some(t)) => unsafe {
                        let n = ggml_n_dims(t) as usize;
                        (0..n).rev().map(|d| (*t).ne[d] as usize).collect()
                    },
                    _ => Vec::new(),
                };
                TensorInfo {
                    shape,
                    dtype,
                    nbytes: unsafe { gguf_get_tensor_size(self.gguf, i) },
                    offset: self.tensor_file_offset(i),
                    name,
                }
            })
            .collect()
    }

    /// Reads tensor `i`'s raw bytes from the file at `path` (this file).
    pub fn read_tensor_bytes(&self, path: impl AsRef<Path>, i: i64) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; unsafe { gguf_get_tensor_size(self.gguf, i) }];
        std::fs::File::open(path)?.read_exact_at(&mut buf, self.tensor_file_offset(i) as u64)?;
        Ok(buf)
    }

    /// The metadata-only ggml tensor for `name`, if the file has one.
    pub(crate) fn tensor(&self, name: &str) -> Result<Option<*mut ggml_tensor>> {
        let name = c_string(name)?;
        let t = unsafe { ggml_get_tensor(self.tensors, name.as_ptr()) };
        Ok((!t.is_null()).then_some(t))
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        unsafe {
            gguf_free(self.gguf);
            if !self.tensors.is_null() {
                ggml_free(self.tensors);
            }
        }
    }
}

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

unsafe fn read_scalar(ty: gguf_type, data: *const std::ffi::c_void, i: usize) -> MetaValue {
    use gguf_type::*;
    unsafe {
        match ty {
            GGUF_TYPE_UINT8 => MetaValue::U8(*data.cast::<u8>().add(i)),
            GGUF_TYPE_INT8 => MetaValue::I8(*data.cast::<i8>().add(i)),
            GGUF_TYPE_UINT16 => MetaValue::U16(*data.cast::<u16>().add(i)),
            GGUF_TYPE_INT16 => MetaValue::I16(*data.cast::<i16>().add(i)),
            GGUF_TYPE_UINT32 => MetaValue::U32(*data.cast::<u32>().add(i)),
            GGUF_TYPE_INT32 => MetaValue::I32(*data.cast::<i32>().add(i)),
            GGUF_TYPE_UINT64 => MetaValue::U64(*data.cast::<u64>().add(i)),
            GGUF_TYPE_INT64 => MetaValue::I64(*data.cast::<i64>().add(i)),
            GGUF_TYPE_FLOAT32 => MetaValue::F32(*data.cast::<f32>().add(i)),
            GGUF_TYPE_FLOAT64 => MetaValue::F64(*data.cast::<f64>().add(i)),
            GGUF_TYPE_BOOL => MetaValue::Bool(*data.cast::<u8>().add(i) != 0),
            _ => MetaValue::String(format!("<unsupported {ty:?}>")),
        }
    }
}

/// A tensor stored in a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    /// ndarray order (outermost first).
    pub shape: Vec<usize>,
    pub nbytes: usize,
    /// Absolute offset of the data in the file.
    pub offset: usize,
}

/// Writes a GGUF file. Tensor data is streamed to the file one tensor at a
/// time, so data added with [`GgufWriter::add_tensor_with`] is only produced
/// (and held in memory) while it is written.
pub struct GgufWriter<'a> {
    gguf: *mut gguf_context,
    tensors: Vec<PendingTensor<'a>>,
    names: std::collections::HashSet<String>,
}

struct PendingTensor<'a> {
    /// no_alloc context holding the tensor's metadata.
    ctx: *mut ggml_context,
    nbytes: usize,
    data: Box<dyn FnOnce() -> Result<Vec<u8>> + 'a>,
}

impl Default for GgufWriter<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> GgufWriter<'a> {
    pub fn new() -> Self {
        GgufWriter { gguf: unsafe { gguf_init_empty() }, tensors: Vec::new(), names: Default::default() }
    }

    pub fn set_str(&mut self, key: &str, value: &str) -> Result<()> {
        let (key, value) = (c_string(key)?, c_string(value)?);
        unsafe { gguf_set_val_str(self.gguf, key.as_ptr(), value.as_ptr()) };
        Ok(())
    }

    pub fn set_u32(&mut self, key: &str, value: u32) -> Result<()> {
        let key = c_string(key)?;
        unsafe { gguf_set_val_u32(self.gguf, key.as_ptr(), value) };
        Ok(())
    }

    /// Stores bytes as a u8 array.
    pub fn set_bytes(&mut self, key: &str, data: &[u8]) -> Result<()> {
        let key = c_string(key)?;
        unsafe { gguf_set_arr_data(self.gguf, key.as_ptr(), gguf_type::GGUF_TYPE_UINT8, data.as_ptr().cast(), data.len()) };
        Ok(())
    }

    /// Embeds a file programs can read with `(asset name)`.
    pub fn set_asset(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.set_bytes(&format!("{TL_ASSET_PREFIX}{name}"), data)
    }

    pub fn remove_key(&mut self, key: &str) -> Result<()> {
        let key = c_string(key)?;
        unsafe { gguf_remove_key(self.gguf, key.as_ptr()) };
        Ok(())
    }

    /// Copies every metadata key of `file` (not its tensors).
    pub fn copy_metadata(&mut self, file: &GgufFile) {
        unsafe { gguf_set_kv(self.gguf, file.gguf) };
    }

    /// Stores `program` in the TL_* keys, replacing any previous program.
    pub fn set_program(&mut self, program: &Program) -> Result<()> {
        self.remove_key(TL_BIN)?;
        self.set_u32(TL_VER, FORMAT_VERSION)?;
        match program {
            Program::Text(text) => self.set_str(TL_TXT, text),
        }
    }

    /// Adds a tensor whose bytes are produced by `data` when the file is written.
    /// `shape` is in ndarray order.
    pub fn add_tensor_with(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
        data: impl FnOnce() -> Result<Vec<u8>> + 'a,
    ) -> Result<()> {
        let c_name = c_string(name)?;
        if c_name.as_bytes().len() >= GGML_MAX_NAME as usize {
            return Err(Error::Input(format!("tensor name longer than {} bytes: {name}", GGML_MAX_NAME - 1)));
        }
        if !self.names.insert(name.to_owned()) {
            return Err(Error::Input(format!("duplicate tensor name {name}")));
        }
        let ne = ne_from_shape(shape)?;
        if ne[0] as usize % dtype.block_size() != 0 {
            return Err(Error::Input(format!(
                "tensor {name}: innermost dimension {} is not a multiple of the {dtype} block size {}",
                ne[0],
                dtype.block_size()
            )));
        }
        let rows: usize = ne[1..].iter().product::<i64>() as usize;
        let nbytes = dtype.row_size(ne[0] as usize) * rows;
        unsafe {
            let ctx = ggml_init(ggml_init_params {
                mem_size: ggml_tensor_overhead(),
                mem_buffer: null_mut(),
                no_alloc: true,
            });
            let t = ggml_new_tensor(ctx, dtype.0, shape.len().max(1) as i32, ne.as_ptr());
            ggml_set_name(t, c_name.as_ptr());
            gguf_add_tensor(self.gguf, t);
            self.tensors.push(PendingTensor { ctx, nbytes, data: Box::new(data) });
        }
        Ok(())
    }

    /// Adds a tensor from raw bytes of `dtype`. `shape` is in ndarray order.
    pub fn add_tensor(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: Vec<u8>) -> Result<()> {
        self.add_tensor_with(name, dtype, shape, move || Ok(bytes))
    }

    pub fn add_f32(&mut self, name: &str, array: ArrayViewD<f32>) -> Result<()> {
        let bytes = array.as_standard_layout().iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_tensor(name, DType::F32, array.shape(), bytes)
    }

    /// Writes the metadata with ggml, then streams each tensor's data to its offset.
    pub fn write(mut self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let path = path.as_ref();
        let c_path = path_c_string(path)?;
        if !unsafe { gguf_write_to_file(self.gguf, c_path.as_ptr(), true) } {
            return Err(Error::Gguf(format!("failed to write {}", path.display())));
        }
        let alignment = unsafe { gguf_get_alignment(self.gguf) };
        let mut out = std::io::BufWriter::new(std::fs::OpenOptions::new().append(true).open(path)?);
        let mut written = 0usize;
        for (i, pending) in std::mem::take(&mut self.tensors).into_iter().enumerate() {
            let name = self.tensor_name(i as i64);
            let bytes = (pending.data)()?;
            unsafe { ggml_free(pending.ctx) };
            if bytes.len() != pending.nbytes {
                return Err(Error::Input(format!(
                    "tensor {name}: got {} bytes of data, expected {}",
                    bytes.len(),
                    pending.nbytes
                )));
            }
            let offset = unsafe { gguf_get_tensor_offset(self.gguf, i as i64) };
            out.write_all(&vec![0u8; offset - written])?;
            out.write_all(&bytes)?;
            written = offset + bytes.len();
        }
        out.write_all(&vec![0u8; written.next_multiple_of(alignment) - written])?;
        out.flush()?;
        Ok(())
    }

    fn tensor_name(&self, i: i64) -> String {
        unsafe { CStr::from_ptr(gguf_get_tensor_name(self.gguf, i)) }.to_string_lossy().into_owned()
    }
}

impl Drop for GgufWriter<'_> {
    fn drop(&mut self) {
        unsafe {
            gguf_free(self.gguf);
            for pending in &self.tensors {
                ggml_free(pending.ctx);
            }
        }
    }
}
