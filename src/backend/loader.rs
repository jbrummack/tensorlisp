use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CStr, CString, NulError, c_int},
    marker::PhantomData,
    os::raw::c_void,
    ptr::{null, null_mut, slice_from_raw_parts},
    rc::Rc,
    slice,
    str::FromStr,
    sync::Arc,
};

use crate::backend::{
    ffi::{
        ggml_backend_alloc_ctx_tensors, ggml_build_forward_expand, ggml_context, ggml_free,
        ggml_get_name, ggml_get_tensor, ggml_graph_compute, ggml_graph_overhead, ggml_graph_plan,
        ggml_is_contiguous, ggml_new_graph, ggml_new_graph_custom, ggml_new_tensor,
        ggml_quantize_chunk, ggml_quantize_requires_imatrix, ggml_row_size, ggml_status,
        ggml_tensor, ggml_tensor_overhead, ggml_type, gguf_context, gguf_free, gguf_get_arr_data,
        gguf_get_arr_n, gguf_get_arr_type, gguf_get_key, gguf_get_kv_type, gguf_get_n_kv,
        gguf_get_n_tensors, gguf_get_tensor_name, gguf_get_tensor_type, gguf_get_val_i8,
        gguf_get_val_str, gguf_get_val_u8, gguf_get_version, gguf_init_from_file, gguf_init_params,
        gguf_type, gguf_write_to_file,
    },
    loader::{LoaderError::GgufFailed, ModelError::DoesntExist},
};
/*pub struct Kv<T>(unsafe extern "C" fn(*const gguf_context, i64) -> T);
impl<T> Kv<T> {
    pub fn exec(&self, ctx: *const gguf_context, key_id: i64) -> T {
        unsafe { (self.0)(ctx, key_id) }
    }
}*/
/*pub struct AnyInt(i64);
impl From<u64> for AnyInt {
    fn from(value: u64) -> Self {
        AnyInt(value as i64)
    }
}

impl From<AnyInt> for u64 {
    fn from(value: AnyInt) -> Self {
        value.0 as u64
    }
}*/
/*#[derive(Debug,thiserror::Error)]
pub enum IntConvErr {

}*/
/*pub trait AnyInt<B> {
    fn conv(self) -> B;
}
impl AnyInt<u8> for u32 {
    fn conv(self) -> u8 {
        self as u8
    }
}
impl AnyInt<u16> for u32 {
    fn conv(self) -> u16 {
        self as u16
    }
}*/
//#[derive(Debug)]
pub enum ModelStruct<'a> {
    Struct(BTreeMap<&'a str, Self>),
    Array(BTreeMap<i16, Self>),
    Leaf(Tensor<'a>),
}
impl<'a> std::fmt::Debug for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let ty = self.ty();
        let typestr = format!("{ty:?}").replace("GGML_TYPE_", "").to_lowercase();
        write!(f, "Tensor<{typestr},{shape:?}>")
        //GGML_TYPE_
        /*f.debug_struct("Tensor")
        .field("model", &self.model)
        .field("tensor", &self.tensor)
        .finish()*/
    }
}
impl<'a> ModelStruct<'a> {
    fn fmt_indented(&self, f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
        let pad = "  ".repeat(indent);
        let inner_pad = "  ".repeat(indent + 1);

        match self {
            Self::Struct(arg0) => {
                writeln!(f, "(struct")?;
                for (key, value) in arg0 {
                    // Indent key-value pair and recursively indent inner values
                    write!(f, "{inner_pad}({} ", key)?;
                    value.fmt_indented(f, indent + 1)?;
                    writeln!(f, ")")?;
                }
                write!(f, "{pad})")
            }
            Self::Array(arg0) => {
                writeln!(f, "(list")?;
                for (_key, value) in arg0 {
                    write!(f, "{inner_pad}")?;
                    value.fmt_indented(f, indent + 1)?;
                    writeln!(f)?;
                }
                write!(f, "{pad})")
            }
            Self::Leaf(tensor) => write!(f, "{tensor:?}"),
        }
    }
}
impl<'a> std::fmt::Display for ModelStruct<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_indented(f, 0)
    }
}
impl<'a> std::fmt::Debug for ModelStruct<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Struct(arg0) => {
                let mut structure = f.debug_map();
                for (key, value) in arg0 {
                    structure.entry(key, value);
                    //structure.field(key, value);
                }
                structure.finish()
            }
            Self::Array(arg0) => {
                //let alt = f.alternate();
                let mut array = f.debug_list();
                for (_key, value) in arg0 {
                    /*let idxd = if alt {
                        format_args!("({key}){value:#?}");
                    } else {
                        format_args!("({key}){value:?}");
                    };*/
                    array.entry(&value);
                }
                array.finish()
            }
            Self::Leaf(tensor) => write!(f, "{tensor:?}"),
        }
    }
}
impl<'a> ModelStruct<'a> {
    pub fn parse(
        model: &'a Model,
        //names: impl Iterator<Item = &'a (impl AsRef<str> + ?Sized + 'a)>,
    ) -> ModelStruct<'a> {
        let mut structure = Self::Struct(BTreeMap::new());
        for (t, s) in model.iter_keys() {
            let st: &'a str = s.as_ref();
            //if let Some(tensor) = model.get_tensor(st) {
            let split = st.split(".");

            structure.insert(split, t);
            //}
        }
        structure
    }

    pub fn insert(&mut self, mut s: std::str::Split<'a, &str>, tensor: Tensor<'a>) {
        let next = if let Some(n) = s.next() { n } else { return };
        let is_number: Option<i16> = next.parse().ok();
        match self {
            ModelStruct::Struct(hash_map) => {
                if let Some(_) = is_number {
                    println!("failed at {hash_map:?}");
                } else {
                    let entry = hash_map.entry(next).or_insert(ModelStruct::Leaf(tensor));
                    entry.insert(s, tensor);
                }
            }
            ModelStruct::Array(btree_map) => {
                if let Some(n) = is_number {
                    let entry = btree_map.entry(n).or_insert(ModelStruct::Leaf(tensor));
                    entry.insert(s, tensor);
                } else {
                    println!("failed at {btree_map:?}")
                }
            }
            ModelStruct::Leaf(t) => {
                if let Some(n) = is_number {
                    let mut arr = BTreeMap::new();
                    arr.entry(n).or_insert(ModelStruct::Leaf(*t)).insert(s, *t);
                    *self = Self::Array(arr);
                } else {
                    let mut map = BTreeMap::new();
                    map.entry(next)
                        .or_insert(ModelStruct::Leaf(*t))
                        .insert(s, *t);
                    *self = Self::Struct(map);
                }
            }
        }
    }
}
pub struct GGTy<T>(PhantomData<T>);
impl<T> GGTy<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
/*impl TryFrom<gguf_type> for Kv<i32> {
    type Error = gguf_type;

    fn try_from(value: gguf_type) -> Result<Self, Self::Error> {
        if value == gguf_type::GGUF_TYPE_INT32 {
            Ok(Kv(super::ffi::gguf_get_val_i32))
        } else {
            Err(value)
        }
    }
}*/

pub trait GgufTypeConvert {
    const ARRAY: bool;
    const GGUF_SCALAR: gguf_type;
    type SelfT;
}

pub struct CtxTy {
    ctx: *const gguf_context,
    key_id: i64,
}
impl TryFrom<CtxTy> for u8 {
    type Error = gguf_type;

    fn try_from(value: CtxTy) -> Result<Self, Self::Error> {
        let CtxTy { ctx, key_id } = value;
        unsafe {
            let checkty = gguf_get_kv_type(ctx, key_id);
            if checkty == gguf_type::GGUF_TYPE_UINT8 {
                Ok(gguf_get_val_u8(ctx, key_id))
            } else {
                Err(checkty)
            }
        }
    }
}
pub struct Loader;
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("unable to convert rust string to c string {0}")]
    StringConv(NulError),
    #[error("Unable to load gguf file")]
    GgufFailed,
}
impl Loader {
    pub fn open(fname: impl AsRef<str>) -> Result<Model, LoaderError> {
        let path: &str = fname.as_ref();
        let initial_box: Box<*mut ggml_context> = Box::new(null_mut());
        let raw_ctx: *mut *mut ggml_context = Box::into_raw(initial_box);
        let params = gguf_init_params {
            no_alloc: true,
            ctx: raw_ctx,
        };
        //let new_box: Box<*mut ggml_context> = unsafe { Box::from_raw(raw_ptr) };
        let cs = CString::new(path).map_err(LoaderError::StringConv)?;
        let fname = cs.as_ptr();
        let ctx = unsafe { gguf_init_from_file(fname, params) };
        if ctx.is_null() {
            return Err(GgufFailed);
        }
        if params.ctx.is_null() {
            return Err(GgufFailed);
        }
        let ggml = unsafe { params.ctx.read() };
        println!("ggml: {ggml:?}, gguf: {ctx:?}");
        Model::new(ctx, ggml)
    }
}
#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    model: PhantomData<&'a ()>,
    tensor: *mut ggml_tensor,
}
impl<'a> Tensor<'a> {
    pub fn name(&'a self) -> &'a str {
        unsafe {
            let ptr = ggml_get_name(self.tensor);
            CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }
    pub fn ty(&self) -> ggml_type {
        unsafe { self.tensor.read().type_ }
    }
    pub fn shape(&self) -> [i64; 4] {
        unsafe { self.tensor.read().ne }
    }
    pub unsafe fn data_ptr(&self) -> *mut c_void {
        unsafe { self.tensor.read().data }
    }
    pub fn data<T: GgmlScalar>(&self) -> Result<&[T], ModelError> {
        unsafe {
            let tensor = self.tensor.as_ref_unchecked();
            let data = tensor.data;
            if !ggml_is_contiguous(self.tensor) {
                Err(ModelError::NotContigouos)?;
            }
            let len = self.shape().iter().product::<i64>() as usize;

            if self.ty() != T::GGML_SCALAR {
                Err(ModelError::InvalidTensorType {
                    got: self.ty(),
                    expected: T::GGML_SCALAR,
                })?;
            }
            Ok(std::slice::from_raw_parts(data as *const T, len))
        }
    }
    pub fn quantise(&self, target_type: ggml_type) -> Result<QuantizedTensor, ModelError> {
        let n_per_row = self.shape()[0];
        let nrows = self.shape().iter().product::<i64>() / n_per_row;
        unsafe {
            if ggml_quantize_requires_imatrix(target_type) {
                Err(ModelError::RequireIMatrix)?;
            }
        }
        let quantised_row_size = unsafe { ggml_row_size(target_type, n_per_row) };
        let mut dest_buf = vec![0u8; quantised_row_size * (nrows as usize)];
        let src_data = self.data::<f32>()?;
        unsafe {
            ggml_quantize_chunk(
                target_type,
                src_data.as_ptr(),
                dest_buf.as_mut_ptr() as *mut _,
                0,
                nrows,
                n_per_row,
                null(),
            );
        };
        let result = QuantizedTensor {
            data: dest_buf,
            ty: target_type,
            shape: self.shape(),
        };
        Ok(result)
    }
    ///stride in bytes:
    /// nb\[0] = ggml_type_size(type)
    /// nb\[1] = nb\[0]   * (ne\[0] / ggml_blck_size(type)) + padding
    /// nb\[i] = nb\[i-1] * ne\[i-1]
    pub fn stride(&self) -> [usize; 4] {
        unsafe { self.tensor.read().nb }
    }
}
pub struct QuantizedTensor {
    data: Vec<u8>,
    shape: [i64; 4],
    ty: ggml_type,
}
pub struct Model {
    ggml: *mut ggml_context,
    ctx: *mut gguf_context,
    //n_kv: i64,
    kv: HashMap<String, i64>,
    tensors: HashMap<String, *const i8>,
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                gguf_free(self.ctx);
            }
            if !self.ggml.is_null() {
                ggml_free(self.ggml);
            }
        }
    }
}

pub struct GgufString {
    inner: *const ::std::os::raw::c_char,
}
impl AsRef<str> for GgufString {
    fn as_ref(&self) -> &str {
        unsafe { &CStr::from_ptr(self.inner).to_str().unwrap_or("") }
    }
}
pub enum GgufValue {
    String(*const ::std::os::raw::c_char),
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    I32(i32),
    U32(u32),
    F32(f32),
    I64(i64),
    U64(u64),
    F64(f64),
    Array(GgufDynArray),
}
/*pub struct Mtree {
    names: lasso::Rodeo,
    root: Vec<Spur>,
}*/
pub struct GgufType<T>(PhantomData<T>);
impl<T> GgufType<T> {
    pub const fn align() -> usize {
        std::mem::size_of::<T>()
    }
}
use crate::backend::ffi::{GgmlScalar, GgufScalar};
impl Model {
    pub fn static_overhead() -> usize {
        unsafe { ggml_tensor_overhead() + ggml_graph_overhead() + 1024 }
    }
    /*pub fn calculate_model_overhead(&self) {
        for (k,v) in self.tensors {
            galloc
        }
    }*/

    pub fn get_array<'a, T: GgufScalar>(
        &'a self,
        key_id: i64,
    ) -> Result<GgufTypedArr<'a, T>, ModelError> {
        unsafe {
            let got = gguf_get_arr_type(self.ctx, key_id);
            let expected = T::GGUF_SCALAR;
            if got != expected {
                Err(ModelError::InvalidType { got, expected })?;
            }
            let data = gguf_get_arr_data(self.ctx, key_id) as *const T;
            let size = gguf_get_arr_n(self.ctx, key_id);
            Ok(GgufTypedArr {
                size,
                data,
                aliveness: PhantomData,
            })
        }
    }
}
pub struct GgufRcArr<T> {
    size: usize,
    data: *const T,
    _aliveness: Rc<Model>,
}
impl<T> AsRef<[T]> for GgufRcArr<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data, self.size) }
    }
}
pub struct GgufTypedArr<'a, T> {
    size: usize,
    data: *const T,
    aliveness: PhantomData<&'a ()>,
}
impl<'a, T> AsRef<[T]> for GgufTypedArr<'a, T> {
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data, self.size) }
    }
}
impl<'a, T> GgufTypedArr<'a, T> {
    pub fn to_slice(&self) -> &[T] {
        self.as_ref()
    }
}

pub struct GgufDynArray {
    ty: gguf_type,
    size: usize,
    data: *const c_void,
}

/*pub enum GgufType {
    Scalar(GgufValue),
    Array(GgufDynArray),
}*/

use crate::backend::ffi::Kv;
//pub struct FFIString([i8;64]);
impl Model {
    fn get_value<T: TryFrom<gguf_type, Error = gguf_type>>(
        &self,
        key_id: i64,
    ) -> Result<T, gguf_type> {
        let ty = unsafe { gguf_get_kv_type(self.ctx, key_id) };
        let result: T = T::try_from(ty)?;
        Ok(result)
    }
    pub fn metadata<T: GgufScalar>(&self, name: impl AsRef<str>) -> Result<T, ModelError> {
        todo!()
    }
    pub fn get_tensor<'a>(&'a self, name: impl AsRef<str>) -> Option<Tensor<'a>> {
        /*if !self.tensors.contains_key(name.as_ref()) {
            return None;
        }*/

        let tensor_id = self.tensors.get(name.as_ref())?;
        unsafe {
            /*let cstr = CString::from_str(name.as_ref()).ok()?;*/
            let tensor = ggml_get_tensor(self.ggml, *tensor_id);
            Some(Tensor {
                model: PhantomData,
                tensor,
            })
        }
    }
    fn _value<T>(&self, name: impl AsRef<str>) -> Result<T, ModelError>
    where
        Kv<T>: TryFrom<gguf_type, Error = gguf_type>,
    {
        let key_id = self.kv.get(name.as_ref()).ok_or(DoesntExist)?;
        self.get_kv(*key_id)
            .map_err(|expected| ModelError::InvalidType {
                got: unsafe { gguf_get_kv_type(self.ctx, *key_id) },
                expected,
            })
    }
    fn get_kv<T>(&self, key_id: i64) -> Result<T, gguf_type>
    where
        Kv<T>: TryFrom<gguf_type, Error = gguf_type>,
    {
        let kv: Kv<T> = self.get_value(key_id)?;
        Ok(kv.exec(self.ctx, key_id))
    }
    pub fn new_tensor<const D: usize, T>(&self, size: [i64; D]) {
        unsafe {
            let tensor = ggml_new_tensor(
                self.ggml,
                ggml_type::GGML_TYPE_F32,
                D as c_int,
                size.as_ptr(),
            );
        }
    }
    /*
    * // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    */
    fn _compute<'a>(&'_ self, tensor: Tensor<'a>) -> Result<Tensor<'a>, ggml_status> {
        unsafe {
            let graph = ggml_new_graph(self.ggml);
            ggml_build_forward_expand(graph, tensor.tensor);

            let plan = Box::new(ggml_graph_plan(graph, 2, null_mut()));
            let plan_ptr = Box::into_raw(plan);
            let result: ggml_status = ggml_graph_compute(graph, plan_ptr);

            let _drop = Box::from_raw(plan_ptr);
            if result != ggml_status::GGML_STATUS_SUCCESS {
                Err(result)?;
            }
            Ok(tensor)
            //ggml_backend_alloc_ctx_tensors(self.ggml, backend)
        }
    }
    fn iter_keys(&self) -> impl Iterator<Item = (Tensor, &str)> {
        unsafe {
            let n_tensors = gguf_get_n_tensors(self.ctx);
            println!("getting {n_tensors} tensors");
            (0..n_tensors).into_iter().flat_map(|tensor_id| {
                let tensor_name = gguf_get_tensor_name(self.ctx, tensor_id);
                let tensor = Tensor {
                    model: PhantomData,
                    tensor: ggml_get_tensor(self.ggml, tensor_name),
                };
                let name = CStr::from_ptr(tensor_name);
                let s = name.to_str();
                s.map(|s| (tensor, s))
            })
        }
    }
    fn new(ctx: *mut gguf_context, ggml: *mut ggml_context) -> Result<Self, LoaderError> {
        unsafe {
            let n_tensors = gguf_get_n_tensors(ctx);
            let n_kv = gguf_get_n_kv(ctx);
            let kv: HashMap<String, i64> = (0..n_kv)
                .into_iter()
                .map(|kv_id| {
                    let key_name = gguf_get_key(ctx, kv_id);
                    //let typ = gguf_get_kv_type(ctx, kv_id);
                    let name = CStr::from_ptr(key_name);
                    let s = name.to_string_lossy().to_string();
                    //println!("{s} {typ:?}");
                    (s, kv_id)
                    //let ty = gguf_get_kv_type(ctx, kv_id);
                })
                .collect();
            let tensors: HashMap<String, *const i8> = (0..n_tensors)
                .into_iter()
                .map(|tensor_id| {
                    let tensor_name = gguf_get_tensor_name(ctx, tensor_id);
                    let ttype = gguf_get_tensor_type(ctx, tensor_id);
                    let name = CStr::from_ptr(tensor_name);
                    let s = name.to_string_lossy().to_string();
                    //println!("{tensor_id}: {s} {ttype:?}");
                    (s.to_string(), tensor_name)
                })
                .collect();
            for tensor_id in 0..n_tensors {
                let tensor_name = gguf_get_tensor_name(ctx, tensor_id);
                let ttype = gguf_get_tensor_type(ctx, tensor_id);
                let name = CStr::from_ptr(tensor_name);
                let s = name.to_string_lossy().to_string();
                println!("{tensor_id}: {s} {ttype:?}");
            }
            /*for kv_id in 0..n_kv {
                let key_name = gguf_get_key(ctx, kv_id);
                let name = CStr::from_ptr(key_name);
                let s = name.to_string_lossy();
                let ty = gguf_get_kv_type(ctx, kv_id);

                //gguf_get_val_str(ctx, k_id);
                //println!("{s}: {ty:?}");
            }*/

            Ok(Self {
                ctx,
                tensors,
                kv,
                ggml,
            })
        }
    }
    pub fn version(&self) -> u32 {
        unsafe { gguf_get_version(self.ctx) }
    }
    /*pub fn index(&self) {
        unsafe { gguf_get_n_tensors(ctx) }
    }*/
}
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Invalid type (got: {got:?}, expected: {expected:?})")]
    InvalidType { got: gguf_type, expected: gguf_type },
    #[error("Invalid tensor type (got: {got:?}, expected: {expected:?})")]
    InvalidTensorType { got: ggml_type, expected: ggml_type },
    #[error("Key not found")]
    DoesntExist,
    #[error("Tensor not contiguous")]
    NotContigouos,
    #[error("Quantise requires IMatrix")]
    RequireIMatrix,
}
pub fn load_model() -> anyhow::Result<()> {
    let model = Loader::open("./models/rfdetr-small-q4_K.gguf")?;
    println!("version: {}", model.version());
    //let image_size: u32 = model.metadata("rfdetr.image_size")?;
    //let tensor = model.get_tensor("backbone.cls_token").unwrap();
    //let structure_iter = model.tensors.keys();
    let structure = ModelStruct::parse(&model);
    println!("{structure:#?}");
    //println!("image size: {image_size}");
    Ok(())
}
