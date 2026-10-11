//! Tree kernels in the native Vulkan executor (docs/design/tree.md): nodes that have a Tree
//! generator are lowered to kernels compiled by the Scheme compiler instead of launches of
//! ggml's precompiled shaders; everything else keeps using ggml's.
//!
//! The kernels crate has no Scheme, so the compiler is a callback: it takes a kernel
//! expression (e.g. `(k:binary 'add)`, see `stdlib/tree_kernels.ss`) and returns SPIR-V
//! bytes. Blobs are cached by expression.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::sched::spirv;
use crate::sched::{KernelBlob, OffsetMode};
use crate::{Error, Result};

type Compile = dyn Fn(&str) -> Result<Vec<u8>> + Send + Sync;

pub struct TreeKernels {
    compile: Box<Compile>,
    cache: Mutex<HashMap<String, Arc<KernelBlob>>>,
    /// Fail on nodes without a Tree kernel instead of falling back to ggml's shaders.
    pub only: bool,
}

impl TreeKernels {
    pub fn new(only: bool, compile: impl Fn(&str) -> Result<Vec<u8>> + Send + Sync + 'static) -> TreeKernels {
        TreeKernels { compile: Box::new(compile), cache: Mutex::new(HashMap::new()), only }
    }

    /// The blob for kernel expression `expr`. Tree kernels are `OffsetMode::Prefix`: the backend
    /// passes each binding's residual byte offset first, the kernel adds it to its indices.
    pub fn blob(&self, expr: &str) -> Result<Arc<KernelBlob>> {
        if let Some(b) = self.cache.lock().unwrap().get(expr) {
            return Ok(b.clone());
        }
        let bytes = (self.compile)(expr)?;
        if bytes.len() % 4 != 0 {
            return Err(Error::Invalid(format!("tree kernel {expr} compiled to a partial word")));
        }
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut blob = spirv::blob(words, OffsetMode::Prefix, vec![])?;
        blob.label = format!("tree:{expr}");
        let blob = Arc::new(blob);
        self.cache.lock().unwrap().insert(expr.to_string(), blob.clone());
        Ok(blob)
    }
}

/// What lowering did with each node: how many became Tree kernels, which ops fell back.
#[derive(Default)]
pub struct TreeStats {
    pub tree_nodes: Cell<usize>,
    pub fallbacks: RefCell<BTreeMap<String, usize>>,
}
