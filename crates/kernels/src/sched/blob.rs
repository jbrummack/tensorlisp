//! Kernel blobs: the only thing the scheduler knows about a kernel.
//!
//! A blob is compiled code plus a [`Signature`] saying how each binding is
//! accessed. Built-in libraries, ggml's kernels, fused kernels and kernels the
//! Scheme compiler generates all arrive as blobs, so the hazard analysis and
//! the backends never look at the op behind one.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Vulkan,
    Metal,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Code {
    Spirv(Vec<u32>),
    Msl(String),
    Cuda(String),
    Ptx(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub fn reads(self) -> bool {
        matches!(self, Access::Read | Access::ReadWrite)
    }

    pub fn writes(self) -> bool {
        matches!(self, Access::Write | Access::ReadWrite)
    }
}

/// Who deals with a binding whose byte offset is not a multiple of the
/// device's storage-buffer offset alignment (tensor views are at arbitrary
/// offsets, descriptor offsets are not).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OffsetMode {
    /// The backend rounds the offset down and prepends the residual byte
    /// offset of every binding (`u32` each) to the push constants.
    Prefix,
    /// The kernel handles it (ggml's shaders do, through `misalign_offsets`):
    /// the backend rounds down and adds nothing, so the lowering computes the
    /// residual itself as `offset % limits.storage_offset_align`.
    Raw,
}

/// How a kernel is called: one entry per bound buffer in binding order, the
/// bytes of its parameter block and its workgroup shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    pub bindings: Vec<Access>,
    /// Bytes of kernel parameters (push constants / argument struct), excluding
    /// any binding-offset words a backend prepends.
    pub push_bytes: u32,
    pub workgroup: [u32; 3],
    pub offsets: OffsetMode,
}

impl Signature {
    /// A signature using the backend-prefixed offset convention.
    pub fn new(bindings: Vec<Access>, push_bytes: u32, workgroup: [u32; 3]) -> Self {
        Signature { bindings, push_bytes, workgroup, offsets: OffsetMode::Prefix }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelBlob {
    pub target: Target,
    pub code: Code,
    pub entry: String,
    /// A name for profiles and traces (the shader variant); not part of the kernel.
    pub label: String,
    pub sig: Signature,
    /// Specialization constants, `constant_id = index`.
    pub spec: Vec<u32>,
}

impl KernelBlob {
    /// Content hash, the key of pipeline caches: two blobs with the same code,
    /// entry, signature and constants share a pipeline.
    pub fn id(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// Which kernel a lowering wants: an op pattern with its dtypes, shape class
/// and tuning parameters. Opaque to the scheduler.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub name: String,
    pub params: Vec<i64>,
}

impl KernelKey {
    pub fn new(name: &str, params: &[i64]) -> Self {
        KernelKey { name: name.to_string(), params: params.to_vec() }
    }
}

/// Produces blobs for keys it knows. Implemented by built-in tables, by the
/// ggml kernel library and (later) by the Scheme kernel compiler.
pub trait KernelSource: Send + Sync {
    /// `Ok(None)` when this source has no kernel for `key` on `target`.
    fn blob(&self, target: Target, key: &KernelKey) -> Result<Option<KernelBlob>>;
}

/// Ordered sources; the first one that has a kernel for a key wins, so a
/// generated kernel registered first overrides a built-in one.
#[derive(Default)]
pub struct Registry {
    sources: Vec<Box<dyn KernelSource>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, source: impl KernelSource + 'static) {
        self.sources.push(Box::new(source));
    }

    /// Like [`Registry::register`] but consulted before the existing sources.
    pub fn register_first(&mut self, source: impl KernelSource + 'static) {
        self.sources.insert(0, Box::new(source));
    }

    pub fn resolve(&self, target: Target, key: &KernelKey) -> Result<KernelBlob> {
        for s in &self.sources {
            if let Some(blob) = s.blob(target, key)? {
                return Ok(blob);
            }
        }
        Err(Error::Invalid(format!("no kernel for {key:?} on {target:?}")))
    }
}

#[cfg(feature = "wgsl")]
pub mod wgsl {
    //! WGSL to SPIR-V through naga: a way to write a compute kernel without
    //! `glslc`. Push constants are written `var<immediate>` (naga's name).

    use super::*;

    /// Translates `source` and returns the SPIR-V words of entry point `entry`.
    pub fn spirv(source: &str, entry: &str) -> Result<Vec<u32>> {
        use naga::back::spv;
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|e| Error::Compile { lib: entry.to_string(), msg: e.emit_to_string(source) })?;
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .map_err(|e| Error::Compile { lib: entry.to_string(), msg: format!("{e:?}") })?;
        let options = spv::Options {
            lang_version: (1, 3),
            flags: spv::WriterFlags::empty(),
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Unchecked,
                buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                image_load: naga::proc::BoundsCheckPolicy::Unchecked,
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
            ..Default::default()
        };
        let pipeline = spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: entry.to_string() };
        spv::write_vec(&module, &info, &options, Some(&pipeline))
            .map_err(|e| Error::Compile { lib: entry.to_string(), msg: format!("{e:?}") })
    }

    /// A Vulkan blob for a WGSL entry point.
    pub fn blob(source: &str, entry: &str, sig: Signature) -> Result<KernelBlob> {
        Ok(KernelBlob { target: Target::Vulkan, code: Code::Spirv(spirv(source, entry)?), entry: entry.to_string(), label: String::new(), sig, spec: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(entry: &str, bindings: Vec<Access>) -> KernelBlob {
        KernelBlob {
            target: Target::Vulkan,
            code: Code::Spirv(vec![1, 2, 3]),
            entry: entry.into(),
            label: String::new(),
            sig: Signature::new(bindings, 4, [64, 1, 1]),
            spec: vec![],
        }
    }

    #[test]
    fn identical_blobs_share_an_id() {
        let a = blob("main", vec![Access::Read, Access::Write]);
        assert_eq!(a.id(), a.clone().id());
        assert_ne!(a.id(), blob("other", vec![Access::Read, Access::Write]).id());
        assert_ne!(a.id(), blob("main", vec![Access::Read, Access::ReadWrite]).id());
    }

    struct Table(&'static str, KernelBlob);
    impl KernelSource for Table {
        fn blob(&self, _: Target, key: &KernelKey) -> Result<Option<KernelBlob>> {
            Ok((key.name == self.0).then(|| self.1.clone()))
        }
    }

    #[test]
    fn registry_prefers_the_first_source() {
        let mut r = Registry::new();
        r.register(Table("add", blob("builtin", vec![Access::Read])));
        r.register_first(Table("add", blob("generated", vec![Access::Read])));
        let key = KernelKey::new("add", &[]);
        assert_eq!(r.resolve(Target::Vulkan, &key).unwrap().entry, "generated");
        assert!(r.resolve(Target::Vulkan, &KernelKey::new("mul", &[])).is_err());
    }
}
