//! ggml's precompiled Vulkan shader library: the `.spv` files that
//! `scripts/build-ggml-vulkan-shaders.sh` writes, one per variant named as in
//! upstream's `vulkan-shaders-gen` (`add_f32_f32_f32`, `mul_mat_vec_q4_0_f32_f32`,
//! `flash_attn_f32_f16_cm2`, ...). Loading a variant reflects its signature from
//! the SPIR-V; ggml's shaders keep their own binding offsets (`OffsetMode::Raw`).
//!
//! Which variant an op needs, its specialization constants, push constants and
//! grid are ggml-vulkan.cpp's job; this module only turns a name into a blob.

use std::path::{Path, PathBuf};

use crate::sched::blob::{KernelBlob, KernelKey, KernelSource, OffsetMode, Target};
use crate::sched::spirv;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct SpvLibrary {
    dir: PathBuf,
}

impl SpvLibrary {
    /// Opens a directory of `.spv` files.
    pub fn open(dir: impl Into<PathBuf>) -> Option<SpvLibrary> {
        let dir = dir.into();
        dir.is_dir().then_some(SpvLibrary { dir })
    }

    /// `$TL_GGML_SPV`, else `target/ggml-vulkan/spv` of this workspace (where
    /// the build script puts it).
    pub fn default_dir() -> PathBuf {
        std::env::var_os("TL_GGML_SPV")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ggml-vulkan/spv"))
    }

    pub fn open_default() -> Option<SpvLibrary> {
        Self::open(Self::default_dir())
    }

    /// All variant names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "spv"))
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .collect();
        names.sort();
        names
    }

    pub fn contains(&self, name: &str) -> bool {
        self.dir.join(format!("{name}.spv")).is_file()
    }

    /// The SPIR-V words of variant `name`.
    pub fn words(&self, name: &str) -> Result<Vec<u32>> {
        let path = self.dir.join(format!("{name}.spv"));
        let bytes = std::fs::read(&path).map_err(|e| Error::Invalid(format!("cannot read {}: {e}", path.display())))?;
        if bytes.len() % 4 != 0 {
            return Err(Error::Invalid(format!("{} is not a whole number of SPIR-V words", path.display())));
        }
        Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Variant `name` as a blob with specialization constants `spec`.
    pub fn load(&self, name: &str, spec: &[u32]) -> Result<KernelBlob> {
        let mut blob = spirv::blob(self.words(name)?, OffsetMode::Raw, spec.to_vec())?;
        blob.label = name.to_string();
        Ok(blob)
    }
}

impl KernelSource for SpvLibrary {
    /// `key.name` is the variant, `key.params` its specialization constants.
    fn blob(&self, target: Target, key: &KernelKey) -> Result<Option<KernelBlob>> {
        if target != Target::Vulkan || !self.contains(&key.name) {
            return Ok(None);
        }
        let spec: Vec<u32> = key.params.iter().map(|&p| p as u32).collect();
        self.load(&key.name, &spec).map(Some)
    }
}
