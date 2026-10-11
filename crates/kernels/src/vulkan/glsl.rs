//! GLSL to SPIR-V by shelling out to `glslc` (Vulkan SDK / shaderc), or to
//! `glslang` when only that is installed (the release zip needs no admin
//! rights). Includes are flattened here, so neither tool needs include-path
//! support, and ggml's `#include "types.glsl"` headers (which carry their own
//! `#ifndef` guards) resolve identically for both.
//!
//! This is a development/build-time tool, not something a shipped model needs:
//! compiled blobs are plain SPIR-V.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Error, Result};

#[derive(Clone, Debug)]
pub enum Tool {
    /// `glslc` (shaderc): `glslc -fshader-stage=compute --target-env=vulkan1.2`.
    Glslc(PathBuf),
    /// `glslang` / `glslangValidator`: `-S comp -V --target-env vulkan1.2`.
    Glslang(PathBuf),
}

#[derive(Clone, Debug)]
pub struct Compiler {
    pub tool: Tool,
}

fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

fn on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe(name))).find(|p| p.is_file())
}

impl Compiler {
    /// Looks for a compiler: `TL_GLSLC` (a path to either tool), then `glslc`
    /// in `$VULKAN_SDK/Bin`, on `PATH` and in `C:\VulkanSDK\*`, then glslang
    /// unpacked in `vendor/glslang` or on `PATH`.
    pub fn find() -> Option<Compiler> {
        let classify = |p: PathBuf| -> Compiler {
            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            Compiler { tool: if name.starts_with("glslc") { Tool::Glslc(p) } else { Tool::Glslang(p) } }
        };
        if let Some(p) = std::env::var_os("TL_GLSLC") {
            return Some(classify(PathBuf::from(p)));
        }
        if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
            for dir in ["Bin", "bin"] {
                let p = Path::new(&sdk).join(dir).join(exe("glslc"));
                if p.is_file() {
                    return Some(classify(p));
                }
            }
        }
        if let Some(p) = on_path("glslc") {
            return Some(classify(p));
        }
        if let Ok(rd) = std::fs::read_dir(r"C:\VulkanSDK") {
            let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            dirs.sort();
            for d in dirs.into_iter().rev() {
                let p = d.join("Bin").join(exe("glslc"));
                if p.is_file() {
                    return Some(classify(p));
                }
            }
        }
        let vendored = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/glslang/bin");
        for name in ["glslang", "glslangValidator"] {
            let p = vendored.join(exe(name));
            if p.is_file() {
                return Some(classify(p));
            }
        }
        on_path("glslang").or_else(|| on_path("glslangValidator")).map(classify)
    }

    pub fn describe(&self) -> String {
        match &self.tool {
            Tool::Glslc(p) | Tool::Glslang(p) => p.display().to_string(),
        }
    }

    /// Compiles one compute shader. `source` must already have its includes
    /// flattened ([`flatten`]); `defines` become `-DNAME=VALUE`.
    pub fn compile(&self, name: &str, source: &str, defines: &[(String, String)]) -> Result<Vec<u32>> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("tl-glsl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| Error::Compile { lib: name.into(), msg: e.to_string() })?;
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let src = dir.join(format!("{id}.comp"));
        let out = dir.join(format!("{id}.spv"));
        std::fs::write(&src, source).map_err(|e| Error::Compile { lib: name.into(), msg: e.to_string() })?;

        let mut cmd;
        match &self.tool {
            Tool::Glslc(p) => {
                cmd = Command::new(p);
                cmd.args(["-fshader-stage=compute", "--target-env=vulkan1.2", "-O"]);
            }
            Tool::Glslang(p) => {
                cmd = Command::new(p);
                cmd.args(["-S", "comp", "-V", "--target-env", "vulkan1.2"]);
            }
        }
        for (k, v) in defines {
            cmd.arg(format!("-D{k}={v}"));
        }
        cmd.arg(&src).arg("-o").arg(&out);
        let output = cmd.output().map_err(|e| Error::Compile { lib: name.into(), msg: format!("cannot run {}: {e}", self.describe()) })?;
        let result = if output.status.success() {
            std::fs::read(&out)
                .map_err(|e| Error::Compile { lib: name.into(), msg: e.to_string() })
                .and_then(|bytes| {
                    if bytes.len() % 4 != 0 {
                        return Err(Error::Compile { lib: name.into(), msg: "SPIR-V output is not a whole number of words".into() });
                    }
                    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
                })
        } else {
            let mut msg = String::from_utf8_lossy(&output.stdout).into_owned();
            msg.push_str(&String::from_utf8_lossy(&output.stderr));
            Err(Error::Compile { lib: name.into(), msg })
        };
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&out);
        result
    }
}

/// Reads `path` and textually replaces every `#include "x"` by the contents of
/// `x` (searched next to the including file, then in `dirs`), recursively.
/// Headers are expected to guard themselves.
pub fn flatten(path: &Path, dirs: &[PathBuf]) -> Result<String> {
    fn go(path: &Path, dirs: &[PathBuf], depth: usize, out: &mut String) -> Result<()> {
        if depth > 16 {
            return Err(Error::Invalid(format!("includes nested too deeply at {}", path.display())));
        }
        let text = std::fs::read_to_string(path).map_err(|e| Error::Invalid(format!("cannot read {}: {e}", path.display())))?;
        let here = path.parent().unwrap_or(Path::new("."));
        for line in text.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("#include") {
                let name = rest.trim().trim_matches(|c| c == '"' || c == '<' || c == '>');
                let found = std::iter::once(here.to_path_buf())
                    .chain(dirs.iter().cloned())
                    .map(|d| d.join(name))
                    .find(|p| p.is_file())
                    .ok_or_else(|| Error::Invalid(format!("{}: cannot find include `{name}`", path.display())))?;
                go(&found, dirs, depth + 1, out)?;
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(())
    }
    let mut out = String::new();
    go(path, dirs, 0, &mut out)?;
    Ok(out)
}
