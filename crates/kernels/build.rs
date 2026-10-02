//! Prepares what the Metal backend takes from vendored ggml (`vendor/ggml`),
//! which stays the single source of truth for kernels and their argument structs:
//! * the flattened `ggml-metal.metal` (what ggml-sys embeds for ggml itself),
//! * `ggml_metal_kargs_*` structs and the `FC_*` / `N_*` / `OP_*` constants, via bindgen.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let ggml_src = manifest.join("../../vendor/ggml/src");
    let metal_dir = ggml_src.join("ggml-metal");
    println!("cargo:rerun-if-changed={}", metal_dir.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let read = |p: &Path| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    let common = read(&ggml_src.join("ggml-common.h"));
    let impl_h = read(&metal_dir.join("ggml-metal-impl.h"));
    let merged: String = read(&metal_dir.join("ggml-metal.metal"))
        .lines()
        .map(|line| {
            if line.contains("__embed_ggml-common.h__") {
                common.clone()
            } else if line.trim() == "#include \"ggml-metal-impl.h\"" {
                impl_h.clone()
            } else {
                format!("{line}\n")
            }
        })
        .collect();
    std::fs::write(out.join("ggml-metal-merged.metal"), merged).unwrap();

    // The header relies on <stdint.h> types the Metal compiler provides itself.
    let wrapper = out.join("kargs_wrapper.h");
    std::fs::write(&wrapper, format!("#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include \"{}\"\n", metal_dir.join("ggml-metal-impl.h").display())).unwrap();
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_str().unwrap())
        .allowlist_type("ggml_metal_kargs_.*")
        .allowlist_var("(FC|OP|N|SZ)_[A-Z0-9_]+")
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .generate()
        .expect("bindgen ggml-metal-impl.h");
    bindings.write_to_file(out.join("ggml_metal_kargs.rs")).unwrap();
}
