use std::path::{Path, PathBuf};

/// Directory holding scheme.h, libkernel.a and the boot files, e.g.
/// vendor/chez/lib/csv10.5.0/tarm64osx. Override with CHEZ_LIB_DIR.
fn chez_lib_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CHEZ_LIB_DIR") {
        return PathBuf::from(dir);
    }
    let prefix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/chez/lib");
    let mut found = std::fs::read_dir(&prefix)
        .unwrap_or_else(|_| {
            panic!(
                "{} not found: run scripts/build-chez.sh or set CHEZ_LIB_DIR",
                prefix.display()
            )
        })
        .flatten()
        .filter(|version| version.file_name().to_string_lossy().starts_with("csv"))
        .flat_map(|version| std::fs::read_dir(version.path()).into_iter().flatten().flatten())
        .map(|machine| machine.path())
        .filter(|machine| machine.join("scheme.h").exists());
    let dir = found.next().expect("no Chez machine directory with scheme.h under vendor/chez/lib");
    assert!(
        found.next().is_none(),
        "several Chez installs under vendor/chez/lib, set CHEZ_LIB_DIR to pick one"
    );
    dir
}

fn generate_bindings(lib_dir: &Path, out_dir: &Path) {
    bindgen::Builder::default()
        .header("csrc/shim.h")
        .clang_arg(format!("-I{}", lib_dir.display()))
        .allowlist_function("S.*")
        .allowlist_function("chez_.*")
        .allowlist_type("ptr|iptr|uptr|octet|string_char")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate Chez bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write Chez bindings");
}

fn main() {
    let lib_dir = chez_lib_dir().canonicalize().unwrap();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=CHEZ_LIB_DIR");
    println!("cargo:rerun-if-changed={}", lib_dir.display());
    println!("cargo:rerun-if-changed=csrc");

    generate_bindings(&lib_dir, &out_dir);

    cc::Build::new()
        .file("csrc/shim.c")
        .include(&lib_dir)
        .compile("chez_shim");

    // Boot files are embedded into the binary with include_bytes!.
    println!("cargo:rustc-env=CHEZ_BOOT_DIR={}", lib_dir.display());

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=kernel");
    println!("cargo:rustc-link-lib=static=lz4");
    println!("cargo:rustc-link-lib=static=z");
    // Matches LIBS in Chez's generated Mf-config.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=iconv");
    println!("cargo:rustc-link-lib=dylib=ncurses");
    println!("cargo:rustc-link-lib=dylib=m");
}
