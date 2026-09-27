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

/// `Path::canonicalize` emits a `\\?\`-prefixed verbatim path on Windows;
/// cl.exe doesn't reliably resolve that prefix in `-I` (fails with "cannot
/// open include file" even though the plain path is correct), so strip it.
fn canonicalize(path: &Path) -> PathBuf {
    let canon = path.canonicalize().unwrap();
    match canon.to_str() {
        Some(s) if cfg!(windows) => PathBuf::from(s.trim_start_matches(r"\\?\")),
        _ => canon,
    }
}

fn main() {
    let lib_dir = canonicalize(&chez_lib_dir());
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
    // Matches LIBS in Chez's generated Mf-config (Unix) / the linker
    // invocation in build.bat's nmake output (Windows: rpcrt4, ole32,
    // advapi32, user32 — no ncurses/libm equivalent needed there).
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=iconv");
    #[cfg(unix)]
    {
        println!("cargo:rustc-link-lib=dylib=ncurses");
        println!("cargo:rustc-link-lib=dylib=m");
    }
    #[cfg(windows)]
    {
        println!("cargo:rustc-link-lib=dylib=rpcrt4");
        println!("cargo:rustc-link-lib=dylib=ole32");
        println!("cargo:rustc-link-lib=dylib=advapi32");
        println!("cargo:rustc-link-lib=dylib=user32");
    }
}
