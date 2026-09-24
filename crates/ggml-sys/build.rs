use std::path::{Path, PathBuf};
fn generate_bindings2(include: impl AsRef<Path>, files: &[impl AsRef<str>]) {
    let headers = files.iter().flat_map(|fname| {
        let fname = fname.as_ref();
        let hname = include.as_ref().join(format!("{fname}.h"));
        let _canonical = std::fs::canonicalize(&hname).expect("header not found");
        hname.to_str().map(String::from)
    });

    let bindings = bindgen::Builder::default()
        //.header(hname_str)
        .headers(headers)
        .rustified_enum(".*")
        //.allowlist_file(&allow_pattern)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs");
    let code = bindings.to_string();

    let wrapped_code = format!(
        "#[allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]#[ggml_codegen::parse_ggml]\npub mod ffi {{\n{code}\n}}",
    );

    std::fs::write(out_path, wrapped_code).expect("Couldn't write bindings!");
}
fn ggml_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ggml")
}
fn main() {
    let include = ggml_root().join("include");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", ggml_root().display());
    generate_bindings2(&include, &["ggml", "gguf", "ggml-alloc", "ggml-cpu"]);

    build_ggml();
}
fn build_ggml() {
    let prefix = ggml_root();
    let include = prefix.join("include");
    let src = prefix.join("src");
    let ggml_cpu = src.join("ggml-cpu");
    let _ggml_metal = src.join("ggml-metal");
    let common_includes = [&include, &src, &ggml_cpu];

    // Create a base builder to share common logic
    let mut base_config = cc::Build::new();
    base_config
        .includes(common_includes)
        .define("GGML_VERSION", "\"unknown\"")
        .define("GGML_COMMIT", "\"unknown\"")
        .define("GGML_USE_CPU", None)
        //.define("GGML_USE_METAL", None)
        .flag_if_supported("-O3");

    // --- PLATFORM SPECIFIC DEFINES FOR ARM64 MAC ---
    #[cfg(target_os = "macos")]
    {
        // This fixes _ggml_critical_section_start/end (uses GCD)
        base_config.define("GGML_USE_COMMON", None);
        // This ensures the backend implementations are actually compiled
        base_config.define("GGML_USE_ACCELERATE", None);
    }
    let arch_arm = ggml_cpu.join("arch/arm");
    // 1. Compile Core (C)
    base_config
        .clone()
        .files([
            //COMMON
            src.join("ggml.c"),
            src.join("ggml-alloc.c"),
            src.join("ggml-quants.c"),
            //CPU COMMON
            ggml_cpu.join("quants.c"),
            ggml_cpu.join("ggml-cpu.c"),
            //CPU ARM
            arch_arm.join("quants.c"),
        ])
        .cpp(false)
        .compile("ggml_core");
    /* */
    // 2. Compile Backend (C++)
    base_config
        .clone()
        .std("c++17") // Ensure C++17 for std::filesystem
        .flag_if_supported("-std=c++17")
        .files([
            //GPU METAL
            /*"vendor/ggml/src/ggml-metal/ggml-metal.cpp",
            "ggml-metal/ggml-metal-common.cpp",
            "ggml-metal/ggml-metal-device.cpp",
            "ggml-metal/ggml-metal-ops.cpp",*/
            //CPU ARM
            arch_arm.join("repack.cpp"),
            arch_arm.join("cpu-feats.cpp"),
            //CPU COMMON
            ggml_cpu.join("repack.cpp"),
            ggml_cpu.join("vec.cpp"),
            ggml_cpu.join("hbm.cpp"),
            ggml_cpu.join("traits.cpp"),
            ggml_cpu.join("binary-ops.cpp"),
            ggml_cpu.join("unary-ops.cpp"),
            ggml_cpu.join("ops.cpp"),
            ggml_cpu.join("ggml-cpu.cpp"),
            //COMMON
            src.join("ggml-backend.cpp"),
            src.join("ggml-backend-meta.cpp"),
            src.join("ggml-backend-dl.cpp"),
            src.join("ggml-backend-reg.cpp"),
            src.join("ggml-threading.cpp"),
            //GGUF
            src.join("gguf.cpp"),
        ])
        .cpp(true)
        .compile("ggml_backend");
    // 1. Link the backends/high-level libs first
    println!("cargo:rustc-link-lib=static=ggml_backend");

    // 2. Link the core lib second (backend depends on core)
    println!("cargo:rustc-link-lib=static=ggml_core");
    // 3. Frameworks
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
