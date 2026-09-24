//! Builds vendor/ggml with plain C/C++ compilers (no CMake) and generates bindings.
//!
//! The CPU backend is always built, since ggml_backend_sched needs it as the
//! fallback. Other backends are Cargo features; the default `platform-backends`
//! feature enables the ones that exist on the target platform (Metal on macOS).
use std::path::{Path, PathBuf};

struct Target {
    os: String,
    arch: String,
}

impl Target {
    fn from_env() -> Self {
        Target {
            os: std::env::var("CARGO_CFG_TARGET_OS").unwrap(),
            arch: std::env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        }
    }
}

struct Backends {
    metal: bool,
}

impl Backends {
    fn from_features(target: &Target) -> Self {
        let feature = |name: &str| std::env::var(format!("CARGO_FEATURE_{name}")).is_ok();
        let platform = feature("PLATFORM_BACKENDS");
        let metal = feature("METAL") || (platform && target.os == "macos");
        if metal && target.os != "macos" {
            panic!("the metal backend is only available on macOS");
        }
        Backends { metal }
    }
}

fn ggml_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ggml")
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
}

fn generate_bindings(include: &Path, backends: &Backends) {
    let mut headers = vec!["ggml", "gguf", "ggml-alloc", "ggml-backend", "ggml-cpu"];
    if backends.metal {
        headers.push("ggml-metal");
    }
    let headers = headers
        .iter()
        .map(|name| include.join(format!("{name}.h")).to_str().unwrap().to_string())
        .chain(["csrc/guard.h".to_string()]);

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include.display()))
        .headers(headers)
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let wrapped_code = format!(
        "#[allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]#[ggml_codegen::parse_ggml]\npub mod ffi {{\n{bindings}\n}}",
    );
    std::fs::write(out_dir().join("bindings.rs"), wrapped_code).expect("Couldn't write bindings!");
}

/// CPU sources that depend on the target architecture.
fn cpu_arch_sources(target: &Target, ggml_cpu: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let arch_dir = match target.arch.as_str() {
        "aarch64" | "arm" => "arm",
        "x86_64" | "x86" => "x86",
        "riscv64" => "riscv",
        "powerpc64" => "powerpc",
        "loongarch64" => "loongarch",
        "s390x" => "s390",
        "wasm32" => "wasm",
        other => panic!("unsupported target architecture for ggml-cpu: {other}"),
    };
    let dir = ggml_cpu.join("arch").join(arch_dir);
    let existing = |names: &[&str]| -> Vec<PathBuf> {
        names.iter().map(|n| dir.join(n)).filter(|p| p.exists()).collect()
    };
    (existing(&["quants.c"]), existing(&["repack.cpp", "cpu-feats.cpp"]))
}

/// Merges the headers the Metal shader includes into one source (what CMake
/// does with sed) and emits assembly that embeds it between the
/// ggml_metallib_start/end symbols that ggml-metal-device.m reads.
fn embed_metal_library(src: &Path) -> PathBuf {
    let metal_dir = src.join("ggml-metal");
    let read = |p: &Path| std::fs::read_to_string(p).unwrap();
    let common = read(&src.join("ggml-common.h"));
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
    let merged_path = out_dir().join("ggml-metal-embed.metal");
    std::fs::write(&merged_path, merged).unwrap();

    let asm_path = out_dir().join("ggml-metal-embed.s");
    std::fs::write(
        &asm_path,
        format!(
            ".section __DATA,__ggml_metallib\n\
             .globl _ggml_metallib_start\n\
             _ggml_metallib_start:\n\
             .incbin \"{}\"\n\
             .globl _ggml_metallib_end\n\
             _ggml_metallib_end:\n",
            merged_path.display()
        ),
    )
    .unwrap();
    asm_path
}

fn build_ggml(target: &Target, backends: &Backends) {
    let prefix = ggml_root();
    let include = prefix.join("include");
    let src = prefix.join("src");
    let ggml_cpu = src.join("ggml-cpu");

    let mut base = cc::Build::new();
    base.includes([&include, &src, &ggml_cpu])
        .define("GGML_VERSION", "\"unknown\"")
        .define("GGML_COMMIT", "\"unknown\"")
        .define("GGML_USE_CPU", None)
        .define("GGML_USE_LLAMAFILE", None)
        .flag_if_supported("-O3")
        .warnings(false);
    if target.os == "macos" {
        base.define("GGML_USE_ACCELERATE", None);
    }
    if backends.metal {
        base.define("GGML_USE_METAL", None)
            .define("GGML_METAL_EMBED_LIBRARY", None);
    }

    let (arch_c, arch_cpp) = cpu_arch_sources(target, &ggml_cpu);

    // Before ggml_core, so linkers that resolve in order find its ggml calls.
    base.clone().file("csrc/guard.c").cpp(false).compile("tl_guard");

    base.clone()
        .files([
            src.join("ggml.c"),
            src.join("ggml-alloc.c"),
            src.join("ggml-quants.c"),
            ggml_cpu.join("quants.c"),
            ggml_cpu.join("ggml-cpu.c"),
        ])
        .files(arch_c)
        .cpp(false)
        .compile("ggml_core");

    base.clone()
        .std("c++17")
        .files([
            ggml_cpu.join("repack.cpp"),
            ggml_cpu.join("vec.cpp"),
            ggml_cpu.join("hbm.cpp"),
            ggml_cpu.join("traits.cpp"),
            ggml_cpu.join("binary-ops.cpp"),
            ggml_cpu.join("unary-ops.cpp"),
            ggml_cpu.join("ops.cpp"),
            ggml_cpu.join("ggml-cpu.cpp"),
            ggml_cpu.join("llamafile/sgemm.cpp"),
            src.join("ggml.cpp"),
            src.join("ggml-backend.cpp"),
            src.join("ggml-backend-meta.cpp"),
            src.join("ggml-backend-dl.cpp"),
            src.join("ggml-backend-reg.cpp"),
            src.join("ggml-threading.cpp"),
            src.join("gguf.cpp"),
        ])
        .files(arch_cpp)
        .cpp(true)
        .compile("ggml_backend");

    if backends.metal {
        let metal = src.join("ggml-metal");
        base.clone()
            .include(&metal)
            .std("c++17")
            .files([
                metal.join("ggml-metal.cpp"),
                metal.join("ggml-metal-common.cpp"),
                metal.join("ggml-metal-device.cpp"),
                metal.join("ggml-metal-ops.cpp"),
            ])
            .cpp(true)
            .compile("ggml_metal_cpp");
        base.clone()
            .include(&metal)
            .files([metal.join("ggml-metal-device.m"), metal.join("ggml-metal-context.m")])
            .file(embed_metal_library(&src))
            .cpp(false)
            .compile("ggml_metal_objc");
    }

    if target.os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        if backends.metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn main() {
    let target = Target::from_env();
    let backends = Backends::from_features(&target);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed={}", ggml_root().display());
    println!("cargo:rustc-check-cfg=cfg(ggml_metal)");
    if backends.metal {
        println!("cargo:rustc-cfg=ggml_metal");
    }

    generate_bindings(&ggml_root().join("include"), &backends);
    build_ggml(&target, &backends);
}
