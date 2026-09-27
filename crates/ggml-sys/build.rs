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
    cuda: bool,
}

impl Backends {
    fn from_features(target: &Target) -> Self {
        let feature = |name: &str| std::env::var(format!("CARGO_FEATURE_{name}")).is_ok();
        let platform = feature("PLATFORM_BACKENDS");
        let metal = feature("METAL") || (platform && target.os == "macos");
        if metal && target.os != "macos" {
            panic!("the metal backend is only available on macOS");
        }
        let cuda = feature("CUDA");
        if cuda && target.os == "macos" {
            panic!("the cuda backend is not supported on macOS; use metal instead");
        }
        Backends { metal, cuda }
    }
}

/// A detected CUDA Toolkit install, used to compile and link `ggml-cuda`.
struct Cuda {
    include: PathBuf,
    lib_dir: PathBuf,
    /// SM architecture number, e.g. "86" for an RTX 3080. Read from
    /// `GGML_CUDA_ARCH` if set, else detected via `nvidia-smi`, else "86".
    arch: String,
}

impl Cuda {
    fn detect(target: &Target) -> Self {
        let root = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
            .into_iter()
            .find_map(|var| std::env::var(var).ok())
            .unwrap_or_else(|| {
                panic!(
                    "the cuda feature needs the CUDA Toolkit: set CUDA_PATH (Windows) or \
                     CUDA_HOME/CUDA_ROOT (Linux) to its install prefix"
                )
            });
        let root = PathBuf::from(root);
        let lib_dir = if target.os == "windows" { root.join("lib").join("x64") } else { root.join("lib64") };
        let arch = std::env::var("GGML_CUDA_ARCH").ok().unwrap_or_else(Self::detect_arch);
        Cuda { include: root.join("include"), lib_dir, arch }
    }

    /// Reads the first GPU's compute capability off `nvidia-smi`, e.g. "8.6" -> "86".
    fn detect_arch() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|s| s.lines().next().map(|l| l.trim().replace('.', "")))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "86".to_string())
    }
}

/// Sources for the `ggml-cuda` backend, matching
/// `vendor/ggml/src/ggml-cuda/CMakeLists.txt`'s file(GLOB ...) selection:
/// every top-level `*.cu`, the always-needed template-instance groups, and
/// (since `GGML_CUDA_FA_ALL_QUANTS` defaults off) just the 4 default
/// FlashAttention vector-kernel instances instead of all of them.
fn cuda_sources(src: &Path) -> Vec<PathBuf> {
    let cuda_dir = src.join("ggml-cuda");
    let cu_files = |dir: &Path, prefixes: &[&str]| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "cu"))
            .filter(|p| {
                prefixes.is_empty()
                    || prefixes.iter().any(|prefix| p.file_name().unwrap().to_str().unwrap().starts_with(prefix))
            })
            .collect()
    };

    let mut files = cu_files(&cuda_dir, &[]);
    let instances = cuda_dir.join("template-instances");
    files.extend(cu_files(&instances, &["fattn-tile", "fattn-mma", "mmq", "mmf"]));
    files.extend(
        ["fattn-vec-instance-f16-f16.cu", "fattn-vec-instance-q4_0-q4_0.cu", "fattn-vec-instance-q8_0-q8_0.cu", "fattn-vec-instance-bf16-bf16.cu"]
            .into_iter()
            .map(|name| instances.join(name)),
    );
    files
}

fn ggml_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ggml")
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
}

/// Bump this when changing the CUDA source-selection or compile flags in
/// `build_ggml` below (defines, -arch handling, etc.) — those aren't hashed
/// into the cache key below, so a stale cached lib from before the change
/// would otherwise get reused as-is.
const CUDA_CACHE_VERSION: u32 = 1;

fn repo_root() -> PathBuf {
    ggml_root().parent().unwrap().parent().unwrap().to_path_buf()
}

/// Persistent cache for the compiled ggml-cuda static lib, outside target/
/// (see .gitignore) so it survives `cargo clean` and rustc/toolchain
/// switches — both invalidate OUT_DIR and would otherwise force a full
/// ~141-file nvcc recompile even though nothing actually changed.
///
/// Recompiles automatically whenever anything it's built from changes: every
/// file cuda_sources() selects, every file directly under vendor/ggml/include
/// and vendor/ggml/src (so pulling in a new vendored ggml version, which
/// touches shared headers those .cu files include, invalidates it too), the
/// GPU arch, and the CUDA Toolkit's include path (so switching CUDA_PATH to
/// a different toolkit version invalidates it as well).
fn cuda_cache_dir() -> PathBuf {
    repo_root().join(".ggml-cuda-cache")
}

fn cuda_cache_key(prefix: &Path, sources: &[PathBuf], cuda: &Cuda) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    CUDA_CACHE_VERSION.hash(&mut hasher);
    cuda.arch.hash(&mut hasher);
    cuda.include.to_string_lossy().hash(&mut hasher);
    // cc-rs reads these cargo-provided vars itself and changes nvcc's flags
    // accordingly (-G / -O0 vs -O3 etc.) — without hashing them, a release
    // build would silently reuse (or clobber) a debug build's cache entry.
    std::env::var("OPT_LEVEL").unwrap_or_default().hash(&mut hasher);
    std::env::var("DEBUG").unwrap_or_default().hash(&mut hasher);

    let list_dir = |dir: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.is_file())
            .collect()
    };
    let mut inputs = sources.to_vec();
    inputs.extend(list_dir(&prefix.join("include")));
    inputs.extend(list_dir(&prefix.join("src")));
    inputs.sort();

    for f in inputs {
        f.to_string_lossy().hash(&mut hasher);
        std::fs::read(&f).unwrap_or_else(|e| panic!("reading {}: {e}", f.display())).hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn static_lib_name(target: &Target, name: &str) -> String {
    if target.os == "windows" { format!("{name}.lib") } else { format!("lib{name}.a") }
}

fn generate_bindings(include: &Path, backends: &Backends) {
    let mut headers = vec!["ggml", "gguf", "ggml-alloc", "ggml-backend", "ggml-cpu"];
    if backends.metal {
        headers.push("ggml-metal");
    }
    if backends.cuda {
        headers.push("ggml-cuda");
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
    if backends.cuda {
        base.define("GGML_USE_CUDA", None);
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

    if backends.cuda {
        let cuda = Cuda::detect(target);
        let sources = cuda_sources(&src);
        let lib_name = static_lib_name(target, "ggml_cuda_impl");
        let cached = cuda_cache_dir().join(format!("{}-{lib_name}", cuda_cache_key(&prefix, &sources, &cuda)));

        if cached.exists() {
            println!("cargo:warning=ggml-cuda: reusing cached build at {}", cached.display());
            std::fs::copy(&cached, out_dir().join(&lib_name)).expect("copying cached ggml-cuda lib");
            println!("cargo:rustc-link-search=native={}", out_dir().display());
            println!("cargo:rustc-link-lib=static=ggml_cuda_impl");
        } else {
            // sccache (or any RUSTC_WRAPPER cc-rs auto-detects as a launcher)
            // doesn't reliably forward the INCLUDE/LIB env nvcc's nested cl.exe
            // invocation needs on MSVC; rustc and the plain C/C++ units above
            // still benefit from it, this just keeps nvcc's own invocation
            // direct and working.
            let rustc_wrapper = std::env::var_os("RUSTC_WRAPPER");
            unsafe { std::env::remove_var("RUSTC_WRAPPER") };
            cc::Build::new()
                .cuda(true)
                // Passed directly to nvcc's own frontend (not just -Xcompiler'd
                // to the host compiler), needed for the C++17 fold expressions
                // and structured bindings ggml-cuda's shared headers use.
                .flag("-std=c++17")
                .includes([&include, &src, &cuda.include])
                .define("GGML_CUDA_PEER_MAX_BATCH_SIZE", "128")
                .flag("-use_fast_math")
                .flag("-extended-lambda")
                .flag(format!("-arch=sm_{}", cuda.arch))
                .warnings(false)
                .files(sources)
                .compile("ggml_cuda_impl");
            if let Some(wrapper) = rustc_wrapper {
                unsafe { std::env::set_var("RUSTC_WRAPPER", wrapper) };
            }
            std::fs::create_dir_all(cuda_cache_dir()).expect("creating .ggml-cuda-cache");
            std::fs::copy(out_dir().join(&lib_name), &cached).expect("caching compiled ggml-cuda lib");
        }

        println!("cargo:rustc-link-search=native={}", cuda.lib_dir.display());
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cuda");
        // Resolves the __fatbinwrap_* symbols nvcc's device-link step
        // (--device-c's relocatable device code) needs at final-link time.
        println!("cargo:rustc-link-lib=cudadevrt");
    }

    if target.os == "windows" {
        // ggml-cpu.c reads CPU info via the Windows Registry API.
        println!("cargo:rustc-link-lib=dylib=advapi32");
    }

    if target.os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        if backends.metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.os != "windows" {
        // MSVC's C++ runtime is part of the CRT link.exe pulls in
        // automatically; there's no separate "stdc++.lib" to name.
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
    println!("cargo:rustc-check-cfg=cfg(ggml_cuda)");
    if backends.cuda {
        println!("cargo:rustc-cfg=ggml_cuda");
    }

    generate_bindings(&ggml_root().join("include"), &backends);
    build_ggml(&target, &backends);
}
