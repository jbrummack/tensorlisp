//! Every generator of `(tl tree kernels)` compiled with representative arguments, checked by the
//! SDK's `spirv-val` when it is installed, and reflected into a blob (so its push block really
//! starts with one prefix word per binding). No GPU needed.
#![cfg(feature = "native-vulkan")]

use std::process::Command;

use chez::{Scheme, Value};
use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::OffsetMode;

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");

fn spirv_val_exe() -> Option<std::path::PathBuf> {
    std::env::var_os("VULKAN_SDK")
        .map(|d| std::path::PathBuf::from(d).join("Bin").join("spirv-val"))
        .into_iter()
        .chain(std::fs::read_dir("C:/VulkanSDK").into_iter().flatten().flatten().map(|e| e.path().join("Bin").join("spirv-val")))
        .chain(std::iter::once("spirv-val".into()))
        .find(|p| Command::new(p).arg("--version").output().is_ok())
}

#[test]
fn every_generator_compiles_and_validates() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree kernels)))").unwrap();
    let exprs: Vec<&str> = include_str!("tree_exprs.txt").lines().filter(|l| !l.trim().is_empty()).collect();
    let validator = spirv_val_exe();
    if validator.is_none() {
        eprintln!("spirv-val not found: only compiling and reflecting");
    }
    for expr in exprs {
        let Value::Bytevector(b) = scheme.eval(&format!("(eval '(compile-kernel {expr}) $env)")).unwrap_or_else(|e| panic!("{expr}: {e}")) else {
            panic!("{expr}: not a bytevector");
        };
        let words: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let blob = spirv::blob(words, OffsetMode::Prefix, vec![]).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert!(blob.sig.workgroup[0] > 0, "{expr}: no workgroup size");
        if let Some(exe) = &validator {
            let path = std::env::temp_dir().join(format!("tree-lib-{}.spv", std::process::id()));
            std::fs::write(&path, &b).unwrap();
            let out = Command::new(exe).args(["--target-env", "vulkan1.2"]).arg(&path).output().unwrap();
            let _ = std::fs::remove_file(&path);
            assert!(out.status.success(), "spirv-val rejected {expr}:\n{}", String::from_utf8_lossy(&out.stderr));
        }
        println!("{expr}: {} bytes, {} bindings, {} push bytes", b.len(), blob.sig.bindings.len(), blob.sig.push_bytes);
    }
}
