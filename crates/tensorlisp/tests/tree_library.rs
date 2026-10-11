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
    let exprs = [
        "(k:binary 'add)",
        "(k:binary 'div)",
        "(k:unary 'gelu 'same)",
        "(k:unary 'copy 'reshape)",
        "(k:unary 'copy 'repeat)",
        "(k:unary 'copy 'same 'f32 'f16)",
        "(k:unary 'copy 'same 'f16 'f32)",
        "(k:unary 'copy 'same 'i32 'f32)",
        "(k:concat 1)",
        "(k:arange)",
        "(k:get-rows 'f32)",
        "(k:get-rows 'f16)",
        "(k:get-rows 'q8_0)",
        "(k:norm)",
        "(k:rms-norm #f)",
        "(k:rms-norm #t)",
        "(k:soft-max #f)",
        "(k:soft-max #t)",
        "(k:sum-rows)",
        "(k:matmul 'f32 64 64 32 4 4)",
        "(k:matmul 'f16 64 64 32 4 4)",
        "(k:matmul 'q8_0 64 64 32 4 4)",
        "(k:matvec 'f32 1)",
        "(k:matvec 'f16 3)",
        "(k:matvec 'q8_0 1)",
        "(k:matvec 'q8_0 8)",
        "(k:rope 'norm)",
        "(k:rope 'neox)",
        "(k:set-rows 'f32)",
        "(k:set-rows 'f16)",
        "(k:argmax)",
        "(k:quantize-q8)",
        "(k:matvec-q8 1 4)",
        "(k:matvec-q8 2 2)",
        "(k:repack-q8)",
        "(k:matvec-q8r 1 4)",
        "(k:flash-attn 'f16 'f16 #t #f 2048)",
        "(k:flash-attn 'f32 'f32 #f #t 512)",
        "(k:rms-norm #t #t)",
        "(k:rms-norm #f #t)",
        "(k:ew '((f32 full) (f32 row) (f32 scalar)) 'f32 '((t0 (+ x0 x1)) (t1 (* t0 x2)) (t2 (unary gelu t1))) 't2 0)",
        "(k:ew '((i32 full)) 'f16 '((t0 x0)) 't0 0)",
        "(k:ew '() 'f32 '((t0 (+ c0 (* (f32 idx) c1))) (t1 (unary step t0))) 't1 2)",
        "(k:ew-q8 '((f32 full) (f32 full)) 'f32 '((t0 (unary gelu x0)) (t1 (* t0 x1))) 't1 0)",
        "(k:rowchain 1536 '((f32 full) (f32 full) (f32 full)) '((norm 1 0) (ew ((t0 (+ x2 x))) t0) (norm #f 1)) '((1 f32) (2 q8)) 2)",
        "(k:rowchain 256 '((f32 full) (f32 row)) '((norm 1 0)) '((0 f32)) 1)",
        "(k:flash-attn-part 'f16 'f16 #t #f 128 256 256)",
        "(k:flash-attn-part 'f32 'f32 #f #t 64 64 64)",
        "(k:flash-attn-combine)",
        "(k:flash-attn-combine #t)",
        "(k:norm-rope 256 #t 'neox 'f32 #f)",
        "(k:norm-rope 256 #t 'neox 'f16 #t)",
        "(k:norm-rope 512 #f #f 'f16 #t)",
        "(k:norm-rope 64 #t 'norm 'f32 #f)",
        "(k:gateup-q8 2 2)",
        "(k:gateup-q8 1 1)",
        "(k:matvec-q8 1 1 3)",
        "(k:matvec-q8 2 2 2)",
    ];
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
            let out = Command::new(exe).args(["--target-env", "vulkan1.1"]).arg(&path).output().unwrap();
            let _ = std::fs::remove_file(&path);
            assert!(out.status.success(), "spirv-val rejected {expr}:\n{}", String::from_utf8_lossy(&out.stderr));
        }
        println!("{expr}: {} bytes, {} bindings, {} push bytes", b.len(), blob.sig.bindings.len(), blob.sig.push_bytes);
    }
}
