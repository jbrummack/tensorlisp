//! Every generator of `(tl tree kernels)` compiled to MSL (`compile-kernel-msl`) and then to a Metal
//! pipeline by the system compiler, with the arguments of `tree_exprs.txt` (shared with
//! `tree_library.rs`). Generators that need cooperative matrices have no Metal form yet and must be
//! rejected by name, not miscompiled.
#![cfg(all(target_os = "macos", feature = "native-metal-tree"))]

use chez::{Scheme, Value};
use tensorlisp_kernels::metal::backend::Metal;
use tensorlisp_kernels::sched::{Access, Backend, Code, KernelBlob, OffsetMode, Signature, Target};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");
const OPT_LIB: &str = include_str!("../src/scheme/stdlib/tree_opt.ss");
const MSL_LIB: &str = include_str!("../src/scheme/stdlib/tree_msl.ss");

/// Kernels built on cooperative matrices (the Metal target does not have them yet).
fn needs_matrix_units(expr: &str) -> bool {
    expr.starts_with("(k:matmul-q8mma")
}

pub fn blob_from_reply(label: &str, reply: Value) -> KernelBlob {
    let Value::List(items) = reply else { panic!("{label}: unexpected reply {reply:?}") };
    let [Value::String(text), Value::String(name), accesses, Value::Int(push), wg] = &items[..] else {
        panic!("{label}: unexpected reply {items:?}")
    };
    let bindings = match accesses {
        Value::List(a) => a
            .iter()
            .map(|v| match v {
                Value::Symbol(s) if s == "ro" => Access::Read,
                Value::Symbol(s) if s == "wo" => Access::Write,
                _ => Access::ReadWrite,
            })
            .collect(),
        _ => vec![],
    };
    let workgroup = [0, 1, 2].map(|i| match wg {
        Value::List(w) => match w.get(i) {
            Some(Value::Int(n)) => *n as u32,
            _ => 1,
        },
        _ => 1,
    });
    KernelBlob {
        target: Target::Metal,
        code: Code::Msl(text.clone()),
        entry: name.clone(),
        label: label.to_string(),
        sig: Signature { bindings, push_bytes: *push as u32, workgroup, offsets: OffsetMode::Prefix },
        spec: vec![],
    }
}

#[test]
fn every_generator_compiles_to_a_metal_pipeline() {
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB, MSL_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme.eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels) '(tl tree msl)))").unwrap();
    let metal = Metal::system_default().unwrap();
    let exprs: Vec<&str> = include_str!("tree_exprs.txt").lines().filter(|l| !l.trim().is_empty()).collect();
    let mut failed = Vec::new();
    for expr in exprs {
        for passes in ["", "(fold licm cse dce)"] {
            let src = if passes.is_empty() {
                format!("(eval '(compile-kernel-msl {expr}) $env)")
            } else {
                format!("(eval '(compile-kernel-msl (optimize {expr} '{passes})) $env)")
            };
            match scheme.eval(&src) {
                Err(e) => {
                    if !needs_matrix_units(expr) || !e.to_string().contains("not available on the Metal") {
                        failed.push(format!("{expr}: {e}"));
                    }
                }
                Ok(reply) => {
                    let blob = blob_from_reply(expr, reply);
                    if let Err(e) = metal.pipeline(&blob) {
                        failed.push(format!("{expr} {passes}: {e}\n{}", blob_text(&blob)));
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "{} generators failed:\n{}", failed.len(), failed.join("\n---\n"));
}

fn blob_text(blob: &KernelBlob) -> String {
    match &blob.code {
        Code::Msl(t) => t.clone(),
        _ => String::new(),
    }
}
