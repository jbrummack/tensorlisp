//! The scheduler conformance suite (tests/common) over the Vulkan backend,
//! under the validation layer with synchronization validation when installed.
//! Kernels are WGSL translated to SPIR-V by naga.
//!
//! `cargo test -p tensorlisp-kernels --features vulkan,wgsl --test vulkan_sched`
//! (`TL_ALLOW_NO_GPU=1` skips when there is no Vulkan device).
#![cfg(all(feature = "vulkan", feature = "wgsl"))]

mod common;

use std::sync::Arc;

use common::Kernels;
use tensorlisp_kernels::sched::blob::wgsl;
use tensorlisp_kernels::sched::{Access, Engine, Signature};
use tensorlisp_kernels::vulkan::{validation_errors, Options, Vulkan};
use tensorlisp_kernels::Error;

/// An engine opened with the validation layer (when installed); the test
/// fails if the layer reported an error.
struct Guard(Engine<Vulkan>);

impl std::ops::Deref for Guard {
    type Target = Engine<Vulkan>;
    fn deref(&self) -> &Engine<Vulkan> {
        &self.0
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert_eq!(validation_errors(), 0, "the Vulkan validation layer reported errors");
        }
    }
}

fn engine() -> Option<Guard> {
    match Vulkan::open(Options { validate: true }) {
        Ok(v) => {
            println!("{} (validation: {})", tensorlisp_kernels::sched::Backend::name(&v), v.validated());
            Some(Guard(Engine::new(Arc::new(v))))
        }
        Err(Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Vulkan device, skipping");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

// Vulkan blobs get the residual byte offset of each binding as leading push
// constants (`OffsetMode::Prefix`), so the kernels declare them first.
const AXPB: &str = r#"
struct PC { oa: u32, ob: u32, oo: u32, n: u32, s: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn axpb(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < pc.n) {
        o[pc.oo / 4u + i] = a[pc.oa / 4u + i] * pc.s + b[pc.ob / 4u + i];
    }
}
"#;

const ACC: &str = r#"
struct PC { oacc: u32, ox: u32, n: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@compute @workgroup_size(64)
fn acc_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < pc.n) {
        acc[pc.oacc / 4u + i] = acc[pc.oacc / 4u + i] + x[pc.ox / 4u + i];
    }
}
"#;

const MATMUL: &str = r#"
struct PC { oa: u32, ob: u32, oc: u32, m: u32, n: u32, k: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if (row < pc.m && col < pc.n) {
        var sum = 0.0;
        for (var i = 0u; i < pc.k; i = i + 1u) {
            sum = sum + a[pc.oa / 4u + row * pc.k + i] * b[pc.ob / 4u + i * pc.n + col];
        }
        c[pc.oc / 4u + row * pc.n + col] = sum;
    }
}
"#;

fn kernels() -> Kernels {
    let blob = |src, entry, bindings, push, wg| Arc::new(wgsl::blob(src, entry, Signature::new(bindings, push, wg)).unwrap());
    Kernels {
        axpb: blob(AXPB, "axpb", vec![Access::Read, Access::Read, Access::Write], 8, [64, 1, 1]),
        acc: blob(ACC, "acc_add", vec![Access::ReadWrite, Access::Read], 4, [64, 1, 1]),
        matmul: blob(MATMUL, "matmul", vec![Access::Read, Access::Read, Access::Write], 12, [8, 8, 1]),
    }
}

conformance!(engine, kernels);
