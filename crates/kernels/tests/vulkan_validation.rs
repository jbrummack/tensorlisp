//! Proves the validation oracle works: dependent launches submitted in ONE
//! group (no barrier between them) must make the layer's synchronization
//! validation complain, while the same launches in two groups must not. Its
//! own test binary, because the error counter is process-global.
#![cfg(all(feature = "vulkan", feature = "wgsl"))]

use tensorlisp_kernels::sched::blob::wgsl;
use tensorlisp_kernels::sched::backend::{Bound, Launch, MemKind};
use tensorlisp_kernels::sched::{Access, Backend, Signature};
use tensorlisp_kernels::vulkan::{validation_errors, Options, Vulkan};
use tensorlisp_kernels::Error;

const COPY: &str = r#"
struct PC { oa: u32, ob: u32, n: u32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < pc.n) { b[pc.ob / 4u + id.x] = a[pc.oa / 4u + id.x] + 1.0; }
}
"#;

#[test]
fn a_missing_barrier_is_reported() {
    let v = match Vulkan::open(Options { validate: true }) {
        Ok(v) => v,
        Err(Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => return,
        Err(e) => panic!("{e}"),
    };
    if !v.validated() {
        eprintln!("validation layer not installed, skipping");
        return;
    }
    let sig = Signature::new(vec![Access::Read, Access::ReadWrite], 4, [64, 1, 1]);
    let pipeline = v.pipeline(&wgsl::blob(COPY, "copy", sig).unwrap()).unwrap();
    let n = 1 << 16;
    let bytes = 4 * n as u64;
    let (x, y, z) = (v.alloc(bytes, MemKind::Device).unwrap(), v.alloc(bytes, MemKind::Device).unwrap(), v.alloc(bytes, MemKind::Device).unwrap());
    let push = (n as u32).to_ne_bytes();
    let launch = |a, b| Launch::<Vulkan> {
        pipeline: &pipeline,
        bindings: vec![Bound { buffer: a, offset: 0, len: bytes }, Bound { buffer: b, offset: 0, len: bytes }],
        push: &push,
        grid: [(n as u32).div_ceil(64), 1, 1],
    };

    // Control: an illegal dispatch (y group count over the device limit) must be
    // caught by core validation, or nothing below means anything.
    let before = validation_errors();
    let mut bad = launch(&x, &y);
    bad.grid = [1, 70_000, 1];
    let p = v.submit(&[vec![bad]]).unwrap();
    v.wait(&p).unwrap();
    drop(p);
    println!("core validation control: {} error(s)", validation_errors() - before);

    // y = x + 1, then z = y + 1, with a barrier between the two groups: clean.
    let before = validation_errors();
    let p = v.submit(&[vec![launch(&x, &y)], vec![launch(&y, &z)]]).unwrap();
    v.wait(&p).unwrap();
    drop(p);
    assert_eq!(validation_errors(), before, "a correct barrier must not be reported");

    // The same two launches in one group: a read-after-write hazard.
    let p = v.submit(&[vec![launch(&x, &y), launch(&y, &z)]]).unwrap();
    v.wait(&p).unwrap();
    drop(p);
    assert!(validation_errors() > before, "synchronization validation should flag the missing barrier");
    println!("synchronization validation reported {} error(s) for the missing barrier", validation_errors() - before);
}
