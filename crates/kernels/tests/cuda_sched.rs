//! The scheduler conformance suite (tests/common) over the CUDA backend:
//! NVRTC-compiled kernels, one in-order stream, event-based tickets.
//!
//! `cargo test -p tensorlisp-kernels --features cuda --test cuda_sched`
//! (`TL_ALLOW_NO_GPU=1` skips when there is no CUDA device or NVRTC).
#![cfg(feature = "cuda")]

mod common;

use std::sync::Arc;

use common::Kernels;
use tensorlisp_kernels::cuda::backend::Cuda;
use tensorlisp_kernels::sched::{Access, Backend, Code, Engine, KernelBlob, Signature, Target};
use tensorlisp_kernels::Error;

struct Guard(Engine<Cuda>);

impl std::ops::Deref for Guard {
    type Target = Engine<Cuda>;
    fn deref(&self) -> &Engine<Cuda> {
        &self.0
    }
}

fn engine() -> Option<Guard> {
    match Cuda::system_default() {
        Ok(c) => {
            println!("{}", c.name());
            Some(Guard(Engine::new(Arc::new(c))))
        }
        Err(Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => {
            eprintln!("no CUDA device, skipping");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

// Pointers arrive already offset, followed by the parameters as one struct.
const SOURCE: &str = r#"
struct AxpbPC { unsigned n; float s; };
extern "C" __global__ void axpb(const float* a, const float* b, float* o, AxpbPC pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pc.n) o[i] = a[i] * pc.s + b[i];
}

struct AccPC { unsigned n; };
extern "C" __global__ void acc_add(float* acc, const float* x, AccPC pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pc.n) acc[i] += x[i];
}

struct MatmulPC { unsigned m, n, k; };
extern "C" __global__ void matmul(const float* a, const float* b, float* c, MatmulPC pc) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < pc.m && col < pc.n) {
        float sum = 0.0f;
        for (unsigned i = 0; i < pc.k; i++) sum += a[row * pc.k + i] * b[i * pc.n + col];
        c[row * pc.n + col] = sum;
    }
}
"#;

fn kernels() -> Kernels {
    // Three separate blobs from one source: each is compiled as its own library keyed
    // by its content, so give each its own entry's source slice.
    let blob = |entry: &str, bindings, push, wg| {
        Arc::new(KernelBlob {
            target: Target::Cuda,
            code: Code::Cuda(SOURCE.to_string()),
            entry: entry.to_string(),
            label: String::new(),
            sig: Signature::new(bindings, push, wg),
            spec: vec![],
        })
    };
    Kernels {
        axpb: blob("axpb", vec![Access::Read, Access::Read, Access::Write], 8, [64, 1, 1]),
        acc: blob("acc_add", vec![Access::ReadWrite, Access::Read], 4, [64, 1, 1]),
        matmul: blob("matmul", vec![Access::Read, Access::Read, Access::Write], 12, [8, 8, 1]),
    }
}

conformance!(engine, kernels);
