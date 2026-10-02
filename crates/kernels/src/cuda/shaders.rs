//! Vendored CUDA sources (see ../../LICENSE-mistral-rs) and the include
//! flattener NVRTC needs for our own headers.
//!
//! Unlike the Metal flattener, an include NVRTC can resolve itself (anything
//! from the CUDA Toolkit: `cuda_fp16.h`, `cuda_bf16.h`, `stdint.h`, ...) is
//! left as a real `#include` rather than being an error: NVRTC bundles these
//! as built-in headers, so only the handful of small vendored headers below
//! need inlining.
//!
//! Changes from upstream (`mistralrs-paged-attn/src/cuda`, itself vendoring
//! vLLM's CUDA kernels, Apache-2.0, headers kept): each file is cut right
//! after its kernel/device-code namespace ends, dropping the host-side
//! launcher (grid/block/shared-mem sizing, the `CALL_*`/`LAUNCH_*` macros,
//! `cudaStream_t`/`dim3`/`<algorithm>`). NVRTC only ever sees `__global__`/
//! `__device__` code; [`crate::cuda::paged_attn`] reimplements the launcher
//! math in Rust (mirroring [`crate::metal::paged_attn`], which already does
//! this for Metal) and dispatches through [`crate::cuda::Device::launch`].

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    PagedAttention,
    ReshapeAndCache,
    CopyBlocks,
    GatherKvCache,
}

impl Module {
    pub fn name(self) -> &'static str {
        match self {
            Module::PagedAttention => "pagedattention",
            Module::ReshapeAndCache => "reshape_and_cache",
            Module::CopyBlocks => "copy_blocks",
            Module::GatherKvCache => "gather_kv_cache",
        }
    }

    fn text(self) -> &'static str {
        match self {
            Module::PagedAttention => include_str!("shaders/pagedattention.cuh"),
            Module::ReshapeAndCache => include_str!("shaders/reshape_and_cache_kernel.cu"),
            Module::CopyBlocks => include_str!("shaders/copy_blocks.cu"),
            Module::GatherKvCache => include_str!("shaders/gather_kv_cache_kernel.cu"),
        }
    }

    /// The flattened source followed by `tail` (e.g. explicit instantiations
    /// for kernels that don't go through NVRTC name expressions).
    pub fn source(self, tail: &str) -> String {
        let mut seen = Vec::new();
        let mut out = String::from(PRELUDE);
        flatten(self.text(), &mut seen, &mut out);
        out.push('\n');
        out.push_str(tail);
        out
    }
}

/// NVRTC has no real libc to resolve these C-runtime headers against (its
/// built-in headers cover CUDA's own `cuda_fp16.h`/`cuda_bf16.h`/`cuda_fp8.h`,
/// not `<stdint.h>` etc.), and passing `-I` to the CUDA Toolkit's own
/// `include/` (for `cuda_bf16.h`) doesn't help since none of these live there
/// either. Fixed-width integers and `FLT_MAX` are defined by hand instead;
/// `<type_traits>`/`<algorithm>` are dropped because nothing we compile
/// actually uses `std::` (the real libc headers would need a host toolchain's
/// include path, which NVRTC has no notion of).
const SKIP_SYSTEM_HEADERS: &[&str] =
    &["stdint.h", "cstdint", "stdio.h", "float.h", "assert.h", "cassert", "type_traits", "algorithm", "mutex", "vector", "map"];

pub(crate) const PRELUDE: &str = "\
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif
#ifndef assert
#define assert(x) ((void)0)
#endif
";

/// Our own small headers, inlined so NVRTC (given only one source string,
/// no header callback) can see them. Looked up by basename: the vendored
/// files use relative `../` includes we don't need to replicate on disk.
fn header(name: &str) -> Option<&'static str> {
    match name {
        "cuda_compat.h" => Some(include_str!("shaders/cuda_compat.h")),
        "attention_dtypes.h" => Some(include_str!("shaders/attention_dtypes.h")),
        "attention_generic.cuh" => Some(include_str!("shaders/attention_generic.cuh")),
        "attention_utils.cuh" => Some(include_str!("shaders/attention_utils.cuh")),
        "dtype_bfloat16.cuh" => Some(include_str!("shaders/dtype_bfloat16.cuh")),
        "dtype_float16.cuh" => Some(include_str!("shaders/dtype_float16.cuh")),
        "dtype_float32.cuh" => Some(include_str!("shaders/dtype_float32.cuh")),
        "dtype_fp8.cuh" => Some(include_str!("shaders/dtype_fp8.cuh")),
        "quant_utils.cuh" => Some(include_str!("shaders/quant_utils.cuh")),
        _ => None,
    }
}

fn flatten(text: &str, seen: &mut Vec<String>, out: &mut String) {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(path) = trimmed.strip_prefix("#include \"").and_then(|r| r.strip_suffix('"')) {
            let base = path.rsplit('/').next().unwrap_or(path);
            // `quantization/fp8/amd/quant_utils.cuh` shares a basename with the
            // nvidia one we vendored; this flattener doesn't evaluate the
            // `#ifdef USE_ROCM` around it, so resolving it too would both wins
            // the dedup race against the real (nvidia) include *and* land the
            // content in a branch NVRTC's real preprocessor then discards
            // (USE_ROCM is never defined). Leave the dead ROCm branch alone.
            match if path.contains("/amd/") { None } else { header(base) } {
                Some(h) => {
                    if seen.iter().any(|s| s == base) {
                        continue;
                    }
                    seen.push(base.to_string());
                    flatten(h, seen, out);
                }
                // A toolkit header (e.g. "cuda_fp8.h"): NVRTC resolves these itself.
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        } else if let Some(path) = trimmed.strip_prefix("#include <").and_then(|r| r.strip_suffix('>')) {
            if !SKIP_SYSTEM_HEADERS.contains(&path) {
                out.push_str(line);
                out.push('\n');
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
}
