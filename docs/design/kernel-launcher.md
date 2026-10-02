# Kernel launcher (`crates/kernels`)

A thin layer below any op graph: compile native GPU kernels, bind buffers and
scalars, dispatch. No tensors, no shapes, no ggml, no torch/candle. Typed
wrappers on top (`metal::paged_attn`) pick the kernel variant and grid the way
the upstream host code does.

Status: **Metal implemented and tested**, ggml-Metal interop implemented and
tested, **CUDA planned (below), untested**, Scheme binding not started.

## Model

| concept | Metal (now) | CUDA (planned) |
|---|---|---|
| `Device` | `MTLDevice` + `MTLCommandQueue` | primary `CUcontext` + `CUstream` |
| library | MSL text, flattened includes, `newLibraryWithSource` | NVRTC text, or embedded cubin/fatbin |
| `Kernel` | entry point + function constants (`Const`) | `CUfunction`; constants become template args / `-D` |
| `Buffer` | `MTLBuffer`, shared storage | `CUdeviceptr` (`cuMemAlloc`) |
| `Arg` at slot | `setBuffer/setBytes(slot)` | entry in the `void**` param array, in order |
| `Dims` | threadgroups x threads, threadgroup mem at slot 0 | grid x block, dynamic shared mem |
| `sync` | commit + `waitUntilCompleted` | `cuStreamSynchronize` |

Libraries are registered by key and compiled once; pipelines are cached by
`(library, entry, constants)`. Upstream instantiates the full cross product of
templates up front (288 paged-attention variants on Metal, minutes of
compile). We strip those lines from the vendored `.metal` and append exactly
the instantiation a launch needs, so a variant compiles on first use
(`cargo test`, 8 distinct variants: 7 s total).

Dispatches in one encoder are serial (`MTLDispatchTypeSerial`): the v2
paged-attention partial kernel and its reduce kernel need no barrier.

## Vendored kernels and what we changed

`src/metal/shaders/*.metal` come from mistral.rs `mistralrs-paged-attn`
(MIT, `LICENSE-mistral-rs`); the paged-attention file is itself adapted from
vLLM and MLX (Apache-2.0, headers kept). Changes:

* removed the bulk `instantiate_*(...)` invocations (macros stay);
* `copy_blocks`: `gid` was `thread_position_in_grid`, which indexes the wrong
  pair as soon as a group has more than one thread (reproduced: the test fails
  with the original binding). Now `threadgroup_position_in_grid`; also added the
  missing `[[buffer(3)]]`/`[[buffer(4)]]` on the two scalar args.

Kernels: `reshape_and_cache`, `paged_attention` (v1, and v2 + reduce for
> 512 tokens when few rows), `copy_blocks`, `gather_kv_cache`. fp8 (e4m3)
caches plug into the same entry points via the scales; not covered by tests
yet. `kv_scale_update.metal` is vendored but has no wrapper. Supported:
head size {64,80,96,112,128,192,256,512}, block size {8,16,32}, f32/f16/bf16,
GQA, softcapping, ALiBi and sink bindings.

Tests (`cargo test -p tensorlisp-kernels`) fill a shuffled paged cache through
`reshape_and_cache`, check `gather_kv_cache` round-trips it bit-exactly, and
compare `paged_attention` against an f32 softmax(qk)v reference (v1 and v2,
f32/f16/bf16, GQA, softcap, odd head size, block 32).

## ggml's Metal backend (what we found)

* `ggml_metal_device` owns **one global `MTLCommandQueue`**, shared by every
  Metal backend instance; command buffers on it run in commit order. We adopt
  that queue (`Device::from_raw`), so a launch + `sync` is ordered against ggml
  graphs computed before/after it. Residency sets are attached to that queue,
  which also keeps ggml's private buffers resident for our dispatches.
* A tensor's `MTLBuffer` + offset is `ggml_metal_buffer_get_id(tensor->buffer
  ->context, tensor)` (views resolve through `view_src`). Not in any public
  header, so `crates/ggml-sys/csrc/metal_interop.{h,cpp}` (compiled with the
  Metal group) exposes `tl_ggml_metal_{device,queue,tensor_buffer}`.
  `ggml_metal_device_get(i)` is *not* usable for this: it builds a new device
  object on every call; the shim goes through the registered backend device.
* ggml weight buffers are Metal *private* storage unless the device uses
  shared buffers; never rely on `Buffer::contents()` for them, read via ggml.
* ggml graph compute is synchronous, our `sync` is too, so interleaving is just
  "graph, launch, sync, graph" (`tests/ggml_interop.rs` does ggml-op ->
  our kernel -> ggml-op on the same tensors). Finer overlap would need a shared
  `MTLEvent` (ggml has `ggml_metal_event_*` internally) and is not needed yet.

Enable with feature `ggml` (off by default; the crate builds standalone).

## CUDA backend plan

Goal: same `Device`/`Buffer`/`Kernel`/`Arg`/`Dims` surface, so the typed
wrappers and tests carry over (extract a `Backend` trait from `metal::Device`
when the second implementation lands; don't abstract before).

1. **Driver API via `libloading`** (`libcuda`: `cuInit`, `cuDevicePrimaryCtxRetain`,
   `cuMemAlloc`, `cuMemcpy{H2D,D2H}`, `cuModuleLoadData`, `cuModuleGetFunction`,
   `cuLaunchKernel`, `cuFuncSetAttribute`, `cuStreamSynchronize`). Nothing to
   link at build time, so the crate still builds on machines without a
   toolkit; `Device::system_default()` just returns `NoDevice`. (This is what
   `cudarc` does; writing the ~15 calls ourselves avoids its build/feature
   matrix.)
2. **Two source paths behind one `Library`:**
   * *NVRTC* (`libnvrtc`, dynamic): text in, cubin out for the device's
     `sm_XX`. Template instantiation on demand maps to
     `nvrtcAddNameExpression("paged_attention<half,half,128,16,...>")` +
     `nvrtcGetLoweredName`, mirroring what the Metal side does by appending
     instantiations. Fits vLLM's `pagedattention*.cu`, `reshape_and_cache`,
     `copy_blocks`, `gather_kv_cache` (small headers, embedded and flattened
     like the Metal includes; `cuda_fp16.h`/`cuda_bf16.h` come from the
     toolkit's include dir, found via `CUDA_PATH`/`CUDA_HOME` like ggml-sys).
   * *AOT cubin/fatbin* via nvcc and the `cc` crate (`.cuda(true)`; ggml-sys
     already drives nvcc this way, same arch detection): for CUTLASS-heavy
     sources (FlashAttention-3, FlashInfer, Marlin) that NVRTC can't handle
     comfortably. Embedded with `include_bytes!` and loaded with
     `cuModuleLoadData`. Phase 2, per kernel family, behind a cargo feature.
3. **Launch details:** args are packed into the `void*[]` parameter array in
   slot order (slots are positional on CUDA, so gaps for optional buffers must
   be filled with null pointers where the kernel signature has them; the typed
   wrappers own this). Pointer args pass `base + offset`. Dynamic shared memory
   above 48 KiB (paged attention v1 with long contexts) needs
   `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` once per function.
   Function constants become template arguments, so they are part of the
   NVRTC name expression and the cache key, as they already are in `Kernel`.
4. **ggml-cuda interop**, same shape as the Metal shim: `tensor->data` is a
   device pointer in ggml's context, which is the device's *primary* context
   (we retain the same one, so pointers are valid). The stream is the
   difference: ggml-cuda computes asynchronously on its own non-blocking
   stream, so `csrc/cuda_interop.cpp` should export that backend's stream
   (`ggml_backend_cuda_context::stream()`) and our `Device` launches on it.
   Until then, `ggml_backend_synchronize` + `cuStreamSynchronize` around each
   hop is correct, just slow.
5. **Reuse upstream's CUDA host code where it is simpler.** mistral.rs ships
   `.cu` files with `extern "C"` host launchers (`paged_attention_v1_f16(...)`,
   called through its `ffi.rs`). The fastest first milestone is compiling those
   with nvcc/`cc` and calling the launchers directly, then moving to the
   NVRTC path only if build time or flexibility matters.
6. **Testing:** the Rust reference in `tests/paged_attn.rs` is
   backend-neutral; parameterize the harness over `Device` so the identical
   cases run on CUDA. Gate on `HAS_CUDA` like ggml-sys. No NVIDIA hardware was
   available here, so none of the CUDA plan has been run.

## Using it from Scheme (next step)

Kernel launches cannot live *inside* a ggml graph: the scheduler only runs ggml
ops, and `ggml_map_custom` is CPU-only. The existing structure already has the
seam: `(pipeline ...)` runs entries in sequence from Scheme, and
`(define-state ...)` tensors are device buffers that outlive a run. So a paged
KV cache is a state tensor; a decode step is `entry A (ggml: qkv, rope)` ->
`reshape-and-cache` + `paged-attention` (kernel steps on the state buffers) ->
`entry B (ggml: o-proj, mlp)`. Binding work: foreign procs for
`kernel-library`/`kernel-launch` plus typed `reshape-and-cache`,
`paged-attention`; tensor/state -> `(Buffer, offset)` through `tensor_buffer`;
a pipeline step kind that is a kernel call instead of `(run entry ...)`.

## Quantization

Paged attention only touches the KV cache (f16/bf16/f32/fp8), so weight quants
don't affect it. For GEMM kernels brought in later (e.g. Marlin), keep ggml
block quants as the on-disk canonical form (`tl quantize` already writes them)
and repack at load per backend, as llama.cpp does for CPU.
