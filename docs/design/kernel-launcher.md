# Kernel launcher (`crates/kernels`)

A thin layer below any op graph: compile native GPU kernels, bind buffers and
scalars, dispatch. No tensors, no shapes, no ggml, no torch/candle. Typed
wrappers on top (`metal::paged_attn`) pick the kernel variant and grid the way
the upstream host code does.

Status: **Metal launcher, ggml-Metal interop and the native Metal executor
(`--device native`, below) implemented and tested**; **CUDA launcher and
paged-attention kernels and the native CUDA executor implemented and tested**
(RTX 3080, sm_86, driver 591.86 / CUDA 12.8); ggml-cuda interop is not started.

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

## CUDA backend (`crates/kernels/src/cuda`, feature `cuda`)

Same `Device`/`Buffer`/`Kernel`/`Arg`/`Dims` *shape* as Metal (a `Backend`
trait hasn't been extracted yet — the two modules don't share code, by
design, until a third backend would actually need it), implemented as follows;
deviations from the plan this section used to describe are called out inline.

1. **Driver API via `libloading`** (`crates/kernels/src/cuda/sys.rs`):
   `cuInit`, `cuDevicePrimaryCtxRetain`, `cuMemAlloc_v2`, `cuMemcpy{H,D}toD_v2`,
   `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuLaunchKernel`,
   `cuFuncSetAttribute`, `cuStreamCreate`/`Synchronize`, plus the matching
   NVRTC entry points, all `dlopen`'d (`nvcuda.dll`/`libcuda.so.1`,
   `nvrtc64_120_0.dll`/`libnvrtc.so`) rather than linked, so the crate builds
   with `--features cuda` on a machine with no NVIDIA driver at all;
   `Device::system_default()` returns `NoDevice`. The *driver* calls use the
   `_v2`-suffixed symbol names explicitly: `nvcuda.dll` still exports the
   unversioned ones too, but only as a compatibility alias of uncertain ABI,
   so don't rely on `GetProcAddress("cuMemAlloc")` resolving to the right thing.
2. **NVRTC only** (no AOT/cubin path yet — not needed: the vLLM paged-attention
   kernels aren't CUTLASS-heavy). `Device::library` registers one name
   expression per template instantiation (`nvrtcAddNameExpression` before
   `nvrtcCompileProgram`, `nvrtcGetLoweredName` after) and compiles to PTX
   (`--gpu-architecture=compute_XY`, arch read once via
   `cuDeviceGetAttribute(COMPUTE_CAPABILITY_{MAJOR,MINOR})` — more reliable
   than ggml-sys's `nvidia-smi` shell-out, since we already have a `CUdevice`);
   `cuModuleLoadDataEx` JIT-compiles the PTX to SASS at load time. An
   `extern "C" __global__` kernel (`copy_blocks_kernel_*`) needs no name
   expression: its symbol is already unmangled.
3. **NVRTC's header situation needed more than `-I`.** Its built-in headers
   cover CUDA's own `cuda_fp16.h`/`cuda_bf16.h`/`cuda_fp8.h` (confirmed:
   `cuda_bf16.h` resolves once `-I<CUDA_PATH>/include` is passed — the
   *toolkit's* headers, not libc's), but not `<stdint.h>`, `<cstdint>`,
   `<float.h>`, `<assert.h>`, `<type_traits>`, `<algorithm>`, `<mutex>`,
   `<vector>`, `<map>`: those are the *host* C/C++ runtime's, which NVRTC has
   no path to (no `cl.exe`/`gcc` invocation backs it). `cuda/shaders.rs`
   strips these `#include`s and prepends a ~10-line prelude defining the
   fixed-width int typedefs, `FLT_MAX` and a no-op `assert` by hand — simpler
   and more portable than hunting down the host toolchain's include dir (MSVC's
   on Windows, glibc's on Linux) just for a handful of typedefs. See that
   file's doc comment for the one sharp edge this flattener has: it does not
   evaluate `#ifdef`/`#else`, so a vendored file with two same-basename
   `#include "..."`s behind a `USE_ROCM` branch (there's exactly one such
   case, `quantization/fp8/{amd,nvidia}/quant_utils.cuh`) needs the dead
   branch's include excluded explicitly rather than just deduplicated.
4. **Launch details**, as planned: args are positional in the `void*[]`
   kernel-params array (`Device::launch` takes `&[Arg]`, not Metal's
   `&[(slot, Arg)]`; an absent optional buffer is `Arg::NullPtr`, matching the
   kernel signature's own null-checked pointer params). Dynamic shared memory
   is a `cuLaunchKernel` argument, not a bound slot; `Device::launch` always
   calls `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` first when it's
   nonzero (vLLM's host code caches this per-function behind a mutex; calling
   it unconditionally is simpler and cheap enough not to bother). Function
   constants have no CUDA equivalent — they're folded into the template args,
   hence the name expression, hence `Kernel`'s cache key; CUDA's `Kernel` has
   no `consts` field.
5. **`crates/kernels/src/cuda/paged_attn.rs`** reimplements vLLM's host
   launcher math in Rust (grid/block/shared-memory sizing, the v1-vs-v2
   selection heuristic) exactly like `metal::paged_attn` does for the Metal
   port — the CUDA `.cu` host launchers (`paged_attention_v1_f16`, ...) are
   *not* compiled or linked; only the `__global__` kernels they'd call are,
   via NVRTC. One correction to this doc's earlier assumption: vLLM's CUDA
   default is `NUM_THREADS = 128` (`WARP_SIZE = 32`), not the Metal port's
   256 — the two backends use different thread-block sizes for the same op,
   each matching its own upstream default. `copy_blocks` is hand-written
   (`cuda/shaders/copy_blocks.cu`), not vendored: upstream's CUDA version
   copies one block pair across *every model layer* in one launch
   (`int64_t *key_cache_ptrs[layer]`), which the Metal port had already
   simplified to one `key_cache`/`value_cache` pair per launch; the CUDA side
   now matches that simplification so `CopyBlocks` is one struct, not two.
6. **Tested**: `cargo test -p tensorlisp-kernels --features cuda` runs
   `tests/paged_attn_cuda.rs`, line-for-line the same cases as
   `tests/paged_attn.rs` (Metal) — `reshape_and_cache` + `gather_kv_cache`
   round-trip, `paged_attention` v1 and v2 against an f32 reference, f32/f16/
   bf16, GQA, softcap, odd head size, block 32, `copy_blocks`. All pass on an
   RTX 3080 (sm_86). Gated the same way as the Metal tests
   (`TL_ALLOW_NO_GPU=1` to skip without one).
7. **Not done**: ggml-cuda interop (the `metal::ggml` feature's equivalent —
   sharing ggml-cuda's primary context/stream so a kernel launch and a ggml
   graph can interleave without an extra sync; needs `csrc/cuda_interop.cpp`
   exporting `ggml_backend_cuda_context::stream()`, same shape as
   `metal_interop.*`), fp8 KV cache (the fp8 conversion paths are vendored but
   gated behind `ENABLE_FP8`, which nothing defines yet), ALiBi/sinks (wired
   through but untested — no port needs them yet).
8. **Native CUDA executor** (`cuda/exec.rs`, `cuda/lower.rs`,
   `cuda/native/{common.cuh,core.cu,mm.cu,fa.cu}`; `tensorlisp` feature
   `native-cuda`, or `cuda` which also builds ggml-cuda): the counterpart of
   the native Metal executor, same IR/planner (`graph.rs`, `plan.rs`), same
   `--device native`. Unlike Metal there is no ggml kernel library to reuse,
   so every op is our own NVRTC kernel with one uniform signature
   `(Op op, src0..src3, dst)`: `Op` is a 456-byte by-value descriptor holding
   shape/strides/type of up to five tensors plus 12 scalar slots (floats as
   bit patterns), packed per op in `lower.rs`. One launch per node, no fusion,
   scratch memory only for split-KV attention (below). Matmul (`mm.cu`, specialized per weight type, f32
   accumulate, dequantizing on the fly for f16/bf16/Q4_0..Q8_0/K-quants) is
   chosen in Rust (`mm_vec` warp-per-row for N <= 8, else a 64x64x16
   shared-memory tile); flash attention (`fa.cu`) is an online-softmax block
   per (query, head, batch), f32 Q, any plain/quantized K and V, f16 mask,
   logit softcap, ALiBi, GQA. Not supported (reported by name at compile
   time like Metal): xielu, mrope/vision rope, circular pad, soft_max/FA
   sinks, quantized cpy destinations, FA head size above 512.
   Verified by `crates/tensorlisp/tests/native.rs` (op families) and
   `native_tips.rs` (TIPSv2 text/vision, YOLO11n, PP-OCRv6, T5Gemma2
   generation) against the ggml CPU backend, plus
   `crates/kernels/tests/native_cuda_compile.rs` (every library builds).
   RTX 3080, release: TIPSv2 vision f16 224px 17.6 ms (ggml CPU 3.8 s),
   YOLO11n 640px 9.8 ms (CPU 592 ms), T5Gemma2 greedy "Paris" identical to
   the CPU backend. Matmul is plain fp32 FMA (no tensor cores yet); WMMA
   tiles and vectorized quantized matvec are the obvious next steps.

   Split-KV decoding: a one-query flash attention over >= 256 keys (the
   T5Gemma2 decode step, 4 heads x 1088 cache rows) would run on nh * nb
   blocks, 4 of 68 SMs. It lowers instead to `fa_split` (grid P x nh x nb,
   each block an online softmax over a slice of >= 64 keys; slices whose mask
   is all -inf, i.e. unused cache rows, exit at once) plus `fa_reduce`
   merging the (max, sum, acc) partials, which live in planned arena scratch.
   This is the structure of the paged-attention v2 kernel, reading ggml's
   `[hd, rows]` KV state tensors directly instead of vLLM's block-table
   layouts (those need a layout the ggml graph does not produce, so
   `cuda::paged_attn` stays a standalone launcher). `TL_NATIVE_FA=plain`
   disables it; `TL_NATIVE_TIME=1` / `TL_NATIVE_PROFILE=1` print per-run wall
   time / per-op time (syncing after each launch).
   T5Gemma2 f16 decode step (median, RTX 3080): one block per head 12.0 ms,
   split-KV 6.5 ms (83 -> 153 tokens/s, same tokens), unfused
   matmul + softmax + matmul 7.0 ms. Attention itself went 6.2 -> 1.0 ms of
   a step.
   Paged attention (vLLM kernel, f16 cache, 128 threads) on the same geometry,
   `tests/decode_attn_bench.rs` (`--ignored`, kernel time with launches queued,
   paged given only the valid keys as a compacted block table): one block per
   head 230 us, split-KV 28 us, paged 27 us with all 1088 rows valid; with 20
   valid keys 26 us vs 10 us; 290 valid 25 us vs 16 us; 8192 rows 74 us vs
   39 us. So paged attention only wins where it can skip keys through the
   block table, worth about 0.3 ms of a 6.5 ms T5Gemma2 step, and it would
   need the KV caches in vLLM's layout rather than ggml state tensors.

   Paged attention inside the native executor (CUDA only): Scheme's
   `(paged-attention q key-cache value-cache tables lens block-size max-context scale [kv-heads])`
   and `(paged-cache-write key value key-cache value-cache slots block-size [kv-heads])`
   (core.ss) build ggml `CUSTOM` nodes with no ggml function; the kind, kv heads,
   block size, max context and scale travel packed in the node's `userdata`
   (`op_params[4..6]`) and `lower.rs::Ctx::custom` turns them into the vendored
   vLLM kernels (`Custom::PagedAttention` / `Custom::CacheWrite` dispatches, kernel
   handles and v2 scratch resolved once at compile time). Pools are f16 states with
   vLLM's layouts (`[blocks, kv_heads, hs/8, bs, 8]` keys, `[blocks, kv_heads, hs, bs]`
   values; any state shape of the right size); block tables and context lengths are
   i32 inputs, slot mappings are one i64 per token written as two i32. Other
   devices fail to compile these nodes by name. `ports/t5gemma2/concurrency.ss`
   (appended to t5gemma2.ss by `tests/native_concurrency.rs`) has the batched
   entries `decode-dense` / `decode-paged`; both give the same tokens as the
   single-sequence `decode-step` (4 prompts of 6 to 206 tokens).
   Aggregate throughput, T5Gemma2 f16, RTX 3080, 32 generated tokens per
   sequence, host loop included, tokens/s:

   | sequences | decode-step per sequence | dense batch | paged batch |
   |---|---|---|---|
   | 1 | 83-150 | 168 | 164 (long) / 160 (short) |
   | 4 | | 488-497 | 497 |
   | 16 | | 777-781 | 836-862 |
   | 64 | | 2281-2285 | 3027 (956-token prompts) / 3302 (6-token prompts) |

   The single-sequence number is noisy (6.7-12 ms per step between runs). Batching
   is the big win (about 15-25x at 64 sequences). Paged beats dense by 1.3x
   (long prompts) to 1.45x (short prompts) at 64 sequences, mostly by reading only
   each sequence's own keys, and holds 0.8 MiB per short sequence instead of the
   dense 19 MiB (17 MiB when prompts fill the cache). Run with
   `cargo test --release -p tensorlisp --features native-cuda --test native_concurrency -- --ignored --nocapture`.

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

## Native Metal executor (`Device::Native`, `tl ... --device native`)

ggml's *backend* is replaced by this crate's; ggml's *graph* is not. A Scheme
program still builds a ggml cgraph (shape inference, the op vocabulary every
port already uses), but nothing is allocated or computed by ggml. Pipeline:

```
Scheme program --> ggml cgraph (metadata only, no_alloc)
   --> tensorlisp_kernels::graph::Graph          IR: ops by ggml name, ne/nb, op_params, views
   --> plan::plan                                 liveness + first-fit arena, input/weight/state placement
   --> metal::lower (Rust) + scheme/native.ss     one or more Dispatch per node, pipelines created
   --> Program::run                               encode the recorded list, commit, wait
```

Everything shape-dependent happens once per `(entry, input shapes, taps)` when
the graph is compiled; `run` only uploads inputs, encodes the dispatch list into
one command buffer and reads the results back from the arena (shared storage).
Weights are read from the GGUF straight into `MTLBuffer`s (split across buffers
past the device's maximum buffer length); `(define-state ...)` tensors get their
own zeroed buffer.

The kernels are ggml's, unchanged: `build.rs` flattens `vendor/ggml/.../ggml-metal.metal`
and bindgens `ggml_metal_kargs_*`, `FC_*`, `OP_*` from `ggml-metal-impl.h`, so
there is one source of truth and a ggml bump flows through. `metal/lower.rs` is
a port of the host half of ggml-metal (`ggml_metal_op_*` plus the pipeline
getters): kernel variant, function constants, threadgroup shapes, argument
structs. Not ported: op fusion (norm+mul+add, add chains) and concurrent
dispatch; every node is its own serial dispatch.

**Ops** (what the TIPS ports need and their natural neighbours): ADD SUB MUL DIV,
SCALE SQR SQRT SIN COS LOG CLAMP FILL LEAKY_RELU and all UNARY ops, SUM_ROWS MEAN,
CONCAT REPEAT, GET_ROWS, SOFT_MAX (mask, no sinks), CPY CONT DUP, NORM RMS_NORM,
IM2COL, ARANGE, UPSCALE, ROPE (norm, neox, multi, vision), POOL_2D, CONV_2D_DW, SET_ROWS, ARGMAX, PAD, PAD_REFLECT_1D, MUL_MAT (f32/f16/bf16 and the common quantized types),
FLASH_ATTN_EXT (matrix and vector kernels, mask, sinks, softcap, f16/f32 K/V);
RESHAPE VIEW PERMUTE TRANSPOSE are free. Anything else fails at compile time with the op
and node name (`native Metal: ROPE ...: op not implemented`). Missing for the
other ports: CONV_2D, CONV_TRANSPOSE_*, MUL_MAT_ID, GLU, PAD, ...
(each is a function in `lower.rs` modelled on its `ggml_metal_op_*`).

**Ports on `--device native`:** TIPSv2 text and vision, YOLO11n (same detections as
the README: bus 0.94, person 0.90/0.85/0.83/0.38; head cosine 0.999999 vs the
Ultralytics reference taps; 47.9 ms vs 48.2 ms on ggml Metal) and T5Gemma 2
(`caption` gives " a bumble bee in a flower bed.", `generate` the same tokens as
ggml Metal; KV cache in state tensors via SET_ROWS/CPY, flash attention with an
f16 cache, argmax in the graph) and PP-OCRv6 (detector taps and probability map match the
reference at f16 level, the full `ocr` pipeline returns the same boxes, scores and text as
ggml Metal). `tests/native_tips.rs` covers all four.

PP-OCRv6 found the one real gap: `nn:conv2d` pads with leading amounts (`ggml_pad_ext`),
which ggml-metal's pad kernel does not support (ggml silently runs it on the CPU).
Such ops get our own kernel: `metal/shaders/extra.metal`, a second library compiled next
to ggml's, with the same calling convention (argument struct at buffer 0). It is where
anything ggml-metal lacks goes. Missing ops are now all collected and reported in one error
(`not implemented: PAD x11 (e.g. ...); ...`) instead of one per run; `TL_NATIVE_TRACE=1`
prints each compiled graph's op histogram.

### Matrix multiplication lives in Scheme

`scheme/native.ss` (`$tl-native-lower-mul-mat`) decides, from the operand types,
shapes and strides, between the small-batch `mul_mv_ext`, the simdgroup-matrix
`mul_mm` and `mul_mv` kernels, and returns plain data: kernel name, function
constants, argument slots (tensors by role, kernel-argument structs as typed
fields), grid, threads, threadgroup memory. Rust packs the structs like C
(`kargs::pack`, size-checked against the bindgen'd `sizeof`), creates the
pipeline and records the dispatch. Replies are cached per operand signature.
This is the seam for the next steps: a variant search or a generated kernel
changes `native.ss`, not the Rust executor.

### Verification

* `crates/tensorlisp/tests/native.rs`: native vs the ggml CPU backend (and exact
  dequantized references for matmul) per op family: every matmul kernel x
  {f32, f16, bf16, q4_0, q8_0, q4_K, q6_K} x batch {1,3,5,8,9,40}; norms,
  activations, reductions, embedding, positions; attention with/without mask,
  GQA, flash vector and matrix kernels with and without KV padding; im2col patch
  embedding; unsupported ops are reported by name.
* `crates/tensorlisp/tests/native_tips.rs`: the real TIPSv2 text (f32) and vision
  (f16, flash attention, antialiased position resize at 224 px) towers, every tap
  and output, vs the CPU backend (skipped without `models/tipsv2-b14`).
* `crates/kernels/src/plan.rs`: arena planner invariants on random graphs.
* `tl compare --device native` against the PyTorch references of
  `ports/tipsv2` gives the same errors as ggml's Metal backend (text f32:
  embedding max abs 5.9e-4 vs 6.5e-4; vision f16 448 px: patches max abs 1.5e-2,
  cosine 0.999999; quantized variants match the accuracy in the port README).

### Numbers (M1 Pro)

| | native | ggml Metal |
|---|---|---|
| text f32, 3 texts | 16.1 ms | 16.4 ms |
| vision f16, 2 images 448 px | 168 ms | 164 ms |
| load (incl. compiling ggml's Metal library) | 0.9 s | 0.9 s |

The ~3% on vision is the missing fusion/concurrency; parity first, speed later.
The Metal library is compiled from source on every load (about 0.8 s); caching
the metallib/binary archive on disk is straightforward and not done.

### Not done / next

* Fusion and concurrent dispatch (needs memory-range tracking like ggml's
  `ggml_mem_ranges`, or a barrier-minimizing pass over the IR).
* The remaining ops above; then the t5gemma2 and yolo11 ports on `--device native`.
* Autotuning: only after op coverage. `native.ss` is where variants get enumerated.
