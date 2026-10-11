# Scheduler: concurrent sessions over kernel blobs and arenas

Status: **design + first implementation** (`crates/kernels/src/sched`,
`crates/kernels/src/vulkan`). The sched core and the Vulkan backend are
implemented and tested on an RTX 3080; the existing Metal and CUDA executors
still use their own synchronous replay and are not yet ported (see
[Migration](#migration)). The server, the v2 program format and yielding
pipelines build on this and are described in the last sections.

Implemented so far:

| piece | where | tests |
|---|---|---|
| generational handles | `sched/handle.rs` | unit |
| kernel blobs, signatures, registry (first source wins), WGSL->SPIR-V helper | `sched/blob.rs` (`wgsl` feature) | unit |
| hazard analysis into barrier-separated groups | `sched/hazard.rs` | unit + randomized independence check |
| size-class block pools (session state, step arenas) | `sched/pool.rs` | unit |
| engine: compile, sessions, async submit, tickets | `sched/engine.rs` | `tests/vulkan_sched.rs` |
| Vulkan backend | `vulkan/mod.rs` (`vulkan` feature) | `tests/vulkan_sched.rs` |
| SPIR-V reflection: signature (bindings, access, push size, workgroup) from any blob | `sched/spirv.rs` | unit + `tests/ggml_vulkan.rs` |
| GLSL compiler wrapper (`glslc` or glslang, include flattening) | `vulkan/glsl.rs` | `tests/ggml_vulkan.rs` |
| CUDA backend (`Backend` for NVRTC blobs, one in-order stream, event tickets) | `cuda/backend.rs` (`cuda` feature) | `tests/cuda_sched.rs` |
| backend-generic conformance suite, instantiated per backend | `tests/common/mod.rs` | 10 cases x {Vulkan, CUDA} |
| precompiled ggml shader library as a `KernelSource` (name + spec constants -> blob) | `vulkan/ggml.rs`, `scripts/build-ggml-vulkan-shaders.sh` | all 1976 modules reflect; registry blob == source blob |
| ggml's Vulkan shaders dispatched as blobs | `vendor/ggml/src/ggml-vulkan`, `tests/ggml_vulkan.rs` | add/sub/mul/div, copy, scale, rms_norm (+fused mul), mat-vec f32 and Q4_0 |

```
cargo test -p tensorlisp-kernels --lib sched
cargo test -p tensorlisp-kernels --features vulkan,wgsl --test vulkan_sched --test vulkan_validation -- --nocapture
cargo test -p tensorlisp-kernels --features cuda --test cuda_sched -- --nocapture
scripts/build-ggml-vulkan-shaders.sh        # glslc + MSVC/c++: ~2 min, 1976 .spv in target/ggml-vulkan/spv
cargo test -p tensorlisp-kernels --features vulkan --test ggml_vulkan -- --nocapture --test-threads=1
```

The GPU tests cover dependent chains (a 300-pass ping-pong over 4 MiB, which
corrupts without barriers), independent launches sharing a group, unaligned
binding offsets, weights in `Space::Model` with a matmul checked against the
CPU, per-session state with pool reuse (a dropped session's block serves the
next, zeroed), two tickets in flight with arena pooling, four threads driving
their own sessions, and rejection of malformed plans. The suite lives in
`tests/common` and runs unchanged on every backend (see
[The dispatcher](#the-dispatcher)).

**Validation.** The Vulkan tests run under `VK_LAYER_KHRONOS_validation`
(`Options { validate: true }`, or `TL_VULKAN_VALIDATE=1`), with errors counted
and a test failing if any occurred. Two things were not obvious: the layer's
synchronization validation ignores compute-shader accesses unless
`syncval_shader_accesses_heuristic` is on, so the backend sets `validate_sync`
and that heuristic through `VK_EXT_layer_settings`; and
`tests/vulkan_validation.rs` proves the oracle works by submitting two
dependent launches in one group and expecting a `READ_AFTER_WRITE hazard`
error, while the correctly barriered version stays clean. (Before the
heuristic was on, a clean sync-validation run meant nothing.)

## Why

Today one `Model` has one `Mutex<Inner>`, one allocated graph per entry, global
`define-state` tensors and a synchronous `run`. That cannot serve concurrent
requests, and each GPU backend (Metal, CUDA, now Vulkan) re-implements "plan,
record, replay". A server needs the opposite split:

* **what a plan reads and writes** is data, so a backend-neutral scheduler can
  reorder, batch and overlap work;
* **what a kernel is** is an opaque blob behind one interface, so built-in
  kernels, ggml's, SPIR-V from a compiler and fused kernels are all the same
  thing to the scheduler;
* **who owns a byte** is explicit (model, session, step), so memory is reused
  across requests when it fits;
* **submission is asynchronous**, with fences, so the host never blocks a
  device that has work to do.

ggml stays as a graph frontend (shape inference, the op vocabulary), a kernel
provider where it has kernels, and the reference implementation for parity
tests. Its scheduler and allocator are not used on this path (the native
executors already work that way).

## Layers

```
Scheme (control plane)     builds graphs, runs pipelines as coroutines,
   |  handles only          holds (slot, generation) handles, never pointers
   v
Graph IR (graph.rs)        ops by ggml name, ne/nb, views; fusion rewrites it
   v
Lowering (per backend)     node -> Launch { kernel blob, bindings, push, grid }
   v
Plan (sched::plan)         launches + read/write sets -> hazard analysis ->
   |                        groups separated by barriers; arena layout
   v
Engine<B: Backend>         sessions, pools, in-flight submissions, fences
   v
Backend (vulkan, ...)      buffers, pipelines from blobs, command recording
```

Scheme never dispatches kernels. It asks the engine to run a plan and gets a
ticket back; a pipeline is a coroutine that yields on tickets (see
[Pipelines](#pipelines)). A GC pause in Scheme therefore cannot stall the
device, and a stale handle is an error, not a crash.

## Memory: three lifetime classes

Every buffer range a launch touches is in one of four *spaces*. This replaces
`plan::BufId { Weights, State, Io, Arena }`:

| space | owner | lifetime | allocated | shared between sessions |
|---|---|---|---|---|
| `Model(i)` | `Arc<ModelMem>` | model load .. drop | once | yes, read-only |
| `Session(i)` | `Session` | session create .. drop | from the `SessionPool` | no |
| `Step` | one in-flight submission | one `submit` .. its fence | from the `StepPool` | no, but the *buffer* is reused |

`Step` contains the graph inputs and outputs (the old `Io` buffer, at the
start) and the planned arena (activations, scratch). A plan states its
`step_bytes`; the engine hands every submission a step arena of at least that
size and takes it back when the fence signals. Memory is
`weights + sessions x state + (in-flight submissions) x max step`, not a sum
over graphs. Planner invariants (liveness, first fit) are unchanged: see
`plan.rs`.

**Session memory** is a size-bucketed pool: blocks are powers of two (or
1.25-steps for big ones), free lists per bucket. A session that is dropped
returns its blocks; a new session whose requirement fits a free block takes it
(zeroed on reuse when the state demands it). KV caches are the main customer.
The paged-attention layout (fixed-size blocks per sequence) is the same idea
one level down, owned by the program: the pool hands out the *pages*, a block
table inside the program maps tokens to pages.

**Handles.** `Handle { index: u32, generation: u32 }` into a `HandleTable<T>`.
Scheme holds handles to tensors/sessions; the table owns the `T`. Freeing bumps
the generation, so a stale handle fails `get` cleanly. Handles are `Copy` and
fit a Scheme fixnum.

## Kernel blobs

A kernel is a **blob plus a signature**; nothing else is visible to the
scheduler.

```rust
struct KernelBlob { target: Target, code: Code, entry: String, sig: Signature, spec: Vec<u32> }
enum Code   { Spirv(Vec<u32>), Msl(String), Cuda(String), Ptx(String) }
struct Signature { bindings: Vec<Access>, push_bytes: u32, workgroup: [u32; 3] }
enum Access { Read, Write, ReadWrite }
```

`sig` is what the hazard analysis uses: binding *i* is read, written or both.
That makes any blob schedulable without knowing the op. A
`KernelSource` produces blobs for a `(Target, key)`:

```rust
trait KernelSource { fn blob(&self, target: Target, key: &KernelKey) -> Result<KernelBlob>; }
```

Implementations: built-in tables (the NVRTC/MSL/SPIR-V libraries), the ggml
Metal library, and later the Scheme kernel compiler, which returns SPIR-V or
MSL text. The registry maps `KernelKey` (op pattern + dtypes + shape class +
tuning parameters) to a source. Pipelines are cached by blob content hash, so
a generated kernel that equals a built-in one costs nothing.

**Conventions every Vulkan blob follows** (other backends define theirs):

* descriptor set 0, bindings `0..n` are storage buffers in signature order;
* push constants: `u32 offs[n]` first (see below), then `push_bytes` of kernel
  parameters;
* workgroup size is a specialization constant triple, or fixed in the blob.

**Binding offsets.** Storage buffer descriptor offsets must be multiples of
`minStorageBufferOffsetAlignment`, but tensor views are at arbitrary byte
offsets. The backend rounds the descriptor offset *down* to the alignment and
passes the residual (in bytes) in `offs[i]`; kernels add `offs[i]` to their
indexing. Arena allocations are 256-byte aligned, so the residual is zero
except for views, and the cost is one add.

### Fusion and node substitution

Fusion is an **IR rewrite before lowering**, not a backend feature:

1. A pass matches a subgraph (e.g. `RMS_NORM -> MUL`) by pattern.
2. It asks the registry for a kernel covering the pattern and the operands.
   If none, the subgraph is left alone.
3. It replaces the subgraph with one node whose sources are the subgraph's
   boundary inputs and whose result is the boundary output. Interior tensors
   disappear from liveness, so the arena planner reclaims their memory too.
   Planning runs after fusion.
4. The substituted node carries the `KernelKey`, so lowering emits exactly one
   launch with the blob the pass chose.

Generated kernels register patterns and blobs together; the Rust side stays
generic. The pass lives with the IR, not in a backend.

## Plans, groups and barriers

A **launch** is `{ kernel, bindings: [(Space, off, len)], push, grid }`. A
**plan** is launches plus their step layout. The hazard planner (`sched::hazard`,
GPU-free, property-tested) walks the launches in program order and cuts them
into **groups**: launches in a group may run in any order or concurrently; a
barrier separates groups. A launch joins the current group unless

* it reads a range the group writes (RAW), or
* it writes a range the group reads (WAR) or writes (WAW).

Ranges are compared per space and as byte intervals, so two independent
branches (the K and V projections) share a group and need no barrier between
them. Backends emit one barrier per group boundary: Vulkan one global compute
memory barrier, CUDA stream order (or a second stream per branch later), Metal
`memoryBarrier` on a concurrent encoder.

A fixed global barrier is also recorded at the start of every submission so
that consecutive submissions on a queue are memory-ordered (Vulkan does not
order memory between submissions implicitly).

The planner is in-order and greedy; reordering independent launches to widen
groups is a later optimization that only touches this module.

## Submission and fences

```rust
let plan   = engine.compile(&launches, layout)?;         // once per (entry, shape class)
let ticket = engine.submit(&plan, &Run { session, inputs })?;  // returns at once
ticket.poll() / ticket.wait()? ; ticket.read(range)?
```

`submit` takes a step arena, writes inputs, records one command buffer, submits
with a fence and returns a `Ticket`. `Ticket::wait` blocks on the fence;
dropping a ticket waits, then returns the arena to the pool. Command buffers,
pools and descriptor pools are recycled per submission. Queue access is
serialized by a mutex (Vulkan requires external synchronization); several
queues are a later step, and the engine API does not change.

Scheduling policy (which ticket next, batching compatible launches of
different sessions into one launch with a larger batch dimension, chunked
prefill) lives above the engine, in the server's scheduler loop. The engine is
mechanism, not policy.

## The dispatcher

One `Engine<B: Backend>` per device. It owns no kernels and knows no ops; it
turns *launch descriptions* into asynchronous work.

```
LaunchDesc { kernel: Arc<KernelBlob>, bindings: [Range], push: bytes, grid: [u32;3] }
        |
        |  Engine::compile(&[LaunchDesc], step_bytes)            once per (entry, shape class)
        |    - the blob's target must equal B::TARGET
        |    - bindings/push/grid must match the blob's signature and the device limits
        |    - every Step range must lie inside step_bytes
        |    - one pipeline per distinct blob (cached by content hash): Backend::pipeline
        |    - footprints from the signature's Read/Write/ReadWrite, then hazard::group
        v
CompiledPlan { pipelines, launches, groups: [[launch]], step_bytes, needs: Space -> extent }
        |
        |  Engine::submit(&plan, &Arc<ModelMem>, Option<&Arc<Session>>, inputs) -> Ticket
        |    - check Model/Session extents against `needs`
        |    - take a step arena (size-class pool), write the inputs into it
        |    - resolve each Range to Bound { buffer, offset, len } by Space
        |    - Backend::submit(groups) records and submits, returns Pending at once
        v
Ticket { pending, arena, model, session, pipelines }     poll() / wait() / read(off, len)
        drop: wait for the device, then the arena returns to the pool
```

`Backend` is the only thing a GPU API implements:

| method | Vulkan | CUDA |
|---|---|---|
| `alloc(bytes, kind)` | `vkAllocateMemory` (device-local; `Step` prefers host-visible VRAM when the heap is big) | `cuMemAlloc` |
| `write` / `read` / `zero` | memcpy if mapped, else staging buffer + copy command; `vkCmdFillBuffer` | stream-ordered `cuMemcpy*Async` / `cuMemsetD8Async`; `read` synchronizes the stream |
| `pipeline(blob)` | SPIR-V -> shader module, descriptor set layout (n storage buffers), pipeline layout (push range), compute pipeline with the blob's spec constants | NVRTC-compile `Code::Cuda` (spec constants as `TL_SPEC_<i>` defines), `cuModuleGetFunction` |
| `submit(groups)` | one command buffer: global barrier, then per launch a descriptor set (offsets aligned down, residual pushed), push constants, dispatch; a global compute barrier between groups; `vkQueueSubmit` with a fence | launches in program order on the one in-order stream, then an event; groups need no barrier |
| `poll` / `wait` | `vkGetFenceStatus` / `vkWaitForFences` | `cuEventQuery` / `cuEventSynchronize` |
| calling convention | set 0 bindings 0..n storage buffers, push = residual offsets (`OffsetMode::Prefix`) + params, or params only (`Raw`, ggml's shaders) | `extern "C" __global__`: n pointers, then one by-value parameter struct |

Everything that differs between APIs sits behind those methods; the plan,
hazard analysis, pools, sessions and tickets are shared. That is what the
conformance suite checks: `tests/common` holds ten scenarios (dependent chain,
independent launches sharing a group, unaligned offsets, weights + matmul,
session state and pool reuse, async tickets and arena pooling, threads with
sessions, a 300-pass ordering stress, malformed plans, wrong-target blobs) as
generic functions over `Backend`, and each backend's test file supplies an
engine plus the same three tiny kernels in its own language (WGSL via naga for
Vulkan, CUDA C for CUDA) and instantiates them with `conformance!`. A third
backend (Metal) passes if it passes the same file.

What is deliberately not in the dispatcher: choosing the next ticket, batching
across sessions, chunked prefill (policy, above it); choosing a kernel for an
op (lowering and the `KernelSource` registry, below it); the CUDA backend uses
one stream, so groups do not yet buy concurrency there.

## Pipelines (control plane)

A pipeline today blocks the Scheme thread for a whole generation. For a server
it must yield between steps:

* `(run entry ...)` inside a pipeline does not call into Rust and block; it
  **returns a request** to the pipeline driver. The driver (Rust) submits the
  plan, and resumes the coroutine when the ticket completes.
* Scheme side: pipelines are coroutines on one-shot continuations
  (`call/1cc`), captured at `run` and resumed by the driver from the Scheme
  thread's own loop. No continuation ever spans a Rust frame: `run` unwinds to
  the driver loop (it is the *driver* that calls back into Scheme), which is
  the part that made a naive design unsafe.
* One Scheme thread interleaves many pipelines; the device interleaves many
  tickets. Neither blocks the other.

## Program format v2

`TL_VER = 2` splits what v1 mixes in one Scheme text:

| key | content |
|---|---|
| `TL_MODEL` | weights table, entries (pure graph builders: inputs, outputs, state declarations), no control flow |
| `TL_PIPE` | pipelines and postprocess (orchestration, may call entries) |
| `TL_VER` | `2` |

v1 files (`TL_TXT` only) keep loading: the loader splits them by form. `tl
convert` rewrites a v1 file as v2. State declarations become *templates*
(`name`, type, dims): the session instantiates them, so `(state name)` resolves
against the running session. Entries are the unit the plan cache is keyed on;
pipelines are what the server schedules.

## Server (`tl serve`)

`axum` over the `Engine` API: `POST /v1/run` (entry or pipeline, inputs,
session id optional), streaming for pipelines that yield text. A session is
created explicitly (`POST /v1/sessions`) or implicitly per request. The
server's scheduler loop owns the policy (continuous batching, prefill
chunking, admission against pool memory). Cancellation drops the ticket and the
coroutine.

## Vulkan backend (`crates/kernels/src/vulkan`, feature `vulkan`)

* **Loader:** `ash` with its `loaded` feature (dlopen of `vulkan-1.dll` /
  `libvulkan.so.1`), same stance as CUDA: the crate builds with no SDK and
  `Device::system_default()` returns `NoDevice` if there is no loader.
* **Device choice:** discrete over integrated over software; `TL_VULKAN_DEVICE`
  selects by index. A compute-capable queue family, preferring a compute-only
  one (async compute on NVIDIA/AMD).
* **Memory:** `Model`/`Session` buffers are device-local. `Step` arenas prefer
  `DEVICE_LOCAL | HOST_VISIBLE` (resizable BAR) so inputs and outputs are plain
  memcpy; otherwise device-local plus a cached staging buffer and copy
  commands.
* **Pipelines:** SPIR-V blobs only. No `glslc` is needed at run time. Kernels
  for tests and fallbacks are written in WGSL and translated with `naga`
  (feature `wgsl`); the Scheme compiler will emit SPIR-V directly.
* **Descriptors:** one descriptor set per launch from a per-submission pool
  (reset on recycle); `VK_KHR_push_descriptor` is not required.
* **Not yet:** cooperative matrices, per-vendor subgroup tuning
  (`VK_EXT_subgroup_size_control`, which ggml's mat-vec uses on some GPUs),
  multiple queues, timeline semaphores between submissions, pipeline cache to
  disk, 64-bit indexing pipelines for tensors above `maxStorageBufferRange`.
  A lowering from the graph IR to SPIR-V kernels (the "native Vulkan executor")
  is the next step and the main remaining work for this backend.

## ggml's Vulkan kernels as blobs

`vendor/ggml/src/ggml-vulkan` is upstream ggml v0.20.1's Vulkan backend
(the version `vendor/ggml` already is): 167 GLSL shaders and `ggml-vulkan.cpp`,
the host half we port from. The scheduler uses the shaders unchanged and
replaces the host half: ggml's pipeline cache, descriptor management and
command recording are what `Engine` and `Backend` do.

**Compiling.** `vulkan::glsl::Compiler::find()` looks for `glslc`
(`TL_GLSLC`, `$VULKAN_SDK`, `PATH`, `C:\VulkanSDK`), then glslang unpacked in
`vendor/glslang` (gitignored; the Khronos release zip, no admin rights needed),
then glslang on `PATH`. glslang is the front end glslc wraps, so the SPIR-V is
equivalent; the difference is that glslang has no implicit `#include`, so
`glsl::flatten` inlines includes itself (ggml's headers guard themselves with
`#ifndef`). Defines are `-DNAME=VALUE` as in `vulkan-shaders-gen.cpp`. The
shaders target Vulkan 1.2, so the device is created as 1.2 with 16/8-bit
storage, float16, int8/16/64 enabled where supported.

**Signatures come from the SPIR-V.** `sched::spirv::reflect` reads descriptor
bindings (with `NonWritable` / `NonReadable` for access), the push-constant
block size and the literal `LocalSize`, so a blob needs no hand-written
description: `add.comp` reflects as 4 bindings (the fourth is the optional
`partial_sums` buffer), 116 push bytes, workgroup 256. A binding the op does
not use is passed as a zero-length range, which the hazard analysis ignores.

**Offsets.** ggml's shaders take the misalignment of each binding in elements
inside `misalign_offsets` (and per-op fields), so they are compiled with
`OffsetMode::Raw`: the backend rounds the descriptor offset down and the
lowering computes `offset % storage_offset_align` itself, exactly like
`get_misalign_bytes` in ggml-vulkan.cpp.

**Host conventions we had to reproduce** (each is a function in
`tests/ggml_vulkan.rs` that becomes the Rust lowering):

* elementwise ops (`wg_denoms = {512,1,1}`): workgroups are `(1, ceil(ne/512), 1)`
  past 512 elements, `(1, 512, ceil(ne/262144))` past 262144, because the
  shaders index `z*262144 + y*512 + x`; getting this wrong silently covers only
  1024 elements;
* strides are passed in elements (`nb / type_size`); unary ops decompose the
  index with `init_fastdiv_values` magic numbers;
* `scale.comp` and friends index linearly and are only valid for contiguous
  sources, `copy.comp` honours strides;
* `rms_norm.comp` is one shader for plain and fused-with-mul, selected by the
  specialization constant `do_multiply` (index 1);
* mat-vec (`mul_mat_vec.comp`): five bindings (weights, activations,
  destination, two optional bias buffers bound to the destination), the
  specialization constants are `{workgroup size, rows per workgroup, columns}`,
  the grid is `ceil(rows / rows_per_wg)`, the first four push words are
  `{K, K, K, N}` (ggml passes `ne10` for the stride) and batching columns
  changes the batch strides. The weights stay in the on-disk quantized layout.

**Precompiled library.** `scripts/build-ggml-vulkan-shaders.sh` mirrors the
CMake logic: probe `glslc` for coopmat, coopmat2, integer dot, bfloat16 and
FP4/FP8 support (`feature-tests/*.comp`), build upstream's
`vulkan-shaders-gen` (MSVC through a generated `.bat` on Windows, `c++`
elsewhere), and run it once per `.comp` in parallel. With shaderc v2026.4 all
seven extensions are supported and 1976 variants compile in about two minutes
(54 MB, `target/ggml-vulkan/spv`, not committed). `vulkan::ggml::SpvLibrary`
loads one by name; it is a `KernelSource`, so `registry.resolve(Vulkan,
KernelKey { name: "mul_mat_vec_q4_0_f32_f32", params: spec })` returns the
blob. Every module goes through the reflector in `tests/ggml_vulkan.rs`.

**Not done yet:** the variant tables (the lists in `vulkan-shaders-gen.cpp`
and `ggml_vk_op_get_pipeline` that map an op and its dtypes to a shader and
defines), push-constant builders for the remaining op families (soft_max,
rope, flash attention, matmul with tiles, im2col, ...), and embedding the
`.spv` set in the binary instead of reading the build directory. Each is a
mechanical port of the corresponding part of `ggml-vulkan.cpp` into a
`KernelSource` plus a lowering function.

## Models on ggml's Vulkan backend

The native Vulkan executor (graph IR to launches on the engine) does not exist
yet, so models run on Vulkan today through ggml's own backend, which `ggml-sys`
now builds (`--features vulkan`). It is the reference for the native path and
the proof that the generated shader library is sound.

```
scripts/build-ggml-vulkan-shaders.sh                      # once: glslc -> target/ggml-vulkan/{spv,gen}
cargo build --release -p tensorlisp-cli --features vulkan # ~1 min incl. 138 generated shader units
target/release/tl run MODEL.gguf --device gpu ...
```

`ggml-sys` needs the SDK headers (`VULKAN_SDK` or `C:\VulkanSDK\*`) and the
generated sources (`GGML_VULKAN_SHADERS` overrides `target/ggml-vulkan/gen`);
the backend loads the Vulkan loader itself. With only this backend enabled,
`--device gpu` and `auto` pick `Vulkan0`. (With both `cuda` and `vulkan` built,
ggml's device order decides; there is no per-backend selector yet.)

Results on an RTX 3080, release build, outputs checked against the CPU backend
or the PyTorch/HF reference:

| model | check | Vulkan | CPU |
|---|---|---|---|
| TIPSv2 B/14 text, f16, 2 prompts | cosine 0.9999998 vs CPU (max abs 0.008) | 3.35 ms | 1782 ms |
| YOLO11n, 640 px, bus.jpg | same boxes as the README (bus 0.94, persons 0.90/0.85/0.83/0.38) | 5.6 ms | 599 ms |
| T5Gemma2 270M f16, `generate` | same tokens as CPU ("Paris.") | 54.8 ms | 344.9 ms |
| Gemma 4 E2B q8_0, `generate`, 32 tokens | all 32 tokens identical to the HF reference | 256 ms (8 ms/token) | 5487 ms |
| Gemma 4 E2B f16 (9.3 GB on a 10 GB card) | runs | 609 ms (19 ms/token) | |

`GGML_VK_PERF_LOGGER=1` prints per-op GPU time: a Gemma 4 q8_0 decode step is
7.1 ms of kernels, the mat-vec kernels at about 1 TFLOP/s (memory bound).

**Cold start.** ggml-vulkan creates pipelines lazily and the driver compiles
SPIR-V on first use: the first run of each model took 1.3 to 11 s (T5Gemma2
`generate` 11 s, Gemma 4 4.2 s), later runs and later processes (driver shader
cache) are fast. A server must warm its pipelines at startup; the engine's
`compile` already creates pipelines eagerly for exactly this reason.

## The native Vulkan executor

`--device native` with `--features native-vulkan` runs a model's ggml graph on the
scheduler instead of a ggml backend. Nothing of ggml's runtime is involved: the
Scheme program builds a ggml cgraph (shape inference), `native_ir.rs` converts it to
the backend-neutral IR, `plan.rs` plans one arena, `vulkan/lower.rs` lowers every node
to launches of ggml's precompiled shaders, and `Engine` runs them.

```
cargo build --release -p tensorlisp-cli --features native-vulkan   # no ggml-vulkan, no SDK needed to build
scripts/build-ggml-vulkan-shaders.sh                               # once, needs glslc: the .spv library
tl run model.gguf --device native ...                              # TL_GGML_SPV overrides the .spv directory
```

| piece | where |
|---|---|
| IR -> launches (the host half of ggml-vulkan.cpp, ported) | `crates/kernels/src/vulkan/lower.rs` |
| memory plan, `Engine` compile/submit, program I/O | `crates/kernels/src/vulkan/exec.rs` |
| runtime glue (weights, states, `NativeProgram`) | `crates/tensorlisp/src/native_vulkan.rs` |

Memory maps onto the scheduler's spaces: weights are `Space::Model(chunk)` (1 GiB
chunks, larger tensors get their own buffer), all `(define-state ...)` tensors one
zeroed `Space::Session(0)` block (`reset_state` swaps in a fresh one), and inputs
plus the activation arena the per-submission `Space::Step` arena. The model keeps one
compiled program per input shape (prefill and decode alternate), which also fixed a
recompile-on-every-shape-change in the Metal/CUDA paths.

**Ops lowered** (everything ggml's Vulkan backend has a shader for that the ports
use): ADD SUB MUL DIV (broadcast, f32/f16), SCALE and the unary family (abs, sgn,
neg, step, tanh, elu, relu, sigmoid, gelu, gelu_quick, silu, hardswish,
hardsigmoid, exp, expm1, gelu_erf, floor, ceil, round, trunc, sqr, sqrt, sin, cos,
log, clamp, leaky_relu, fill), RMS_NORM, NORM, SUM_ROWS, MEAN, SOFT_MAX (up to 16384
columns), MUL_MAT (mat-vec up to 8 columns for f32/f16/bf16/Q4_0..Q8_0/K-quants,
matrix-matrix on the scalar tiles with alignment variants and split-K, f32/f16/
quantized weights), ROPE (normal, neox), FLASH_ATTN_EXT (scalar path, f16/f32 K/V,
mask, softcap, GQA grouping, split-K), SET_ROWS, GET_ROWS, CPY/CONT/DUP, REPEAT,
CONCAT, ARANGE, ARGMAX, IM2COL, POOL_2D, CONV_2D_DW (whcn), PAD, UPSCALE. Views are
free. Anything else fails at compile time with the op's name and every missing op
at once. **Not lowered:** PAD_REFLECT_1D (ggml-vulkan has no kernel either),
multimodal/vision RoPE, bf16 weights in the matrix-matrix path, attention sinks,
soft_max beyond 16384 columns, channels-last depthwise conv, antialiased upscale (TIPSv2's
position-embedding resize) and GLU ops. Fusion: see below.

**Results**, RTX 3080, release, validation layer off. Outputs checked against the CPU
backend (or the HF reference where there is one):

| model | native Vulkan | ggml's Vulkan | CPU |
|---|---|---|---|
| Gemma 4 E2B q8_0, 32 tokens (all 32 identical to the HF reference) | 350 ms (368 unfused) | 256 ms | 5487 ms |
| T5Gemma2 270M f16, `generate` | 68.7 ms | 54.8 ms | 345 ms |
| T5Gemma2 `caption` (vision + cross attention + decode) | same 33 tokens as CPU | | |
| TIPSv2 B/14 text f16, 2 prompts | 5.3 ms | 3.35 ms | 1782 ms |
| TIPSv2 B/14 vision f16, 448 px | 35 ms | | 17058 ms |
| YOLO11n, 640 px (same detections) | 11.8 ms | 5.6 ms | 599 ms |

Per-op correctness is `crates/tensorlisp/tests/native.rs`, the same op-family suite the
Metal and CUDA executors run: 8 of its 9 tests pass under the validation layer with
zero errors (matmul by weight type f32/f16/q4_0/q8_0/q4_K/q6_K at n = 1..40 against
exact dequantized references; norms, activations, reductions, positions;
masked/unmasked, plain and flash attention; rope, pooling, depthwise conv, argmax;
patch embedding through im2col; state and set_rows across runs and `reset_state`).
The ninth needs PAD_REFLECT_1D.

**Fusion.** `vulkan/fuse.rs` rewrites the graph before planning, for the sequences ggml's
Vulkan backend fuses and whose shaders exist in the precompiled library: `RMS_NORM` +
`MUL` by a weight (the `rms_norm` shader's `do_multiply` specialization constant) and
`RMS_NORM` + `MUL` + `ROPE` (`rms_norm_mul_rope_*`, the rows pass through shared memory).
The last node of a sequence becomes a pseudo-op (`RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE`) that
`lower.rs` maps onto the fused shader, and the earlier nodes leave `graph.nodes`, so the
planner sees the real reads (the input stays alive until the fused node, the intermediates
are never allocated) and no launch can clobber a source it still needs. A sequence fuses
only when each intermediate has exactly one reader and is not an output, and nothing but
views sits between the nodes. `TL_NATIVE_NOFUSE=1` turns it off (or a comma-separated list
of pseudo-op names), `TL_NATIVE_TRACE=1` prints the op histogram and what was formed.

| Gemma 4 E2B q8_0 decode step | launches | device time | 32-token generate |
|---|---|---|---|
| no fusion | 1281 | 6.58 ms | 368 ms |
| `RMS_NORM_MUL` (227) | 1054 | 6.27 ms | |
| + `RMS_NORM_MUL_ROPE` (40) | 1014 | 6.25 ms | 350 ms |

Tokens stay identical to transformers; `native.rs::norm_weight_rope_chains_match_unfused`
covers the chains (both rope modes, a second reader that blocks fusion, a weight narrower
than the row). Not formed: `ROPE` + `VIEW` + `SET_ROWS` (ggml's shader reads i64 row
indices, the programs use i32) and `multi_add` (the decode graph has almost no adjacent
adds).

**Profiling.** `TL_VULKAN_PROFILE=1` writes two timestamps around every dispatch (blobs carry
the shader variant name as `label`) and prints GPU time per kernel when the device closes:
`busy` is start stamp to end stamp, `gap` the time between one launch's end and the next
one's start. Each stamp pair costs about 1-2 us, so tiny kernels read high. Gemma 4 q8_0
decode, per step (32 steps): q8_0 mat-vecs 4.3 ms for 2.42 GB of weights (~565 GB/s, 74% of
the 3080's 760 GB/s), rms_norm 0.8, elementwise add/mul/gelu/scale/copies ~1.0 (about 400
launches of ~2.4 us, mostly the serial drain between dependent launches), attention 0.56,
fused norm+rope 0.19, argmax 0.33 (one 262144-wide row on a single workgroup), gaps 0.26.

**Where the rest of the gap to ggml's backend is.** The device time is already close to
the card's memory bandwidth (the mat-vecs read the q8_0 weights once per token), so fusion
buys launches, not much time. What is left is on the host: 2.0 ms per step to record the
command buffer (descriptor sets and push constants for ~1000 launches), the logits
readback, and one global barrier between dependent launches. Next steps in order of
payoff: record the command buffer once per plan and replay it (push constants that change
per run, like positions, would need an update path), skip the readback of outputs nothing
reads, lower the GLU ops so programs can use `geglu`/`swiglu` (one launch instead of
`gelu` + `mul`).

**What debugging it taught** (each is a small thing that gave silently wrong answers):

* ggml's reduction-mode array is `{shmem, subgroup, subgroup_no_shmem}` against the
  enum `{SHMEM, HYBRID, SUBGROUP}`: a one-subgroup workgroup is `_subgroup_no_shmem`,
  a four-subgroup one is `_subgroup`. Swapped, large mat-vecs produced garbage with no
  error. Found by running one Gemma layer on the CPU and on the executor and diffing
  each intermediate (`dbg.ss` pattern: copy the port's definitions, add models that
  output the intermediates).
* The flash-attention shader shuffles 16-bit values between lanes: the device needs
  `shaderSubgroupExtendedTypes` enabled; the validation layer said so.
* A readback through write-combined staging memory crawled at 350 MB/s; the readback
  staging buffer must be `HOST_CACHED` (8 ms -> 1.5 ms for YOLO's 2.8 MB output).
* ggml sizes its push-constant ranges by the host struct, which can end in fields only
  some variants declare (matmul's `padded_N`); ours follow the shader, so trailing
  words are dropped.

### Engine changes that made the Gemma step faster (host side)

Measured on Gemma's decode step with ggml's own shaders (9.1 ms before, 7.2 ms after; Tree
gets the same):

* **Replay of recorded submissions** (`vulkan/mod.rs`, `Replay`): a step that repeats the previous
  one exactly (same pipelines, buffers, offsets, push constants and grids; only the data differs)
  resubmits the recorded command buffer. The key is a flat `Vec<u64>` of everything recorded; eight
  recordings are kept; any buffer destroyed clears them (a new buffer may be given the same
  handle). Recording 900 dispatches (a descriptor set allocated and updated per launch) cost 1.7 ms
  per step, replay 0.5 ms. Off while profiling.
* **Reads folded into the submission** (`Backend::submit_reading`, `Engine::submit_reading`): the
  executor asks for its outputs and taps to be copied into host-visible memory at the end of the
  same command buffer, so reading the token costs a memcpy, not a second submit and fence wait
  (0.8 ms on this driver). Backends that do not implement it fall back to the blocking read.
* **Level scheduling** (`hazard::level_group`): launches are grouped by dependency depth, not by
  runs of program order, so independent branches (the q, k and v paths) share a group.
  `TL_SCHED_PROGRAM_ORDER=1` restores the old grouping.
* **No memory reuse in small arenas** (`plan::plan_spread`, Vulkan executor only): a dead tensor's
  bytes given to a new one make the new writer wait for the old readers, a barrier between launches
  that need none. A decode step's arena is 2 MB with reuse and 32 MB without (`TL_PLAN_REUSE=1`
  brings reuse back): 679 barrier groups became 581, 3% faster.
* `TL_NATIVE_DUMP=<file>` appends the node list of every compiled program (after fusion) to a file.

## Migration

1. `sched` + Vulkan with hand-written kernels, ggml's Vulkan shaders as blobs (done).
2. CUDA implements `Backend` (done: `cuda/backend.rs`, same conformance suite).
   Still to do: port `native_cuda` onto the engine: `Dispatch` becomes
   `LaunchDesc`, `Program` becomes a `CompiledPlan`, `Shared { weights, state }`
   becomes `Model`/`Session` spaces. Its uniform `(Op, src0..3, dst)` kernels fit
   the CUDA convention as five pointers plus one 456-byte parameter struct.
3. The native Vulkan executor (done, above) is the first model path on the engine. The same for the
   Metal and CUDA executors; ggml's Metal kernels become a `KernelSource`.
4. v2 format, per-session state, `Model::new_session`.
5. Yielding pipelines, then `tl serve`.

ggml's own backends remain behind `--device cpu|gpu` as the reference for
`tl compare` and the native tests.

## Risks

* **Continuations across the host boundary.** Mitigated by making `run` return
  a request instead of re-entering Scheme; to be prototyped before step 5.
* **Shape churn.** Plans are keyed by shape class (padded sequence lengths,
  batch in powers of two), cached per entry; the step arena is the max over
  plans in flight.
* **Fairness.** A long prefill must not starve decode steps of other sessions:
  chunked prefill in the server loop.
* **Kernel coverage.** The native executors fail on ops they lack; check the
  coverage matrix per model before a server depends on it. Gemma 4 and
  docling are not yet verified on `--device native`.
* **Training** uses ggml's backward graph and stays on the ggml path for now.
