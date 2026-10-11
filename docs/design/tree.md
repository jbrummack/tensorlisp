# Tree: the target-agnostic kernel layer

Status: first prototype running (SPIR-V only, compiler written in Scheme); see "Prototype results" at the end. Written after the native Vulkan executor
(docs/design/scheduler.md) showed where its GPU time goes: the mat-vecs run at ~74% of the
card's bandwidth and the rest is ~700 tiny dependent launches, so the next gains come from
generating fused kernels instead of dispatching ggml's one-op shaders.

## What it is

Tree is the last layer that knows nothing about a target. It plays the role C/CUDA/SYCL play
for hand-written GPU code: a typed, first-order, statically shaped language with explicit
threads, warps, workgroups and memory spaces. Below it sit one asm dialect per target.

```
Scheme (metaprogram: stages, fuses, specializes)
   │  quoted Tree s-expr
   ▼
Tree      typed, structured, SIMT-aware        ← authoring, fusion, inlining
   │  lower (one per target)
   ├──► SPIR-V     Vulkan compute (this machine: RTX 3080)
   ├──► PTX        CUDA, JIT-loaded by the driver (this machine)
   └──► AMDGCN     via LLVM IR (no AMD hardware here: checked offline only)
        │  decode / passes / encode (later, per target, by agents)
        ▼
      target asm  peepholes, scheduling, hand edits; SASS is reachable only through ptxas output
```

Everything below Tree is entered only when profiles say the Tree level has run out of room.
The scheduler's `KernelBlob` stays the interface: a Tree kernel compiles to a blob with a
`Signature`, so generated kernels sit next to ggml's precompiled ones and replace graph nodes
(or fused groups of nodes) one by one.

## The language

A static subset of Scheme: what remains of Scheme once nothing is dynamic.

* Values: `f32 f16 bf16 i32 u32 i64 u64 bool`, vectors `(vec f32 4)`, fixed-size arrays,
  pointers into a memory space. No pairs, strings, closures or allocation at kernel run time.
* Forms: `let`, `if`, `cond`, `set!` on locals, `begin`, loops (`for` over a range, `while`),
  arithmetic, comparison, casts, math functions. Calls are first-order and always inlined or
  specialized at staging time (no recursion, no function values in the generated code).
* Types are monomorphic after staging and checked bidirectionally. A pointer type carries its
  address space: `(ptr global f32)`, `(ptr shared f32)`, `(ptr const f32)`, `(ptr local f32)`.
* The host Scheme *is* the macro/generic layer. `define-syntax`, ordinary functions and
  closures run at staging time and produce Tree; nothing about them survives into a kernel.
  This is staging in the sense of Terra or LMS: fusion is function composition in Scheme,
  inlining a producer into a consumer is substitution.

```scheme
(define (gelu x) ...)                          ; staging-time function returning Tree
(define (fuse-unary f g) (lambda (x) (g (f x))))

(kernel add-gelu
  (bindings [a (ro (array f32))] [b (ro (array f32))] [d (wo (array f32))])
  (push [n u32])
  (launch (workgroup 256) (grid (ceil-div n 256)))
  (parallel-for i n                              ; grid-stride; the thread index form is explicit too
    (store d i (gelu (+ (load a i) (load b i))))))
```

## The SIMT wrappers

CUDA-like, explicit, and the only place the machine model shows through:

* Identity: `(thread-id d) (block-id d) (block-dim d) (grid-dim d) (lane) (warp-id)`,
  `(subgroup-size)`. A warp is a subgroup; the target sets its width (32 on NVIDIA, 32/64 on
  AMD, queryable on Vulkan), and kernels may require a width or be specialized per width.
* Cooperation: `(block-sync)`, `(shared T n)` allocations, warp/subgroup shuffles and
  reductions (`(warp-reduce + x)`, `(warp-shuffle x lane)`), ballots, atomics.
* Matrix units: one intrinsic, `(mma D A B C shape)`, lowered per target (`mma.sync` on PTX,
  cooperative matrix on SPIR-V, MFMA/WMMA on AMD). Targets without it fall back or reject.
* Async/bulk copy (`cp.async` and friends) are intrinsics with a capability check, not
  hidden behind loads.
* The checker enforces what is statically checkable: barriers only in provably uniform control
  flow (conservative), address-space discipline, no out-of-range constant indices, kernel
  parameters match the declared `Signature`.

Intrinsics carry a capability requirement; a target that lacks one fails at compile time naming
the kernel and intrinsic, the same way the native executor reports unsupported ops.

## Pipeline and where it lives

* Rust owns the IR (arena of typed nodes), the s-expr parser, the checker and each target's
  lowering and encoder. They are pure functions from IR to words/text, so they unit-test
  without Scheme. Scheme builds quoted trees and calls one host function (`kernel-compile`).
  The text form of the tree is the boundary: it gives source positions in errors and keeps the
  Rust enum the single definition of what a valid kernel is.
* A tracing front end is sugar on the same IR: in `(tl tree)` operators on symbolic values
  record nodes, so `(+ (load a i) (load b i))` written in ordinary Scheme builds the tree.
* Output is a `KernelBlob` (`Code::Spirv`, `Code::Ptx`, later `Code::Cubin`, `Code::Llvm`) and
  a `Signature`; `Registry`/`KernelSource` already know how to serve it.

## Targets

| target | emit | needs | testable here |
|---|---|---|---|
| SPIR-V | binary words by our own encoder (ids, interned types, structured control flow lowered to merge blocks) | nothing | yes, `spirv-val` + the Vulkan conformance suite |
| PTX | text; loaded with `cuModuleLoadData`, which JITs PTX (no toolkit needed at run time) | the driver | yes, RTX 3080 |
| AMDGCN | LLVM IR, compiled by LLVM or ROCm's comgr | LLVM dependency | compile-only: no AMD GPU on this machine |

SPIR-V first (hardware, validator and conformance suite are all here), PTX second.

## Verification (the gate, shared by every target and every pass)

Generated code is untrusted, including code that agents write or transform:

1. Structural check in the Tree checker; `spirv-val` for SPIR-V; the driver's JIT log for PTX.
2. Differential test against a reference (the ggml shader, the CPU result) on random inputs,
   under the Vulkan sync-validation layer.
3. Dispatch timeouts (a hung kernel on Windows can trigger a driver reset).
4. Accept only if faster, measured with the per-kernel timestamps (`TL_VULKAN_PROFILE`).

Round-trip tests protect the encoders: decoding all 1976 ggml SPIR-V shaders and re-encoding
must reproduce the words; the same for every cubin `ptxas` emits once SASS is in play.

## Below Tree (later, driven by profiles)

* Each target gets an asm dialect as typed s-expr (SSA values, explicit blocks) with exact
  encode/decode, so agents can rewrite it: peepholes, scheduling, vectorizing loads. The driver
  recompiles SPIR-V/PTX to the real ISA, so only structural changes (memory access, tiling,
  fusion, launch shape) are expected to pay; instruction-level ones often will not.
* SASS is reached by decoding `ptxas` output (unknown instructions kept as opaque bits, control
  codes preserved), never by a from-scratch selector. The encodings are reverse engineered,
  per architecture (sm_86 only), and need a scheduler pass; read the CUDA EULA first.
* AMD's ISA is documented and LLVM has an assembler, so the AMD asm layer needs no reverse
  engineering, but it cannot be tested on this machine.

## Milestones

1. IR, parser, checker, SPIR-V lowering and encoder. Reproduce `add`, `gelu`, `rms_norm` as
   Tree kernels and pass the conformance suite against the ggml shaders.
2. Elementwise chain fusion from the graph (one generated kernel per chain).
3. PTX target; the same kernels run on CUDA.
4. Row kernels (`rms_norm` + `mul` + `rope` as composition), then mat-vec epilogue fusion.
5. Asm layers and the pass API for agents, SPIR-V first.

## Prototype results

The first compiler is two Scheme libraries, `(tl tree spirv)` (module builder and binary
encoder, ~120 lines) and `(tl tree)` (type inference with literal polymorphism, structured
control flow, builtins, subgroup ops, codegen, ~450 lines), loaded with the stdlib; programs
and tests call `(compile-kernel '(kernel ...))`, which returns a bytevector. The decision to
write it in Scheme instead of Rust (the earlier plan) held up: kernels are quasiquoted data,
tiling and unrolling are ordinary list code, and nothing needed a Rust-side IR yet. The
Rust side only reflects the signature from the SPIR-V (`sched/spirv.rs`), as for ggml's
shaders. Kernels live in `crates/tensorlisp/tests/tree_kernels.ss`, the harness in
`tests/tree_vulkan.rs` (`TL_VULKAN_PROFILE=1 cargo test --release -p tensorlisp --features
native-vulkan --test tree_vulkan -- --nocapture`).

Every kernel passes `spirv-val` and matches a CPU reference, including odd sizes, under the
Vulkan validation layer with sync validation (zero errors). RTX 3080, GPU time from
timestamps, ggml's precompiled `soft_max_f32` / `matmul_f32_f32` as reference:

| softmax (rows x cols) | ggml | Tree (shared reduce) | Tree (subgroup reduce) |
|---|---|---|---|
| 4096 x 1024 | 63 us | 52 us | 52 us |
| 512 x 8192 | 261 us | 95 us | 93 us |
| 64 x 16384 | 252 us | 15 us | 15 us |
| 2048 x 128 | 11 us | 18 us | 9 us |

| matmul (m x n x k) | ggml | best Tree tile | ratio |
|---|---|---|---|
| 1024 x 1024 x 1024 | 297 us | 128x64x16, 8x4 regs: 268 us | 0.90 |
| 2048 x 512 x 2048 | 584 us | 128x64x16, 8x4: 536 us | 0.92 |
| 1536 x 256 x 1536 | 441 us | 64x64x32, 4x4: 274 us | 0.62 |
| 1536 x 64 x 1536 | 95 us | 32x32x16, 2x2: 167 us | 1.75 |
| 12288 x 16 x 1536 | 136 us | 64x16x16, 4x1: 295 us | 2.17 |
| 4096 x 32 x 4096 | 216 us | 64x64x32, 4x4: 455 us | 2.11 |

The one-thread-per-output matmul is 10-15x slower than the tiled one. Reading the table:
a ~35 line staged tiled matmul (shared tiles with a padded stride, register blocking, bounds
guards) reaches ggml's scalar kernel on large shapes, and the tile is a parameter, so the best
configuration per shape is a loop in Scheme. Where ggml still wins (skinny n) it uses split-K
and smaller tiles to fill the 68 SMs; the Tree kernel has neither yet. The softmax gains come
from ggml's kernel being general (masks, ALiBi, sinks, up to 16384 columns) while the Tree
one is specialized; that is the point of generating kernels per use.

Not done yet: the uniformity check for barriers, `f16`, 2D/3D grids beyond what `global-id`
gives, vector loads (`vec4`), a Tree-level lowering of graph nodes (so the executor can use
these), the asm layer, PTX.

## A whole model on Tree kernels: TIPSv2 text encoder

`TL_NATIVE_TREE=1` makes the native Vulkan executor lower every node that has a Tree generator
(`stdlib/tree_kernels.ss`) to a kernel compiled by the Scheme compiler, and every other node to
ggml's shader as before; `TL_NATIVE_TREE=only` fails on nodes without one. The Rust side
(`vulkan/lower/tree_ops.rs`) packs push constants in the order each generator declares them and
asks Scheme for the SPIR-V by expression (`(k:matmul 64 64 32 4 4)`), cached per expression. All
generated kernels are `(offsets prefix)`: the backend passes each binding's residual byte offset
as leading push words and the kernel adds it to its indices, so views of any alignment work.

Generators: strided 4-D binary ops with broadcast, unary maps and copies (including reshaping
copies), repeat, concat, arange, row gather, layer norm, masked softmax, row sum/mean, and a
batched tiled f32 matmul with broadcast weights (~100 lines of Scheme in total for all of them,
the matmul being a third). f16/quantized weights, rope, flash attention, convolutions and the
rest fall back to ggml's shaders.

All 336 nodes of the TIPSv2 text encoder (f32 weights, 3 x 64 tokens, 12 layers) run on Tree
kernels with no fallback, and its output matches the CPU backend on every tap (cosine to the
CPU: 1.00000000, against 0.99999979 for ggml's shaders). With Tree lowering on, the op-family
suite (`tests/native.rs`), T5Gemma2, YOLO11n and PP-OCR still match the CPU under the validation
layer; the only failures are the two ops nobody lowers (reflect pad, antialiased upscale).

| TIPSv2 text, per run | ggml's shaders | Tree only |
|---|---|---|
| wall time | 5.9 ms | 12.0 ms |
| GPU: matmul (73 launches) | 4.0 ms | 9.9 ms |
| GPU: everything else (263 launches) | 1.2 ms | 1.3 ms |

Everything except the matmul is on par (Tree's layer norm is faster, its masked softmax slower).
The matmul is 2.5x slower on these shapes (m up to 3072, n = 192, k up to 3072). No tile shape of
the generator beats 64x64x32 with 4x4 registers per thread (12 other shapes tried, including
many outputs per thread and tiles that fill the GPU better), so the limit is the kernel
structure: scalar global and shared loads. ggml's kernel loads `vec4`, double buffers the tiles
and uses split-K. The next steps for Tree are therefore language features, not tuning: vector
types and aliased vec4 views of a buffer, then a way to pipeline loads with compute.
Merging a batch over shared weights into the column count (what ggml does) took the model from
15.6 to 12.0 ms.

## The Gemmas on Tree kernels

`TL_NATIVE_TREE=only` runs Gemma 4 E2B (q8_0) and T5Gemma2 (f16) with no node left to ggml's
shaders; Gemma's 32 greedy tokens equal the transformers reference
(`native_gemma4::greedy_tokens_match_transformers`) and T5Gemma2's text equals the CPU backend's
(`native_tips::t5gemma2_...`). What it took, on top of the TIPSv2 kernels:

* **Language:** 16-bit storage (`u16` buffer elements, `half->f32` / `f32->half` through
  `UnpackHalf2x16` / `PackHalf2x16`, so no Float16 arithmetic is needed) and bit-field
  extraction (`sbits`, `ubits`) for q8_0's int8 quants; f16 and q8_0 data are `u16` arrays, a
  q8_0 block being 17 words (scale + 32 quants).
* **Generators:** matmul and gather with f16 / q8_0 weights (dequantized while filling the shared
  tile), a mat-vec for up to 8 columns, rms_norm (with its weight multiply), rope (normal and
  neox, frequency factors, YaRN), set_rows into an f16 cache, typed copies (f32 / f16 / i32),
  argmax, step/sgn/floor/ceil, and flash attention (f16 K/V, mask, softcap, GQA). All 33
  generator variants are compiled and checked by `spirv-val` in `tests/tree_library.rs`.

Gemma 4 E2B q8_0, GPU time per decode step on the RTX 3080 and the 32-token `generate`:

| | device time / step | generate |
|---|---|---|
| ggml's shaders (with fusion) | 6.25 ms | 337 ms |
| Tree, first version | 27.6 ms | 1101 ms |
| + flash attention skips fully masked keys and bounds its loops by the last valid one | 12.6 ms | 607 ms |
| + mat-vec: a subgroup takes 4 rows and shares the activation loads | 7.9 ms | 408 ms |

Two findings. Flash attention went from 442 us to 8.9 us per call (ggml: ~10 us) by not doing
work for the ~2000 masked-out cache rows of a decode step: the mask is already in memory, the
kernel only has to look at it first. The mat-vec was bound by re-reading the activation vector
once per row (4x the traffic of the weights); sharing one load across several rows took it from
38.9 us to 21.5 us per launch (ggml's: ~15 us). What remains: the mat-vec's per-chunk instruction
count (u16 loads for q8_0 blocks, scalar loads of the activations), the weighted rms_norm as its
own launch (ggml fuses it into rope), and the prefill mat-vec for 6 columns, which has no row
sharing.

## PTX target

`(tl tree ptx)` (`stdlib/tree_ptx.ss`) compiles the same kernels to PTX text; the CUDA backend
loads `Code::Ptx` blobs through the driver JIT (`Device::library_ptx`), no NVRTC or toolkit
needed at run time. Calling convention: one pointer per buffer, then one by-value struct of the
push words (the `(offsets prefix)` words are dropped: CUDA binds exact addresses). Registers are
typed and mutable, subgroup reductions are warp shuffles, exp/tanh/sin/cos are built from the
approximate hardware instructions with corrections. `tests/tree_ptx.rs`: all 33 generator
variants load; add, gelu, both softmaxes and the matmuls match the CPU with the same errors as
the SPIR-V build. RTX 3080, 200 launches per timing: softmax 4096x1024 64 us (Vulkan 52),
matmul 1024^3 128x128 tile 9.0 TFLOP/s (Vulkan 10.3). A short timing run under-reports (the GPU
had not reached full clocks after 20 launches). Findings: unrolled, independent tile loads
(staged in the generator) gave +40% on PTX and +30% on Vulkan; the `guard` form compiles to a
branch on PTX (predication was slower). Not done: wiring Tree kernels into the CUDA native
executor (the kernels run through the scheduler only), a full-model run on CUDA.

### Tree kernels in the native CUDA executor (`native-cuda-tree`)

`--features native-cuda-tree` (on `tensorlisp` / `tensorlisp-cli`, implies `native-cuda`, builds `tensorlisp-kernels` with
`vulkan` for the shared lowering; ash is dlopen'd, no SDK needed) runs `--device native` through the Vulkan executor's lowering
with PTX instead of SPIR-V: `vulkan::exec::Executor<B>` and `Program<B>` are generic over the scheduler backend,
`cuda::tree::executor` describes the device (`Cuda::tree_caps`: 32-wide warps, dp4a, `mma.sync` standing in for cooperative
matrices on sm_80+), `TreeKernels::from_blobs` takes a compiler that returns whole blobs (`compile-kernel-ptx`, raw offsets), and
`src/native_cuda_tree.rs` is the glue. Every node must have a Tree kernel (`only` mode; the rest fail by name: IM2COL, PAD,
POOL_2D, CONV_2D_DW, some UNARY, UPSCALE, odd MUL_MAT types). Checked on the RTX 3080: TIPSv2 text, PP-OCRv6 det + rec, T5Gemma2
text generation (same text as the CPU backend), Gemma 4 E2B q8_0 (32 greedy tokens equal to transformers), batched dense and
paged decoding of T5Gemma2; Gemma 4 paged differs from dense/sequential at one near-tie token in one of four sequences.
Gemma 4 E2B q8_0 aggregate tokens/s, 6-token prompts (the NVRTC executor / Tree on PTX): 1 sequence 36 / 131 (sequential
decode-step), batched dense 1: 37 / 140, 4: 90 / 517, 8: 110 / 737, 16: 186 / 937, 32: 347 / 1118, 64: 635 / 945.
The last three used to be 67, 350 and 689 on Tree: with more than 8 columns a q8_0 product went to the tensor-core matmul, which
dequantizes the weights inside the tile and sat at ~85 ms a step whatever the batch (the weights are read at a fraction of the
bandwidth). The dp4a mat-vec is generic in the column count and now takes up to 64 columns (`TL_TREE_DOT_MAXN`, default 64):
16 sequences 17 ms a step, 32 28 ms, 64 68 ms (against 93 ms for the tensor-core path). Beyond 64 columns (prefill) the matmul
is compute-bound and the tensor cores stay. `native.rs::paged_attention_matches_dense` checks the paged kernels against dense
attention on the same K/V (hs 64..512, grouped heads, shuffled blocks, unequal lengths): 2.5e-4 on PTX, 7e-4 on Vulkan.

PTX tuning that followed (Gemma 4 E2B q8_0, RTX 3080 at a 300 W cap; `TL_CUDA_PROFILE=1` prints a per-kernel table
when the backend drops, `TL_CUDA_GRAPHS=0` and `TL_TREE_NOMMA=1` switch the first and last item off):

* **CUDA graphs.** `cuda::backend` captures a submission that equals the previous one (same kernels, buffers, push
  words, grids) into a graph and replays it with one `cuGraphLaunch`; single-sequence decode 7.2 -> 5.7 ms.
* **Two-pass argmax** for rows of 16384 or more (`k:argmax-part` / `k:argmax-final`): the logits argmax took 300 us on one
  workgroup. Ties still resolve to the first maximum (ggml's CPU backend picks the last).
* **dp4a mat-vec rows by column count**: 1 row per subgroup up to 4 columns, 4 up to 16, 2 above. 64 sequences 66 -> 41 ms.
* **Int8 tensor cores for prefill** (`mma-s8` statement, `k:matmul-q8mma`, `Caps::int_mma`): a q8_0 product with more than 64
  columns quantizes the activations to int8 (scratch) and runs `mma.sync.m16n8k32.s8`; one q8_0 block is one k step, scaled
  per block into an f32 accumulator. 12288x406x1536: 0.65 ms against 1.59 ms for the f16 tile matmul (21 against 10 T-ops/s);
  the long-prompt sweep's GPU time went from 72.5 s to 46 s. Ablation of the kernel: without global loads it runs at ~70
  T-ops/s, without the mma the loads alone cost half the time, so the remaining cost is staging the 34-byte q8_0 blocks (two
  16-bit loads per word). Next step there: 16-byte loads of raw row chunks (8 blocks, k % 256 == 0) with the realignment done
  at fragment load. ncu is installed but the driver denies counter access (ERR_NVGPUCTRPERM) without admin rights.
* Single-sequence decode is now 5.2 ms a step (190 tokens/s, 36 on the NVRTC executor). Left in the profile: ~270 launches
  of the mat-vec at ~60% of the weights' bandwidth bound, prefill flash attention (~20% of a long-prompt run).

Bug found on the way: the PTX `exp` returned NaN for -inf (its error-correction term is inf - inf), which only shows with masked
keys *inside* a valid range (masked suffixes are never evaluated); fixed, regression test `attention_with_large_head_sizes`.

## Learning from ggml-cuda's PTX

Method: `scripts/ggml-cuda-ptx.bat mmvq softmax norm rope mmf template-instances/mmq-instance-q8_0
template-instances/fattn-vec-instance-f16-f16` compiles the industry kernels to PTX (sm_86) and
the instruction mix says what they rely on:

| kernel | in the PTX | Tree before |
|---|---|---|
| `mul_mat_vec_q` (q8_0 mat-vec) | `dp4a` (13k), activations pre-quantized to int8 (`ld.v2`), 32-bit weight words built from two `u16` loads | dequantize every weight to f32 (bfe + cvt + FMA each) |
| `mmq` (quantized GEMM) | `mma.sync.m16n8k32.s32.s8.s8` int8 tensor cores, `ldmatrix`, `ld.shared.v4` | no tensor cores |
| flash-attn vec | `ld.global.nc.v4` (683), `st.shared.v4`, many `shfl.sync`, `ex2.approx.ftz` | scalar loads |
| rope | `st.global.v2` for the rotated pairs | scalar stores |
| softmax, rms_norm | `ex2.approx.ftz`, `shfl.sync` butterflies, `bar.sync` | the same |

**Applied: `dp4a`.** The language gained `(dp4a a b acc)` (PTX `dp4a.s32.s32`, SPIR-V `OpSDot`
from `SPV_KHR_integer_dot_product`, which the Vulkan device now enables when the driver has it:
`Caps::int_dot`), `(shuffle-xor v mask)`, and generators `k:quantize-q8` (activations to int8
blocks: eight lanes share a block, a three-step butterfly finds its amax) and `k:matvec-q8`
(weights read as 16-bit pairs like ggml, four `dp4a` per 16 weights). Results, RTX 3080:

* PTX, weight bandwidth of the q8_0 mat-vec (GB/s, f32-dequant path -> dp4a path): 2048x1536
  310 -> 444, 6144x1536 476 -> 607, 12288x1536 536 -> 647, 1536x12288 285 -> 477; on the
  DRAM-sized logits matrix (262144x1536, 428 MB) both are at 690-780 GB/s, ~90-100% of the 760 GB/s
  peak, so the gain is on the medium matrices that dominate a decode step. A plain streaming read
  probe reaches 600-640 GB/s through Tree/PTX.
* Vulkan, Gemma 4 E2B q8_0 on Tree kernels (`TL_NATIVE_TREE=only TL_TREE_DOT=1`): device time per
  step 7.83 -> 7.23 ms even with 277 extra quantize launches, 32-token generate 414 -> 363 ms; all 32
  tokens still equal transformers. Activation quantization changes results by ~0.5% (ggml's CPU and
  CUDA backends do the same), so it is opt-in (`TL_TREE_DOT=1`), and the exact-reference matmul tests
  stay exact by default.

**Tried and dropped** (measured, no consistent win): repacking q8_0 weights into aligned words +
`ld.v4` (`k:repack-q8`, `k:matvec-q8r`, `vload`: equal results, speed within noise and not
systematically better), unrolling constant-trip loops in the PTX emitter, predicating guarded loads
instead of branching (slower). **Measurement lesson:** a timing that re-uploads its
inputs per submit hides the kernel: the harness now times a plan of 2N launches minus a plan of N,
which cancels the upload (the earlier "234 GB/s" for the logits mat-vec was upload time).

**Not done yet, in the order I would do them:** quantize each distinct activation once instead of
per mat-vec (Gemma's q/k/v and gate/up share inputs; needs scratch that lives past one node in the
planner); `ld.shared.v4` with transposed tiles for the f32 matmul; int8 `mma.sync` (`mmq`) and a
tf32/f16 `mma` intrinsic; `cp.async` double buffering; packed `half2` math in attention.

## Passes, fusion, and the Gemma decode step against ggml

The compiler (`(tl tree)`, `(tl tree spirv)`, `(tl tree ptx)`) translates a kernel in one go; what
makes the code better than that translation happens before it, as Tree-to-Tree functions and as
rewrites of the model graph.

### Passes (`stdlib/tree_opt.ss`, `(tl tree opt)`)

A pass is a function from a kernel to a kernel, registered by name:

```scheme
(register-pass! 'my-pass (lambda (kernel) ... kernel))
(optimize kernel '(fold my-pass dce))          ; default (fold licm cse dce)
```

| pass | what it does |
|---|---|
| `fold` | constant folding, `x+0`, `x*1`, `select` of a literal, dead `when`/`if` arms. An integer division of two literals folds only when exact: a literal takes its type from its context. |
| `licm` | loop-invariant arithmetic (no loads: a load may sit behind a bounds check) is hoisted out of `for` |
| `cse` | a pure, *typed* expression computed more than once is bound once; reuse stops at an assignment to a variable it mentions or a shadowing binder; names used only once are inlined back (a name is kept when its use is in a deeper loop). Loads count as pure only from `ro` buffers. |
| `dce` | unread `let`/`var` bindings and the `set!`s of an unread var go |

The traversals the passes are built from are exported (`map-stmt`, `expr-map`, `mentions`,
`assigned-vars`, `binders`, `pure?`, `op-count`), so an agent writes a pass in a few lines.
`TL_TREE_PASSES=fold,cse` selects passes for the native executor (`none` turns them off).
`tests/tree_opt.rs` checks each pass structurally and that every generator survives the pipeline.

**Measured honestly: they do almost nothing on Vulkan or PTX.** Gemma's decode step is 6.11 ms
without them and 6.08 ms with them; the SPIR-V of `matvec q8_0` shrinks 12% (30.2 to 26.5 KB). The
drivers' own compilers already do constant folding, CSE and LICM. They are the substrate for
what does matter: kernels that are *composed* by Scheme code (below) contain duplicated
subexpressions, and an agent that works on the asm layer later wants clean input.

### Fusion: what the executor rewrites before planning (`vulkan/fuse_tree.rs`)

Run when `TL_NATIVE_TREE` is on (`TL_TREE_NOFUSE=1` turns all of it off):

* **`cse`**: nodes that compute the same thing from the same inputs are computed once. Gemma
  converts one attention mask to f16 in every layer: 35 copies become one (75 nodes per step).
* **`elementwise`**: a tree of elementwise nodes over dense f32 tensors (add, mul, scale, unary
  functions, copies to f16, a repeated row or scalar, `arange`) becomes one `FUSED_EW` node whose
  kernel (`k:ew`) evaluates the whole expression per element. A node is absorbed into its consumer
  when it has exactly that consumer and is not an output. The expression is built from the nodes and
  named locally, so the 35 layers share one compiled kernel. The 12-node attention-mask chain
  becomes two kernels.
* **`row_chains`**: an `RMS_NORM` followed by any run of elementwise nodes (residual add, scale)
  and further norms over the same rows becomes one `FUSED_ROW` node (`k:rowchain`): a workgroup per
  row, the row in registers, the stages applied in turn, and any stage's value that something else
  still reads is written out as well (a tap). Gemma's `post-attn norm, add residual, pre-ffn norm`
  (three launches), `post-ffn norm, add` and the layer end `norm, add, scale, norm` are one
  launch each. A chain may only grow while every outside reader of a tapped value runs after the
  stage being added, and its other inputs must come from before the chain starts.
* **`mark_q8_outputs`**: when every reader of a fused node's result is a dp4a mat-vec, the node
  writes int8 blocks (`xq`, `xd`) instead of f32, inside its own allocation (4 bytes per element is
  more than the 1.125 an int8 block needs). No f32 copy, no separate quantize launch, and one
  quantization shared by all the readers (q, k, v read one). When some readers want the f32 values
  too, a sidecar tensor holds the blocks and both are written. Norms, `k:ew`, the flash-attention
  merge and row chains can all produce them.

The graph keeps its tensor ids: the last node of a region keeps its tensor and is rewritten in
place, the absorbed ones leave `graph.nodes`, and a tap or sidecar becomes an extra source of the
fused node so the planner allocates it from the fused node's position.

### Split-KV flash attention (`k:flash-attn-part` + `k:flash-attn-combine`)

One query over a long cache gives only `heads` workgroups (8 on a 68-SM GPU). Pass one takes a
workgroup per (128-key chunk, head), pass two merges the chunks. A chunk with no unmasked key does
nothing, so Gemma's 2048-slot cache at position 10 costs one chunk's work. Head sizes are
compile-time: each lane keeps its slice of the query in registers, four keys' loads are issued
before the first multiply, the value sum takes eight keys at a time (a loop with a dynamic trip
count waits for each load in turn): 14.2 to 9.5 us at position 600 (`tests/tree_attention.rs`).

### Results: Gemma 4 E2B q8_0, one decode step, RTX 3080, wall time (`tests/tree_gemma.rs`)

| | position 10 | position 600 |
|---|---|---|
| ggml's Vulkan shaders, before this work (engine as it was) | 9.1 ms | 9.1 ms |
| ggml's shaders on the engine now | 7.2 ms | 7.2 ms |
| Tree kernels | **6.1 ms** | **6.3 ms** |
| Tree, exact f32 mat-vec (`TL_TREE_DOT=0`) | 7.7 ms | 7.9 ms |
| Tree without graph fusion (`TL_TREE_NOFUSE=1`) | 7.3 ms | 7.6 ms |
| Tree without split-KV attention | 6.2 ms | 10.2 ms |
| Tree, arena memory reused (`TL_PLAN_REUSE=1`) | 6.3 ms | 6.5 ms |

Greedy tokens still equal the transformers reference. Two things to know about the comparison:

* **The mat-vec quantizes its activations to int8 by default now** (`TL_TREE_DOT=0` keeps f32).
  That is what ggml's CPU and CUDA backends do; it changes results by about 0.5% of the scale
  of a mat-vec, and it is where much of the margin over ggml's *Vulkan* shaders comes from
  (they keep f32 activations): with exact f32 activations Tree is 7% *slower* than ggml.
* GPU time per step (timestamps): Tree 5.7 ms, ggml 7.1 ms. The rest of the wall time is the host
  (see "Engine changes" in scheduler.md): 1.7 ms to record 900 dispatches and 0.8 ms to read one
  token back, both now close to zero.

What the profile says is left (Tree, per step): the q8_0 mat-vecs are 4.0 of 5.7 ms, at ~600 GB/s
of the 760 GB/s the card has; 683 launches in 546 barrier groups; norm + rope + `set_rows` of the
K path are still 3 launches, gate/up/gelu-mul 3 (a mat-vec pair with the gelu-mul as epilogue, a
32-row workgroup producing exactly one int8 block, would be 1).

### Two more fusions that did not move the step (measured)

* **`norm_rope`** (`fuse_tree.rs`, `k:norm-rope`, node `FUSED_NR`, on by default): `RMS_NORM[_MUL]`,
  then an optional `ROPE`, then an optional single-row `SET_ROWS` into the KV cache, one workgroup per
  row with the normalized row in shared memory for the rotation. 65 chains per Gemma step, launches
  684 to 604. Step time 6.08 to 6.00 ms at position 10 (about 1-2%, near the noise).
* **`gate_up`** (`k:gateup-q8`, node `FUSED_GU`, opt-in `TL_TREE_GU=1`; `TL_TREE_GU_ROWS`,
  `TL_TREE_GU_UNROLL`): `gelu(W_gate x) * (W_up x)` in one launch writing int8 blocks (a workgroup of
  32/R subgroups owns 32 rows = one block; needs subgroup size 32). 35 pairs, launches 604 to 534,
  and no change in time for any R/U tried (6.00 ms either way). Off by default.

Conclusion: with the mat-vecs at ~80% of the card's bandwidth, removing launches no longer pays on
this driver; the next gains are in the f32 mat-vec path and in prefill (matrix products).

## Tensor cores: cooperative matrices (`k:matmul-cm`)

Tree has `f16` as a storage/matrix type and cooperative-matrix forms (SPIR-V target only, for now):

```scheme
(cmat acc c f32) (cmat fa a f16) (cmat fb b f16)      ; 16x16 matrices, uses a | b | c
(cm-zero acc)
(cm-load fa buf idx stride row|col)                   ; buf: a buffer or shared array
(cm-muladd acc fa fb acc)                             ; acc = fa * fb + acc
(cm-store acc buf idx stride row|col)  (cm-convert dst src)
```

They compile to `VK_KHR_cooperative_matrix` (capability CooperativeMatrixKHR, which needs the Vulkan
memory model, so such modules declare it). The optimizer leaves kernels with `cmat` alone. The device
enables the extension when it lists f16 x f16 + f32 16x16x16 at subgroup scope (`Caps.coopmat`;
`TL_VULKAN_NOCOOPMAT=1` turns it off).

`k:matmul-cm atype BM BN BK WM WN` has the buffers and push words of `k:matmul` (f32, f16 and q8_0
weights, batching, edges): shared f16 tiles filled like the scalar kernel (dequantizing while
filling), WM x WN subgroups with (BM/WM/16) x (BN/WN/16) accumulators each, results through shared
memory to the destination. The native executor uses it for products with 16 or more columns
(`TL_TREE_NOCM=1` keeps the exact f32 kernel, `TL_TREE_CM="bm bn bk wm wn"` forces a tile): 64x64x16
(2x2 subgroups) when there are at least ~2.25 tiles per SM, 32x32x32 otherwise (an n = 192 product
of TIPSv2 has only 36 64x64 tiles for 68 SMs), 64x16 under 32 columns.

Measured on the RTX 3080 (GPU microseconds per launch, `tests/tree_matmul_cm.rs`):

| m x n x k | scalar tiled | tensor cores |
|---|---|---|
| 768 x 192 x 768 | 78 | 29 (32x32x32) |
| 3072 x 192 x 768 | 205 | 58 |
| 768 x 192 x 3072 | 292 | 128 |
| 768 x 1024 x 768 | 187 | 62 |
| 3072 x 1024 x 768 | 694 | 168 (29 TFLOPS) |

TIPSv2 text encoder, whole model on Tree kernels: 9.98 ms to 4.78 ms per run (ggml's shaders 5.53 ms),
cosine to the CPU backend 0.9999998 (the inputs of the products are rounded to f16). Gemma's decode
step is unchanged (its products have one column). The PTX target has no equivalent yet (`mma.sync`).
