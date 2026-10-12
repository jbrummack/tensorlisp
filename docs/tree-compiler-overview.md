# Tree: a retargetable kernel compiler written in Scheme

> Write a GPU kernel once as quoted Scheme data. Compile it to SPIR-V, PTX or MSL. Rewrite it with small passes that an agent can write and check in minutes.

Companion to `docs/overview-presentation.md`. Details: `docs/design/tree.md`.

---

## 1. What Tree is

Tree is a typed, statically shaped kernel language with explicit threads, subgroups, workgroups and memory spaces. It plays the role CUDA C plays for hand-written GPU code. The difference is that it is **plain Scheme data**, and the compiler is a set of **Scheme libraries** loaded with the stdlib at boot.

```
Scheme metaprogram (stages, fuses, specializes)
        │  quoted kernel s-expression
        ▼
   Tree IR  (typed, SIMT-aware)
        │  (tl tree opt): fold / licm / cse / dce / your passes
        ▼
   Tree IR (cleaned)
        ├──► (tl tree spirv)  SPIR-V binary    → Vulkan
        ├──► (tl tree ptx)    PTX text         → CUDA (driver JIT, no toolkit at run time)
        └──► (tl tree msl)    MSL source text  → Metal (runtime compiler)
                │
                ▼
          KernelBlob + Signature  →  backend-neutral scheduler  →  GPU
```

Key facts:

- **Real compilers, small ones.** The front end (`tree.ss`, ~700 lines) and the SPIR-V encoder (~130 lines) are Scheme. PTX is ~700 lines and MSL ~500.
- **No Rust IR is needed.** Kernels are quasiquoted data. Tiling and unrolling are ordinary list code.
- **One kernel library for every target.** `tree_kernels.ss` (~1900 lines) holds the generators: matmul, mat-vec, softmax, norms, rope, flash attention, convolutions, dequantization and more.
- **Output is a `KernelBlob` with a `Signature`.** Generated kernels sit next to precompiled ggml ones and replace graph nodes one by one.

---

## 2. The language in one screen

```scheme
(define (gelu x) ...)                       ; staging-time function that returns Tree

(kernel add-gelu
  (bindings [a (ro (array f32))] [b (ro (array f32))] [d (wo (array f32))])
  (push [n u32])
  (launch (workgroup 256) (grid (ceil-div n 256)))
  (parallel-for i n
    (store d i (gelu (+ (load a i) (load b i))))))
```

- **Types:** `f32 f16 bf16 i32 u32 i64 u64 bool`, vectors, fixed arrays, and pointers that carry an address space (`global`, `shared`, `const`, `local`).
- **Control flow:** `let`, `if`, `cond`, `for`, `while`, `set!` on locals. Calls are inlined at staging time.
- **SIMT:** thread, block and lane ids, `block-sync`, `shared` memory, subgroup reductions and shuffles, atomics.
- **Matrix units:** one intrinsic family (`cmat`, `cm-load`, `cm-muladd`, `cm-store`; `mma-s8` for int8) mapped to each target's tensor hardware.
- **Capabilities are checked at compile time.** A target that lacks an intrinsic fails by naming the kernel and the intrinsic.
- **The host Scheme is the macro layer.** Fusion is function composition and inlining a producer into a consumer is substitution. Nothing from this layer survives into the kernel.

---

## 3. How agents build passes

A kernel is data, so **a pass is a function from kernel to kernel.** The pass API is small enough for an agent to hold in context.

### Register and run

```scheme
(register-pass! 'my-pass (lambda (kernel) ... kernel))
(optimize kernel '(fold my-pass dce))     ; default pipeline: (fold licm cse dce)
```

From the environment, `TL_TREE_PASSES=fold,cse` selects passes for the native executor and `none` disables them.

### Built-in passes

| Pass | Effect |
|---|---|
| `fold` | Constant folding, algebraic identities (`x+0`, `x*1`), dead `if`/`when` arms |
| `licm` | Hoists loop-invariant arithmetic out of `for` |
| `cse` | Binds a repeated pure typed expression once |
| `dce` | Drops unread `let`/`var` bindings and their `set!`s |

### Traversal toolkit an agent composes from

| Export | Use |
|---|---|
| `map-stmt` | Rebuild a statement with a function applied to its own expressions and its nested statement lists |
| `expr-map` | Bottom-up rewrite of an expression |
| `mentions` | Which symbols an expression uses |
| `assigned-vars` | Which vars are mutated |
| `binders` | Which names are bound |
| `pure?` | Safe to reorder, share or delete? |
| `op-count` | Cheap cost metric |

### A pass in a few lines (sketch)

```scheme
;; strength-reduce: (* x 2) -> (+ x x)
(register-pass! 'strength
  (lambda (k)
    (map-body-of k
      (lambda (stmt)
        (map-stmt stmt
          (lambda (e) (expr-map (lambda (n)
                        (if (and (pair? n) (eq? (car n) '*) (equal? (caddr n) 2))
                            `(+ ,(cadr n) ,(cadr n))
                            n))
                      e))
          ...)))))
```

(The exact entry points are in `stdlib/tree_opt.ss`; this sketch is illustrative.)

### Why this suits agents

- **Tiny search space.** A pass is one function, and a rewrite is a tree transformation, not text editing.
- **Safe by construction.** `pure?` and `assigned-vars` let the pass prove a rewrite is legal. Subgroup operations, loads from writable buffers and shared-memory loads are never treated as pure.
- **A hard verification gate.** Generated code is untrusted, whoever wrote it. Every pass and kernel must clear four checks:
  1. The structural checker, plus `spirv-val` for SPIR-V or the driver JIT log for PTX.
  2. A differential test against the CPU or ggml result on random inputs, including odd sizes, under the Vulkan sync-validation layer.
  3. Dispatch timeouts, so a hung kernel cannot hang the machine.
  4. Accept only if faster, measured with per-kernel GPU timestamps (`TL_VULKAN_PROFILE=1`, `TL_METAL_PROFILE=1`).
- **Existing harnesses.** `tests/tree_opt.rs` checks each pass structurally. `tests/tree_library.rs` and `tests/tree_msl_library.rs` compile and validate every generator, with and without passes, from the shared list `tests/tree_exprs.txt`. A new pass is covered by dropping it into the pipeline.
- **Graph-level passes too.** `vulkan/fuse_tree.rs` rewrites the model graph before planning: node CSE, `FUSED_EW` elementwise regions, `FUSED_ROW` norm/add/scale chains, norm+rope, optional gate+up, and int8 producer outputs. These are the same kind of agent task one level up.
- **Honest measurement culture.** Local passes do almost nothing on Vulkan or PTX (Gemma's step is 6.11 ms without them and 6.08 ms with), because the drivers already do the same. The wins come from composition and graph fusion. The docs say so, which keeps agents from chasing false gains.
- **Below Tree (planned).** Each target gets a typed asm dialect with exact encode/decode so agents can rewrite peepholes and scheduling. SASS would be reached only by decoding `ptxas` output.

### Agent loop

```
propose pass / kernel variant
   → compile to every available target
   → structural check + differential test + validation layer
   → time on the GPU (per-kernel timestamps)
   → keep if correct and faster, else discard
```

Everything in the loop is already scriptable. Parameters such as the matmul tile (`(k:matmul 64 64 32 4 4)`) are plain arguments, so "the best configuration per shape" is a loop in Scheme.

---

## 4. Porting to a new architecture

The design keeps target knowledge in **two small, separable pieces**, so a new target is bounded work.

### The recipe

| Step | What you write | Size reference in this repo |
|---|---|---|
| 1. **Emitter** | A Scheme library that walks the already-typed, already-optimized Tree and prints the target's code. | SPIR-V ~130 lines + shared encoder, PTX ~710, MSL ~490 |
| 2. **Backend** | Implement the `sched::Backend` trait: `alloc`, `write`/`read`/`zero`, `pipeline(blob)`, `submit(groups)`, `poll`/`wait`, plus `limits()`. Declare the calling convention (`OffsetMode`). | `metal/backend.rs`, `cuda/backend.rs`, `vulkan/` |
| 3. **Caps** | State what the device has: subgroup width, matrix units, int dot, f16. Kernel selection reads these. | `metal/tree.rs` builds `Caps` for Metal |
| 4. **Glue** | A model-level file that selects the backend as a native device. | `native_metal_tree.rs`, `native_cuda_tree.rs` |

Everything else comes for free:

- The **whole kernel library** (`tree_kernels.ss`).
- The **fusion and planning layer**: hazard analysis, barrier groups, memory pools, the `Engine`/`Ticket` async API, replayed command buffers.
- The **lowering of ggml graph nodes** to kernels (`vulkan/lower`, shared across targets).
- **Test suites**: the backend-generic conformance suite, plus the shared generator lists that check every kernel on the new target.
- All **models, ports, CLI and preprocessing**. Nothing above the backend changes.

### Evidence that it ports

- **Three targets already run the same kernels.** SPIR-V first, then PTX, then MSL, each reusing the previous kernel library.
- **Metal went from nothing to the full suite quickly.** The MSL emitter emits one assigned-once local per value and lets Metal's compiler allocate registers, so no register state is kept. All of `tests/native.rs` and `native_tips.rs` match the CPU backend, including TIPSv2, T5Gemma 2, YOLO11n and PP-OCRv6.
- **Per-target differences are small and explicit.** Examples: PTX builds `exp`/`tanh` from approximate instructions plus corrections, and MSL uses the precise library compiled without fast math. Subgroups map to warps and SIMD-groups. `cmat` maps to a 2×2 grid of 8×8 `simdgroup_matrix` tiles on Metal. `dp4a` is emulated with `char4` products where no int dot exists.
- **Capability gating, not forks.** Unsupported intrinsics fail at compile time by name. A target without tensor cores still runs the scalar kernels.
- **A WGSL path exists in the scheduler** (feature `wgsl`, via naga), so WebGPU-style targets are within reach.

### What a new target looks like in practice

| Target | Path | Status |
|---|---|---|
| Vulkan (NVIDIA/AMD/Intel) | SPIR-V | Done; runs Gemma 4 E2B, T5Gemma 2, TIPSv2, YOLO11n |
| CUDA | PTX through the driver JIT | Done; Gemma 4 E2B q8_0 at 131 tokens/s on an RTX 3080 |
| Metal | MSL | Done; matches the CPU on the full native test suite |
| AMD native | LLVM IR / AMDGCN | Planned. Checked offline only, with no AMD GPU in the lab |
| WebGPU | WGSL | Possible via the scheduler's naga path |
| Others (OpenCL, SYCL, CPU SIMD) | Another emitter plus a `Backend` | The same four steps |

### Why it is fast to do

1. **Typed, structured input.** The emitter never sees untyped or unstructured code. Type inference and control flow are done once, in `tree.ss`.
2. **Optimizations are target-independent.** `fold`, `licm`, `cse` and `dce` run before any emitter, so a new target inherits clean input.
3. **The driver does the last mile.** SPIR-V, PTX and MSL are all recompiled to the real ISA by the vendor, so an emitter only needs to be correct and structurally good, not instruction-scheduled.
4. **Tests are shared.** The first day of a port is: emit `add`, then `gelu`, then `rms_norm`, and run the same conformance and differential tests as the other targets.
5. **Agents can do the porting loop.** The verification gate (structure, differential test, validation, timing) is mechanical, which is what an agent needs to iterate without a human in the loop.

> Caveat for the slides: "any architecture" means any target with a compute shading language or an LLVM backend. The three-target result is proven. AMD native is designed but not built or tested.

---

## 5. Results that show the approach works

| Result | Number |
|---|---|
| Gemma 4 E2B q8_0 decode step (RTX 3080, Vulkan) | 6.1 ms with Tree against 7.2 ms with ggml's shaders on the same engine |
| Gemma 4 E2B q8_0 on CUDA (Tree PTX) | 131 tokens/s against 36 on the NVRTC executor |
| TIPSv2 text encoder with tensor cores (Vulkan) | 9.98 ms to 4.78 ms (ggml shaders: 5.53 ms) |
| YOLO11n on Metal (Tree MSL) | 9.7 ms against 15.4 ms on the ggml-kernel executor |
| Softmax 64 × 16384 (Vulkan) | 15 µs against 252 µs for ggml's general kernel |
| Correctness | TIPSv2 text encoder on Tree: cosine 1.00000000 to the CPU on every tap; Gemma's 32 greedy tokens equal the transformers reference |

Speed comes from **specialization**: a generated kernel can be exactly as general as one use needs. Tiling, fusion and epilogues are parameters to Scheme functions, not rewrites of a C++ template.

Honest limits to show: Tree matmul on large shapes is on par with ggml's scalar kernel and needed tensor-core forms to win. Some Metal numbers are slower than ggml's hand-tuned kernels (TIPSv2 text encoder 18.2 ms vs 15.9 ms). Not every op is lowered yet.

---

## 6. Suggested slides

1. **Tree in one sentence**: kernels as Scheme data, three GPU targets.
2. **The stack diagram** (section 1).
3. **The language**: the `add-gelu` example next to the SPIR-V/PTX/MSL it produces.
4. **Staging**: fusion as function composition.
5. **Passes**: the API (section 3) and the pass sketch.
6. **The agent loop**: propose, compile, verify, time, keep.
7. **Verification gate**: the four checks.
8. **Porting recipe**: the four-step table (section 4).
9. **Three targets, one kernel library**: the status table.
10. **Results**: the numbers in section 5, with the honest limits.
11. **Roadmap**: asm layers, AMD, WebGPU, executors moved onto the shared engine.
