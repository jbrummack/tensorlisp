# tensorlisp: Overview

> Neural network models written as small Scheme programs, run on CPU, Metal, CUDA and Vulkan through one runtime, with a self-hosted GPU kernel compiler.

---

## 1. The Idea

- A model is a **program**: a Scheme (Chez Scheme) source file stored inside a **GGUF** file, next to its weights.
- The program builds a ggml compute graph, declares its preprocessing and postprocessing, and can hold state such as a KV cache.
- One runtime (`Model::load` / `Model::run`, or the `tl` CLI) executes it on many backends.
- The goal is portable, inspectable, scriptable model deployment with no Python at inference time.

**GGUF keys:** `TL_TXT` (UTF-8 program), `TL_BIN` (binary, planned), `TL_VER` (version), and `TL_ASSET.<name>` for bundled assets.

---

## 2. Architecture at a Glance

| Layer | Crate / location | What it does |
|---|---|---|
| Scheme engine | `crates/chez` | Embeds Chez Scheme (bindgen'd `scheme.h`, boot files compiled in) |
| Tensor backend | `crates/ggml-sys` | Vendored ggml built with `cc` only (no cmake). Backends: CPU, Metal, CUDA, Vulkan |
| FFI generator | `crates/ggml-codegen` | Generates the Scheme FFI: raw foreign procedures, implicit-context `ggml-*` ops, enum constants |
| Runtime | `crates/tensorlisp` | Model loading and running, Scheme on a dedicated thread, standard library |
| CLI | `crates/tensorlisp-cli` (`tl`) | inspect, convert, pack, check, run, compare, quantize (all with `--json`) |
| Kernel layer | `crates/kernels` | ggml-independent kernel launcher, native executors, async scheduler, Tree compiler targets |
| Importers | `crates/importers` | ONNX and CoreML readers |
| Pre/postprocessing | `../autopro` (sibling repo) | Images, audio, text, detection, clustering, OCR |

---

## 3. The Programming Model

- **Standard library:** R6RS libraries `(tl tensor)`, `(tl nn)`, `(tl attn)`, `(tl vision)` and `(tl util)`. A program starts with `(import (tl nn))` and gets canonical prefixes such as `nn:linear` and `attn:sdpa`.
- **Convolutions:** non-square kernels, `(h w)` strides and padding, banded im2col for large inputs, and `nn:conv-transpose2d`.
- **Multiple entry points:** `(model name ...)` declares several graphs in one program. `(pipeline name ...)` chains them from Scheme with `(run entry ...)`. The CLI selects one with `--entry`.
- **Stateful models:** `(define-state ...)`, `(state name)` and `(effect t)` give KV caches and other state tensors. They live on the primary device, and `Model::reset_state` clears them.
- **Device-aware programs:** `(device)` returns `cpu` or `gpu`, so a program can pick kernels per device (for example im2col convs on Metal and direct convs on CPU).
- **Pre- and postprocessing in the program:** `(preprocess ...)` and `(postprocess ... (results ...))`, backed by Rust host functions.
- **Error recovery:** ggml asserts are caught and turned into recoverable errors, both at graph build and on the calling thread in Rust.

---

## 4. Ported Models (the proof it works)

| Port | Model | Notes |
|---|---|---|
| `ports/tipsv2` | TIPSv2 B/14 text + vision encoders | Matches PyTorch |
| `ports/t5gemma2` | T5Gemma 2 270M-270M | Image captioning and text generation, greedy decoding, KV cache as state tensors |
| `ports/yolo11` | YOLO11n detector | Ultralytics-style NMS postprocessing, matches the reference |
| `ports/ppocrv6` | PP-OCRv6 medium detection + recognition | Two entries in one GGUF with `ocr` and `recognize` pipelines. Matches transformers 5.17 |
| Gemma 4 E2B | LLM, q8_0 and f16 | Runs on ggml Vulkan and on the native executors. Output identical to the HF reference |

Every port is validated against its original framework (PyTorch, Ultralytics, transformers) using reproducible reference environments (Nix flakes).

---

## 5. Pre- and Postprocessing (`autopro`)

- **Images:** Pillow-exact resize.
- **Text:** native SentencePiece, `tokenizer.json`.
- **Audio:** Whisper mel spectrograms.
- **Detection:** torchvision NMS, Ultralytics detector NMS, letterbox.
- **Clustering:** sklearn-matching DBSCAN (Accelerate sgemm via `platform-blas`).
- **OCR:** bit-exact ports of OpenCV's `findContours`, `minAreaRect`, `fillPoly` and `warpPerspective`, plus CTC greedy decoding and vocabularies.
- All of it is validated against the Python originals and exposed to programs as host functions.

---

## 6. The `tl` CLI

| Command | Purpose |
|---|---|
| `tl inspect` | Show program, tensors, graph info |
| `tl convert` | Convert weights into GGUF (`--replace-prefix`, `--offset GLOB=V`) |
| `tl pack` | Bundle program, weights and assets into one GGUF |
| `tl check` | Validate a program |
| `tl run` | Run a model and show results (`-i` inputs, `--entry`, `--no-post`, `--device`) |
| `tl compare` | Compare outputs between backends or references |
| `tl quantize` | Graph-aware quantization (needs `-i` shapes). Works across all entries (`ENTRY:NAME=`) |

All commands support `--json` for scripting.

---

## 7. Backends and Devices

| Backend | Status |
|---|---|
| CPU | Reference path |
| Metal (macOS) | Default GPU on Apple silicon (ggml-Metal) |
| CUDA | ggml-cuda vendored and built with `nvcc` through `cc` (feature `cuda`) |
| Vulkan | ggml's own Vulkan backend (feature `vulkan`) |
| **Native executors** | `--device native`: our own graph, plan and dispatch pipeline per GPU API (see below) |

### Native executors

The ggml cgraph is turned into an IR, planned into an arena, and run as a recorded dispatch list:

- **Metal native:** over ggml's own `.metal` kernels. TIPSv2 runs on it.
- **CUDA native:** own NVRTC kernels. Matches ggml CPU on TIPSv2, T5Gemma 2, YOLO11n and PP-OCRv6.
- **Vulkan native:** lowers to ggml's precompiled SPIR-V shaders on the shared scheduler engine. Gemma 4 E2B, T5Gemma 2, TIPSv2 and YOLO11n run on it.
- **Per-input-shape caching** of compiled native programs.
- Ops that a native executor lacks fail at compile time, by name.

---

## 8. Kernel Launcher and Scheduler (`crates/kernels`)

- Paged attention, `reshape_and_cache`, `gather_kv_cache` and `copy_blocks`, ported from vLLM via mistral.rs, on Metal and on CUDA (NVRTC, with `libloading` so no toolkit is needed to build).
- **Backend-neutral async scheduler:** kernel blobs with signatures, hazard analysis that produces barrier groups, memory spaces (`Model` / `Session` / `Step`) with size-class pools, and `Engine::submit` returning a `Ticket`.
- Engine optimizations: replayed command buffers, reads folded into the submission, level-scheduled groups.
- **Vulkan validation:** a conformance suite runs under the validation layer, and a test proves a missing barrier is caught.
- Shader tooling: SPIR-V reflection, GLSL compiled with `glslc` or glslang, WGSL via naga, and a script that builds all 1976 ggml Vulkan shader variants.

---

## 9. Tree: a Kernel Language Written in Scheme

The headline research piece. GPU kernels are written in a small kernel language, compiled by **Scheme libraries** (`(tl tree)`, `(tl tree spirv)`) into multiple targets:

| Target | Library |
|---|---|
| SPIR-V (Vulkan) | `tree_spirv` |
| PTX (CUDA, via the driver JIT) | `tree_ptx.ss` |
| MSL (Metal) | `tree_msl.ss` |

Features:

- Tree-to-Tree optimization passes (`(tl tree opt)`: fold, licm, cse, dce, user-registered passes).
- Graph fusion before planning: node CSE, fused elementwise regions, fused norm/add/scale chains, norm+rope, and int8 producer outputs.
- Split-KV flash attention.
- dp4a int8 q8_0 mat-vec, plus int8 tensor-core matmul (`mma.sync`) on PTX.
- Cooperative-matrix (tensor core) forms on SPIR-V: `cmat`, `cm-load`, `cm-muladd`, `cm-store`, `k:matmul-cm`.
- Dequantization kernels for q4, q5, q4_K and q6_K, plus conv, pool, pad and upscale kernels.
- Profiling and debugging: `TL_VULKAN_PROFILE=1` (per-kernel GPU time), `TL_NATIVE_TRACE`, `TL_NATIVE_DUMP`, and `tests/tree_library.rs`, which compiles and validates every generator.

### Highlights (measured, RTX 3080 unless noted)

- **Gemma 4 E2B decode step:** 6.1 ms with Tree, against 7.2 ms with ggml's shaders on the same engine (9.1 ms before the engine work).
- **Gemma 4 E2B q8_0 single-sequence decode on CUDA:** 131 tokens/s with Tree on PTX, against 36 on the NVRTC executor.
- **TIPSv2 text encoder with tensor cores:** 9.98 ms down to 4.78 ms (ggml shaders: 5.53 ms).
- **YOLO11n on Metal:** 9.7 ms with Tree MSL against 15.4 ms on the ggml-kernel executor.
- Tree runs whole models (TIPSv2 text, Gemma 4 E2B q8_0, T5Gemma 2 f16) with outputs matching the references.

---

## 10. Importers and Formats

- ONNX and CoreML readers (`crates/importers`).
- `crates/tensorlisp-aot` for CoreML and ANE (macOS only).
- `experiments/` archives earlier attempts: hand-written Lisps and a first GGUF loader.

---

## 11. Engineering Practices

- Cargo workspace, vendored dependencies (ggml, Chez Scheme submodule), and a build with no cmake.
- Every port and every backend is tested against a reference (PyTorch, HF, Ultralytics, the ggml CPU backend).
- Backend-generic conformance tests, and Vulkan validation layers in CI-style tests.
- Docs: `README.md` (program format), `docs/stdlib.md`, `docs/porting-notes.md`, `docs/design/{kernel-launcher,scheduler,tree}.md`.

---

## 12. Roadmap

- Port the Metal and CUDA executors onto the shared scheduler engine.
- v2 program format split, per-session state, yielding pipelines, and `tl serve`.
- `TL_BIN` binary programs.
- C bindings.
- Rust "autoprocessors" for pre- and postprocessing.

---

## Suggested Slide Structure

1. Title and one-line pitch
2. The idea: models as Scheme programs inside GGUF
3. Architecture diagram (layers table, section 2)
4. Programming model (a short code snippet would help)
5. Ported models (section 4)
6. Pre/postprocessing pipeline, with OCR as the example
7. The `tl` CLI
8. Backends matrix
9. Native executors and scheduler
10. Tree: one kernel language, three GPU targets
11. Performance highlights (section 9)
12. Roadmap and closing
