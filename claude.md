# tensorlisp
tensorlisp
## info
chezscheme is loaded with `git clone https://github.com/cisco/ChezScheme.git --filter=blob:none` to avoid loading unneccessary loaders

## lisp
I evaluated several LISPs, even tried implementing a lisp by hand but i tried already known lisps and settled on ChezScheme.

## other file formats
i looked at other file formats/backends where support may later be added.

## layout
Cargo workspace, see README.md (also documents the program format). `crates/chez` embeds Chez (bindgen'd scheme.h + C shim for its macros, boot files via include_bytes!). `crates/ggml-sys` builds vendored ggml (`vendor/ggml`) with cc only (no cmake), backend features (`platform-backends` default = metal on macOS), bindgen into OUT_DIR. `crates/ggml-codegen` generates the Scheme FFI (raw foreign-procedures, implicit-ctx `ggml-*` ops, enum constants, symbol table). `crates/tensorlisp` is the runtime (Model::load/run, Scheme on a dedicated thread, library in src/scheme/core.ss); `crates/tensorlisp-cli` is `tl` (inspect/convert/pack/check/run/compare/quantize, all with --json). `crates/importers` has onnx/coreml readers; `experiments/` holds archived attempts (hand-written lisps, first gguf loader).
ggml asserts are caught: at graph build via a Chez foreign-callable abort handler (Chez unwinds), on the calling thread in Rust via setjmp wrappers in `crates/ggml-sys/csrc/guard.c`; worker-thread asserts still abort.
ChezScheme is the submodule `vendor/ChezScheme`, built with `scripts/build-chez.sh` into `vendor/chez` (gitignored).

## plan
done: runtime (Model::load/run, taps, graph info), assert recovery, `tl` CLI. Later: TL_BIN binary programs, C bindings, rust preprocessors (autoprocessors), vulkan/cuda backends.
GGUF keys: TL_TXT (utf8 program), TL_BIN (binary), TL_VER (u32 version).
