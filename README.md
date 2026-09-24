# tensorlisp

Network definitions as Scheme programs stored inside a GGUF file.

The GGUF holds the weights as usual, plus a (Chez) Scheme program that builds
the ggml compute graph (`cgraph`). That gives the flexibility of a portable
graph format like ONNX while running on ggml. A Rust host loads the GGUF,
runs the program to build the graph, and later handles preprocessing and other
host-side work.

## Layout

```
crates/
  chez/            embedded ChezScheme: raw bindings (chez::sys) + safe Scheme/Value API
  ggml-sys/        compiles vendor/ggml (CPU backend) and generates bindgen FFI
  ggml-codegen/    proc macro applied to the bindings (typed GGUF/ggml helpers)
  tensorlisp/      core library: GGUF loading, tensor access, (later) Chez host
  importers/       ONNX and CoreML/MIL readers, for translating models to tensorlisp
experiments/
  handrolled-lisp/ archived hand-written lisp interpreters (superseded by Chez)
examples/          tensorlisp programs (*.tl)
scripts/           build helpers
vendor/
  ggml/            vendored ggml sources
  ChezScheme/      ChezScheme submodule (blob-filtered)
  chez/            ChezScheme install prefix (generated, gitignored)
models/            local test models (gitignored)
```

Generated code (ggml bindings, protobuf types) is written to Cargo's `OUT_DIR`,
not into `src/`.

## Setup

```sh
git submodule update --init --filter=blob:none vendor/ChezScheme
./scripts/build-chez.sh          # builds and installs into vendor/chez
vendor/chez/bin/scheme           # REPL
cargo build --workspace
```

The `chez` crate links `vendor/chez/lib/csv<version>/<machine>/libkernel.a`
and embeds the boot files (`petite.boot`, `scheme.boot`) into the binary, so
nothing has to be installed at runtime. Set `CHEZ_LIB_DIR` to use a different
Chez install.

## Running the experiments

These expect models in `models/`:

```sh
cargo run -p chez --example repl                 # Chez REPL embedded in a Rust binary
cargo run -p tensorlisp --example scheme_graph   # ggml graph built from Scheme
cargo run -p tensorlisp --example inspect_gguf   # models/rfdetr-small-q4_K.gguf
cargo run -p importers  --example onnx           # models/model_fp16.onnx
cargo run -p importers  --example coreml         # models/model.mlmodel
cargo run -p handrolled-lisp --example hashcons
cargo run -p handrolled-lisp --example stupid_lisp
```
