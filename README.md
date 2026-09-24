# tensorlisp

Network definitions as Scheme programs stored inside a GGUF file.

The GGUF holds the weights as usual, plus a (Chez) Scheme program that builds
the ggml compute graph (`cgraph`). That gives the flexibility of a portable
graph format like ONNX while running on every backend ggml supports. The goal
is a runtime library plus a CLI that lets agents port arbitrary networks.

## Layout

```
crates/
  chez/            embedded ChezScheme: raw bindings (chez::sys) + safe Scheme/Value API
  ggml-sys/        builds vendor/ggml with cc (no CMake), bindgen FFI, backend features
  ggml-codegen/    proc macro on the bindings; generates the Scheme FFI layer
  tensorlisp/      runtime: Model::load(gguf) + Model::run(ndarray) -> ndarray
  tensorlisp-cli/  `tl`: inspect, convert, pack, check, run, compare, quantize
  importers/       ONNX and CoreML/MIL readers, for translating models to tensorlisp
experiments/
  handrolled-lisp/ archived hand-written lisp interpreters (superseded by Chez)
  gguf-loader/     archived first GGUF loader (superseded by tensorlisp::gguf and `tl`)
ports/             real model ports: program, PyTorch reference script, nix flake
  tipsv2/          TIPSv2 B/14 text and vision encoders (match PyTorch, run on Metal)
docs/              design notes, e.g. stdlib-candidates.md
examples/          tensorlisp programs (*.tl)
scripts/           build helpers
vendor/
  ggml/            vendored ggml sources
  ChezScheme/      ChezScheme submodule (blob-filtered)
  chez/            ChezScheme install prefix (generated, gitignored)
models/            local test models (gitignored)
```

Generated code (ggml bindings, Scheme FFI, protobuf types) is written to
Cargo's `OUT_DIR`, not into `src/`.

## Setup

```sh
git submodule update --init --filter=blob:none vendor/ChezScheme
./scripts/build-chez.sh          # builds and installs into vendor/chez
vendor/chez/bin/scheme           # REPL
cargo build --workspace
cargo test --workspace
```

The `chez` crate links `vendor/chez/lib/csv<version>/<machine>/libkernel.a`
and embeds the boot files (`petite.boot`, `scheme.boot`) into the binary, so
nothing has to be installed at runtime. Set `CHEZ_LIB_DIR` to use a different
Chez install.

### ggml backends

`ggml-sys` compiles ggml with plain C/C++/Objective-C compilers. The CPU
backend is always built (it is the scheduler's fallback). The default
`platform-backends` feature adds every backend the target supports: Metal on
macOS (its shader source is embedded in the binary). Use
`default-features = false` plus e.g. `features = ["metal"]` to choose.

## Runtime

```rust
let model = tensorlisp::Model::load("net.gguf", tensorlisp::Device::Auto)?;
let out = model.run(&[("x", x.view().into_dyn())])?;   // named ndarray in
let logits = &out["logits"];                            // named ndarray out
```

`Device::Auto` uses the first GPU if there is one; ops it can't run fall back
to the CPU. The graph for the current input shapes stays allocated; a shape
change rebuilds it.

### File format

A tensorlisp GGUF is a normal GGUF (weights as tensors) plus three keys:

| key      | type   | content                                  |
|----------|--------|------------------------------------------|
| `TL_VER` | u32    | program format version, currently `1`    |
| `TL_TXT` | string | UTF-8 program source                     |
| `TL_BIN` | u8[]   | binary program (reserved, not supported) |

### Programs

```scheme
(define (linear x name)
  (ggml-add (ggml-mul-mat (weight (string-append name ".weight")) x)
            (weight (string-append name ".bias"))))

(model (inputs [x f32 (784 batch)])
  (define h (ggml-relu (linear x "fc1")))
  (outputs [logits (linear h "fc2") 2]))
```

- Programs run in a sandbox: R6RS `(rnrs base)`, `lists`, `control`, fixnum
  and flonum arithmetic, the R5RS `quotient remainder modulo
  exact->inexact inexact->exact`, Chez's `iota list-head format fold-left
  fold-right`, plus `(tensorlisp)`. No eval, ports, files or FFI.
- **Shapes inside Scheme are in ggml order**, innermost first: a numpy array
  of shape `(batch, 784)` is `(784 batch)`. The Rust API uses ndarray order;
  the memory layout is the same.
- `(model (inputs [name dtype (dims ...)] ...) body ...)` declares the inputs;
  dims are optional and may be `#f` or a symbol to accept any size. Input
  dtypes are `f32` and `i32` (e.g. token ids); the Rust API and `.npy` files
  pass i32 inputs as whole-number floats.
- `(outputs [name tensor] ...)` ends the body. An optional integer after the
  tensor sets the output's rank; by default trailing size-1 dimensions are
  dropped, as ggml does.
- `(weight "name")` / `(weight? "name")` access tensors of the GGUF file.
- Every ggml graph op is available with an implicit context, named with
  dashes: `ggml_conv_2d(ctx, a, b, ...)` is `(ggml-conv-2d a b ...)`. Pass `#f`
  for optional tensor arguments (e.g. a mask). ggml enum values are available
  under their C names (`GGML_TYPE_F16`, ...).
- `(shape t)`, `(strides t)` (byte strides, as `ggml-view-*` takes them; both
  ggml order) and `(dtype t)` inspect tensors.
- `(tap "name" t [rank])` returns `t` and marks it as an intermediate result
  that `tl run --taps` / `tl compare` (or `RunOptions::taps`) can read back.

### ggml assertions

A failing `GGML_ASSERT` becomes an error instead of aborting the process:

```
Exception in ggml-add: ggml assertion failed: ggml.c:2062: GGML_ASSERT(ggml_can_repeat(b, a)) failed
  a = #<tensor f32 (5 2)>
  b = #<tensor f32 (8)>
```

- **While the graph is built** (shape checks in ops, graph or context full):
  the abort callback raises a Scheme error and Chez unwinds through ggml's
  frames. The model stays usable, e.g. for other input shapes.
- **During allocation, compute and tensor transfers on the calling thread**:
  guarded C wrappers (`crates/ggml-sys/csrc/guard.c`) `longjmp` back and
  return `Error::Ggml`. After a compute failure the model refuses to run and
  leaks its ggml objects, since backend threads may still use them; load it
  again.
- **On backend worker threads** (CPU threadpool, Metal/GCD) there is nowhere
  to return to: the message is printed and ggml aborts.

## CLI (`tl`)

```sh
cargo install --path crates/tensorlisp-cli   # or: cargo run -p tensorlisp-cli -- <args>
```

Every command takes `--json` (errors too, as `{"error": ...}`) and `-v` for
ggml's logs. Shapes on the command line and in `.npy` files are in numpy order.
A typical port:

```sh
tl convert model.safetensors --strip-prefix model. -o weights.gguf  # weights only
tl inspect weights.gguf                        # tensor names, types, shapes
# write net.ss; tap intermediate tensors that PyTorch also exposes
tl check weights.gguf --program net.ss -i x=1,3,224,224   # graph nodes + shapes, no compute
tl compare weights.gguf --program net.ss -i x=x.npy --reference-dir ref/
#   ref/ holds NAME.npy for outputs and taps, dumped from the reference model;
#   reports each in tap order and the first failure; exit code 2 on mismatch
tl pack weights.gguf --program net.ss -o model.gguf        # final file
tl quantize model.gguf model-q8.gguf -t q8_0 -i x=1,3,224,224
#   builds the program's graph and converts only weights that feed matmuls /
#   embedding lookups (others, e.g. added position embeddings, stay f32)
tl run model-q8.gguf -i x=x.npy -o out/ --repeat 20        # outputs as .npy + timing
```

Note: ggml's f16/bf16 matmuls round the activations to f16/bf16 as well, so
compare such weights with `--rtol 1e-2`, or convert with `--dtype f32` for an
exact comparison.

## Examples

```sh
cargo run -p tensorlisp --example mlp            # write + run a tiny MLP GGUF
cargo run -p ggml-sys --example dump_scheme      # print the generated Scheme FFI
cargo run -p chez --example repl                 # Chez REPL embedded in a Rust binary
cargo run -p gguf-loader --example inspect_gguf  # models/rfdetr-small-q4_K.gguf
cargo run -p importers  --example onnx           # models/model_fp16.onnx
cargo run -p importers  --example coreml         # models/model.mlmodel
cargo run -p handrolled-lisp --example hashcons
cargo run -p handrolled-lisp --example stupid_lisp
```
