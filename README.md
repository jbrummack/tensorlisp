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
                   (preprocessing via ../autopro, a sibling repository)
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
- `(device)` is `cpu` or `gpu` (the primary backend), for choosing kernels:
  e.g. convolutions as im2col + matmul on the GPU, ggml's direct convolution
  on the CPU (`ports/yolo11`).

### Entries and pipelines

A program can define several named entries that share the weights, e.g. an
encoder run once and a decoder run per generated token, and pipelines: host
code that runs them (generation loops, multi-stage models).

```scheme
(model encode (inputs [ids i32 (_ 1)]) ... (outputs [memory ... 3]))
(model decode (inputs [tokens i32 (_ 1)] [memory f32 (640 _ 1)]) ... (outputs [logits ... 2]))

(pipeline generate ([prompt string])
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let ([memory (output (run encode [ids ids]) 'memory)])
      (let loop ([tokens (list bos)])
        (let* ([r (run decode [tokens (list->array tokens)] [memory memory])]
               [next (car (array->list (array-argmax (output r 'logits))))])
          (if (= next eos)
              (results [text (detokenize tok tokens)])
              (loop (append tokens (list next)))))))))
```

- An unnamed `(model ...)` is the entry `main`. The default entry (for
  `preprocess`, `postprocess` and runs that name none) is `main`, or else the
  first one defined.
- Each entry keeps its own graph allocated for its current input shapes, so
  alternating between entries doesn't rebuild anything.
- `(run entry [input array] ...)` runs an entry on host arrays and returns its
  outputs; `(output r 'name)` picks one. A missing leading batch dimension of
  1 is added.
- State: `(define-state "name" f32 dim ...)` declares a tensor that keeps its
  contents between runs (e.g. a KV cache), allocated next to the weights and
  zeroed at load (`model.reset_state()` zeroes it again). In an entry,
  `(state "name")` is the tensor; writes go through `(effect t)` (e.g.
  `(effect (ggml-set-rows (state "k.0") k pos))`), which adds them to the
  graph immediately, so ops built afterwards read the new contents. An entry
  may have only effects: `(outputs)`.
- Pipeline results may be strings as well as arrays and numbers.
  `(detokenize tok ids)` and `(token-id tok "<eos>")` help with text.
- Rust: `model.entries()`, `model.run_entry(name, inputs, &options)`,
  `model.pipelines()`, `model.pipeline(name, raw_example)` (returns
  `Value::Array` / `Value::Text` results).
- See `ports/t5gemma2` for an image-to-text model built this way.

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

### Preprocessing

Programs can turn raw inputs into their tensor inputs with `(preprocess ...)`,
using [autopro](../autopro) (Hugging Face processor compatible) from Scheme.
Files such as tokenizers are embedded in the GGUF as assets (`TL_ASSET.<name>`,
see `tl pack --asset`).

```scheme
(define tok (tokenizer (asset "tokenizer.model")))      ; SentencePiece or tokenizer.json

(preprocess ([text string] [photo image])
  (let-values ([(ids mask) (tokenize tok text 'lowercase #t 'pad-to 64)])
    (model-inputs [ids ids]
                  [pixels (image->array (image-resize photo 448 448 'bilinear))])))
```

- Raw input kinds: `string`, `image`, `audio`, `array`. The body ends in
  `(model-inputs [name array] ...)` naming every model input; one example is
  processed at a time and examples are stacked into the batch.
- Text: `tokenizer`, `(tokenize tok text 'lowercase b 'special-tokens b
  'max-length n 'pad-to n 'pad-id n)` → ids and attention mask.
- Images: `image-size`, `image-resize` (static), `image-resize-shortest`,
  `image-resize-longest`, `image-resize-multiple` (dynamic), `image-center-crop`,
  `(image->array img 'scale s 'mean (...) 'std (...) 'channels rgb|bgr 'layout chw|hwc)`.
  Filters: `nearest bilinear bicubic lanczos box hamming` (Pillow-exact).
- Audio: `audio-rate`, `audio-length`, `audio-resample`, `audio-pad`,
  `audio->array` (waveform + mask), `log-mel` (options `n-fft hop win mels
  f-min f-max power center scale norm log floor`), `whisper-features`.
- Images for detectors: `(image-letterbox img w h 'fill (114 114 114))` (YOLO).
- Arrays: `array-shape`, `array-affine`, `array-reshape`.
- Rust: `model.run_raw(examples, &options)`, `model.preprocess(example)`.

### Postprocessing

`(postprocess (name ...) ... (results [name value] ...))` turns the model's
outputs into results, per example, on the host. Each name is a model output
(the example's slice: the first dimension is the batch) or a raw input of
preprocess (e.g. the original image, for its size).

```scheme
;; YOLOv8-style head [84, 8400] -> detections on the original image
(postprocess (head photo)
  (let ([rows (array-transpose head)])
    (let-values ([(boxes scores classes _) (detect (array-slice rows 0 4 'axis 1) (array-slice rows 4 #f 'axis 1)
                                                   'score-threshold 0.25 'iou 0.45)])
      (let ([size (image-size photo)])
        (results [boxes (boxes-unletterbox boxes 640 640 (car size) (cadr size))]
                 [scores scores] [classes classes])))))

;; ViT patch tokens [1024, 768] -> segments (DBSCAN, cosine)
(postprocess (patches)
  (let ([labels (dbscan patches 0.08 8 'metric 'cosine)])
    (results [segments (array-reshape labels 32 32)] [centroids (cluster-centroids patches labels)])))
```

- Detection: `detect` (Ultralytics `non_max_suppression`: options `format
  score-threshold iou class-agnostic multi-label max-candidates max`; returns
  boxes xyxy, scores, classes and source rows), `nms` (torchvision `nms` /
  `batched_nms` with `'classes`), `boxes-convert` (`xyxy xywh cxcywh`),
  `boxes-scale`, `boxes-clip`, `boxes-unletterbox`.
- Clustering: `(dbscan x eps min-samples 'metric euclidean|cosine)` (scikit-learn
  labels, -1 = noise; distances on BLAS, Accelerate on Apple),
  `cluster-centroids`.
- Arrays: `array-slice` (`'axis`, negative bounds), `array-transpose`,
  `array-take` (rows at indices), `array-argmax` (last axis), `array-length`,
  `array->list`, `list->array`. Results may also be numbers or lists of numbers.
- Rust: `model.infer(examples, &options)` (preprocess, run, postprocess →
  `Inference { examples, run }`), `model.postprocess(&outputs)`,
  `model.postprocess_with_raw(&outputs, examples)`.

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
tl pack weights.gguf --program net.ss --asset tokenizer.model=tok.model -o model.gguf
tl run model.gguf --raw text="a photo" --raw image=@cat.jpg  # through (preprocess ...)
tl process model.gguf --raw image=@cat.jpg -o pre/          # only preprocess: arrays as .npy
#   run applies the program's postprocess (results per example; small ones
#   printed in full, -o writes out/results/<example>/NAME.npy; --no-post skips it)
tl run model.gguf --entry decode -i ...                     # another entry (also check/compare)
tl run model.gguf --entry caption --raw image=@bee.jpg --raw "prompt=..."   # a pipeline
tl quantize model.gguf q8.gguf -t q8_0 -i ids=1,64 -i decode:tokens=1,64
#   analyzes every entry; ENTRY:NAME= sets one entry's input
tl convert m.safetensors --replace-prefix encoder.vision_tower.vision_model.=vision. ...
#   shortens names past ggml's 63-byte limit; --offset 'GLOB=1' adds a constant
#   (e.g. Gemma's 1 + w norms)
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
