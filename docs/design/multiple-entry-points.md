# Multiple entry points per file (planned)

Status: not implemented. Today a program defines exactly one `(model ...)`,
so TIPSv2 ships as two files (`tipsv2-b14-text-*.gguf`,
`tipsv2-b14-vision-*.gguf`) although it is one checkpoint.

## Motivation

- Multi-tower models: CLIP/SigLIP/TIPS text + vision encoders, retrieval
  models with query and document encoders.
- Encoder/decoder models and multi-stage pipelines (e.g. an encoder run once
  and a decoder run per step, or a detector backbone and a head).
- Shared weights: towers or stages that share tensors should load them once.
- One artifact to version, quantize and ship.

## Proposal

### Program

Named models; an unnamed `model` stays valid and is the entry `main`.

```scheme
(model text (inputs [ids i32 (64 batch)] [paddings f32 (64 batch)])
  ...
  (outputs [embedding ... 2]))

(model vision (inputs [image f32 (_ _ 3 batch)])
  ...
  (outputs [cls ... 2] [register ... 2] [patches ... 3]))
```

Helpers defined at the top level are shared by all entries.

### Runtime

- `Model::load` loads the weights once (one weights buffer per device) and
  evaluates the program once.
- `model.entries()` lists entry names with their input specs.
- `model.run_entry("vision", inputs)`; `model.run(inputs)` keeps working when
  there is a single entry (or an entry named `main`).
- One scheduler, one live graph per entry (the "graph stays allocated while
  shapes don't change" rule applies per entry). Needs care: resetting the
  shared scheduler invalidates every allocated graph, so either one scheduler
  per entry or rebuild-on-switch.

### CLI

- `--entry NAME` on `check`, `run`, `compare`; default: the only entry or `main`.
- `inspect` lists entries and their inputs.
- `quantize` must union the weight uses of all entries (a tensor is quantized
  only if every entry uses it as a matmul / get-rows weight), so it needs input
  shapes per entry: `-i vision:image=1,3,448,448 -i text:ids=1,64`.
- Taps are per entry; reference dirs per entry (`--reference-dir ref/vision`).

### File format

The program text stays in `TL_TXT`. Entry names come from evaluating the
program, so no new key is required; optionally cache them in `TL_ENTRIES`
(string array) so tools can list entries without running Scheme. Bump
`TL_VER` to 2 only if old runtimes must refuse multi-entry files (they would
otherwise fail with "a program can define only one model", which is an
acceptable error).

## Open questions

- Joint graphs: running two entries in one graph (e.g. text + vision for a
  similarity score) — a third entry that calls the others' bodies may be enough.
- Per-entry host processing (tokenizer for `text`, image resize for `vision`)
  once Rust preprocessors exist: attach them to entries in metadata?
- Should entries be able to declare which weights they use (to allow loading
  only one tower)?

## Related: per-tensor quantization rules

Mixed precision currently takes two `tl quantize` passes (the second skips
already-quantized tensors):

```sh
tl quantize f32.gguf tmp.gguf   -t q4_K --include '*mlp*' -i image=1,3,448,448
tl quantize tmp.gguf mixed.gguf -t q8_0                   -i image=1,3,448,448
```

A single pass with ordered rules (`--rule '*mlp*=q4_K' --rule '*=q8_0'`) and
importance-matrix calibration (ggml supports imatrix-based quantization; needed
for usable 4-bit on dense features — plain q4 gave cosine 0.91-0.94 on TIPSv2
patch tokens) would make this practical. Note that for compute-bound models
(ViT at 1026 tokens) quantization saves memory, not time: see
ports/tipsv2/README.md.
