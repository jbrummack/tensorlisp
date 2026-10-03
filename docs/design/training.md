# Training LoRA adapters

`tl train` fine-tunes LoRA adapters of a tensorlisp program on JSON lines and writes them in PEFT's format.
The base weights stay frozen in the GGUF file; only the adapters (`(define-param ...)` tensors) are
trained, with gradients from ggml's autodiff and AdamW.

Status: **works on the ggml backends (CPU, and the GPU through ggml-cuda: `cargo build --release
-p tensorlisp-cli --features cuda`, `--device gpu`), validated against PyTorch on T5Gemma 2 270M**. Not done yet: the native CUDA/Metal executors (no backward ops), fused Unsloth-style
kernels, QLoRA (quantized base). See "Plan".

## Using it

```sh
tl train t5gemma2-270m-f16.gguf \
    --program ports/t5gemma2/t5gemma2.ss --append ports/t5gemma2/train.ss \
    --data data.jsonl -o my-adapter \
    --define train-src-len=64 --define train-tgt-len=32 \
    --rank 8 --alpha 16 --lr 1e-4 --epochs 3 --accum 4 --device gpu
```

`data.jsonl` has one object per line with the fields `input` and `output` (the raw string inputs of the
program's `train-example` pipeline, which tokenizes and pads one example). The run prints one line per
optimizer step (`--json` for JSON lines) and writes `my-adapter/adapter_model.safetensors` and
`adapter_config.json`. Then, with any command that loads a model:

```sh
tl run t5gemma2-270m-f16.gguf --program ports/t5gemma2/t5gemma2.ss --append ports/t5gemma2/train.ss \
    --adapter my-adapter --entry generate --raw "prompt=hello"
```

The adapter's rank and alpha come from its `adapter_config.json`. `--resume DIR` continues from an
adapter, `--save-every N` checkpoints, `--schedule cosine|linear|constant` and `--warmup` shape the
learning rate, `--accum N` accumulates N examples per step (the entry takes one example; the gradient is
the mean).

Example (8 examples, `uppercase`, 24 steps on the CPU): the loss goes from 1.8 to 0.01 and the adapter
turns `hello`, `tensor` and the unseen `python` into `HELLO`, `TENSOR` and `PYTHON`; the base model
produces garbage for them.

## How it works

- **Parameters**: `(define-param name init dim ...)` is a state (f32) the trainer initializes (`'zeros` or
  uniform in [-b, b]) and updates. `(lora-config! rank alpha)` and `(lora-attach! module in out)` declare
  `lora.<module>.A` (`[in, r]`, uniform ±1/sqrt(in), PEFT's kaiming a=sqrt 5) and `lora.<module>.B`
  (`[r, out]`, zeros); `nn:linear` adds `alpha/r * B (A x)` to every attached module, so a program's
  inference entries also run with the adapter. In ggml order `A` and `B` are torch's `lora_A.weight`
  `[r, in]` and `lora_B.weight` `[out, r]`, byte for byte.
- **Training entry**: a `(model train ...)` that ends in `(outputs [loss ...])`, a scalar. T5Gemma 2's
  (ports/t5gemma2/train.ss) is teacher-forced cross entropy, `sum_t w_t * -log softmax(logits_t)[y_t]`, with
  per-token weights (0 for padding and prompt), the exact GELU (no table) and unfused attention, since
  only ops with a backward pass can appear.
- **Graphs** (`crates/tensorlisp/src/model/train.rs`, `Model::trainer`): the entry is built with gradient
  slots, the parameters are flagged (`ggml_set_param`, before the build: ggml only gives a flagged leaf a
  node), `ggml_build_backward_expand` adds the backward pass and one `opt_step_adamw` per parameter.
  This is `ggml_opt`'s construction, kept static because `ggml_opt`'s static mode is limited to 2048 nodes
  (T5Gemma's graph has ~15k). One graph (forward + backward into persistent gradient accumulators + the
  optimizer step) serves every micro-batch; on the micro-batches that don't end an accumulation period
  the AdamW step is made a no-op through its parameter tensor (alpha 0, beta 1, no bias correction), and
  the loss gradient starts at 1 / accum_steps so the update uses the mean. (A separate gradient-only
  graph, as `ggml_opt` has, makes ggml's scheduler re-reserve its buffers when switching, which stalled on
  the CUDA backend.) Everything is allocated once; a step is "set inputs, compute".
  The gradients stay readable (`Trainer::grad`): they are what a native executor is checked against.
- **Adapters on disk** (`adapter.rs`): PEFT's `adapter_model.safetensors`, keys
  `<prefix><module>.lora_A.weight` / `.lora_B.weight` (default prefix `base_model.model.model.` for Gemma
  style checkpoints) and an `adapter_config.json` (`r`, `lora_alpha`, `target_modules`).

### Patches to the vendored ggml

`vendor/ggml/src/ggml.c`, marked `tensorlisp:`: backward passes for `CONCAT` (slices of the gradient) and
`UNARY(TANH)` (1 - tanh²), which upstream lacks and T5Gemma 2's decoder (merged self/cross keys) and the
exact GELU need; and the input gradient of a `mul_mat` with a frozen f16/bf16 weight is a `mul_mat` with the
transposed copy of the weight instead of `out_prod`, which the GPU backends only run for an f32 weight (the
scheduler copied every weight, 320 MB for the embedding, to the CPU at each step: 1.4 s instead of 90 ms). Everything else T5Gemma 2 uses (mul_mat, rms_norm, rope, soft_max, get_rows, scale, mul,
add, log, sum, ...) already has one. Not supported by ggml's autodiff, so not usable in a training entry:
`flash-attn-ext` (use `attn:sdpa` without `'flash`), the fused `ggml-gelu` (use `nn:gelu-tanh-exact`),
`ggml-norm` (layer norm), quantized frozen weights (their transpose can't be copied for the input
gradient).

## Validation

`ports/t5gemma2/train_reference.py` re-implements the text encoder-decoder and PEFT-style LoRA in PyTorch
from the checkpoint's tensors (transformers can't load the gated `google/t5gemma-2-270m-270m` without its
config). `crates/tensorlisp/tests/train_t5gemma2.rs` (`--ignored`; needs `TL_TRAIN_REF`, optionally
`TL_TRAIN_GGUF`, `TL_TRAIN_DEVICE`) compares a batch of 12 encoder / 8 decoder tokens with all 504 LoRA
tensors (rank 8 on all seven linear layers of 36 layers), nonzero A and B:

| | loss vs PyTorch f64 | gradients, mean / worst relative error |
|---|---|---|
| ggml CPU, f32 weights | 19.139309 vs 19.139311 | 9.5e-4 / 2.2e-3 |
| PyTorch f32 vs f64 (noise floor) | 19.139256 | 4.3e-4 / 8.4e-4 |
| ggml CPU, f16 weights (the GGUF) | 19.1602 | 1.1e-2 / 2.9e-2 |
| ggml-cuda (RTX 3080), f32 weights | 19.1386 | 6.6e-3 / 1.7e-2 |
| ggml-cuda (RTX 3080), f16 weights | 19.1559 | 1.5e-2 / 3.8e-2 |

The f16 file's error is ggml rounding the activations to f16 in matmuls, not a logic error (the f32 file
removes it). Three AdamW steps (lr 1e-3) end within ~1% of PyTorch's parameters; Adam amplifies gradient
noise, so the trajectories are only compared loosely. `adapter_round_trip` saves, loads into a fresh model
and compares bit for bit.

Timing, one example (12 + 8 tokens), forward + backward (+ AdamW): CPU ~1.4 s with f32 weights, ~2.5 s with
f16; RTX 3080 ~90 ms with f32 or f16 weights (the toy run above: ~6 examples/s end to end, 24 steps of 2).

## Plan

1. ~~GPU through ggml-cuda~~: done (see above). The scheduler still splits the graph ~560 times between CPU
   and GPU for small ops (mask construction, views); fusing or moving them is a cheap speed-up.
2. **Native executors**: add the backward ops (`RMS_NORM_BACK`, `ROPE_BACK`, `SOFT_MAX_BACK`, `OUT_PROD`,
   `REPEAT_BACK`, `SILU_BACK`, `GET_ROWS_BACK`, `CROSS_ENTROPY_LOSS(_BACK)`, `OPT_STEP_ADAMW`, `SUM_ROWS`,
   ...) to `crates/kernels` and compare each against the ggml CPU gradients with this test. Then fuse as
   Unsloth does (Apache-2.0 parts of `unsloth/kernels`: fused LoRA matmuls, RMSNorm, RoPE, SwiGLU/GeGLU,
   chunked cross entropy; they are Triton, so the math is ported to CUDA C, with the license notice in a
   `NOTICE` when code is adapted).
3. **QLoRA**: frozen quantized base (q8_0/q4 GGUF) with f32 adapters. ggml can't differentiate through a
   quantized `mul_mat` (the input gradient needs the transposed weight); the native path needs a dequantizing
   backward matmul kernel (what Unsloth does with bitsandbytes' `dequantize_4bit`).
4. Batches > 1, packing, gradient checkpointing for long sequences, other ports (Gemma 4, decoder-only LLMs).
