# Gemma 4 E2B (text)

[google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B): `gemma4.ss` ports the
text decoder of `Gemma4ForCausalLM` from transformers 5.x (`modeling_gemma4.py`),
including tokenization and greedy generation:

```sh
tl run gemma4-e2b-q8_0.gguf --entry generate --raw "prompt=The capital of France is"
#   text    " Paris.\n\nThe capital of France is Paris. ..."
```

## Program

| entry / pipeline | inputs (numpy order) | outputs |
|---|---|---|
| `step` | token, pos `[1, n]` i32 | logits `[1, 262144]` (last token, softcapped), next (argmax) |
| `generate` (pipeline) | prompt | text, tokens (greedy, at most 32) |

`step` writes the n tokens' K/V into the cache at `pos` and attends over the whole cache
(2048 rows, masks built in the graph from positions), so it serves as prefill (n = prompt
length) and as decode step (n = 1).

What differs from Gemma 3:

- Head size 256 on sliding layers (window 512), 512 on every 5th (global) layer; 8 query
  heads, 1 KV head; attention scale is 1.0.
- q/k norms have weights, V gets a scale-free RMS norm. RMSNorm uses the plain weight
  (no `+1`, so no `--offset` when converting).
- Proportional RoPE on global layers: only the first 64 of 256 rotation pairs rotate. It
  is ggml's NEOX rope with a frequency-factor tensor built in the graph (the rest get
  factor 1e30).
- Per-layer embeddings (PLE): a 262144 x 8960 table plus a projection of the embedding feed
  a gate in every layer. Then a per-layer scalar multiplies the layer's output.
- K/V sharing: layers 15..34 have no K/V projections and attend over the cache of layer 13
  (sliding) or 14 (global). Only 15 layers keep a cache.
- Final logit softcapping (30), tied embeddings; the embedding scale is rounded to bf16
  (39.25) like transformers does.

## Building the files

```sh
tl convert model.safetensors --include 'model.language_model.*' --strip-prefix model.language_model. \
    --dtype f16 -o weights-f16.gguf
tl pack weights-f16.gguf --program gemma4.ss --asset tokenizer.json=tokenizer.json -o gemma4-e2b-f16.gguf
tl quantize gemma4-e2b-f16.gguf gemma4-e2b-q8_0.gguf -t q8_0 -i token=1,8 -i pos=1,8
```

The f16 weights are 8.7 GiB (the PLE table is most of it), q8_0 4.6 GiB, which fits an
RTX 3080 (10 GB).

## Validation

`reference.py` (venv with torch CPU + transformers 5.18) prints the token ids, top-10
logits and 32 greedy tokens in bf16. On the native CUDA device with the q8_0 file, `generate`
gives the same 32 tokens for `"The capital of France is"` and for a 406-token prompt
(80 filler sentences + the question): `tests/native_gemma4.rs`, `greedy_tokens_match_transformers`.

## Concurrent decoding

`concurrency.ss` is an add-on (appended to `gemma4.ss` by the test, not packed) that decodes
B sequences per step, with the K/V kept either way:

- **dense**: a `[hd, 640]` cache per sequence slot, every sequence attends over all 640
  rows under a mask (exact sliding window).
- **paged**: one pool of 16-token blocks, a sequence takes the blocks it needs and attends
  over exactly its own length with the vendored paged-attention kernels. Those have no
  sliding window, so this is exact only up to 512 tokens of context.

Prefill attends over the prompt's own K/V and writes the cache, so the host only supplies
tokens, positions and cache indices. Entries: `prefill-dense`, `prefill-paged`,
`decode-dense`, `decode-paged`; pipelines `fill-dense`, `fill-paged`.

`cargo test --release -p tensorlisp --features native-cuda --test native_gemma4 -- --ignored --nocapture`
checks that dense and paged decoding of four prompts (6 to 206 tokens) equal sequential
`generate`, then sweeps the number of sequences. RTX 3080, q8_0, 32 generated tokens each,
tokens/s over the decode steps:

| sequences | sequential | dense, short | paged, short | dense, 406 tokens | paged, 406 tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 38 | 39 | 37 | 38 |
| 2 | 37 | 63 | 65 | 63 | 63 |
| 4 | 37 | 91 | 94 | 90 | 92 |
| 8 | 37 | 112 | 115 | 111 | 114 |
| 16 | 37 | 188 | 202 | 188 | 200 |
| 32 | 37 | 348 | 397 | 348 | 391 |
| 64 | 37 | 636 | 745 | 633 | 714 |

Batching is the big win (17x at 64 sequences: the weights are read once per step instead of
once per sequence). Paged adds 12-17% at 32-64 sequences and needs far less cache memory for
short prompts (0.8 MiB vs 11.2 MiB per sequence at 6 tokens; 7.9 MiB at 406 tokens, since
the block table is sized to the longest sequence). Gemma 4's KV cache is small (one KV head,
15 layers), so attention is a minor part of a step and the paged kernel's advantage over
dense flash attention is small here.
