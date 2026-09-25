# T5Gemma 2 (270M-270M)

[google/t5gemma-2-270m-270m](https://huggingface.co/google/t5gemma-2-270m-270m)
(gated): a Gemma 3 encoder-decoder with a SigLIP vision tower. `t5gemma2.ss`
ports `T5Gemma2ForConditionalGeneration` from transformers 5.x
(`modeling_t5gemma2.py`), including tokenization and greedy generation, so
the GGUF goes from raw image + prompt to text:

```sh
tl run t5gemma2-270m-f16.gguf --entry caption \
    --raw image=@bee.jpg --raw "prompt=<start_of_image> in this image, there is"
#   text    " a bumble bee in a flower bed."
tl run t5gemma2-270m-f16.gguf --entry generate --raw "prompt=The capital of France is"
```

## Program

| entry / pipeline | inputs (numpy order) | outputs |
|---|---|---|
| `encode` | ids, gather, pos `[1, L]` i32, mask `[1, L]` | memory `[1, L, 640]` |
| `encode-image` | pixels `[1, 3, 896, 896]` + the above | memory, image features `[1, 256, 640]` |
| `decode` | tokens, pos `[1, T]` i32, memory, memory-mask, at `[1, 1]` | logits `[1, 262144]` at `at` (no cache) |
| `cross` | memory `[1, M, 640]` | (fills the cross-attention cache) |
| `decode-step` | token, pos `[1, 1]` i32, self-mask `[1, 64]`, memory-mask `[1, 1024]` | logits, next (argmax) |
| `caption` (pipeline) | image, prompt | text, tokens |
| `generate` (pipeline) | prompt | text, tokens |

- The pipelines do what `Gemma3Processor` + `generate(do_sample=False)` do:
  `<start_of_image>` becomes `\n\n<start_of_image>` + 256 image tokens +
  `<end_of_image>\n\n`, the prompt is tokenized with the embedded
  `tokenizer.json`, the image resized to 896 x 896 (bilinear, Pillow) and
  normalized to [-1, 1]. Then the encoder runs once, and the decoder once
  per token until `<eos>` (at most 32 new tokens).
- `gather` places the image features and the `<end_of_image>` embedding
  (T5Gemma 2 replaces that token's embedding) with one `get-rows` over
  [token embeddings; image features; eoi].
- Masks are built in the graph from positions: bidirectional with a
  sliding window (256 left, 257 right) on the encoder's local layers, causal
  on the decoder's self part, plus the memory's padding mask for the merged
  cross part.
- Decoding uses a KV cache in state tensors (`define-state`): per layer one
  f16 key and one value cache of 64 + 1024 rows, the decoder's own tokens
  first, then the encoder output's. `cross` writes the encoder part once per
  prompt; `decode-step` embeds one token, writes its keys/values at `pos`
  (`ggml-set-rows` in an `effect`) and attends over the whole cache with
  `flash-attn-ext`, masked by `[self-mask; memory-mask]`. It returns the
  logits and the argmax (so greedy decoding reads back one number). All shapes
  are fixed, so every graph is built once. Set `use-kv-cache` to `#f` to decode
  with `decode` instead (every position recomputed per token).
- Flags at the top: `flash-attention` (vision tower, default on) and
  `exact-gelu` (ggml's CPU GELU uses an f16 table, ~1e-3; on for comparisons).

## Building the files

```sh
tl convert model.safetensors --strip-prefix model. \
    --replace-prefix encoder.vision_tower.vision_model.=vision. \
    --replace-prefix encoder.multi_modal_projector.=mm. \
    --offset 'model.encoder.layers.*norm.weight=1' --offset 'model.encoder.norm.weight=1' \
    --offset 'model.decoder.*norm.weight=1' \
    --offset 'model.encoder.multi_modal_projector.mm_soft_emb_norm.weight=1' \
    --dtype f16 -o weights-f16.gguf
tl pack weights-f16.gguf --program t5gemma2.ss --asset tokenizer.json=tokenizer.json \
    -o t5gemma2-270m-f16.gguf
tl quantize t5gemma2-270m-f16.gguf t5gemma2-270m-q8_0.gguf -t q8_0 \
    -i ids=1,267 -i gather=1,267 -i pos=1,267 -i mask=1,267 \
    -i decode:tokens=1,64 -i decode:pos=1,64 -i memory=1,267,640 -i memory-mask=1,267
```

- `--replace-prefix`: the vision tower's names exceed ggml's 63 bytes otherwise.
- `--offset ...=1`: Gemma's RMSNorm scales by `1 + w`; the file stores `1 + w`
  (like llama.cpp's Gemma conversion), which saves an op per norm and lets
  Metal fuse the norm with the multiplication. The program requires it. The
  vision tower's LayerNorms (`vision.*`) are not RMSNorms and keep their weights.
- `quantize` builds every entry's graph; `ENTRY:NAME=` sets one entry's
  input, unprefixed shapes apply to entries whose declared dims accept them.

## Validation

`reference.py` (in the CPU-only torch flake here; `nix store gc` removes it)
dumps the processor outputs, activations and a greedy generation for the
model card's captioning prompt on `bee.jpg` and for a text-only prompt.

On the CPU with f32 weights and the exact flags (`flash-attention #f`,
`exact-gelu #t`), against PyTorch float32:

| | max abs error | cosine |
|---|---|---|
| text encoder (embeddings / layer 0 / output) | 0 / 1.4e-4 / 9.2e-5 | 1.000000 |
| vision tower output (4096 x 1152) | 3.8e-3 (values ~20) | 1.000000 |
| image features, encoder output with image | 2.3e-3, 3.7e-2 | 1.000000 |
| decoder hidden states, logits (all positions) | 1.3e-4, 1.4e-4 | 1.000000 |

Cached decoding (teacher-forced over the reference tokens): logits within
0.04 of PyTorch (the f16 cache), argmax equal at every position, on the CPU
and on Metal. Generated tokens are identical to transformers' for both
prompts with f32, f16 and q8_0 weights on Metal, with and without the cache
(" a bumble bee in a flower bed.").

Found while porting: transformers applies the embedding scale sqrt(640) in
the checkpoint's bf16 in the encoder (25.25) but in f32 in the decoder
(25.298); the port does the same.

## Performance (M1 Pro, Metal)

| | f32 | f16 | q8_0 |
|---|---|---|---|
| file | 2.9 GiB | 1.5 GiB | 923 MiB |
| `encode-image` | 4.3 s | 3.9 s | |
| `caption` (9 tokens) | 4.7 s | 4.3 s | 4.3 s |

The SigLIP tower at 896 px (4096 tokens, ~5.5 TFLOP) dominates, split about
evenly between attention and MLPs; that is compute-bound on this GPU.

Decoding, per token (`generate`, 26 tokens, f16, median of 7 runs):

| | per token |
|---|---|
| without cache (`decode`, 64 positions recomputed) | 13 ms (18 ms GPU time per step) |
| KV cache | 5.5 ms GPU time per step; 8.6-11 ms end to end on a busy machine |

With the cache a step is GPU-bound (the process waits on the GPU; host
overhead is small): ~500 kernels for one token, and the tied output
projection (262144 x 640) is the largest read. Getting there needed: norm
weights stored as `1 + w` (108 fewer ops), one cache tensor per layer
instead of concatenating self and cross caches, flash attention instead of
permute/cont/matmul/softmax, and the argmax on the device.

Not done: sampling
(top-k/top-p like `generation_config.json`; greedy only), batches > 1,
Gemma 3's pan-and-scan crops.
