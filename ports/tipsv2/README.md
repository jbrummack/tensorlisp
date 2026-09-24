# TIPSv2 B/14

Both towers of [google/tipsv2-b14](https://huggingface.co/google/tipsv2-b14),
ported from the checkpoint's `text_encoder.py` and `image_encoder.py`.

## Text encoder — `text.ss`

12 layers, width 768, 12 heads, ReLU MLP, sinusoidal positions, masked mean pooling.

- Inputs (numpy order): `ids [batch, 64]` SentencePiece ids of the lowercased
  text without BOS/EOS, zero-padded; `paddings [batch, 64]`, 1 at padding.
- Output: `embedding [batch, 768]`, not L2-normalized (like `encode_text`).
- Taps: `embed`, `layer.00` .. `layer.11`, `final_ln`.

Tokenization is not part of the program yet (the host will provide it).

## Vision encoder — `vision.ss`

DINOv2-style ViT-B/14 with one register token and LayerScale.

- Input (numpy order): `image [batch, 3, H, W]`, RGB in [0, 1] (the processor
  only rescales; no mean/std normalization). H and W must be multiples of 14;
  448 is native, other sizes interpolate the position embeddings (bilinear,
  antialiased) like the reference.
- Outputs: `cls [batch, 768]`, `register [batch, 768]`,
  `patches [batch, H/14 * W/14, 768]` (row-major over the patch grid), all
  after the final norm.
- Taps: `embed`, `block.00` .. `block.11`, `norm`.
- `flash-attention` (top of the file, default `#t`) uses ggml's fused attention
  with f16 K/V: ~30% faster on Metal. Set it to `#f` for exact comparisons on
  the CPU (all taps then match PyTorch at the default 1e-4/1e-3 tolerance).

## Reproduce

```sh
M=models/tipsv2-b14
mkdir -p $M && for f in model.safetensors text_encoder.py image_encoder.py tokenizer.model; do
  curl -fL -o $M/$f https://huggingface.co/google/tipsv2-b14/resolve/main/$f
done

# One GGUF per tower
tl convert $M/model.safetensors --include 'text_encoder.*' --strip-prefix text_encoder. -o $M/text-f32.gguf
tl convert $M/model.safetensors --include 'vision_encoder.*' --exclude vision_encoder.mask_token \
   --strip-prefix vision_encoder. -o $M/vision-f32.gguf
tl pack $M/text-f32.gguf --program ports/tipsv2/text.ss -o $M/tipsv2-b14-text-f32.gguf
tl pack $M/vision-f32.gguf --program ports/tipsv2/vision.ss -o $M/tipsv2-b14-vision-f32.gguf
tl quantize $M/tipsv2-b14-text-f32.gguf $M/tipsv2-b14-text-f16.gguf -t f16 -i ids=1,64 -i paddings=1,64
tl quantize $M/tipsv2-b14-vision-f32.gguf $M/tipsv2-b14-vision-f16.gguf -t f16 -i image=1,3,448,448

# PyTorch references (CPU-only torch from the flake; no CUDA, no transformers)
(cd ports/tipsv2 && for part in text vision; do
   nix develop --command python reference.py $part ../../$M ../../$M/data; done)

T=$M/data/text V=$M/data/vision-448
tl compare $M/tipsv2-b14-text-f32.gguf --device cpu -i ids=$T/ids.npy -i paddings=$T/paddings.npy --reference-dir $T/ref
tl compare $M/tipsv2-b14-vision-f32.gguf --device gpu -i image=$V/image.npy --reference-dir $V/ref --atol 1e-2 --rtol 1e-2
```

`data/vision-224` holds the same test images at 224 px to check the position
embedding interpolation.

## Results (M1 Pro)

Text, batch of 3 texts:

| file   | size    | device | embedding max abs error | cosine    | time    |
|--------|---------|--------|-------------------------|-----------|---------|
| f32    | 418 MiB | CPU    | 4.8e-6 (all taps pass at 1e-4/1e-3) | 1.0000000 | 89 ms |
| f32    | 418 MiB | Metal  | 6.5e-4                  | 1.0000000 | 16.4 ms |
| f16    | 209 MiB | Metal  | 8.6e-4                  | 1.0000000 | 15.8 ms |
| q8_0   | 111 MiB | Metal  | 1.1e-2                  | 0.9999924 | 16.5 ms |

Embedding values are within about ±15.

Vision on Metal with flash attention, 2 test images at 448 px (patch values within about ±6):

| file   | size    | cls cosine | patches max abs error | patches cosine | 448 px, batch 2 |
|--------|---------|------------|-----------------------|----------------|-----------------|
| f32    | 329 MiB | 0.9999998  | 1.3e-2                | 0.9999992      | 168 ms |
| f16    | 166 MiB | 0.9999998  | 1.3e-2                | 0.9999992      | 162 ms |
| q8_0   | 91 MiB  | 0.9998483  | 1.3e-1                | 0.9996514      | 194 ms |

f16, one image: 81 ms at 448 px, 21 ms at 224 px. The CPU takes ~1.5 s for
two images at 448 px. At 224 px the results are as close (patches cosine
0.9999987 for f32/f16).

### Quantization and speed (vision, one 448 px image)

| weights | size | Metal | CPU | patches cosine |
|---|---|---|---|---|
| f16 | 166 MiB | **90 ms** | **422 ms** | 1.00000 |
| q8_0 | 91 MiB | 94 ms | 658 ms | 0.99965 |
| mixed (MLP q4_K, rest q8_0) | 64 MiB | 93 ms | 792 ms | 0.957 |
| q4_K | 50 MiB | 97 ms | 690 ms | 0.940 |
| q4_0 | 50 MiB | 93 ms | 1021 ms | 0.909 |

At 1026 tokens every weight is reused ~1000 times per matmul, so the model
is compute-bound: ggml's Metal quantized kernels dequantize tiles to half
(same speed as f16), and the CPU quantized paths also quantize activations,
which is slower than native f16 on Apple silicon. Use f16; q8_0 only to save
memory. Plain 4-bit (no importance matrix) is too lossy for patch features.

## Cleanup

The flake only adds a Python with torch, numpy, safetensors and sentencepiece
to the Nix store; nothing is installed into a profile. To remove it:

```sh
nix store gc            # deletes store paths no longer referenced (the torch env included)
rm -rf models/tipsv2-b14
```
