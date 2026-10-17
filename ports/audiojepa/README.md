# Audio-JEPA

The encoder of [ltuncay/Audio-JEPA](https://huggingface.co/ltuncay/Audio-JEPA)
(ICME 2025), ported from the upstream `VisionTransformer`
([LudovicTuncay/Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA)). The checkpoint also holds the
predictor and the EMA target encoder, which are only used for pretraining and dropped here.

ViT-B (12 layers, width 768, 12 heads, GELU MLP, `nn.LayerNorm` eps 1e-5) over a 1-channel
256 x 128 log-mel image: 16 x 16 patch convolution, fixed 2D sin-cos position embeddings
(stored in the checkpoint), no class token.

## `encoder.ss`

- Raw input: `clip`, audio of any rate (resampled to 32 kHz). `(preprocess ...)` is
  `compute_mel_spec` from the model card's `inference_example.py`: Kaldi fbank of the
  waveform minus its mean (97.65625 ms frames every 39.0625 ms, 128 mels from 20 Hz to Nyquist,
  Hanning window), zero padded or truncated to 256 frames. It uses the host function
  `kaldi-fbank` (autopro `KaldiFbank`, matches torchaudio).
- Input (numpy order): `mel [batch, 256, 128]` (time, mel bins).
- Output: `embedding [batch, 128, 768]`, after the final norm: 16 time x 8 frequency patches,
  time-major (patch `p` is time step `p / 8`, frequency band `p % 8`). Clips longer than 10 s
  are cut to 10 s; the model has no other input size (the position embedding is fixed).
- Taps: `embed`, `block.00` .. `block.11`, `norm`.

```sh
tl run audiojepa-f16.gguf --raw clip=@speech.wav -o out/
```

## Reproduce

```sh
M=models/audiojepa
mkdir -p $M && git clone --depth 1 https://github.com/LudovicTuncay/Audio-JEPA $M/src
for f in JEPA.ckpt inference_example.py; do
  curl -fL -o $M/$f https://huggingface.co/ltuncay/Audio-JEPA/resolve/main/$f
done

# PyTorch reference (CPU-only torch + torchaudio from the flake): writes $M/encoder.safetensors
# (the `encoder.*` weights), test clips, their mels and the reference activations
(cd ports/audiojepa && nix develop path:. --command python -I reference.py ../../$M/src ../../$M ../../$M/data)

tl convert $M/encoder.safetensors -o $M/f32.gguf
tl pack $M/f32.gguf --program ports/audiojepa/encoder.ss -o $M/audiojepa-f32.gguf
tl quantize $M/audiojepa-f32.gguf $M/audiojepa-f16.gguf -t f16 -i mel=1,256,128

tl compare $M/audiojepa-f32.gguf --device cpu -i mel=$M/data/mel.npy --reference-dir $M/data/ref
tl run $M/audiojepa-f32.gguf --device cpu --raw clip=@$M/data/clip0.wav --raw clip=@$M/data/clip1.wav -o out/
```

(`nix develop path:.` because a new flake isn't tracked by git yet; plain `nix develop` works once it is.)

## Results (M1 Pro)

Two 16-bit WAV test clips (a 10 s chirp with noise, a 4 s two-tone), embedding values within about -8 .. 18.
The residual stream has activations up to ~1500 in some channels, so per-block taps are compared
with a relative tolerance; the final norm is what matters.

| file | size | device | embedding max abs error (from WAV) | cosine | time (1 clip) |
|------|------|--------|------------------------------------|--------|---------------|
| f32 | 325 MiB | CPU | 4.1e-5 (batch of 2) | 1.0000000 | 190 ms |
| f32 | 325 MiB | Metal | 7.6e-3 | 0.9999999 | 29 ms |
| f16 | 163 MiB | Metal | 7.6e-3 | 0.9999999 | 27 ms |
| f16 | 163 MiB | native Metal | 9.8e-3 | 1.0000001 | 35 ms |
| f16 | 163 MiB | CPU | 6.6e-2 | 0.9999958 | 344 ms |

On the CPU in f32 every tap passes at 1e-4/1e-3 up to `block.01`; later blocks differ by
~1e-3 absolute on values around 1000 (1e-6 relative), and `norm` is within 3.7e-5.
The mel front end is checked against torchaudio by comparing the model output from the WAV file
with the output from torchaudio's mel; resampling other rates uses autopro's resampler, which
has not been compared against torchaudio's.
