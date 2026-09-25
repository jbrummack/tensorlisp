# tl stdlib candidates

Helpers that came up while porting models and should move into a shared
library that programs can import (e.g. `(tensorlisp nn)`), plus the pitfalls
they encapsulate. Each entry names the port it comes from; the code there is
the current reference implementation.

Conventions: shapes are ggml order (innermost first); `D` = width,
`L` = sequence length, `B` = batch. Weight names follow PyTorch state dicts
with a `prefix` ending in `.`.

## Tensor utilities

### `(dim t i)` — size of dimension `i`, 1 past the rank
*From: ports/tipsv2/text.ss.* `shape` drops trailing size-1 dimensions (ggml's
`n_dims`), so batch 1 makes `(list-ref (shape x) 2)` fail. Every layer that
reads a batch or sequence dimension needs this.

### `(nb t i)` — byte stride of dimension `i`, extended past the rank
*From: ports/tipsv2/vision.ss.* `strides` drops trailing size-1 dimensions like
`shape`, so views that take a batch stride (`nb2`) break for batch 1. Past the
rank, the stride is the previous stride times the previous size.

### `(tokens->grid tokens w h)` / `(grid->tokens grid)`
*From: TIPSv2 vision.* Converts `[C, w*h]` tokens (row-major grid) to an image
layout `[w, h, C]` for image ops like `ggml-interpolate`, and back.

### `(format "layer.~2,'0d" i)` for tap names
Zero-padded tap names keep `tl compare` output and reference files sorted.
`format` is available in programs; the TIPSv2 port used a hand-written
`layer-name` before that.

## Layers (PyTorch semantics)

### `(linear x prefix)` — `torch.nn.Linear`
`(ggml-add (ggml-mul-mat W x) b)` with `W` = `prefix.weight` `[out, in]`
(ggml `[in, out]`) and `b` = `prefix.bias`. Variant without bias needed for
bias-free layers (check `(weight? ...)`).

### `(layer-norm x prefix eps)` — `torch.nn.LayerNorm` over dim 0
`norm` then `mul` weight, `add` bias. PyTorch's default eps is 1e-5, many ViTs
use 1e-6: always pass eps explicitly.

### `(embedding table ids)` — `torch.nn.Embedding` with any batch shape
*From: TIPSv2.* `ggml-get-rows` asserts `table.ne[2] == ids.ne[1]` (it looks
rows up per batch entry of a 3-D table), so batched ids `[L, B]` must be
flattened to `[L*B]` and the result reshaped to `[D, L, B]`.

### `(sinusoidal-positions size len max-timescale)` — transformer position signal
*From: TIPSv2.* `[sin(p f_i) ..., cos(p f_i) ...]`, `f_i = exp(-i ln(T)/(n-1))`,
built from `arange`, `exp`, an outer product as a K=1 `mul-mat`, `sin`, `cos`,
`concat`. Watch the variants: T5X/TIPS divide by `n-1`, ggml's
`timestep_embedding` (diffusion) divides by `n` and orders cos before sin.
Should take a `sin-first?` / divisor option.

### `(patch-embed image prefix p)` — ViT patch embedding (`Conv2d`, kernel = stride = p)
*From: TIPSv2 vision.* `ggml-im2col` with `GGML_TYPE_F32` and one matmul,
oriented so the result is directly `[D, patches, B]` in row-major patch order;
no permutes. Do not use `ggml-conv-2d` for f32 models: it hardcodes an f16
im2col for f32 kernels, rounding the input pixels.

### `(fused-attention x qkv-prefix out-prefix mask heads)` — timm/DINOv2 `Attention`
*From: TIPSv2 vision.* Same q/k/v split as below with timm names
(`attn.qkv`, `attn.proj`), mask optional (`#f`). Has a `ggml-flash-attn-ext`
path (q f32, K/V cast to f16, mask f16/#f): ~30% faster for ViT-B at 1026
tokens on Metal, result `[hd, heads, N, B]` is already `[D, N, B]` in memory.
Should become one attention helper with a `flash?` option and a name scheme
argument covering both `in_proj`/`out_proj` and `qkv`/`proj`.

### `(multi-head-attention x prefix mask heads)` — `torch.nn.MultiheadAttention`
*From: TIPSv2.* Fused `in_proj_weight` `[3D, D]` split with `ggml-view-4d` into
q, k, v of `[hd, heads, L, B]` (needs `strides` for byte offsets), permuted to
`[hd, L, heads, B]`, `soft-max-ext` with scale `1/sqrt(hd)`, then
`out_proj`. Everything stays within 4 dimensions (head dim, sequence, heads,
batch). Variants to add: separate q/k/v projections (HF style), causal mask,
`ggml-flash-attn-ext` fast path, rotary positions.

### `(key-padding-mask paddings)` — additive attention mask `[L, L, 1, B]`
*From: TIPSv2.* `soft-max-ext` needs the mask contiguous with a full row per
query (`mask.ne[1] >= L_q`), broadcast only over heads and batch, so the
`[L, B]` padding flags are reshaped to `[L, 1, 1, B]` and repeated.
Use `-1e30`, not `-inf`: scaling 0/1 flags by `-inf` gives `0 * -inf = NaN`.

### Position embedding interpolation (DINOv2 `interpolate_pos_encoding`)
*From: TIPSv2 vision (`patch-positions`).* `ggml-interpolate` with
`(+ GGML_SCALE_MODE_BILINEAR GGML_SCALE_FLAG_ANTIALIAS)` reproduces
`F.interpolate(mode="bilinear", antialias=True)` to ~1e-6. DINOv2 variants
with `interpolate_offset` use `scale_factor` instead of `size`; not covered yet.

### `(masked-mean x valid eps)` — mean over the sequence of valid positions
*From: TIPSv2 (`GlobalAvgPooling`).* `sum-rows` only reduces dim 0, so
transpose `[D, L, B]` to `[L, D, B]` first; counts come from the mask the same
way; `+ eps` in the denominator like the reference.

### Gemma-style pieces (from `ports/t5gemma2`)

- `(rms-norm x name)`: Gemma's RMSNorm scales by `(1 + weight)`:
  `(ggml-mul (ggml-rms-norm x eps) (ggml-scale-bias (weight name) 1.0 1.0))`.
- `(attend q k v mask)`: multi-query / grouped attention without repeating K/V:
  `ggml-mul-mat` broadcasts a single K head over the query heads.
- `(rope x pos layer)`: `ggml-rope-ext` with mode 2 (NEOX, `rotate_half`);
  HF "linear" rope scaling by `factor` is `freq_scale = 1/factor`.
  `GGML_ROPE_TYPE_*` are `#define`s the codegen doesn't export.
- `(distance-mask n-k n-q conditions)`: additive masks built in the graph from
  `ggml-arange` positions and `ggml-step`, e.g. causal, sliding windows
  (bidirectional ones too); no host-side mask inputs needed.
- `(sandwich x p pre post f)`: Gemma 2/3 blocks norm before and after each sublayer.
- `gelu`: ggml's CPU `ggml-gelu` uses an f16 lookup table (~1e-3 error); an
  exact tanh GELU from ops for comparisons.
- Placing image features into a token sequence: one `ggml-get-rows` over
  `[token embeddings; image features; special embeddings]` with a host-computed
  gather index, instead of a scatter.
- Pipelines: `string-replace`, greedy decoding with a fixed-length decoder
  (`at` input + `get-rows` to take one position's logits).
- KV cache: one state tensor per layer holding self rows then cross rows
  (no concat), `ggml-set-rows` at `pos` in an `effect`, `flash-attn-ext` with an
  f16 mask `[n_kv, 1]` for the single query, argmax on the device.
- Per-op overhead dominates single-token steps on Metal (~500 kernels): fold
  constant weight transforms into the file (`tl convert --offset`), avoid
  per-run weight math.

## Rank > 4

ggml tensors have at most 4 dimensions. Neither TIPSv2 tower needed more
(both are ported): attention is `[head-dim, tokens, heads, batch]`, the patch
embedding works on `[W, H, C, B]` images and `[p*p*C, gw, gh, B]` im2col
columns. Where 5-D would appear (windowed attention with batch, video,
multi-query groups), the usual polyfill is folding two adjacent dimensions
into one with `reshape` (e.g. `heads*batch`) for ops that treat dims 2-3 as
independent batches (`mul-mat`, `soft-max`, elementwise), and unfolding after.
A `(fold t i)` / `(unfold t i n)` pair would make that less error-prone.

## Pitfalls seen while porting

- **Definitions after expressions**: a model body (like any R6RS body) may not
  have `(define ...)` after an expression such as a `(unless ... (error ...))`
  check; wrap the check in a definition.
- **Exact GELU**: `nn.GELU()` is the erf form, `ggml-gelu` the tanh
  approximation; use `ggml-gelu-erf`.
- **Chez FFI, bool parameters (fixed in codegen)**: Chez 10.x on arm64 macOS
  misplaces stack arguments following a `stdbool` passed on the stack
  (e.g. `ggml_im2col`'s `dst_type` arrived as garbage). The generated bindings
  pass C `bool` parameters as int-sized `boolean`, which is layout-compatible.
  Worth reporting upstream.
- **Quantization needs the graph**: which tensors may be quantized depends on
  how the program uses them (`pos_embed` is added, not multiplied: ggml has no
  f32 + q8_0 add, and the CPU fallback aborts on a worker thread). `tl quantize`
  therefore builds the graph and only converts weights consumed by
  `mul-mat`/`get-rows` (through views/reshapes).
- **R6RS names**: `quotient` is R5RS (R6RS has `div`); programs now also get
  `quotient remainder modulo exact->inexact inexact->exact`, `iota`,
  `list-head`, `format`, `fold-left`, `fold-right`.
- **Metal precision**: ggml's Metal matmuls stage tiles in half precision.
  TIPSv2 f32 matches PyTorch to 1e-4 on the CPU but only to ~1.5e-4 relative
  on Metal, which fails absolute tolerances on outlier channels (values
  ~1000). Compare Metal runs with `--rtol 1e-2 --atol 1e-2` and look at
  cosine similarity.
- **bf16/f16 weights** round activations too (see README).
