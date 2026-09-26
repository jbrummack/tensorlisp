# tensorlisp stdlib

Shared building blocks for programs, as five R6RS libraries. A program
imports them at the top; each library's names then carry its prefix, so
every helper is recognizably from the stdlib and from which part:

```scheme
(import (tl nn) (tl attn))

(model (inputs [x f32 (768 _ batch)])
  (define h (nn:layer-norm x "blocks.0.norm1" 'eps 1e-6))
  (outputs [y (attn:multi-head h "blocks.0.attn" 12 'names 'timm) 3]))
```

| library | prefix | contents |
|---|---|---|
| `(tl tensor)` | `tensor:` | shape/stride helpers, slicing, concatenation, casts |
| `(tl nn)` | `nn:` | layers with PyTorch semantics: linear, norms, embedding, GELU, convolutions, pooling, positions |
| `(tl attn)` | `attn:` | masks, heads, scaled dot-product attention (GQA, flash), RoPE, fused multi-head blocks |
| `(tl vision)` | `vision:` | token/image layouts, position-embedding resizing, detection-head decoding |
| `(tl util)` | `util:` | plain Scheme helpers for pipelines: strings, lists |

Everything unprefixed is the core (`(tensorlisp)`, always available): `model`,
`weight`, `shape`, `tap`, the `ggml-*` ops, the host functions (`tokenize`,
`image-resize`, `detect`, ...), `pipeline`, `run`, `define-state`, ...
Source: `crates/tensorlisp/src/scheme/stdlib/*.ss`; tests:
`crates/tensorlisp/tests/stdlib.rs` (each function against a direct
implementation of the PyTorch semantics).

## Imports

- `(import (tl nn))` binds `nn:linear`, `nn:layer-norm`, ... The prefix is
  always the library's; that's the canonical spelling used in docs and ports.
- Standard R6RS import sets work for other names: `(only (tl tensor) dim)`,
  `(prefix (tl nn) layer/)`, `(rename (tl attn) (sdpa attention))`,
  `(except ...)`. An explicit import set is used as written (no prefix added).
- Imports must come first in the program. Only `(tl ...)` libraries can be
  imported (programs stay sandboxed).

## Conventions

- **Shapes and axes are ggml order** (innermost first): tokens `[D, L, B]`,
  images `[W, H, C, B]` (torch's NCHW memory), per-head tensors
  `[hd, L, heads, B]`. `B` may be 1 or missing.
- **Weight prefixes** are module paths in the state dict **without** the
  trailing dot: `(nn:linear x "blocks.0.mlp.fc1")` reads
  `blocks.0.mlp.fc1.weight` and, if the file has it, `blocks.0.mlp.fc1.bias`.
- **Options** are `'name value` pairs after the positional arguments:
  `(nn:layer-norm x "norm" 'eps 1e-6)`. Unknown options are errors.
- **Masks** are additive: 0 where attention is allowed, `attn:blocked`
  (-1e30) elsewhere, `[L_k, L_q(, 1, B)]`; combine them with `ggml-add`.
- Errors name the function with its prefix (`nn:conv2d: ...`).

## `(tl tensor)` — `tensor:`

| function | |
|---|---|
| `(tensor:dim t i)` | size of axis `i`; 1 past the rank (`shape` drops trailing size-1 axes, e.g. batch 1) |
| `(tensor:stride t i)` | byte stride of axis `i`, extended past the rank |
| `(tensor:as-f32 t)` | `t` cast to f32 unless it already is (e.g. f16 weights added to activations) |
| `(tensor:flatten t)` | all elements as one axis (copies if not contiguous) |
| `(tensor:slice t axis from n)` | view of elements `[from, from + n)` along `axis`, e.g. channels 64..127 of an image: `(tensor:slice x 2 64 64)` |
| `(tensor:concat ts axis)` | concatenation of a list of tensors |

## `(tl nn)` — `nn:`

| function | PyTorch | notes |
|---|---|---|
| `(nn:linear x prefix)` | `nn.Linear` | bias optional |
| `(nn:layer-norm x prefix 'eps 1e-5)` | `nn.LayerNorm` over axis 0 | weight/bias optional; many ViTs use eps 1e-6 |
| `(nn:rms-norm x prefix 'eps 1e-6 'offset 0)` | RMSNorm | `'offset 1` for Gemma's `(1 + weight)`; cheaper to store `1 + w` in the file (`tl convert --offset`) |
| `(nn:embedding table ids)` | `nn.Embedding` | `table` a weight name or tensor `[D, vocab]`; ids `[L]` or `[L, B]` → `[D, L, B]` |
| `(nn:gelu x)` | `nn.GELU()` (erf) | |
| `(nn:gelu-tanh x)` | `GELU(approximate="tanh")` | ggml's kernel: f16 table on the CPU (~1e-3), exact on GPUs |
| `(nn:gelu-tanh-exact x)` | same | exact everywhere, a few more ops (for comparisons) |
| `(nn:conv2d x prefix 'stride 1 'padding k/2 'method 'auto)` | `nn.Conv2d` (groups 1) | `[W, H, C, B]` → `[W', H', C_out, B]`; kernels may be non-square; `'stride` / `'padding` an integer or `(height width)` like PyTorch's tuples (padding defaults to kernel/2 per axis); `'method`: `auto` (im2col + matmul on GPUs, direct kernel on the CPU — each ~3x faster on its device), `im2col` (in bands of output rows when the patches would exceed 256 MiB), `direct` |
| `(nn:conv2d-depthwise x prefix 'stride 1 'padding k/2)` | `nn.Conv2d(groups=C)` | stride / padding / kernels as `conv2d` |
| `(nn:conv-transpose2d x prefix 'stride 1)` | `nn.ConvTranspose2d` (groups 1, padding 0) | batch 1; kernel = stride (upsampling) runs as a matmul + pixel shuffle, others through ggml's op (very slow on Metal) |
| `(nn:patch-embed image prefix patch)` | ViT `Conv2d(kernel=stride=patch)` | → tokens `[D, patches, B]`, row-major; f32 im2col (`ggml-conv-2d` would round pixels to f16) |
| `(nn:upsample-nearest x factor)` | `nn.Upsample(mode="nearest")` | |
| `(nn:max-pool x k 'stride k 'padding 0)` | `nn.MaxPool2d` | |
| `(nn:avg-pool x k 'stride k 'padding 0)` | `nn.AvgPool2d(count_include_pad=True)` | |
| `(nn:sinusoidal-positions size len 'max-timescale 10000 'divisor 'n-1 'order 'sin-cos)` | transformer position signal `[size, len]` | `'divisor 'n` and `'order 'cos-sin` for tensor2tensor / diffusion timestep variants |
| `(nn:masked-mean x valid 'eps 0)` | mean over valid positions | `x [D, L, B]`, `valid [L, B]` (1 = keep) → `[D, B]` |

## `(tl attn)` — `attn:`

| function | |
|---|---|
| `attn:blocked` | the additive value for blocked positions (-1e30; not -inf, which gives NaN when multiplied by 0) |
| `(attn:padding-mask valid n-q)` | keys that may be attended: `valid [L_k]` or `[L_k, B]` (1 = token) → `[L_k, n-q, 1, B]` |
| `(attn:causal-mask n 'window #f)` | `[n, n]`: a query sees itself and earlier keys; with a window only the last `w` (HF sliding window: `0 <= q - k < w`) |
| `(attn:window-mask n 'left l 'right r)` | bidirectional window `[n, n]` (HF: `0 <= q - k < l` or `0 < k - q < r`) |
| `(attn:relative-mask n-k n-q conditions)` | general form: allowed where `scale * (q - k) + bias > 0` for every `(scale . bias)`; built in the graph (`arange`, `step`) |
| `(attn:split-heads x heads)` | `[heads * hd, L, B]` → `[hd, L, heads, B]` (view; head-major channels like torch) |
| `(attn:merge-heads x)` | `[hd, L, heads, B]` → `[heads * hd, L, B]` |
| `(attn:sdpa q k v 'mask #f 'scale 1/sqrt(hd) 'flash #f)` | scaled dot-product attention: `q [hd, L_q, heads, B]`, `k`/`v [hd, L_k, kv-heads, B]` → `[heads * hd, L_q, B]`. `kv-heads` may divide `heads` (grouped/multi-query attention; consecutive query heads share K/V like HF `repeat_kv`, without copies). `'flash #t`: `ggml-flash-attn-ext` with f16 K/V/mask (cast unless already f16, e.g. an f16 KV cache) — faster on GPUs for long sequences and single-token decoding, slightly less exact |
| `(attn:rope x pos 'base 10000 'scale 1 'mode 'neox 'dims hd)` | rotary embedding on `x [hd, heads, L]` (before `split-heads`), `pos` i32 `[L]`; `'scale` multiplies positions (HF "linear" scaling by `f` is `1/f`); `'mode 'neox` rotates halves (HF `rotate_half`), `'normal` adjacent pairs |
| `(attn:multi-head x prefix heads 'names 'torch 'mask #f 'flash #f)` | self-attention with a fused `[q; k; v]` projection on `[D, L, B]`: `'torch` names (`nn.MultiheadAttention`: `prefix.in_proj_weight`, `prefix.out_proj`) or `'timm` (`prefix.qkv`, `prefix.proj`) |

## `(tl vision)` — `vision:`

| function | |
|---|---|
| `(vision:tokens->grid tokens w h)` / `(vision:grid->tokens grid)` | tokens `[C, w*h]` (row-major) ↔ image layout `[w, h, C]`, for image ops on token sequences (pooling, interpolation) |
| `(vision:resize-positions pos w h)` | position embeddings `[C, g*g]` of a square grid resized to `w x h`, like `F.interpolate(mode="bilinear", antialias=True)` (ViT/DINOv2 `interpolate_pos_encoding` with a target size) |
| `(vision:anchor-grid w h)` | two values: cell centers x + 0.5 and y + 0.5, each `[w*h]` |
| `(vision:dfl raw 'bins 16)` | YOLOv8+ distribution focal loss decoding: `[A, 4 * bins]` → expected distances `[A, 4]` (l t r b) |
| `(vision:decode-ltrb dist w h pixels-per-cell)` | distances from the cell centers of a `w x h` grid → boxes `[A, 4]` cx, cy, w, h in pixels (Ultralytics `dist2bbox`) |

## `(tl util)` — `util:`

| function | |
|---|---|
| `(util:string-replace s from to)` | every occurrence replaced |
| `(util:string-repeat s n)`, `(util:string-join strings separator)` | |
| `(util:make-list n x)`, `(util:pad-list xs n value)`, `(util:last xs)` | |
| `(util:->integers xs)` | numbers (e.g. from `array->list`, which returns floats) as exact integers |

## What stays in programs

Model-specific structure: Gemma's pre/post "sandwich" norms, Ultralytics'
attention layout, how image features are placed into a token sequence, KV
cache layouts and generation loops. `docs/porting-notes.md` collects those
patterns and the pitfalls found while porting.
