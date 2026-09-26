# Porting notes

Patterns and pitfalls from the ports (`ports/tipsv2`, `ports/t5gemma2`,
`ports/yolo11`, `ports/ppocrv6`) that aren't stdlib functions (see `docs/stdlib.md` for
those): model-specific structure, performance lessons, and traps.

## Patterns

- **Pre/post-normed blocks** (Gemma 2/3): norm before and after each
  sublayer, then the residual — `sandwich` in `ports/t5gemma2`.
- **Image features in a token sequence**: one `ggml-get-rows` over
  `[token embeddings; image features; special embeddings]` with a
  host-computed gather index, instead of a scatter (`ports/t5gemma2`).
- **KV cache**: one state tensor per layer holding self rows then cross rows
  (no concat), `ggml-set-rows` at `pos` in an `effect`, `attn:sdpa ... 'flash #t`
  for the single query, the argmax on the device (`ports/t5gemma2`).
- **Fixed-shape decoding**: a decoder of fixed length with an `at` input and
  `get-rows` to take one position's logits builds its graph once.
- **Structure from the weights**: count repeated blocks with `weight?`
  instead of hard-coding a model size's depth (`ports/yolo11`).
- **Detection heads in pixels**: DFL decoding and anchor grids in the graph
  (`vision:dfl`, `vision:decode-ltrb`), so the output matches the reference's.
- **Kernels per device**: `(device)` is `cpu` or `gpu`; `nn:conv2d` uses it to
  pick im2col or direct convolution.
- **Per-op overhead** dominates small steps on Metal (~500 kernels per decoded
  token): fold constant weight transforms into the file (`tl convert
  --offset`), avoid per-run weight math, prefer fused ops (flash attention).
- **Batch norms folded at export**: `reference.py` folds each BN into the
  convolution before it (`ports/yolo11` via Ultralytics' `fuse()`,
  `ports/ppocrv6` by hand; a `ConvTranspose2d` weight is `[in, out, k, k]`, so
  its BN scales dim 1).
- **Two models, one file**: export both state dicts with prefixes (`det.`,
  `rec.`) from the reference script and shorten names there (ggml's 63
  bytes); pipelines then run both entries (`ports/ppocrv6`).
- **Per-shape graphs**: an entry's graph is rebuilt whenever an input shape
  changes (e.g. one text line per width in `ports/ppocrv6`); fine at
  ~100 rebuilds, keep shapes fixed where the reference does.
- **Postprocessing that must match OpenCV**: port the OpenCV code, not the
  math. `ports/ppocrv6`'s DB boxes are bit-exact only with OpenCV's hull
  order (Sklansky chains, then a cyclic shift by input index), its float32
  rotating calipers, `fillPoly`'s fixed-point spans and clang's FMA
  contraction on arm64 (`a*b + c` fused, `-ffp-contract=on`): a corner at
  1242.9999 instead of 1243 floors to another row and moves a box score by 0.1.
- **Zero-padded tap names** (`(format "layer.~2,'0d" i)`) keep `tl compare`
  output and reference files sorted.

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

- **Pickled PyTorch checkpoints can differ from current source**: activation
  modules are stored with the model (YOLO11's SPPF `cv1` has SiLU in the
  checkpoint, not in today's code). Check the loaded model, not the code.
- **Framework quirks are part of the reference**: transformers applies T5Gemma
  2's embedding scale in bf16 in the encoder but in f32 in the decoder.
- **`ggml-conv-2d`** hardcodes an f16 im2col for f32 kernels, rounding the
  input; `nn:patch-embed` / `nn:conv2d` use f32 im2col or the direct kernel.
- **`ggml-get-rows`** looks rows up per batch entry of a 3-D table: flatten
  batched ids (`nn:embedding` does).
- **`soft-max-ext` masks** need a full row per query (`mask.ne[1] >= L_q`):
  `attn:padding-mask` repeats the flags.
- **Metal matmul order**: an f32 × f16 matmul must have the f16 tensor first
  (there's no kernel for f32 × f16 the other way round).
- **`GGML_ROPE_TYPE_*`** are `#define`s the codegen doesn't export; `attn:rope`
  takes `'mode`.

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
- **im2col memory**: patches take `4 * kw * kh * C_in` bytes per output pixel
  (10 GB for PP-OCRv6's 9x9 neck convs at 400 x 304); `nn:conv2d` makes them
  in bands of rows beyond 256 MiB. ggml's Metal direct conv is ~20x slower
  than banded im2col there.
- **`ggml-conv-transpose-2d-p0` on Metal** took ~50 s for PP-OCRv6's two
  2x2/2 head convs at 1216 x 1600; kernel = stride is a matmul + pixel
  shuffle (`nn:conv-transpose2d` does that).
- **transformers' processors aren't PaddleOCR's**: its PP-OCR detector resizes
  float images (torchvision, antialiased), grows boxes by a polygon offset
  instead of pyclipper, and its recognizer resizes 8-bit images with
  `antialias=False` (torch's separable int16 path, `bilinear-no-antialias`).
  The port follows transformers; Pillow's 8-bit resize of the detector input
  still moves a few map pixels across the threshold.
