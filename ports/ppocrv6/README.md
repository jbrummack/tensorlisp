# PP-OCRv6 (medium)

[PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors)
(text detection, 15.5M parameters) and
[PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_safetensors)
(text recognition, 18,710 characters, 48 languages) in one GGUF, ported from
transformers 5.17 (`pp_lcnet_v4`, `pp_ocrv6_medium_det`, `pp_ocrv6_small_rec`,
which the medium recognizer uses). Apache-2.0, like the weights.

```sh
tl run ppocrv6.gguf --entry ocr --raw image=@page.png       # boxes, scores, text-scores, text
tl run ppocrv6.gguf --raw image=@page.png                   # detection only: boxes, scores
tl run ppocrv6.gguf --entry recognize --raw image=@line.png # one text line: text, score
```

`ocr` on the model card's sample page (1224 x 1584, 98 lines):

```
device MTL0, load 806.2 ms, pipeline ocr: 8685.8 ms
  boxes        [98, 4, 2] [354, 107, 878, 107, 878, 128, 354, 128, …]
  scores       [98]       [0.8882, 0.9446, 0.8823, 0.8792, 0.8129, 0.8840, …]
  text-scores  [98]       [0.9900, 0.9999, 0.9973, 0.9910, 0.9957, 0.9952, …]
  text         "Algorithms for the Markov Entropy Decomposition\nAndrew J. Ferris and David Poulin\nDépartement de Physique, …"
```

## Program

| entry / pipeline | inputs (numpy order) | outputs |
|---|---|---|
| `det` (default) | image `[1, 3, H, W]`, BGR, ImageNet-normalized, H and W multiples of 32 | prob `[1, 1, H, W]`; postprocess: boxes `[n, 4, 2]`, scores `[n]` |
| `rec` | image `[1, 3, 48, W]`, BGR, in [-1, 1] | probs `[1, W/8, 18710]` (CTC, blank first) |
| `ocr` (pipeline) | image | boxes, scores (reading order), text-scores, text (one line per box) |
| `recognize` (pipeline) | image of one line | text, score |

- **Detector**: LCNetV4 backbone (large stem, depthwise-separable blocks with
  squeeze-excitation and a GELU channel MLP), RepLKFPN neck (top-down and
  bottom-up paths, 9x9 convolutions, "intraclass" blocks of 7/5/3 square,
  vertical and horizontal kernels), DB head (3x3 conv, two 2x2/2 transposed
  convs, sigmoid). Taps `det.stage1` .. `det.stage4`, `det.neck`.
- **Recognizer**: LCNetV4 with (2, 1) strides in stages 3 and 4, a (3, 2)
  average pool to one row, SVTR encoder (1x1 convs, a 1x7 depthwise conv, two
  pre-LN transformer blocks with 8 heads, SiLU), linear CTC head, softmax.
  Inputs narrower than 320 are zero-padded to 320 in the graph, like the
  processor. Taps `rec.backbone`, `rec.pooled`, `rec.encoder`.
- **Preprocessing** (transformers' processors): the detector's input keeps
  the shorter side at least 736 (at most 4000 on the longer), rounded to
  multiples of 32, bilinear; the recognizer's is 48 high and as wide as the
  aspect ratio gives (at most 3200), `bilinear-no-antialias`.
- **Postprocessing**: DB boxes (threshold 0.2, box threshold 0.45, unclip
  1.4, up to 3000 candidates, from `inference.yml`), PaddleOCR's reading order
  and min-area-rectangle crops (rotated when 1.5 times taller than wide), greedy
  CTC with the `characters` asset. These are host functions (`text-boxes`,
  `text-boxes-order`, `image-crop-text`, `ctc-greedy`, `vocabulary-text`)
  from autopro's `ocr` module, which reproduces OpenCV bit for bit.
- Not included: document orientation, unwarping and text-line orientation
  (PaddleOCR's optional modules). Vertical text is read as PaddleOCR reads it
  without them (the sample's rotated arXiv margin comes out as "rX 12").

## Building the file

```sh
huggingface-cli download PaddlePaddle/PP-OCRv6_medium_det_safetensors --local-dir det
huggingface-cli download PaddlePaddle/PP-OCRv6_medium_rec_safetensors --local-dir rec
nix develop --command python reference.py det rec page.png out     # -> out/ppocrv6.safetensors, characters.txt, data/
tl convert out/ppocrv6.safetensors -o weights.gguf
tl pack weights.gguf --program ppocrv6.ss --asset characters=out/characters.txt -o ppocrv6.gguf
```

`reference.py` loads both models with transformers (CPU, float32), folds
every batch norm into its convolution (for the transposed convs along the
output channels), shortens the names below ggml's 63 bytes (`det.backbone.2.1.se.fc1`,
`rec.svtr.blocks.0.attn.qkv`, …; see `SHORT`), writes the recognizer's
vocabulary and dumps the detector's and the first line's intermediate
outputs plus the whole OCR result (PaddleOCR's sorting and crops copied
from PaddleOCR 3.x). The flake takes transformers, torch and OpenCV from
nixpkgs; remove them with `nix store gc`.

```sh
tl compare ppocrv6.gguf --entry det -i image=out/data/det.image.npy --reference-dir out/data -r prob=out/data/det.prob.npy
tl compare ppocrv6.gguf --entry rec -i image=out/data/rec.image.npy --reference-dir out/data -r probs=out/data/rec.probs.npy
```

## Validation

Against transformers (PyTorch float32, CPU), same input tensors:

| | CPU max abs error | Metal max abs error | cosine |
|---|---|---|---|
| detector stages, neck | <= 4.8e-5 | <= 1.2e-2 | 1.000000 |
| detector probability map | 2.5e-5 | 5.8e-3 | 1.000000 |
| recognizer backbone, encoder | <= 4.5e-5 | <= 5.4e-2 (values ~7) | 0.999997 |
| recognizer probabilities | 1.1e-5 | 4.7e-3 | 1.000000 |

End to end from the PNG (`ocr` against `ocr.json`), CPU and Metal alike: all
98 texts identical, boxes identical except one corner 1 pixel off, text
scores within 0.004, detection scores within 0.033. The postprocessing
itself is exact: on the same probability map, the DB boxes and scores equal
transformers' bit for bit. The remaining differences come from the detector's
input: Pillow resizes the 8-bit image, transformers the float image, and that
moves a few pixels of the map across the 0.2 threshold.

f16 weights (`tl convert --dtype f16`, 79 MiB instead of 157 MiB) give the
same 98 texts, but a page takes ~24 s on Metal instead of ~9 s (stages 3-4 of
the recognizer's backbone get slower); prefer f32.

## Performance (M1 Pro, busy machine: load average ~13)

| | CPU | Metal | PyTorch CPU |
|---|---|---|---|
| detector, 1216 x 1600 | ~6.3-7.4 s | ~2.2 s | ~8.7 s |
| recognizer, one line (48 x 1197) | ~270 ms | ~52 ms | ~74 ms |
| `ocr`, sample page (98 lines) | ~34 s | ~8.7 s | |

The detector is ~290 GMAC at this size, mostly the neck's 9x9 convolutions
at stride 4. On Metal they run as im2col + matmul in bands of rows (the
patches alone would take 10 GB), and the head's 2x2/2 transposed convs as a
matmul plus pixel shuffle (ggml's Metal kernel took ~50 s). Each line has
its own width, so the `rec` graph is rebuilt for most lines. On the CPU,
ggml's spinning threads suffer most from the load; expect better numbers on
an idle machine.
