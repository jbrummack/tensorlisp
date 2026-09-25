# YOLO11 (Ultralytics)

[YOLO11n](https://docs.ultralytics.com/models/yolo11/) object detection
(80 COCO classes), ported from `ultralytics/nn/modules` (8.4). `yolo11.ss`
builds the network from the weights, so the n/s/m/l/x sizes should all
work (only n is tested). The weights are AGPL-3.0, like Ultralytics.

```sh
tl run yolo11n.gguf --raw image=@bus.jpg
#   boxes    [5, 4]   [16.72, 228.39, 798.10, 734.07] ...   (xyxy on the original image)
#   scores   [5]      [0.9396, 0.9015, 0.8454, 0.8264, 0.3844]
#   classes  [5]      [5, 0, 0, 0, 0]
#   labels   "bus 0.94, person 0.9, person 0.85, person 0.83, person 0.38"
```

- Preprocess: letterbox to 640 x 640 (fill 114, Ultralytics' geometry,
  Pillow's bilinear resampling instead of OpenCV's), RGB in [0, 1].
- Model: `head [1, 84, 8400]` like Ultralytics' inference output: cx, cy, w, h
  in input pixels (DFL decoding and anchors are in the graph), then the 80
  class probabilities. Taps `layer.00` .. `layer.22` are the blocks' outputs.
  Inputs may be any size that's a multiple of 32.
- Postprocess: Ultralytics' `non_max_suppression` (confidence 0.25, IoU 0.7,
  max 300) through autopro's `detect`, boxes mapped back with
  `boxes-unletterbox`, class names in the program.
- Convolutions: im2col + matmul on the GPU, ggml's direct convolution on the
  CPU, picked with `(device)`.

## Building the file

```sh
nix develop --command python reference.py path/to/model-dir   # yolo11n.pt + bus.jpg -> yolo11n.safetensors, data/
tl convert yolo11n.safetensors --strip-prefix model. -o weights.gguf
tl pack weights.gguf --program yolo11.ss -o yolo11n.gguf
```

`reference.py` exports the state dict after `fuse()` (batch norms folded
into the convolutions) and dumps every layer's output for a fixed
640 x 640 letterbox of `bus.jpg`, the head output and the final detections.
The flake builds Ultralytics from nixpkgs (not in the binary cache, ~1 min);
remove it with `nix store gc`.

## Validation

Against Ultralytics (PyTorch float32, CPU), same input tensor:

| | max abs error | cosine |
|---|---|---|
| layers 0-22 (CPU) | <= 1.5e-4 | 1.000000 |
| head (CPU) | 4.8e-3 (box coordinates up to ~640 px) | 1.000000 |
| head (Metal) | 1.06 px on some box coordinates | 1.000000 |

End to end from `bus.jpg` the detections are the same five objects as
Ultralytics' (bus, 4 persons), scores within 0.015 and boxes within a few
pixels; the differences come from Pillow vs OpenCV resizing in the letterbox.

Found while porting: the released checkpoint has SiLU after SPPF's `cv1`,
while current Ultralytics source builds that conv without an activation
(module activations are pickled with the checkpoint). The program follows
the checkpoint (`sppf-cv1-activation`).

## Performance (M1 Pro, 640 x 640)

| | per image |
|---|---|
| Metal, im2col convolutions | ~48 ms |
| Metal, direct convolutions | ~134 ms |
| CPU, direct convolutions | ~110-170 ms (busy machine) |
| CPU, im2col | ~390 ms |

About 450 kernels for 6.5 GFLOP, so it's dominated by per-op overhead and
layout copies: every 1x1 conv transposes its input to channels-innermost
(54 copies), and 3x3 convs go through im2col. A channels-last layout
throughout, fused conv + bias + SiLU kernels, or a compiling backend
(CoreML / Neural Engine) are the ways to go faster. f16 weights don't help
(the network is 10 MiB; activations dominate).
