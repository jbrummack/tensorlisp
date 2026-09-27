"""Reference outputs for the PP-OCRv6 port, from transformers (CPU, float32).

    nix develop --command python reference.py DET_DIR REC_DIR IMAGE [OUT_DIR]

DET_DIR / REC_DIR are the Hugging Face repos PaddlePaddle/PP-OCRv6_medium_det_safetensors
and ..._rec_safetensors. Writes OUT_DIR (default: IMAGE's directory):

- ppocrv6.safetensors: both models with their batch norms folded into the
  convolutions and short names (det.*, rec.*; see `short_name`), for tl convert;
- characters.txt: the recognizer's vocabulary, one character per line (line 0
  is CTC's blank);
- data/: the detector's input, backbone stages, neck and probability map, the
  recognizer's input, backbone, encoder and probabilities for the first text
  line, and ocr.json: every box (PaddleOCR's order), its score, text and text
  score, as the OCR pipeline computes them (PaddleOCR 3.x's crops, the model's
  own processor settings, DB parameters from inference.yml).
"""

import json
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForTextRecognition

det_dir, rec_dir, image_path = sys.argv[1:4]
out = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(image_path))
data = os.path.join(out, "data")
os.makedirs(data, exist_ok=True)
torch.set_grad_enabled(False)


def save(name, t):
    t = t if isinstance(t, np.ndarray) else t.detach().float().numpy()
    np.save(os.path.join(data, f"{name}.npy"), np.ascontiguousarray(t.astype(np.float32)))


det = AutoModelForObjectDetection.from_pretrained(det_dir, torch_dtype=torch.float32).eval()
rec = AutoModelForTextRecognition.from_pretrained(rec_dir, torch_dtype=torch.float32).eval()
det_processor = AutoImageProcessor.from_pretrained(det_dir)
rec_processor = AutoImageProcessor.from_pretrained(rec_dir)

# --- weights: batch norms folded into the preceding (transposed) convolution


def fold(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    # Conv2d weights are [out, in/groups, kh, kw]; ConvTranspose2d [in, out, kh, kw].
    shape = (1, -1, 1, 1) if isinstance(conv, nn.ConvTranspose2d) else (-1, 1, 1, 1)
    return conv.weight * scale.reshape(shape), (bias - bn.running_mean) * scale + bn.bias


def folded_state(model):
    state = {}
    folded = set()
    for name, module in model.named_modules():
        conv = getattr(module, "convolution", None)
        bn = getattr(module, "normalization", None) or getattr(module, "norm", None)
        if isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(bn, nn.BatchNorm2d):
            w, b = fold(conv, bn)
            state[f"{name}.weight"], state[f"{name}.bias"] = w, b
            folded.add(name)
    for name, t in model.state_dict().items():
        module = name.rsplit(".", 2)[0]
        if module in folded or name.endswith("num_batches_tracked"):
            continue
        state[name] = t
    return state


# Names are shortened to stay below ggml's 63 bytes.
SHORT = [
    (r"^model\.backbone\.encoder\.convolution\.", "backbone.stem."),
    (r"^model\.backbone\.encoder\.blocks\.(\d+)\.blocks\.(\d+)\.", r"backbone.\1.\2."),
    (r"token_squeeze_excitation\.convolutions\.0\.", "se.fc1."),
    (r"token_squeeze_excitation\.convolutions\.2\.", "se.fc2."),
    (r"^model\.neck\.input_channel_adjustment_convolution\.", "neck.adjust."),
    (r"^model\.neck\.input_feature_projection_convolution\.", "neck.project."),
    (r"^model\.neck\.path_aggregation_head_convolution\.", "neck.down."),
    (r"^model\.neck\.path_aggregation_lateral_convolution\.", "neck.lateral."),
    (r"^model\.neck\.intraclass_blocks\.", "neck.intra."),
    (r"conv_reduce_channel\.", "reduce."),
    (r"symmetric_conv_long_(long|mid|short)ratio\.", r"square.\1."),
    (r"vertical_long_to_small_conv_(long|mid|short)ratio\.", r"vertical.\1."),
    (r"horizontal_small_to_long_conv_(long|mid|short)ratio\.", r"horizontal.\1."),
    (r"^head\.conv_(down|up|final)\.", r"head.\1."),
    (r"^neck\.intra\.(\d+)\.conv_final\.", r"neck.intra.\1.final."),
    (r"^head\.encoder\.conv_block\.0\.", "svtr.skip."),
    (r"^head\.encoder\.conv_block\.1\.", "svtr.reduce."),
    (r"^head\.encoder\.conv_block\.2\.", "svtr.local."),
    (r"^head\.encoder\.svtr_block\.", "svtr.blocks."),
    (r"^head\.encoder\.norm\.", "svtr.norm."),
    (r"self_attn\.projection\.", "attn.proj."),
    (r"self_attn\.qkv\.", "attn.qkv."),
    (r"layer_norm(\d)\.", r"norm\1."),
    (r"^head\.head\.", "ctc."),
]


def short_name(name):
    for pattern, repl in SHORT:
        name = re.sub(pattern, repl, name)
    return name


weights = {}
for prefix, model in (("det", det), ("rec", rec)):
    for name, t in folded_state(model).items():
        short = f"{prefix}.{short_name(name)}"
        assert len(short) < 64 and short not in weights, short
        weights[short] = t.detach().float().contiguous()
save_file(weights, os.path.join(out, "ppocrv6.safetensors"))
characters = rec_processor.character_list
assert all(len(c) == 1 for c in characters[1:]) and characters[0] == "blank"
with open(os.path.join(out, "characters.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(characters))
print(len(weights), "tensors,", len(characters), "characters")

# --- detection

image = Image.open(image_path).convert("RGB")
rgb = np.array(image)
inputs = det_processor(images=image, return_tensors="pt")
x = inputs["pixel_values"]
save("det.image", x)
stages = det.model.backbone(x).feature_maps
for i, s in enumerate(stages):
    save(f"det.stage{i + 1}", s)
neck = det.model.neck(stages)
save("det.neck", neck)
prob = det.head(neck)
save("det.prob", prob)

with open(os.path.join(det_dir, "inference.yml")) as f:
    post = dict(re.findall(r"^  (box_thresh|max_candidates|thresh|unclip_ratio): (\S+)", f.read(), re.M))
db = dict(threshold=float(post["thresh"]), box_threshold=float(post["box_thresh"]),
          max_candidates=int(post["max_candidates"]), unclip_ratio=float(post["unclip_ratio"]), min_size=3)
result = det_processor.post_process_object_detection(
    type("Out", (), {"last_hidden_state": prob}), target_sizes=inputs["target_sizes"], **db)[0]
boxes = result["boxes"].numpy().astype(np.float32)   # [n, 4, 2] (x, y) corners
scores = result["scores"].numpy()

# --- PaddleOCR 3.x's OCR pipeline: sort boxes, crop, recognize


def sort_quad_boxes(boxes):
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(len(order) - 1):
        for j in range(i, -1, -1):
            a, b = boxes[order[j]], boxes[order[j + 1]]
            if abs(b[0][1] - a[0][1]) < 10 and b[0][0] < a[0][0]:
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return order


def ordered_corners(points):
    points = sorted(list(points), key=lambda p: p[0])
    a, d = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    b, c = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return np.array([points[a], points[b], points[c], points[d]])


def get_rotate_crop_image(img, points):
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    m = cv2.getPerspectiveTransform(points.astype(np.float32), target)
    crop = cv2.warpPerspective(img, m, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if crop.shape[0] * 1.0 / crop.shape[1] >= 1.5:
        crop = np.rot90(crop)
    return crop


def get_minarea_rect_crop(img, points):
    rect = cv2.minAreaRect(np.array(points).astype(np.int32))
    return get_rotate_crop_image(img, ordered_corners(cv2.boxPoints(rect)))


order = sort_quad_boxes(boxes)
lines = []
for n, i in enumerate(order):
    crop = np.ascontiguousarray(get_minarea_rect_crop(rgb, boxes[i]))
    pixels = rec_processor(images=crop, return_tensors="pt")["pixel_values"]
    if n == 0:
        Image.fromarray(crop).save(os.path.join(data, "rec.crop.png"))
        save("rec.image", pixels)
        features = rec.model(pixels).last_hidden_state
        save("rec.backbone", rec.model.backbone(pixels).feature_maps[-1])
        save("rec.pooled", features)
        save("rec.encoder", rec.head.encoder(features).last_hidden_state)
    probs = rec(pixels).last_hidden_state
    if n == 0:
        save("rec.probs", probs)
    text = rec_processor.post_process_text_recognition(type("Out", (), {"last_hidden_state": probs}))[0]
    lines.append({"box": boxes[i].astype(int).tolist(), "score": round(float(scores[i]), 5),
                  "text": text["text"], "text_score": round(text["score"], 5),
                  "crop": [crop.shape[1], crop.shape[0]], "input_width": pixels.shape[-1]})

json.dump({"size": [rgb.shape[1], rgb.shape[0]], "input": list(x.shape[2:]), "db": db, "lines": lines},
          open(os.path.join(data, "ocr.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"det input {tuple(x.shape)}, {len(lines)} lines")
for line in lines[:8]:
    print(f"{line['score']:.3f} {line['text_score']:.3f} {line['text']}")
