"""Reference outputs for the YOLO11 port, from Ultralytics (CPU, float32).

    nix develop --command python reference.py MODEL_DIR [OUT_DIR]

Reads MODEL_DIR/yolo11n.pt and MODEL_DIR/bus.jpg. Writes MODEL_DIR/yolo11n.safetensors
(convolutions with their batch norms folded in, as Ultralytics' fuse() does) and,
to OUT_DIR (default MODEL_DIR/data), the letterboxed input, every layer's output
(layer.NN), the head output and the detections after NMS on the original image.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import save_file
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

model_dir = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "data")
os.makedirs(out, exist_ok=True)


def save(name, t):
    np.save(os.path.join(out, f"{name}.npy"), np.ascontiguousarray(t.detach().float().numpy()))


net = YOLO(os.path.join(model_dir, "yolo11n.pt")).model.float().eval()
net.fuse(verbose=False)
save_file({k: v.contiguous() for k, v in net.state_dict().items()}, os.path.join(model_dir, "yolo11n.safetensors"))
names = net.names

# Preprocessing like the predictor, but a fixed 640 x 640 letterbox (auto=False).
bgr = cv2.imread(os.path.join(model_dir, "bus.jpg"))
boxed = LetterBox(new_shape=(640, 640), auto=False)(image=bgr)
x = torch.from_numpy(boxed[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
save("image", x)

outputs = {}
for i, layer in enumerate(net.model):
    layer.register_forward_hook(lambda m, inp, o, i=i: outputs.__setitem__(i, o))
with torch.no_grad():
    y = net(x)
    y = y[0] if isinstance(y, (list, tuple)) else y
for i, o in outputs.items():
    if torch.is_tensor(o):
        save(f"layer.{i:02d}", o)
save("head", y)  # [1, 4 + 80, 8400]: xywh (pixels) + class probabilities

det = non_max_suppression(y, conf_thres=0.25, iou_thres=0.7)[0]
det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], bgr.shape[:2])
detections = [
    {"box": [round(v, 2) for v in d[:4].tolist()], "score": round(d[4].item(), 4), "class": names[int(d[5])]}
    for d in det
]
json.dump({"size": [bgr.shape[1], bgr.shape[0]], "detections": detections}, open(os.path.join(out, "detections.json"), "w"), indent=1)
json.dump(names, open(os.path.join(model_dir, "names.json"), "w"))
print(len(net.state_dict()), "tensors;", y.shape, "head")
for d in detections:
    print(d)
