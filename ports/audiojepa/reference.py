"""Exports Audio-JEPA's encoder weights and dumps test inputs and reference activations as .npy files.

Run inside the flake's shell (nix develop path:.), with the upstream code cloned and the
checkpoint plus inference_example.py downloaded (see README.md):
    python reference.py SRC MODEL_DIR OUT_DIR

Writes MODEL_DIR/encoder.safetensors (the `encoder.*` weights, prefix stripped) and
OUT_DIR: mel.npy [batch, 256, 128] (Kaldi fbank of the test clips), mel0/mel1.npy [256, 128], clip0/clip1.wav (32 kHz, 16-bit),
ref/ embed, block.00..block.11, norm [batch, 128, 768].
"""

import os
import sys

import wave

import numpy as np
import torch
from safetensors.torch import save_file

src, model_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, model_dir)
import types
sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(hf_hub_download=None))  # only used by its main()
import inference_example as ie  # the model card's shim and constants

ie.install_flash_attn_shim()
root = os.path.abspath(src)
sys.path.insert(0, root)
ie.stub_upstream_inits(__import__("pathlib").Path(root))
from src.models.components.vision_transformer import VisionTransformer

encoder = VisionTransformer(
    input_size=(ie.TARGET_TIME_BINS, ie.N_MELS), patch_size=ie.PATCH_SIZE, in_chans=1,
    embed_dim=ie.EMBED_DIM, depth=ie.DEPTH, num_heads=ie.NUM_HEADS, mlp_ratio=ie.MLP_RATIO,
    use_flash_attn=False,
)
state = torch.load(os.path.join(model_dir, "JEPA.ckpt"), map_location="cpu", weights_only=False)
raw = state.get("state_dict", state)
sd = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.") and not k.startswith("encoder_")}
encoder.load_state_dict(sd, strict=True)
encoder.eval()

out = os.path.join(model_dir, "encoder.safetensors")
if not os.path.exists(out):
    save_file({k: v.contiguous() for k, v in sd.items()}, out)
    print("wrote", out, len(sd), "tensors")

# Test clips: a chirp with harmonics and a noisy two-tone, 10 s and 4 s (zero padded by fbank handling).
def clip(seconds, kind):
    n = int(ie.SAMPLE_RATE * seconds)
    t = torch.arange(n).float() / ie.SAMPLE_RATE
    g = torch.Generator().manual_seed(kind)
    if kind == 0:
        phase = 2 * torch.pi * (200 * t + 300 * t * t)
        w = 0.3 * torch.sin(phase) + 0.1 * torch.sin(2 * phase) + 0.05 * torch.randn(n, generator=g)
    else:
        w = 0.3 * torch.sin(2 * torch.pi * 440 * t) * (t < 2) + 0.2 * torch.sin(2 * torch.pi * 3000 * t) + 0.1 * torch.randn(n, generator=g)
    # 16-bit PCM, as the clips are also written to WAV files (clip{i}.wav) for `tl run --raw`.
    return (torch.round(w.clamp(-1, 1 - 1 / 32768) * 32768) / 32768).unsqueeze(0)

waves = [clip(10, 0), clip(4, 1)]
mel = torch.stack([ie.compute_mel_spec(w) for w in waves])  # [B, 1, 256, 128]
taps = {}
def keep(name):
    return lambda _m, _i, o: taps.__setitem__(name, o.detach().numpy())
encoder.patch_embed.register_forward_hook(lambda _m, _i, o: taps.__setitem__("patch", o.detach().numpy()))
for i, b in enumerate(encoder.blocks):
    b.register_forward_hook(keep(f"block.{i:02d}"))
with torch.inference_mode():
    x = encoder.patch_embed(mel)
    taps["embed"] = (x + encoder.pos_embed).numpy()
    y = encoder(mel)
taps["norm"] = y.numpy()

def save(d, name, v):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, name + ".npy"), np.asarray(v, dtype=np.float32))

save(out_dir, "mel", mel.squeeze(1).numpy())
for i, w in enumerate(waves):
    with wave.open(os.path.join(out_dir, f"clip{i}.wav"), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(ie.SAMPLE_RATE)
        f.writeframes((w.squeeze(0) * 32768).round().short().numpy().astype("<i2").tobytes())
    save(out_dir, f"mel{i}", mel[i].squeeze(0).numpy())
for k, v in taps.items():
    if k != "patch":
        save(os.path.join(out_dir, "ref"), k, v)
print("mel", tuple(mel.shape), "out", tuple(y.shape), "range", float(y.min()), float(y.max()))
