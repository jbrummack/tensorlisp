"""Dumps TIPSv2 inputs and reference activations as .npy files.

Run inside the flake's shell:
    nix develop --command python reference.py text   ../../models/tipsv2-b14 ../../models/tipsv2-b14/data
    nix develop --command python reference.py vision ../../models/tipsv2-b14 ../../models/tipsv2-b14/data

text   -> OUT/text:        ids.npy, paddings.npy [batch, 64]; ref/ embed, layer.00..11,
                           final_ln [batch, 64, 768] and embedding [batch, 768]
vision -> OUT/vision-SIZE: image.npy [batch, 3, SIZE, SIZE] in [0, 1]; ref/ embed,
                           block.00..11, norm [batch, tokens, 768], cls and register
                           [batch, 768], patches [batch, patches, 768]
"""

import os
import sys
import warnings

import numpy as np
import torch
from safetensors.torch import load_file

mode, model_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, model_dir)
torch.manual_seed(0)


def save(directory, name, value):
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, f"{name}.npy"), np.asarray(value, dtype=np.float32))


def load_tower(module, prefix):
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return module.eval()


def text():
    from text_encoder import TextEncoder, Tokenizer  # shipped with the checkpoint

    texts = [
        "a photo of a bus",
        "A dog running on the beach at sunset, with waves in the background.",
        "tips",
    ]
    encoder = load_tower(
        TextEncoder({"hidden_size": 768, "mlp_dim": 3072, "num_heads": 12, "num_layers": 12}, vocab_size=32000),
        "text_encoder.",
    )
    ids, paddings = Tokenizer(os.path.join(model_dir, "tokenizer.model")).tokenize(texts, max_len=64)
    taps = {}

    def keep(name, layout):
        def hook(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            # Transformer blocks work in [L, N, D]; the program's taps are [N, L, D].
            taps[name] = (x.permute(1, 0, 2) if layout == "LND" else x).detach().numpy()
        return hook

    for i, block in enumerate(encoder.transformer.resblocks):
        block.register_forward_hook(keep(f"layer.{i:02d}", "LND"))
    encoder.ln_final.register_forward_hook(keep("final_ln", "NLD"))
    encoder.transformer.register_forward_pre_hook(
        lambda _m, args: taps.__setitem__("embed", args[0].permute(1, 0, 2).detach().numpy())
    )
    with torch.no_grad():
        embedding = encoder(torch.from_numpy(ids), torch.from_numpy(paddings))

    directory = os.path.join(out_dir, "text")
    save(directory, "ids", ids)
    save(directory, "paddings", paddings)
    for name, value in taps.items():
        save(os.path.join(directory, "ref"), name, value)
    save(os.path.join(directory, "ref"), "embedding", embedding.numpy())
    print("text: tokens per text", (1 - paddings).sum(axis=1).tolist(), "->", directory)


def test_images(size, batch=2):
    """Deterministic images in [0, 1]: smooth color gradients, a disc and noise."""
    y, x = np.mgrid[0:size, 0:size] / size
    rng = np.random.default_rng(0)
    images = []
    for b in range(batch):
        disc = ((x - 0.3 - 0.3 * b) ** 2 + (y - 0.5) ** 2 < 0.05).astype(np.float64)
        channels = [x, y * (1 - x), 0.5 + 0.5 * np.sin(8 * x + 5 * y + b)]
        image = np.stack([np.clip(0.8 * c + 0.2 * disc, 0, 1) for c in channels])
        images.append(np.clip(image + rng.normal(0, 0.02, image.shape), 0, 1))
    return np.stack(images).astype(np.float32)


def vision():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "xFormers is not available"
        from image_encoder import vit_base  # shipped with the checkpoint

    encoder = load_tower(
        vit_base(
            img_size=448, patch_size=14, ffn_layer="mlp", block_chunks=0, init_values=1.0,
            interpolate_antialias=True, interpolate_offset=0.0,
        ),
        "vision_encoder.",
    )
    taps = {}

    def keep(name):
        return lambda _m, _i, output: taps.__setitem__(name, output.detach().numpy())

    for i, block in enumerate(encoder.blocks):
        block.register_forward_hook(keep(f"block.{i:02d}"))
    encoder.norm.register_forward_hook(keep("norm"))
    encoder.blocks[0].register_forward_pre_hook(
        lambda _m, args: taps.__setitem__("embed", args[0].detach().numpy())
    )

    for size in (448, 224):
        images = test_images(size)
        taps.clear()
        with torch.no_grad():
            cls, register, patches = encoder(torch.from_numpy(images))
        directory = os.path.join(out_dir, f"vision-{size}")
        save(directory, "image", images)
        for name, value in taps.items():
            save(os.path.join(directory, "ref"), name, value)
        save(os.path.join(directory, "ref"), "cls", cls[:, 0].numpy())
        save(os.path.join(directory, "ref"), "register", register[:, 0].numpy())
        save(os.path.join(directory, "ref"), "patches", patches.numpy())
        print(f"vision {size}: patches", tuple(patches.shape), "->", directory)


{"text": text, "vision": vision}[mode]()
