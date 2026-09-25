"""Reference outputs for the T5Gemma 2 (270M-270M) port, from transformers (CPU, float32).

    nix develop --command python reference.py MODEL_DIR image|text OUT_DIR

image: the model card's captioning prompt on MODEL_DIR/bee.jpg.
text:  a text-only prompt.

Writes the processor outputs, intermediate activations (numpy order) and a
greedy generation (tokens + text, generation.json) to OUT_DIR.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, T5Gemma2ForConditionalGeneration

model_dir, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)


def save(name, t):
    a = t.detach().float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    np.save(os.path.join(out, f"{name}.npy"), np.ascontiguousarray(a.astype(np.float32)))


processor = AutoProcessor.from_pretrained(model_dir)
model = T5Gemma2ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).float().eval()

if mode == "image":
    prompt = "<start_of_image> in this image, there is"
    image = Image.open(os.path.join(model_dir, "bee.jpg"))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
else:
    prompt = "The capital of France is"
    inputs = processor(text=prompt, return_tensors="pt")

save("input_ids", inputs["input_ids"])
save("attention_mask", inputs["attention_mask"])
if "pixel_values" in inputs:
    save("pixel_values", inputs["pixel_values"])

encoder = model.get_encoder()
with torch.no_grad():
    if "pixel_values" in inputs:
        vision = encoder.vision_tower(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        save("vision_embeddings", vision.hidden_states[0])
        save("vision_layer0", vision.hidden_states[1])
        save("vision_out", vision.last_hidden_state)
        save("image_features", encoder.multi_modal_projector(vision.last_hidden_state))
    enc = encoder(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        pixel_values=inputs.get("pixel_values"), output_hidden_states=True, return_dict=True,
    )
    save("encoder_embeddings", enc.hidden_states[0])
    save("encoder_layer0", enc.hidden_states[1])
    save("encoder_out", enc.last_hidden_state)

    generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    tokens = generated[0].tolist()

    # Teacher-forced decoder pass over the generated sequence.
    dec = model(
        encoder_outputs=enc, attention_mask=inputs["attention_mask"],
        decoder_input_ids=generated, output_hidden_states=True, return_dict=True, use_cache=False,
    )
    save("decoder_tokens", generated)
    save("decoder_layer0", dec.decoder_hidden_states[1])
    save("decoder_hidden", dec.decoder_hidden_states[-1])
    save("logits", dec.logits)

json.dump({
    "prompt": prompt,
    "tokens": tokens,
    "text": processor.decode(generated[0], skip_special_tokens=True),
    "raw": processor.decode(generated[0]),
}, open(os.path.join(out, "generation.json"), "w"), indent=1)
print(json.load(open(os.path.join(out, "generation.json"))))
print({f: np.load(os.path.join(out, f)).shape for f in sorted(os.listdir(out)) if f.endswith(".npy")})
