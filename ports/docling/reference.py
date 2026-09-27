"""Reference outputs for the Granite Docling 258M port (Idefics3 architecture),
from transformers (CPU, float32).

    python reference.py MODEL_DIR IMAGE [OUT_DIR]

MODEL_DIR is a local checkout of ibm-granite/granite-docling-258M (config.json,
model.safetensors, tokenizer.json, chat_template.jinja). Writes, to OUT_DIR
(default MODEL_DIR/data):

- the preprocessed pixel values;
- vision tower: patch embeddings, position-embedded input, every encoder
  layer's output, the post-layernorm output;
- pixel_shuffle's output in isolation (the riskiest op to port - dumped on
  its own so it can be checked before anything downstream is trusted);
- the connector's final projection (image_hidden_states);
- the merged input embeddings (text + spliced image features);
- every decoder layer's hidden state and the final logits, teacher-forced
  over the full prompt + a short greedy continuation;
- the greedy-generated text itself, for an end-to-end sanity check.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

model_dir = sys.argv[1]
image_path = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(model_dir, "data")
os.makedirs(out, exist_ok=True)
torch.set_grad_enabled(False)


def save(name, t):
    t = t if isinstance(t, np.ndarray) else t.detach().float().numpy()
    np.save(os.path.join(out, f"{name}.npy"), np.ascontiguousarray(t))


model = Idefics3ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
processor = AutoProcessor.from_pretrained(model_dir)
inner = model.model  # Idefics3Model: vision_model, connector, text_model

# --- preprocessing: chat template + single global image (no splitting) ---

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image = Image.open(image_path).convert("RGB")

# Force "no splitting": the processor's image_seq_len/global-image expansion
# still applies, only the sub-tile grid is skipped.
inputs = processor(text=prompt, images=[image], return_tensors="pt", do_image_splitting=False)
save("input_ids", inputs["input_ids"][0])
save("pixel_values", inputs["pixel_values"][0, 0])  # [num_images=1, frames=1, 3, 512, 512] -> [3, 512, 512]
print("prompt:", prompt)
print("input_ids shape:", tuple(inputs["input_ids"].shape), "pixel_values shape:", tuple(inputs["pixel_values"].shape))

pixel_values = inputs["pixel_values"][:, 0]  # [1, 3, 512, 512], drop the (unsplit) frame axis

# --- vision tower, dumped stage by stage ---

vision = inner.vision_model
patch_embeds = vision.embeddings.patch_embedding(pixel_values)  # [1, 768, 32, 32]
patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [1, 1024, 768]
save("vision.patch_embeds", patch_embeds)

position_ids = torch.arange(patch_embeds.shape[1]).unsqueeze(0)
x = patch_embeds + vision.embeddings.position_embedding(position_ids)
save("vision.embeddings", x)

layer_outputs = {}


def hook(i):
    def fn(module, inp, output):
        layer_outputs[i] = output[0] if isinstance(output, tuple) else output

    return fn


handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(vision.encoder.layers)]
vision_out = vision(pixel_values=pixel_values).last_hidden_state  # already post_layernorm'd
for h in handles:
    h.remove()
for i, o in layer_outputs.items():
    save(f"vision.layer{i:02d}", o)
save("vision.post_layernorm", vision_out)

# --- connector: pixel_shuffle in isolation, then the projection ---

connector = inner.connector
shuffled = connector.pixel_shuffle(vision_out, connector.scale_factor)
save("connector.pixel_shuffle", shuffled)
image_hidden_states = connector.modality_projection(shuffled)
save("connector.projection", image_hidden_states)
print("image_hidden_states shape:", tuple(image_hidden_states.shape))

# --- merged embeddings (text + spliced image features) ---

text_embeds = inner.text_model.embed_tokens(inputs["input_ids"])
image_token_id = model.config.image_token_id
mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(text_embeds)
merged = text_embeds.clone()
merged = merged.masked_scatter(mask, image_hidden_states.to(merged.dtype))
save("merged_embeds", merged)
n_image_tokens = int((inputs["input_ids"] == image_token_id).sum())
print("image token count:", n_image_tokens, "(expect 64 for one unsplit 512x512 image)")

# --- decoder: teacher-forced hidden states + logits over prompt, then a short greedy run ---

decoder_layer_outputs = {}


def dhook(i):
    def fn(module, inp, output):
        decoder_layer_outputs[i] = output[0] if isinstance(output, tuple) else output

    return fn


dhandles = [layer.register_forward_hook(dhook(i)) for i, layer in enumerate(inner.text_model.layers)]
full_out = model(inputs_embeds=merged, attention_mask=inputs["attention_mask"])
for h in dhandles:
    h.remove()
for i, o in decoder_layer_outputs.items():
    save(f"decoder.layer{i:02d}", o)
save("logits", full_out.logits[0])

gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
text = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
json.dump({"prompt": prompt, "generated": text, "n_image_tokens": n_image_tokens}, open(os.path.join(out, "generation.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("generated:", text)
