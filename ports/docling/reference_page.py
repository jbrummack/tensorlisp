"""Token-level reference for docling.ss: transformers (CPU, float32), the processor's
defaults (image splitting on, longest edge 2048, 512 px tiles), greedy decoding.

    python reference_page.py MODEL_DIR IMAGE [PROMPT] > reference-page.json
"""
import json
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

model_dir, image_path = sys.argv[1], sys.argv[2]
prompt_text = sys.argv[3] if len(sys.argv) > 3 else "Convert this page to docling."
torch.set_grad_enabled(False)
model = Idefics3ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
processor = AutoProcessor.from_pretrained(model_dir)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)[0, inputs["input_ids"].shape[1]:]
print(json.dumps({
    "prompt": prompt,
    "tiles": inputs["pixel_values"].shape[1],
    "length": inputs["input_ids"].shape[1],
    "generated": gen.tolist(),
    "text": processor.tokenizer.decode(gen),
}))
