"""Reference outputs for gemma4.ss: transformers' Gemma4ForCausalLM (CPU, bf16).

    python reference.py MODEL_DIR "The capital of France is" > reference.json

Prints the prompt's token ids, the top-10 next-token logits (after softcapping)
and the 32 greedy tokens, as JSON.
"""
import json
import sys

import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM

model_dir, prompt = sys.argv[1], sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = Gemma4ForCausalLM.from_pretrained(
    model_dir, dtype=torch.bfloat16, key_mapping={r"^model.language_model.": "model."}
).eval()

ids = tokenizer(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1].float()
    generated = model.generate(ids, max_new_tokens=32, do_sample=False)[0, ids.shape[1]:]

top = torch.topk(logits, 10)
print(json.dumps({
    "ids": ids[0].tolist(),
    "top_ids": top.indices.tolist(),
    "top_logits": top.values.tolist(),
    "generated": generated.tolist(),
    "text": tokenizer.decode(generated),
}))
