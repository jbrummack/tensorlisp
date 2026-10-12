"""Reference for the audio path: Gemma4ForConditionalGeneration (transformers 5.19, CPU bf16) on a 16 kHz wav.
Prints the audio tower's output for `--tower` (an .npy of [tokens, 1536] to compare with the port's `emb`) and
the greedy transcript. Needs torch, transformers, pillow, soundfile and google/gemma-4-E2B-it-qat-q4_0-unquantized
(or google/gemma-4-E2B-it) downloaded to the directory given as MODEL.

  python reference_audio.py MODEL speech.wav [--tower emb.npy]
"""
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

model_dir, wav = sys.argv[1], sys.argv[2]
processor = AutoProcessor.from_pretrained(model_dir)
model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
audio, rate = sf.read(wav, dtype="float32")
assert rate == 16000, "the model takes 16 kHz audio"
messages = [{"role": "user", "content": [{"type": "text", "text": "Transcribe this audio."}, {"type": "audio", "audio": audio}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audio=[audio], return_tensors="pt")
if "--tower" in sys.argv:
    with torch.no_grad():
        tokens = model.model.get_audio_features(inputs["input_features"], inputs["input_features_mask"], return_dict=True).pooler_output
    np.save(sys.argv[sys.argv.index("--tower") + 1], tokens[0].float().numpy())
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(repr(processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
