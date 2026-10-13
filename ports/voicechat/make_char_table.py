"""The character-aware subword encoder (NeMo CharAwareSubwordEncoder / build_vocabs) embeds a subword from the characters
of its token string. This writes the per-token character ids for the Nemotron tokenizer (tokenizer.json) as a safetensors
file for `tl convert` (merged into the model GGUF with `tl merge`):

  tts.sub.chars    [vocab, 76] int32   ids into the 256-entry character vocabulary (the single-character tokens, dense,
                                       ordered by token id), padded with 256
  tts.sub.charlen  [vocab]     int32   how many (tokens with no known character have 0)

  python3 -I make_char_table.py tokenizer.json OUT.safetensors
"""
import json, struct, sys
import numpy as np

tok = json.load(open(sys.argv[1]))
vocab = dict(tok["model"]["vocab"])
for a in tok["added_tokens"]:
    vocab.setdefault(a["content"], a["id"])
size = max(vocab.values()) + 1
single = {k: v for k, v in vocab.items() if len(k) == 1}
char_ids = {c: i for i, c in enumerate(sorted(single, key=lambda k: single[k]))}
assert len(char_ids) == 256
width = max(sum(c in char_ids for c in k) for k in vocab)
chars = np.full((size, width), len(char_ids), np.int32)
lens = np.zeros(size, np.int32)
for token, i in vocab.items():
    ids = [char_ids[c] for c in token if c in char_ids]
    chars[i, :len(ids)] = ids
    lens[i] = len(ids)

tensors = {"tts.sub.chars": chars, "tts.sub.charlen": lens}
header, blob, off = {}, b"", 0
for name, a in tensors.items():
    b = np.ascontiguousarray(a, "<i4").tobytes()
    header[name] = {"dtype": "I32", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
    blob += b
    off += len(b)
hb = json.dumps(header).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
open(sys.argv[2], "wb").write(struct.pack("<Q", len(hb)) + hb + blob)
print("tokens", size, "max chars", width)
