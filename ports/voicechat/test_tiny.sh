#!/usr/bin/env bash
# Checks backbone.ss and audio.ss on a tiny random model against reference_tiny.py (numpy, float64):
# logits of a prefill (f32 KV cache variant: exact; the shipped f16 cache: ~2e-3), 48 greedy tokens
# through the generate pipeline (state carried across calls), and the speech encoder (1e-6).
set -euo pipefail
cd "$(dirname "$0")"
TL=${TL:-tl}; D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
python3 -I reference_tiny.py make "$D"
"$TL" convert "$D/tiny.safetensors" -o "$D/tiny.gguf" --type '*conv1d.weight=f32' --type '*pos_bias_*=f32' >/dev/null
cat "$D/backbone_tiny.ss" "$D/audio_tiny.ss" > "$D/all_tiny.ss"
# exact variant: f32 cache, plain attention
sed -e 's/f16 kv-dim capacity/f32 kv-dim capacity/' -e "s/(\* 2 kv-dim) (\* 2 head-dim) (\* 2 kv-dim capacity)/(* 4 kv-dim) (* 4 head-dim) (* 4 kv-dim capacity)/" \
    -e "s/'flash #t/'flash #f/" "$D/backbone_tiny.ss" > "$D/exact.ss"
cp "$D/tiny.gguf" "$D/exact.gguf"
"$TL" pack "$D/exact.gguf" --program "$D/exact.ss" --asset tokenizer.json=tokenizer.json >/dev/null
"$TL" compare "$D/exact.gguf" -e step -i token="$D/token.npy" -i pos="$D/pos.npy" -r logits="$D/logits.npy" --device cpu --atol 1e-4
"$TL" pack "$D/tiny.gguf" --program "$D/all_tiny.ss" --asset tokenizer.json=tokenizer.json --asset rnnt_tokenizer.model=rnnt_tokenizer.model >/dev/null
"$TL" compare "$D/tiny.gguf" -e encode -i mel="$D/mel.npy" -r emb="$D/emb.npy" -r proj="$D/proj.npy" --device cpu
"$TL" compare "$D/tiny.gguf" -e step -i token="$D/token.npy" -i pos="$D/pos.npy" -r logits="$D/logits.npy" --device cpu --atol 5e-3 --rtol 1e-2
"$TL" run "$D/tiny.gguf" -e generate --raw prompt="The capital of France is" --device cpu --json > "$D/gen.json"
python3 -I - "$D" <<'PY'
import json, subprocess, sys, os
d = sys.argv[1]
r = {x['name']: x for x in json.load(open(f'{d}/gen.json'))['results'][0]}
ids = [int(v) for v in r['prompt_ids']['values']]
got = [int(v) for v in r['tokens']['values']]
out = subprocess.run(['python3', '-I', 'reference_tiny.py', 'greedy', d, *map(str, ids)], capture_output=True, text=True,
                     env={**os.environ, 'N': str(len(got))}, check=True).stdout
ref = eval(out)[len(ids):]
print('generate: %d tokens, %s' % (len(got), 'equal to numpy' if ref == got else 'MISMATCH %s vs %s' % (ref, got)))
sys.exit(ref != got)
PY
