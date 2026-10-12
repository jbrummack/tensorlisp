#!/bin/sh
# build_audio.sh OUT.gguf TEXT.gguf MMPROJ.gguf TOKENIZER.json
#
# Gemma 4 E2B with audio from Google's QAT q4_0 files (https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf:
# gemma-4-E2B_q4_0-it.gguf and gemma-4-E2B-it-mmproj.gguf; tokenizer.json from google/gemma-4-E2B-it):
# one GGUF with the text model under the names gemma4.ss reads, the audio tower (f16), and the program.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=$1; text=$2; mm=$3; tokenizer=$4
tl=${TL:-tl}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
TL=$tl "$here/merge.sh" "$tmp/merged.gguf" "$text" "$mm"
cat "$here/gemma4.ss" "$here/audio_clips.ss" "$here/audio.ss" "$here/asr.ss" > "$tmp/program.ss"
$tl pack "$tmp/merged.gguf" --program "$tmp/program.ss" --asset tokenizer.json="$tokenizer" -o "$out"
