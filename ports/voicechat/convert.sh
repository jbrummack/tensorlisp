#!/usr/bin/env bash
# Streams nvidia/NVIDIA-NemotronLabs-VoiceChat-11B (44 GB, all f32) from Hugging Face
# into a mixed-precision GGUF without the checkpoint ever touching the disk.
#   convert.sh q8 OUT.gguf    # ~12 GB
#   convert.sh q4 OUT.gguf    # ~8 GB
# Env: TL (path to the tl binary), HF_TOKEN (or ../../.env).
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../../.env ] && { set -a; . ../../.env; set +a; }
plan=${1:?q8 or q4}; out=${2:?output .gguf}; TL=${TL:-tl}

# Short names (ggml allows 63 bytes): llm.* backbone, enc.* FastConformer, perc.* its
# preprocessor/projection, rnnt_*, tts.* TTS decoder (tts.sub.* text embedding), codec.*.
# The first match applies, so the specific prefixes come first.
names=(
  --replace-prefix stt_model.llm.=llm.
  --replace-prefix stt_model.perception.encoder.=enc.
  --replace-prefix stt_model.perception.=perc.
  --replace-prefix stt_model.rnnt_decoder.prediction.=rnnt_dec.
  --replace-prefix stt_model.rnnt_decoder.=rnnt_dec.
  --replace-prefix stt_model.rnnt_joint.=rnnt_joint.
  --replace-prefix stt_model.=
  --replace-prefix tts_model.tts_model.embed_subword.backbone.encoder.=tts.sub.enc.
  --replace-prefix tts_model.tts_model.embed_subword.=tts.sub.
  --replace-prefix tts_model.tts_model.=tts.
  --replace-prefix tts_model.audio_codec.=codec.
  --replace-prefix tts_model.=tts_
)

# Precision (--type: first match wins, patterns use the original names). Everything not
# listed is q8_0 when it is a matrix whose rows fit 32-value blocks, f16 for other
# conv/matrix shapes, and f32 for 1-D tensors (norms, biases, dt_bias, A_log, D).
# Rows of 4480 and 15680 are no multiple of 256, so K-quants don't apply to the backbone.
common=(
  --type '*pos_bias_*=f32'                 # tiny, added to attention scores
  --type '*featurizer*=f32'                # mel filterbank and window
  --type 'stt_model.llm.*conv1d.weight=f32' # depthwise conv of the Mamba blocks (kernel 4)
  --type '*prvq*=f32' --type '*rvq_embs=f32' --type '*audio_prompt_latents*=f32'  # codebooks
  --type 'stt_model.rnnt_*=f16'            # small ASR transducer
  --type '*embed_subword.embed_tokens*=f16'
)
case $plan in
  q8) plan_types=() ;;
  q4) plan_types=(
        # The 4 attention layers are small and the lm/function heads decide every token: keep q8_0.
        # Mamba projections feed the recurrent state, so they get 5 bits; the MLPs (a third of the weights) 4.
        --type 'stt_model.llm.layers.*.mixer.in_proj.weight=q5_0'
        --type 'stt_model.llm.layers.*.mixer.out_proj.weight=q5_0'
        --type 'stt_model.llm.layers.*.mixer.up_proj.weight=q4_0'
        --type 'stt_model.llm.layers.*.mixer.down_proj.weight=q4_0'
      ) ;;
  *) echo "plan must be q8 or q4" >&2; exit 2 ;;
esac

"$TL" convert hf:nvidia/NVIDIA-NemotronLabs-VoiceChat-11B -o "$out" --dtype q8_0 \
  "${names[@]}" "${common[@]}" ${plan_types[@]+"${plan_types[@]}"} \
  --meta general.architecture=nemotron-voicechat --meta "tl.plan=$plan"
