#!/bin/sh
# merge.sh OUT.gguf TEXT.gguf MMPROJ.gguf: the text model under the names ports/gemma4/gemma4.ss reads, plus the audio tower
T=${TL:-tl}
OUT=$1; TEXT=$2; MM=$3
exec $T merge "$TEXT" "$MM" -o "$OUT" --dtype f16 \
 --drop 'v.*' --drop 'mm.input_projection.weight' --drop 'rope_freqs.weight' \
 --map 'token_embd.weight=embed_tokens.weight' --map 'per_layer_token_embd.weight=embed_tokens_per_layer.weight' \
 --map 'per_layer_model_proj.weight=per_layer_model_projection.weight' --map 'per_layer_proj_norm.weight=per_layer_projection_norm.weight' \
 --map 'output_norm.weight=norm.weight' \
 --map 'blk.*.attn_norm.weight=layers.*.input_layernorm.weight' --map 'blk.*.attn_q.weight=layers.*.self_attn.q_proj.weight' \
 --map 'blk.*.attn_k.weight=layers.*.self_attn.k_proj.weight' --map 'blk.*.attn_v.weight=layers.*.self_attn.v_proj.weight' \
 --map 'blk.*.attn_output.weight=layers.*.self_attn.o_proj.weight' --map 'blk.*.attn_q_norm.weight=layers.*.self_attn.q_norm.weight' \
 --map 'blk.*.attn_k_norm.weight=layers.*.self_attn.k_norm.weight' --map 'blk.*.post_attention_norm.weight=layers.*.post_attention_layernorm.weight' \
 --map 'blk.*.ffn_norm.weight=layers.*.pre_feedforward_layernorm.weight' --map 'blk.*.ffn_gate.weight=layers.*.mlp.gate_proj.weight' \
 --map 'blk.*.ffn_up.weight=layers.*.mlp.up_proj.weight' --map 'blk.*.ffn_down.weight=layers.*.mlp.down_proj.weight' \
 --map 'blk.*.post_ffw_norm.weight=layers.*.post_feedforward_layernorm.weight' --map 'blk.*.inp_gate.weight=layers.*.per_layer_input_gate.weight' \
 --map 'blk.*.proj.weight=layers.*.per_layer_projection.weight' --map 'blk.*.post_norm.weight=layers.*.post_per_layer_input_norm.weight' \
 --map 'blk.*.layer_output_scale.weight=layers.*.layer_scalar'
