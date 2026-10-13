# NemotronLabs VoiceChat 11B

`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`: a full-duplex speech-to-speech model. A cache-aware streaming
FastConformer reads the user's audio (80 ms frames), a Nemotron-H 9B hybrid (27 Mamba2, 25 ReLU² MLP, 4 attention
layers) predicts the agent's text and function-call channels frame by frame, a Gemma3-style TTS decoder (EAR-TTS,
mixture-of-Gaussians head) turns the text into codec latents and a 31-quantizer codec decodes them to 22.05 kHz audio.
The reference code is NeMo's `speechlm2` (branch `nemotron-labs-voicechat` of NVIDIA-NeMo/Speech).

## Status

| part | state |
|---|---|
| streaming conversion (`convert.sh`: 44 GB f32 from Hugging Face -> mixed-precision GGUF, never on disk) | done, q8 (11.3 GiB) and q4 (8.3 GiB) |
| text backbone (`backbone.ss`: `step`, `generate`) | done; tiny model equals numpy to 3e-6; real q8 writes coherent text |
| log-mel front end (`crates/tensorlisp/src/nemo_mel.rs`, `nemo-mel-*`) | done; equals NeMo's recipe in numpy to 1e-5 |
| FastConformer encoder + projection (`audio.ss`: `encode`) | done; tiny model equals numpy to 1e-6; real model transcribes (below) |
| RNNT ASR head (`transcribe`) | done: "Hello, do you know what color the sky is" |
| duplex loop over the backbone (`duplex`, `converse`) | done: agent text channel is coherent ("Hi! How can I help you today? Yes, the sky is blue.") |
| codec decoder (`tts.ss`: `codec-decode`) | done; real weights vs numpy f32: cosine 0.99996 (the 1x1 convs round activations to f16) |
| TTS: character-aware subword embedding, fusion, 28-layer Gemma3 decoder with guidance, MoG sampler, residual quantizer (`tts.ss`: `speak`, `speak-text`) | done; the speech transcribes back exactly ("The quick brown fox jumps over the lazy dog") |
| the whole turn (`converse-speak`: user audio -> agent text channel -> agent speech) | done on q4 (43 s for 15.7 s of audio, ~2.7x real time); the agent says "Hi, how can I help you today? Yes, the sky is blue." |
| streaming (`stream.ss`: `listen`; encoder with caches, per-frame mel/encoder/LM/TTS/codec, `tl-serve` pipeline) | done; the streamed encoder equals `encode` (cosine 0.9999998 on 196 frames); runs over RPC; **2.1x slower than real time** on an M-series Mac (below) |
| system prompt and tool calling (`listen-tools`: prompt prefill, function channel, silence frames, constrained call with outlines-core, response injection) | done; the model calls `generate_random_number` with min 1, max 50 for `sample_fc.wav` and speaks "Your random number is twenty" |
| split over two machines (`brain` = encoder + LM, `voice` = speech decoder + codec; frames relayed between two servers) | done; same text, same tool call, speech transcribes back (below) |
| user-text (ASR) channel in the LM, sampling options for the LM | not started |

## Files

- `convert.sh` -- `q8` or `q4` GGUF (`tl convert hf:...` with a `--type` plan; K-quants don't fit the backbone's row sizes).
- `backbone.ss`, `audio.ss` -- one program: `cat backbone.ss audio.ss`; assets `tokenizer.json` (the Nemotron
  tokenizer) and `rnnt_tokenizer.model`. Run without repacking the 11 GB file:
  `tl run MODEL.gguf -e converse --program voicechat.ss --asset tokenizer.json=tokenizer.json --asset rnnt_tokenizer.model=rnnt_tokenizer.model --raw audio=@clip.wav`
  (`-e transcribe` for the ASR head, `-e generate --raw prompt=...` for the plain LLM). Pipeline state (Mamba, KV
  cache, LSTM) lives in the process: one conversation per run for now.
- `stream.ss`, `fc_prompt.ss` -- the live path and the function-calling prompt (append both: `cat backbone.ss audio.ss tts.ss fc_prompt.ss stream.ss`), see below.
- `tts.ss` -- the speech output; `make_char_table.py` writes the per-token character ids its subword encoder needs
  (`tts.sub.chars`, `tts.sub.charlen`), which `tl merge` adds to the GGUF (`tl merge MODEL.gguf chars.gguf -o MODEL-tts.gguf`).
  `npy2wav.py` turns a pipeline's audio result (`tl run ... -o DIR`) into a wav.
- `reference_tiny.py`, `reference_audio.py`, `reference_codec.py`, `test_tiny.sh` -- numpy references of the HF/NeMo math (float64, read from
  the source, not imported: there is no torch here) and the tests that run them against tiny random models.
- `config.json`, `modeling_nemotron_h.py`, ... -- the Nemotron-Nano-9B-v2 files the backbone was ported from.

## Running it

The full q8 model (11.3 GiB) plus the speech decoder does not fit the GPU of a 16 GiB machine; use the q4 file
(`convert.sh q4`, then `tl merge`) for `converse-speak`, or a speech-only file for the TTS and the ASR check:
`tl merge MODEL-tts.gguf -o speech.gguf --drop 'llm.*' --drop embed_tokens.weight --drop lm_head.weight --drop function_head.weight`.

```sh
cat backbone.ss audio.ss tts.ss fc_prompt.ss stream.ss > voicechat.ss
A="--program voicechat.ss --asset tokenizer.json=tokenizer.json --asset rnnt_tokenizer.model=rnnt_tokenizer.model"
tl run q4-tts.gguf -e converse-speak $A --raw audio=@user.wav -o out    # out/results/0/audio.npy at 22.05 kHz
tl run speech.gguf -e speak-text $A --raw text="Hello there" -o out     # the speech decoder alone
python3 -I npy2wav.py out/results/0/audio.npy out.wav
tl run speech.gguf -e transcribe $A --raw audio=@out.wav                # check it with the model's own ASR
```

## Streaming

`listen` takes `audio` as a stream of 16 kHz chunks of any size and emits `audio` (1764 samples = 80 ms at 22.05 kHz per
frame; the first is 1758 samples and a final 6 follow) and `token` (the agent's words) while it runs;
the results are `text`, `tokens` and `audio` (everything). Per 80 ms frame: mel (`NemoMel`, incremental) -> `enc-sub`
(a window of 25 mel frames; the last output is exact) -> `enc-step` (FastConformer with a 70-frame key/value cache and an
8-frame conv cache per layer) -> `duplex` -> `tts-frame` -> `codec-step` (the codec with states). `stream-reset` zeroes
the states at the start, so one process serves one conversation after another. `stream-encode` runs only the encoder.

```sh
tl run q4-tts.gguf -e listen $A --raw audio=@user.wav#80 -o out    # #80: the wav as 80 ms chunks; emits are timed on stderr
tl run speech.gguf -e stream-encode $A --raw audio=@user.wav#80 -o enc   # compare with `-e encode` on `-e mel`'s output
tl pack COPY.gguf --program voicechat.ss --asset ...                # then tl-serve installs it; the test below drives it
TL_VC_MODEL=COPY.gguf TL_VC_WAV=user.wav cargo test --release -p tensorlisp-serve --test voicechat_live -- --ignored --nocapture
```

Measured on the 16 GiB Mac (q4, sample_general.wav, 15.7 s): the agent text is the same as `converse-speak`'s and the speech
transcribes back through the model's own ASR. Per 80 ms frame: encoder 23 ms, language model 60 ms (the 8 GB of q4
weights at memory bandwidth), speech decoder ~50 ms, codec ~35 ms, so ~170 ms, 2.1x slower than real time. Over RPC with the
audio pushed in real time the output ends 17 s behind after 15.6 s of audio; the first word arrives after 4.3 s.
The codec runs one token at a time with its convolution and overlap-add states (`codec-step`), the speech decoder uses
flash attention and four quantizer graphs of which an iteration runs only those with active levels (these three took
it from 280 ms). Real time on this machine would also need the language model to read less than it does now.
The speech decoder's KV cache holds 1024 frames (82 s) and the language model's attention cache 2048; a conversation
longer than that must be restarted. The streamed waveform is 6 samples behind the codes (the codec's padding); the
chunked `tts-decode` of the offline pipelines differs from `codec-step` by ~1% (correlation 0.993) at its chunk edges.

## Tool calling

`listen-tools` ([audio stream] [system string] [tools string] [reply string]) is `listen` with a system prompt and tools.
`system` is the system message (empty: NVIDIA's default), `tools` a JSON array of tools (`{"name", "description",
"parameters": <JSON schema>}`, or OpenAI-style `{"type": "function", "function": {...}}`) in the order of their
properties, which is the order the model writes the arguments in. The prompt is NVIDIA's template (`fc_prompt.ss`), put in
front of the audio one token per position (`prefill`: each prompt position reads `embed(token) + 3 embed(pad)`).

The model writes a tool call on the *function channel*, a second head next to the text head, one token per frame:
`<SPECIAL_20>` `<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>` `<SPECIAL_21>`. The pipeline then follows NeMo's
layout (`DuplexSTTModel._expand_for_function_calling`): the frames of the call, the response and the closing token get the
encoder's output for digital silence instead of the user's audio (what the user says meanwhile waits in a queue and is read
afterwards), the agent's speech for them is cut out, the call is handed to the client, and the response
`<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` is fed on the function channel one token per frame, followed by one frame that
closes it with `<SPECIAL_22>`. `emit`s: `call` and `response` (text) as they happen; results add `calls`.

The client answers a call with `(client-call "tool" call-text)` -> the response as JSON text (over `tl-serve`:
`Callbacks.call("tool", [text])`); `reply`, if not empty, is used instead (for `tl run`). The model likes short ASCII
responses (`{"result": 20}`), they are spoken.

The call tokens are constrained with outlines-core (`src/guide.rs`; host functions `tool-call-regex`, `guide-new`,
`guide-mask`, `guide-advance!`): a regular expression of `<TOOLCALL>[call(, call){0,2}]</TOOLCALL>` with every `call` a
tool's JSON schema, compiled against the Nemotron tokenizer's byte-level BPE vocabulary (0.4 s), gives the allowed token
ids at each step; `duplex` adds the mask to the function head's logits and the end token is allowed only once the call is
complete. Unconstrained, the model writes the same call (checked for `sample_fc.wav`).

```sh
T='[{"type": "function", "function": {"name": "generate_random_number", "description": "Generate a random integer between min and max (inclusive).", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}}]'
tl run q4-tts.gguf -e listen-tools $A --raw audio=@sample_fc.wav#80 --raw system= --raw tools="$T" --raw reply='{"result": 20}' -o out
```

(`sample_fc.wav` and the template come from NVIDIA-NeMo/Speech, `examples/speechlm2`; leave a few seconds of silence after
the request so the agent has time to answer.) Not like NeMo: the SOTC frame is computed with the user's real audio (NeMo
re-runs the audio with the call frames inserted before it, which needs the call in advance), and the speech decoder runs
on the frames of a call (so its pacing continues) but nothing of them is played.

## Notes

- The speech decoder expects the language model's text channel: `<s>`, then the subwords on consecutive frames (it paces
  the speech itself, from its KV cache), then pad frames while it is spoken, then `</s>`, which cuts the speech off.
  Feeding the subwords spread out, or `</s>` right after the last one, gives one word or a truncated sentence.
- Sampling is stochastic (top-p 0.95 over the mixture, noise 0.001, guidance 0.2, as in the checkpoint's config); the
  seed is fixed in `tts-speak-values`. The T5Gemma encoder's attention-logit softcapping (50) is not applied.
- Gemma-style `(1 + weight)` norms are applied in the graph (`'offset 1`), not folded into the file.

- The model dimensions are literals between `CONFIG-BEGIN/END` markers (a program can't read weight shapes at load
  time); the tiny test substitutes its own.
- The checkpoint's mel window and filterbank are the analytic ones rounded to bfloat16 (checked equal for every
  element); `NemoMel` rounds the same way.
- FastConformer `rel_shift` is a strided view of the position scores (query i, key j reads column t-1-i+j).
- The RNNT weights are f16 in the files (including biases); `audio.ss` adds them as f32.

## Split over two machines

`brain` (encoder, language model, tool calls; needs no `tts.*`/`codec.*` weights) and `voice` (speech decoder and codec;
needs no `llm.*`) are separate pipelines. `brain` takes the same arguments as `listen-tools` and emits one `frame` per 80 ms:
a 2-element f32 array `[agent text token, 1 if inside a tool call]` (plus `token`, `call`, `response`); `voice` takes a `frames`
stream of those and emits `audio` (the frames inside a call are not spoken). Nothing flows back from `voice` to `brain`, so a
relay (the client) only forwards frames: 8 bytes per frame one way, 16 kHz audio (32 KB/s as 16-bit) into `brain` and 22.05 kHz
speech (44 KB/s) out of `voice`.

```sh
tl merge q4-tts.gguf -o brain.gguf --drop 'tts.*' --drop 'codec.*' --drop 'tts_*'   # 7.6 GB; voice: the speech.gguf above (2.1 GB)
tl pack brain.gguf $A; tl pack speech.gguf $A     # A as above; then install each on its own tl-serve
TL_VC_BRAIN=brain.gguf TL_VC_VOICE=speech.gguf TL_VC_WAV=user.wav [TL_VC_TOOLS=tools.json] \
  cargo test --release -p tensorlisp-serve --test voicechat_split -- --ignored --nocapture
```

Checked with both servers in one process on the Mac (same GPU, so not faster than `listen`): on `sample_fc.wav` with the tool
the agent text, the call and the speech (the model's own ASR: "Hi there, how can I help you today your random number is
twenty") are the same as unsplit. Real-time numbers need the two halves on separate machines.
