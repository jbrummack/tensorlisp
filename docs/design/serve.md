# tensorlisp RPC server

`crates/tensorlisp-serve` (`tl-serve`) serves installed models over Cap'n Proto RPC. The schema is
`crates/tensorlisp-serve/schema/tensorlisp.capnp`; building needs the `capnp` tool on `PATH`.

```
tl-serve --listen unix:/tmp/tl.sock --store ~/.cache/tensorlisp/models --device auto
tl-serve --listen tcp:127.0.0.1:7700
```

## What exists

- **Server** (bootstrap): `info`, `list`, `install` (stream a GGUF into an `Upload`, then `finish`),
  `installFrom` (a path the server can read; URLs are not supported), `open`, `remove`.
- **Store**: `<sha256>.gguf` files plus `index.json` (alias -> hash). Installing the same bytes twice is one model;
  `open` takes an alias, a hash or a unique hash prefix of 8+ digits. Removing the last alias deletes the file.
- **Model**: `describe` (entries with typed inputs, pipelines with their raw inputs, the state tensors; output
  types are not known to the runtime yet and stay empty), `load`/`unload`, `run(entry, inputs, session)` (the
  any-model path: tensors in, tensors out), `pipeline(name, args, session)` -> `Run`, `session`.
- **Run**: `status`, `result` (waits), `cancel` (a pipeline already executing is not interrupted, its result is
  dropped). Pipelines run on `spawn_blocking`; arguments are checked when `pipeline` is called.
- **Tensors**: dims are outermost-first. Inputs may be f32, f16, bf16, i8/i16/i32/i64, u8 or bool and are converted
  to f32 (the runtime's boundary type). Results go out as f32 by default; `RunOptions.outputDtype` selects f16 or
  bf16 and halves the bytes. Cap'n Proto has no half-float scalar, so f16 travels as little-endian `Data`.
  `outputsByRef` leaves results on the server as `TensorRef` (host memory for now); a `TensorRef` can also be
  passed as an input, in which case the server reads it back through the capability.
- **Values**: tensor, tensorRef, text, image (rgb8/rgba8/gray8 or an encoded file), audio (channels averaged to
  mono), bytes, numbers. Pipeline results come back as tensors or text.

## Streaming

A pipeline parameter of kind `stream` is an input stream, and `(emit 'name value)` sends a chunk to an
output stream:

```scheme
(pipeline running ([xs stream] [label string])
  (let loop ([n 0])
    (let ([x (stream-recv xs)])          ; blocks; array, audio, image or string; #f at the end
      (if x
          (begin (emit 'y (output (run double [x x]) 'y)) (loop (+ n 1)))
          (results [count n])))))
```

The client calls `Run.input("xs")` for a `Sink` (push tensor, text, audio or image values, then `done`) and
`Run.output("y")` for a `Source` (`pull(max)` waits for chunks and reports `eof` when the pipeline is done;
a failed pipeline turns into an error once the queue is empty). Pushes are flow controlled with a window of
32 chunks and arrive in order. Results of `Run.result` still work. Cancelling a run, letting go of it, or
`RunOptions.timeoutMillis` closes its input streams, so a pipeline that waits for input ends; the pipeline's
next `emit` then fails.

Runtime side: `RawInput::Stream(StreamReader)` from `tensorlisp::input_stream`, `Model::pipeline_streaming(name,
example, Some(emit))`; the model test is `pipelines_stream_in_and_out` in `tests/entries.rs`.

**Yielding.** A pipeline is a coroutine on the Scheme thread (`$tl-pipeline-start` / `-resume` in `core.ss`). It
leaves the Scheme thread whenever it has to wait, by capturing its continuation and escaping to the host, and a
Rust driver (`LoadedProgram::run_pipeline`, on the caller's thread) resumes it later with another call. It waits
for two things: `(stream-recv s)` with nothing queued, where the driver blocks on the stream and resumes when a
chunk arrives or the stream closes; and `(run entry ...)`, where the driver runs the entry itself instead of the
pipeline doing it on the Scheme thread. So the Scheme thread is occupied only while pipeline code (not model
compute or waiting) executes: other pipelines, model loads, and graph builds for new input shapes go on while a
pipeline waits for audio or for the GPU. (`crates/chez/tests/coroutine.rs` checks the Chez side: a continuation
captured in one host call can be resumed from another, errors included, as long as the computation leaves through
the current call's escape continuation instead of returning.) This also removes a deadlock: an entry run from a
pipeline used to wait for the model's lock on the Scheme thread while a client's call held that lock waiting for
the Scheme thread to build a graph.

Two pipelines can't stream into each other's `Run` and a pipeline can wait for one stream at a time (it yields on
the stream it asked for); `emit` never blocks.

## Callbacks and samplers

A pipeline can call back into the client that started it. `Model.pipeline` takes a `Callbacks` capability (and
`Run.setCallbacks` can replace it later); inside the pipeline:

```scheme
(client-call "name" arg ...)                     ; -> string, array, audio, image or void
(client-sample logits [history [k [full?]]])     ; -> token id
```

`client-call` sends strings, numbers (0-d f32 tensors), arrays, audio and images to `Callbacks.call(name, args)`
and gets back none, text, a tensor, audio, an image or a number. `client-sample` sends
`Callbacks.sample(SampleRequest)`: the step number (0 for the first sample of the run), the `k` (default 40)
highest logits with their token ids, the history the pipeline passed, and, with `full?`, a `TensorRef` to the
whole logits array; the client answers with the token. Like `run`, these are yields: the pipeline leaves the
Scheme thread while the client answers, and a failing or missing handler turns into the run's error. A pipeline
started without callbacks that calls one fails with "started without callbacks".

Each sample is a round trip, so a client sampler suits a local socket or short outputs. Anything that can be
decided from the logits alone should be written in the pipeline instead (masks and grammar constraints as tensors
the pipeline applies before picking), which needs no round trip; `SamplerConfig` in the schema is the declarative
version of that and is not implemented. The runtime side is `tensorlisp::Callbacks` (a blocking trait) in
`PipelineOptions::callbacks`; the server's implementation forwards each call to the connection's event loop.

## Sessions

`Model.session()` returns a `Session`: its own zeroed copy of the program's `(define-state ...)` tensors (KV
caches, encoder context). Pass it as the `session` argument of `Model.run` or `Model.pipeline` and every entry
that call runs reads and writes that session's state; without one the model's own default state is used.
`Session.reset()` zeroes it (individual tensors by name are not supported), `Session.info()` lists the state
tensors and their size, dropping the capability frees the memory. `Session.fork()` returns a new session that
starts as a copy of this one's state and then diverges (prefix caching: run the shared prompt once, fork per
request; branching samples). It copies the state (device to device on ggml, through the host on the native
devices) and is independent of the original afterwards, including if the original is dropped.

How the server knows which state a returned capability means: the `Session` has an unguessable `token()`, and a
capability passed back to `run`/`pipeline` is asked for its token, which the server looks up among its live
sessions; anything else is refused.

Runtime side (`tensorlisp::ModelState`, `Model::new_state()`, `RunOptions::state`, `Model::pipeline_in`): on the
native devices each session is its own block and switching is free. On ggml's backends (CPU, Metal, CUDA, Vulkan)
the allocated graphs are tied to one set of state tensors, so running a session other than the last one copies
the previous session's state out and this one's in, on the device. That is a state-sized copy per switch: fine
for a few sessions taking turns per request, wasteful for per-token interleaving of large KV caches. Runs of one
model are serialized in any case. Tested on the CPU, ggml Metal and the native Metal executors
(`sessions_have_their_own_state`); the CUDA and Vulkan native devices have the same change but were not run here.

## On a real model

`crates/tensorlisp-serve/tests/real_model.rs` (ignored; see its header for the command line) runs T5Gemma 2 270M
(q8_0, 955 MB) through the server, in this process or against a separate `tl-serve`. The GGUF is the one from
`ports/t5gemma2` with `ports/t5gemma2/rpc.ss` appended by `tl pack`: `generate-stream` (emits each token as it is
chosen), `generate-sampled` (the client picks every token with `client-sample`) and `chat` (a prompt stream in, one
reply per prompt out). Every result is compared with the same pipeline run directly on the runtime. Passing on
ggml's Metal backend, the native Metal executor, and the native Tree-on-Metal device (`native-metal-tree`, release
build, M1 Pro), in process and against a standalone `tl-serve` over a unix socket:

- Installing the model by streaming the file in 4 MiB chunks, hashing it, storing it and loading it takes 5 s
  (release; the debug build's hashing makes it 50 s). `describe` lists the real pipelines and the 36 state tensors
  (19 MB per session).
- `generate` (three prompts) and `caption` (a JPEG sent as an encoded `Image` argument) give the same text and token
  ids as the local runs. `generate` takes 150 ms for 32 tokens on Tree, same as locally. The first `caption` used to
  take 58 s on a Tree device, local or remote alike, and again on every later call: it was GPU time (not compile
  time), 56.5 s of it in 27 launches of the scalar `k:flash-attn`. With the tensor-core `k:flash-attn-cm`
  (docs/design/tree.md) the vision tower takes 5.1 s, 3.5 s of it attention.
- Two prompts in two sessions at the same time each give exactly their solo result, and again with the sessions
  swapped. The same two prompts at once without sessions share one KV cache and the results differ: sessions are
  what makes concurrent clients correct.
- `generate-stream`: the first token reaches the client 14 ms after the call, the last after 144 ms (32 tokens).
- Client sampler: a client that takes the best candidate reproduces greedy decoding exactly; one that takes the
  second best at step 0 changes the output as the same local sampler does. One round trip per token costs about
  10 ms per sampled token on Tree over a unix socket (including the logits readback and the top-k).
- `chat`: three prompts pushed one at a time each get their reply right after their own turn (29 to 150 ms).

## Reverse mode

`tl-serve --connect tcp:HOST:PORT` (or `unix:PATH`, repeatable, `--name`, default the host name) makes the worker
dial a scheduler instead of waiting to be called, so it needs no inbound port. It offers its `Server` as the
bootstrap capability of the connection it opened and calls `Scheduler.register(worker, info)` on whatever the
scheduler offers; `WorkerInfo` carries the name, version, devices and the models it already holds. From then on the
scheduler has a capability to the worker and uses the ordinary interface: install models (streamed, or from a path
the worker can read), run entries, start pipelines, open streams and sessions. Cap'n Proto connections are
symmetric, so this needed no new transport. The registration lasts as long as the connection: when it drops, the
worker dials again (1 s, doubling to 30 s while dialing fails) and registers again, listing what it now has
installed. `--listen` and `--connect` can be combined.

The `Scheduler` interface is only this registration; placement, work queues and leases are the scheduler's business
and not part of this crate. There is no authentication: run the connection through TLS or an SSH/WireGuard tunnel
until the transport grows it. Tests: `tests/reverse.rs` (an in-process worker and the real `tl-serve` binary
dialing a test scheduler, including redialing after the scheduler drops the connection).

## Streaming audio into Gemma 4

`ports/gemma4` has the `transcribe` pipeline (see its README): a `stream` input of 16 kHz audio chunks, a log-mel
front end that produces frames as the samples arrive, the streaming conformer, and the q4_0 language model filling
its KV cache with the audio tokens block by block, so that when the audio ends only the last partial chunk, the text
after the audio and one forward pass are left. `crates/tensorlisp-serve/tests/real_audio.rs` (ignored; the header has the
command line) pushes a 20.4 s clip in 100 ms pieces over RPC, once paced in real time and once all at once, M1 Pro,
Google's QAT q4_0 files on ggml's Metal backend, release build:

| | first token after the audio ended | the whole answer (353 characters) |
|---|---|---|
| audio streamed in real time | **88 to 104 ms** (4 runs) | 0.94 to 1.15 s |
| the same audio sent at once, at the end | about 735 ms | 1.6 to 1.9 s |

Of the 90 ms, the last partial chunk through the conformer is about 27 ms and the final block (up to 11 leftover audio
tokens + the 6 text tokens, padded to a fixed 24 positions so the graph is built once) another 63 ms: a 35-layer q4_0
decoder pass over a handful of positions costs 27 ms (4 positions) to 60 ms (16 to 32) because it streams the
weights once; each generated token after that is 14 ms. While the audio plays the pipeline keeps up with a lot of
room: with the whole clip delivered at once it takes 0.66 s to run the tower and prefill 20 s of audio (30x real time).
The transcript is word for word what transformers (bf16, unquantized QAT weights) gives for it. The first utterance
builds the graphs (every entry once); the numbers are for the ones after it. Smaller conformer chunks (4 tokens
instead of 12) would cut the 27 ms but the decoder pass dominates and has a floor of about 44 ms for 8 positions, so
60 ms looked like the best that was worth the extra graphs.

## The kernel cache

Tree kernels are compiled by Scheme on first use. `tensorlisp::kernel_cache` stores the compiler's output (SPIR-V,
PTX or MSL plus the launch metadata) in `$TL_KERNEL_CACHE` (default `~/.cache/tensorlisp/kernels`; `off` disables),
one file per kernel, keyed by the kernel expression, the optimization passes, the Tree sources' contents and the
crate version, so editing a generator invalidates exactly the kernels built from the old code. Files carry their
whole key and are replaced atomically, so a stale, truncated or colliding file is a miss and processes can share
the directory. `TL_NATIVE_TIME=1` prints `kernel cache: hits/misses`. On T5Gemma 2's text path (32 kernels,
320 KB) it cuts the Tree compile from 47 ms to 1 ms per process; the rest of a cold start is the driver's own
shader compile (Metal's 240 ms after a rebuild, ~3 ms once its own cache is warm), which is left to the driver.

## What does not exist yet

`SamplerConfig`, `Session.getState`/`setState`,
`Model.describe` output
types, `pipelineSource` on install, `bytes`/`list`/`map` values as arguments, GPU-resident `TensorRef`. Those
need the v2 program format split (per-session state, yielding pipelines) described in the design notes; the schema
already carries their shape. Calls into one model are serialized by the model's own lock.

Test: `cargo test -p tensorlisp-serve` (an in-process server and client over a unix socket).
