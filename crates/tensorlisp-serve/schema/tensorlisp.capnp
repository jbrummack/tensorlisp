@0xc25a7b2ffd2eb6a2;

# tensorlisp RPC (draft), served by crates/tensorlisp-serve.
#
# Shape:
#   Server  -> install / list / open        (bootstrap capability)
#   Model   -> describe / run / session     (one installed program)
#   Session -> per-client state tensors (KV cache, encoder context)
#   Run     -> one execution of a pipeline: push inputs, pull outputs, cancel
#   TensorRef -> a tensor that stays on the server's device
#
# Pipelines are user-defined, so the wire type is generic (Value) and each pipeline
# describes its own signature via Model.describe().


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

enum DType {
  f32 @0;
  f16 @1;
  bf16 @2;
  i32 @3;
  i64 @4;
  i16 @5;
  i8 @6;
  u8 @7;
  bool @8;
  # Quantized blocks travel as raw bytes of the ggml layout.
  q40 @9;
  q41 @10;
  q50 @11;
  q51 @12;
  q80 @13;
  q4K @14;
  q6K @15;
}

struct Tensor {
  # dims are outermost-first (numpy order). f16/bf16/i* are raw little-endian
  # elements in `data` (Cap'n Proto has no half-float scalar, so Data it is).
  dtype @0 :DType;
  dims @1 :List(UInt64);
  data @2 :Data;
}

interface TensorRef {
  # A tensor resident on the server. Dropping the capability frees it.
  info @0 () -> (dtype :DType, dims :List(UInt64), device :Text);
  read @1 (offset :UInt64 = 0, length :UInt64 = 0) -> (data :Data);
  # length 0 = to the end.
}

struct Audio {
  sampleRate @0 :UInt32;
  channels @1 :UInt16;
  samples @2 :List(Float32);   # interleaved
}

struct Image {
  width @0 :UInt32;
  height @1 :UInt32;
  format :union {
    rgb8 @2 :Void;
    rgba8 @3 :Void;
    gray8 @4 :Void;
    encoded @5 :Text;           # "png", "jpeg", ... ; pixels hold the file bytes
  }
  pixels @6 :Data;
}

struct Value {
  union {
    none @0 :Void;
    tensor @1 :Tensor;
    tensorRef @2 :TensorRef;
    text @3 :Text;
    bytes @4 :Data;
    int @5 :Int64;
    float @6 :Float64;
    bool @7 :Bool;
    image @8 :Image;
    audio @9 :Audio;
    list @10 :List(Value);
    map @11 :List(Entry);
    # A stream of Values the receiver pulls from / the sender pushes into.
    source @12 :Source;
    sink @13 :Sink;
  }

  struct Entry {
    key @0 :Text;
    value @1 :Value;
  }
}

struct Named {
  name @0 :Text;
  value @1 :Value;
}

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

interface Sink {
  # Producer pushes chunks. Calls are `-> stream` so the transport applies
  # flow control; the producer may keep a bounded window in flight.
  push @0 (chunk :Value) -> stream;
  # Ends the stream. Returns once everything pushed has been consumed.
  done @1 () -> ();
}

interface Source {
  # Consumer pulls. Empty `chunks` with eof = true means finished.
  pull @0 (max :UInt32 = 1) -> (chunks :List(Value), eof :Bool);
  cancel @1 () -> ();
}

# ---------------------------------------------------------------------------
# Self-description
# ---------------------------------------------------------------------------

struct TypeSpec {
  # What a pipeline argument or result looks like. Advisory; the server checks.
  union {
    any @0 :Void;
    tensor :group {
      dtype @1 :DType;
      dims @2 :List(Dim);     # -1 = dynamic
    }
    text @3 :Void;
    bytes @4 :Void;
    int @5 :Void;
    float @6 :Void;
    bool @7 :Void;
    image @8 :Void;
    audio @9 :Void;
    list @10 :TypeSpec;
    stream @11 :TypeSpec;     # Source/Sink of this element type
  }
  struct Dim { size @0 :Int64; name @1 :Text; }
}

struct Param {
  name @0 :Text;
  type @1 :TypeSpec;
  optional @2 :Bool;
  doc @3 :Text;
}

struct EntryInfo {
  # A raw graph entry: tensors in, tensors out. Callable through Model.run.
  name @0 :Text;
  inputs @1 :List(Param);
  outputs @2 :List(Param);
}

struct PipelineInfo {
  name @0 :Text;
  doc @1 :Text;
  inputs @2 :List(Param);
  outputs @3 :List(Param);
  # streamingIn: some parameter is a `stream` fed through Run.input(). Outputs
  # are not declared: a pipeline may emit under any name (streamingOut is not
  # set yet). Without streams, pipeline() takes everything up front and the
  # result is available from Run.result().
  streamingIn @4 :Bool;
  streamingOut @5 :Bool;
}

struct StateInfo {
  name @0 :Text;
  dtype @1 :DType;
  dims @2 :List(Int64);
  bytes @3 :UInt64;           # at the model's maximum configuration
}

struct ModelInfo {
  id @0 :Text;                # content hash
  name @1 :Text;
  entries @2 :List(EntryInfo);
  pipelines @3 :List(PipelineInfo);
  state @4 :List(StateInfo);
  weightBytes @5 :UInt64;
  assets @6 :List(Text);
  loaded @7 :Bool;
}

struct DeviceInfo {
  name @0 :Text;              # "cpu", "metal", "cuda:0", "vulkan:0"
  memoryTotal @1 :UInt64;
  memoryFree @2 :UInt64;
  native @3 :Bool;            # native (Tree/engine) executor available
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

struct InstallOptions {
  name @0 :Text;              # optional alias; defaults to the program's name
  device @1 :Text;            # "" = server default
  load @2 :Bool = true;       # load weights now; false = register only
  replace @3 :Bool = false;   # overwrite an alias that points at another hash
  # Optionally install a pipeline definition over an existing model.
  pipelineSource @4 :Text;    # Scheme text; empty = use the pipelines in the file
}

interface Server {
  info @0 () -> (version :Text, devices :List(DeviceInfo));

  list @1 () -> (models :List(ModelInfo));

  # Upload a model: stream the GGUF (TL_TXT/TL_BIN program + weights) in Data
  # chunks, then finish(). Content-addressed: the same bytes are one model.
  install @2 (options :InstallOptions) -> (upload :Upload);

  # Same, from a path or URL the server can read (no bytes over the wire).
  installFrom @3 (location :Text, options :InstallOptions) -> (model :Model);

  open @4 (name :Text) -> (model :Model);   # alias or hash
  remove @5 (name :Text) -> ();

  interface Upload {
    push @0 (chunk :Data) -> stream;
    finish @1 () -> (model :Model);
  }

}

# ---------------------------------------------------------------------------
# Model, Session, Pipeline, Run
# ---------------------------------------------------------------------------

struct RunOptions {
  priority @0 :Int8 = 0;
  # Wall-clock limit; 0 = none. Run is cancelled with `timeout`.
  timeoutMillis @1 :UInt32;
  # Return outputs as TensorRef instead of copying them to the client.
  outputsByRef @2 :Bool;
  # Tensors in results are sent as this dtype (f32, f16 or bf16) to halve the bytes.
  outputDtype @3 :DType = f32;
}

interface Model {
  describe @0 () -> (info :ModelInfo);
  load @1 (device :Text) -> ();
  unload @2 () -> ();

  # Raw path: run one graph entry on tensors. Works for any model, no pipeline needed.
  # `session` selects the state tensors the entry reads and writes; without
  # one it uses the model's own default state.
  run @3 (entry :Text, inputs :List(Named), options :RunOptions, session :Session)
      -> (outputs :List(Named));

  # State (KV cache etc.) lives in a Session. Dropping the Session frees it.
  session @4 () -> (session :Session);

  # Start a pipeline. `session` may be omitted for stateless pipelines;
  # the server then creates a throwaway one.
  # `callbacks` is what the pipeline's `(client-call ...)` and `(client-sample ...)` talk to
  # (Run.setCallbacks can replace it later).
  pipeline @5 (name :Text, session :Session, args :List(Named), options :RunOptions,
               callbacks :Callbacks) -> (run :Run);
}

interface Session {
  reset @0 (names :List(Text)) -> ();  # empty = all state
  info @1 () -> (state :List(StateInfo), bytesUsed :UInt64);
  # A new session starting with a copy of this one's state (prefix caching, branching samples).
  fork @2 () -> (session :Session);
  # Direct state access, for debugging and for moving caches between servers.
  getState @3 (name :Text) -> (tensor :TensorRef);
  setState @4 (name :Text, tensor :Tensor) -> ();
  # Identifies the session to the server that made it: when a Session capability
  # comes back in `Model.run` or `Model.pipeline`, the server asks it for its
  # token. Unguessable; hold the capability, not the number.
  token @5 () -> (token :UInt64);
}

struct Progress {
  union {
    queued @0 :UInt32;        # position in the scheduler queue
    running @1 :Void;
    step @2 :Text;            # free-form, e.g. "decode 17/128"
  }
}

enum RunStatus {
  queued @0;
  running @1;
  waitingForInput @2;         # a streaming pipeline is blocked on push()
  finished @3;
  cancelled @4;
  failed @5;
}

interface Run {
  status @0 () -> (status :RunStatus, error :Text);

  # Incremental inputs for pipelines with streamingIn: `name` is a pipeline
  # parameter of kind `stream`. Push chunks (tensor, text, audio or image
  # values), then done(). At most INPUT_WINDOW (32) chunks may be waiting for
  # the pipeline; further pushes wait.
  input @1 (name :Text) -> (sink :Sink);

  # Incremental outputs: whatever the pipeline `(emit 'name value)`s under
  # `name`. Any name may be requested; one nothing is emitted under just ends.
  output @2 (name :Text) -> (source :Source);

  # Final results; resolves when the pipeline returns.
  result @3 () -> (outputs :List(Named));

  cancel @4 () -> ();

  # Pipeline code can ask the client something (custom sampler, tool call).
  # The client sets a handler before or while the run is going.
  setCallbacks @5 (callbacks :Callbacks) -> ();
}

# ---------------------------------------------------------------------------
# Callbacks (client-side logic invoked from pipeline code)
# ---------------------------------------------------------------------------

interface Callbacks {
  # Generic named call from pipeline code: `(client-call "name" arg ...)`. Arguments are
  # strings (text), numbers (0-d f32 tensors), arrays, audio and images; the result may be
  # none, text, a tensor, audio, an image or a number (a 0-d tensor in the pipeline).
  call @0 (name :Text, args :List(Value)) -> (result :Value);

  # Custom sampling. Pipeline code sends the top-k candidates (not the full
  # logits) and the generated-so-far tokens; the client returns a token.
  # One round trip per token, so intended for local sockets or short outputs;
  # remote clients should express constraints as masks in pipeline code.
  sample @1 (request :SampleRequest) -> (token :Int32);
}

struct SampleRequest {
  step @0 :UInt32;
  candidateIds @1 :List(Int32);
  candidateLogits @2 :List(Float32);   # same length, descending
  history @3 :List(Int32);
  # If true, the full vocabulary logits are available as a TensorRef.
  fullLogits @4 :TensorRef;
}

# ---------------------------------------------------------------------------
# Declarative sampling parameters, usable as a pipeline argument
# (Value.map with these keys) or built into stock pipelines.
# ---------------------------------------------------------------------------

struct SamplerConfig {
  temperature @0 :Float32 = 1.0;
  topK @1 :UInt32;                     # 0 = off
  topP @2 :Float32 = 1.0;
  minP @3 :Float32;
  seed @4 :UInt64;
  repetitionPenalty @5 :Float32 = 1.0;
  logitBias @6 :List(Bias);
  allowedTokens @7 :List(Int32);       # non-empty = only these
  bannedTokens @8 :List(Int32);
  # Constrained decoding, compiled server-side.
  constraint :union {
    none @9 :Void;
    jsonSchema @10 :Text;
    gbnf @11 :Text;
    regex @12 :Text;
  }
  struct Bias { token @0 :Int32; bias @1 :Float32; }
}

# ---------------------------------------------------------------------------
# Reverse mode: the worker dials the scheduler
# ---------------------------------------------------------------------------

struct WorkerInfo {
  name @0 :Text;
  version @1 :Text;
  devices @2 :List(DeviceInfo);
  models @3 :List(ModelInfo);     # what the worker has installed (id, name, size, whether loaded)
}

interface Scheduler {
  # The bootstrap capability of whatever a `tl-serve --connect` worker dials. The worker offers
  # its own Server over the connection it opened itself, so the scheduler can call it (install
  # models, run entries and pipelines, stream) without being able to reach the worker's network.
  # The registration lasts as long as the connection; a worker that is disconnected dials again
  # and registers again.
  register @0 (worker :Server, info :WorkerInfo) -> (id :Text);
}
