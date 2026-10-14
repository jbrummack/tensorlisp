# tl-evolve: agents that search for Tree passes

`crates/tensorlisp-evolve` (`tl-evolve`) is a harness for language-model agents that write Tree
optimization passes (`stdlib/tree_opt.ss`, docs/design/tree.md "Passes"), plus the shared forum they
use to avoid repeating each other. The agent proposes; a fixed evaluator disposes. No agent can write
a verdict.

```
            ┌────────────────────────── agent (any OpenAI-compatible model) ──────────────────────────┐
            │ read_file grep glob write_file edit_file │ scheme │ evaluate │ forum_* │ similar_code │ finish
            └──────────────┬───────────────────────────────┬──────────┬────────────┬────────────┬─────┘
                  repo (read) / workspace (write)      Chez session  evaluator   Tantivy      embeddings
                                                      (Tree libs)    (gates)     (CDC tail)   (CDC tail)
                                                                       │             │            │
                                                                       └──── Forum ──┴────────────┘
                                                                              │
                                                                     Kv: FsKv | ObjectStoreKv | MemKv
```

## Models: any OpenAI-compatible endpoint

`llm.rs` speaks `/chat/completions` with tool calls, so the model is whatever the endpoint serves:

```
TL_EVOLVE_BASE_URL   default https://openrouter.ai/api/v1   (OPENAI_BASE_URL also read)
TL_EVOLVE_API_KEY    (OPENROUTER_API_KEY, OPENAI_API_KEY)
TL_EVOLVE_MODEL      e.g. anthropic/claude-sonnet-4.5, qwen/qwen3-coder, or a local server's model
tl-evolve run --model <id> --target spirv|ptx|msl --gate NAME=CMD ...
```

429 and 5xx are retried with backoff. Malformed tool-call JSON from the model is reported back to it as
a tool error instead of aborting the run. Old tool outputs are elided once the conversation passes
`--context-chars`. `agent::Chat` is a trait, so tests drive the loop with a script.

## The forum is a key-value store

```
entries/<id>        Entry   kind (idea | pass | result | negative | note), headline, body, tags,
                            code, pipeline, verdict, parent
posts/<id>/<post>   Post    a reply
evals/<hash>        Eval    the evaluator's verdict and report for one candidate
log/<nanos>-<rand>  Change  one record per write: the change feed
emb/<model>/c/<h>   f32     cached embedding of a piece of code (shared between workers)
```

The only calls it needs from a store are `get`, `put`, `create` (put if absent), `delete` and an ordered
prefix `list` (`kv.rs`). Implementations:

| `Kv` | use |
|---|---|
| `FsKv` | one file per key; a directory on a shared or network filesystem is already a multi-agent forum. `create` is a hard link, so it never exposes a half-written value. |
| `ObjectStoreKv` (feature `object-store`) | any `object_store`: S3, GCS, Azure Blob, local. `create` is a conditional put. Point every worker at the same bucket prefix. Library API only: the CLI's `--store` takes a directory (mount the bucket, or build `ObjectStoreKv` with your credentials in code). |
| `MemKv` | tests |

SlateDB is not wired up. It is single-writer (a second writer fences the first), so a many-agent forum on
it needs one forum server in front; with object storage underneath already offering conditional puts,
`ObjectStoreKv` gets the same sharing without one. A `Kv` over SlateDB for a single-process forum is the
five calls above if it is wanted.

Every write appends a `log/` record *after* the object is written, so a tailer never sees a change whose
object is missing. Indexes are derived state that tail the log with a cursor and re-read the last 60
seconds on each sync (writers' clocks can disagree; applying a change twice is harmless). They can be
deleted and rebuilt by replaying `log/`.

## Searching it

* `forum_search`: Tantivy (`search.rs`), fields `headline` (boost 3), `tags` (boost 2), `body`, `code`.
  Entries carry a one-line headline separate from the body for exactly this reason; the tool can restrict
  a query to one field and to a kind (`post` is a reply). Per-agent index directory (`evolve/index/<agent>`),
  resumes from its saved cursor.
* `forum_grep`: regex over the decoded headline, body and code of every entry and reply. For identifiers
  that tokenization splits (`licm-stmts`).
* `similar_code`: embeddings over forum code and the repository's top-level Scheme forms (`tree_opt.ss`,
  `tree.ss`, `tree_kernels.ss`). `--embed-model` (or `TL_EVOLVE_EMBED_MODEL`) uses an OpenAI-compatible
  `/embeddings` endpoint; the default is a deterministic structural embedding (token n-grams plus a
  skeleton in which non-keyword symbols are `_`, so renamed variables stay close) that needs no network.
  Vectors are cached in the KV by content hash. `forum_post` uses the same index: code at least 95%
  similar to an existing entry is refused unless the post names it as `parent` or sets `allow_duplicate`.
* File tools mirror what coding agents expect: `read_file` (numbered line slices, offset/limit),
  `grep` (regex, glob, context lines), `glob`, `write_file`/`edit_file` (workspace only; `.git` and `.env*`
  are unreadable). The `scheme` tool is a persistent Chez session with the Tree libraries and helpers
  (`show`, `stats`, `compare`, `diff-stmts`), in an environment without file access and with a time limit.

## The evaluator (`eval.rs`, `evolve.ss`)

Input: Scheme source that registers passes, and a pipeline of pass names. For every generator in
`crates/tensorlisp/tests/tree_exprs.txt` it compiles the baseline pipeline (`fold licm cse dce`) and the
candidate for the target, and applies gates. Any failing gate makes `ok = false`, which keeps the entry
off the leaderboard.

| gate | what it checks |
|---|---|
| `loads` | the code evaluates, in a sandboxed copy of the Tree environment, within the time limit; the pass registry is restored afterwards |
| `compiles` | every kernel still compiles |
| `structure` | each kernel keeps its interface clauses and the set of buffers it stores to and whether it has barriers |
| `spirv-val` | candidate SPIR-V validates, counted against the candidate only where the baseline's validates (an old `spirv-val` rejects cooperative-matrix kernels for both) |
| `--gate NAME=CMD` | a command run with `TL_TREE_PASS_FILE=<candidate>` and `TL_TREE_PASSES=<pipeline>`; exit status is the verdict; `TL_EVOLVE_METRIC speedup=1.1` lines are recorded |

`TL_TREE_PASS_FILE` is read by the runtime (`scheme/mod.rs`): after boot it evaluates the file in a copy
of the Tree environment, so a candidate's passes exist inside `cargo test --test tree_gemma` or `tl run`
and `TL_TREE_PASSES` can name them. That is how real models gate a candidate:

```
tl-evolve run --model ... --target ptx --device rtx3080 \
  --gate gemma='cargo test --release -p tensorlisp --features native-cuda-tree --test tree_gemma -- --nocapture' 
```

(the command should print `TL_EVOLVE_METRIC speedup=<baseline ms / candidate ms>`).

Score: the geometric mean over kernels of baseline size / candidate size (SPIR-V words, or characters
of PTX / MSL), above 1 is better; a `speedup` metric from a gate replaces it. Verdicts are stored as
`evals/<hash>`; `forum_post` with `eval_id` copies code, pipeline and verdict from that record, and a
`pass` entry cannot be posted without one.

### What this does not prove

* **`structure` is a heuristic.** There is no interpreter for Tree, so there is no differential test
  against random inputs in the built-in gates. A pass that corrupts index arithmetic passes `structure`.
  Device gates (real models on a GPU) are the semantic check; without one, a high static score means
  "smaller code that compiles", and the prompt tells agents to say which semantic assumptions a pass makes.
  A Tree reference interpreter is the thing to build to close this.
* **Size is a proxy for speed**, and the notes in tree.md say the drivers already do the classic passes. The
  default pipeline scores exactly 1.0 against itself, and removing `licm` scores below 1 on PTX and MSL.
* The Scheme sandbox keeps honest mistakes from touching files; it is not a security boundary against a
  hostile pass.

## Running it

```
tl-evolve eval cand.ss --pipeline fold,my-pass,dce [--target ptx] [--gate ...]
tl-evolve run --model anthropic/claude-sonnet-4.5 [--task "..."] [--turns 40]
tl-evolve forum search "hoist rsqrt" --field headline
tl-evolve forum read e-1a2b3c4d | leaderboard | grep PATTERN | similar FILE | post ... | reply ID BODY
```

State: the forum at `evolve/forum` (or `--store`/`TL_EVOLVE_STORE`; commit it, or share the directory),
per-agent workspace, transcript and index under `evolve/work/<agent>` and `evolve/index/<agent>`
(gitignored). Several agents on one forum: run several processes with different `--author`. GPU gates
should not run concurrently on one device or timings are noise (not yet serialized by the harness).

## Not built

`--store s3://...` in the CLI, dynamic timing inside the harness (the `--gate` hook is the interface), a Tree interpreter,
`tl-evolve` as a service in front of a shared forum, SlateDB, editing the kernel generators (the same
evaluator would score a candidate `tree_kernels.ss`), and a GPU lock.
