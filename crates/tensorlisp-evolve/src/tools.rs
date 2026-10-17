//! The tools an agent gets, as OpenAI-style function specs plus their implementation.
//!
//! Files: `read_file` (line slices), `grep`, `glob`, `write_file` and `edit_file` (workspace only).
//! Scheme: `scheme`, a persistent REPL with the Tree libraries. Evaluation: `evaluate`, the only way to
//! get a verdict. Forum: `forum_search` (Tantivy; headline/body/tags/code fields), `forum_grep`,
//! `forum_read`, `forum_post`, `forum_reply`, `forum_leaderboard`, and `similar_code` (embeddings over
//! forum code and the repository's Scheme). Tool errors come back as text starting with `error:` so the
//! model can correct itself.
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Result, bail};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};

use crate::embed::CodeIndex;
use crate::eval::{self, EvalConfig};
use crate::reference::{self, RefIndex};
use crate::weights;
use crate::forum::{Entry, Forum, Kind, NewEntry};
use crate::implementation;
use crate::scheme;
use crate::search::SearchIndex;

const MAX_OUTPUT: usize = 12_000;
/// The notes are re-sent after every compaction, so they stay short.
const MAX_NOTES: usize = 6000;
/// The agent's editable copy of `tree_kernels.ss`, in its workspace.
const GENERATOR_COPY: &str = "tree_kernels.ss";
/// `vendor/ggml` is searchable (its kernels are the reference); the Chez sources and builds are not.
const SKIP_DIRS: &[&str] = &[".git", "target", "ChezScheme", "chez", "glslang", "node_modules", ".venv", "venv", "__pycache__", "models", ".ggml-cuda-cache"];
/// Most reference windows `similar_code` will embed in one process.
const REFERENCE_BUDGET: usize = 8000;
/// Entries this similar to an existing one are refused unless the agent says it is a deliberate variation.
const DUPLICATE_THRESHOLD: f32 = 0.95;

pub struct ToolCtx {
    pub forum: Forum,
    pub search: Mutex<SearchIndex>,
    pub code: Mutex<CodeIndex>,
    pub eval_cfg: EvalConfig,
    /// Reads are allowed anywhere under this directory (and the workspace).
    pub repo: PathBuf,
    /// Writes are allowed only here.
    pub workspace: PathBuf,
    /// Scheme files of the repository that `similar_code` searches besides the forum.
    pub repo_files: Vec<PathBuf>,
    pub author: String,
    pub session: String,
    /// Offer the tools that show the Rust implementation (`--rust`).
    pub rust_tools: bool,
    notes: Mutex<String>,
    last_eval: Mutex<Option<String>>,
    repo_indexed: Mutex<bool>,
    ref_embedded: Mutex<bool>,
    ref_index: Mutex<Option<RefIndex>>,
    finished: Mutex<Option<String>>,
}

/// `read_file` prints `   703\ttext`; models paste that back. Removes the prefix from every line when all of them have one.
fn strip_line_numbers(text: &str) -> Option<String> {
    let re = Regex::new(r"^\s*\d+\t").unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    let nonempty: Vec<&&str> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    if nonempty.is_empty() || !nonempty.iter().all(|l| re.is_match(l)) {
        return None;
    }
    Some(lines.iter().map(|l| re.replace(l, "").to_string()).collect::<Vec<_>>().join("\n"))
}

/// Where in `text` something like `old` is: the best-matching window of as many lines as `old` has, by the
/// similarity of its first and last non-blank lines to the file's lines. `(first line, last line, similarity)`.
fn closest_region(text: &str, old: &str) -> Option<(usize, usize, f64)> {
    let want: Vec<&str> = old.lines().filter(|l| !l.trim().is_empty()).collect();
    let (first, n) = (want.first()?.trim(), old.lines().count().max(1));
    let lines: Vec<&str> = text.lines().collect();
    let sim = |a: &str, b: &str| similar::TextDiff::from_chars(a, b).ratio() as f64;
    let mut best: Option<(usize, f64)> = None;
    for (i, l) in lines.iter().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let mut score = sim(first, l.trim());
        if want.len() > 1 {
            // the second line of the region corroborates a first-line match
            if let Some(next) = lines[i + 1..].iter().find(|x| !x.trim().is_empty()) {
                score = 0.7 * score + 0.3 * sim(want[1].trim(), next.trim());
            }
        }
        if best.is_none_or(|b| score > b.1) {
            best = Some((i, score));
        }
    }
    best.filter(|b| b.1 >= 0.6).map(|(i, score)| (i + 1, (i + n).min(lines.len()), score))
}

/// The lines around an edit as they are now, so the author sees the effect and where later lines moved.
fn snapshot(text: &str, first: usize, last: usize) -> String {
    let total = text.lines().count();
    let (from, to) = (first.saturating_sub(2).max(1), (last + 2).min(total));
    if from > to {
        return String::new();
    }
    let to = to.min(from + 30);
    format!("\nnow ({total} lines in the file):\n{}", numbered(text, from, to))
}

fn numbered(text: &str, from: usize, to: usize) -> String {
    text.lines().enumerate().skip(from - 1).take(to + 1 - from).map(|(i, l)| format!("{:>6}\t{l}", i + 1)).collect::<Vec<_>>().join("\n")
}

/// For Scheme files: tell the author right away when the parentheses do not balance (a model otherwise finds
/// out after a whole `evaluate`, and tends to rewrite the file unchanged several times).
fn paren_warning(path: &Path, text: &str) -> String {
    if path.extension().is_none_or(|e| e != "ss" && e != "scm") {
        return String::new();
    }
    match crate::lispcheck::explain("parenthesis check", text) {
        m if m.contains("paren check:") => format!("\nWARNING: the parentheses in this file do not balance {}. It will not load until that is fixed.", m.split_once("[paren check: ").map(|x| x.1.trim_end_matches(']')).unwrap_or("")),
        _ => String::new(),
    }
}

fn workspace_notes(workspace: &Path) -> PathBuf {
    workspace.join("notes.md")
}

pub fn clip(mut s: String, max: usize) -> String {
    if s.chars().count() > max {
        let cut: String = s.chars().take(max).collect();
        let dropped = s.chars().count() - max;
        s = format!("{cut}\n[output truncated: {dropped} more characters]");
    }
    s
}

fn spec(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}})
}

/// The tool definitions; `rust` adds the tools that show the Rust implementation around the kernels.
pub fn specs(rust: bool) -> Vec<Value> {
    let s = |d: &str| json!({"type": "string", "description": d});
    let n = |d: &str| json!({"type": "integer", "description": d});
    let mut specs = vec![
        spec("read_file", "Read a slice of a text file with line numbers. Paths are relative to the repository root. Use offset/limit to page through large files (default the first 200 lines, at most 400 per call).",
            json!({"path": s("file path"), "offset": n("first line, 1-based (default 1)"), "limit": n("number of lines (default 200)")}), &["path"]),
        spec("grep", "Regex search over files (Rust regex syntax). Returns path:line:text, with optional context lines. Skips target/, vendor/, .git/.",
            json!({"pattern": s("regular expression"), "path": s("file or directory to search (default: the repository root)"), "glob": s("only files whose path matches, e.g. *.ss or crates/**/*.rs"),
                   "context": n("lines of context around each match (default 0)"), "ignore_case": {"type": "boolean"}, "max_results": n("default 60")}), &["pattern"]),
        spec("glob", "List files whose repository-relative path matches a glob (* within a directory, ** across directories, ?).", json!({"pattern": s("glob pattern")}), &["pattern"]),
        spec("write_file", "Create or overwrite a file in your workspace (the only place you can write).", json!({"path": s("path inside the workspace"), "content": s("file contents")}), &["path", "content"]),
        spec("edit_file", "Replace text in a workspace file. `old` must occur exactly once unless replace_all is set, and must be copied exactly from the file (read it first: do not write it from memory). If it is not found, the error shows the closest real text with line numbers.",
            json!({"path": s("path inside the workspace"), "old": s("text to replace"), "new": s("replacement"), "replace_all": {"type": "boolean"}}), &["path", "old", "new"]),
        spec("edit_lines", "Replace a range of lines (1-based, inclusive, the numbers read_file shows) of a workspace file with new text; empty `new` deletes them. Use it when edit_file cannot find your `old` text. Read the lines first, so you replace what you think you replace.",
            json!({"path": s("path inside the workspace"), "start": n("first line to replace"), "end": n("last line to replace"), "new": s("replacement text (may be empty)")}), &["path", "start", "end", "new"]),
        spec("scheme", "Evaluate Scheme in a persistent Chez session that has the Tree libraries ((tl tree), (tl tree opt), (tl tree kernels), ptx, msl) and helpers: (show x), (stats kernel), (compare kernel pipeline-a pipeline-b), (diff-stmts kernel-a kernel-b), (tree-size t), (spirv-words k). Definitions persist between calls. Prints output and `=> value`. No file access.",
            json!({"code": s("Scheme source, one or more forms"), "session": s("session name (default \"main\")"), "timeout_ms": n("default 10000"), "reset": {"type": "boolean", "description": "start the session from scratch first"}}), &["code"]),
        spec("evaluate", "Evaluate a candidate pass pipeline. `code` (or `path` to a workspace file) is Scheme that defines passes and calls (register-pass! 'name (lambda (kernel) ...)); `pipeline` lists pass names to run, built-ins included: fold, licm, cse, dce (the baseline is all four). Compiles every Tree kernel with the baseline and with your pipeline, checks the gates, and returns a score (above 1 beats the baseline). Only this tool produces verdicts; forum_post copies them by eval_id.",
            json!({"code": s("Scheme source defining passes"), "path": s("workspace file with the Scheme source (alternative to code)"), "pipeline": {"type": "array", "items": {"type": "string"}, "description": "pass names, in order"},
                   "generators": {"type": ["boolean", "string"], "description": "also use an edited copy of the kernel generators: true for tree_kernels.ss in your workspace (see generator_workspace), or the path of another workspace copy. Kernels are then built by your generators and compared with the original ones"},
                   "focus": {"type": "array", "items": {"type": "string"}, "description": "substrings of kernel expressions to report on separately, e.g. [\"matvec\"]; the verdict is unchanged"}}), &["pipeline"]),
        spec("evaluate_batch", "Evaluate several candidates in one call and rank them. Each runs the cheap gates; the survivors are then timed on the real GPU against the reference kernels (when the device has captured launches, see hot_kernels): per kernel the GPU time per launch of the reference and of the candidate on the shapes real runs use, the noise (the reference timed against itself), and whether the outputs agree. Prefer this to evaluating one idea at a time: write your variants (each a short list of `generator_edits` on the original tree_kernels.ss, or copies of it as variants/a.ss, or several pass files), submit them together and keep the winner. The ranking is by the measured speedup, a size ratio only when nothing could be timed.",
            json!({"candidates": {"type": "array", "description": "the candidates to compare", "items": {"type": "object", "properties": {
                "name": s("a label for the table"), "code": s("Scheme source defining passes"), "path": s("workspace file with the Scheme source"),
                "pipeline": {"type": "array", "items": {"type": "string"}}, "generators": {"type": ["boolean", "string"], "description": "true, or the path of an edited copy of tree_kernels.ss in your workspace"},
                "generator_edits": {"type": "array", "description": "the cheap way to give variants: edits applied to the ORIGINAL tree_kernels.ss in order (each `old` must occur exactly once), so a variant is a few lines, not a copy of a 2000-line file; instead of `generators`", "items": {"type": "object", "properties": {"old": s("exact text of the original"), "new": s("replacement")}, "required": ["old", "new"]}}}, "required": ["pipeline"]}}}), &["candidates"]),
        spec("generator_workspace", "Start (or reset) a change to the kernel GENERATORS themselves, not just to passes: puts a pristine copy of tree_kernels.ss (the Scheme that writes every Tree kernel) in your workspace and lists its definitions with line numbers. Edit the copy with read_file / edit_file, then run `evaluate` with generators=true: the evaluator builds every kernel with your generators and with the original ones and compares them, and a device gate (if configured) runs the real executor with your file. Keep each kernel's buffers, push constants and workgroup the same: the Rust launch sites depend on them (the `structure` gate checks), so a generator change is a change to how the kernel is written, not to its contract.", json!({}), &[]),
        spec("generator_diff", "The unified diff of your workspace tree_kernels.ss against the original, and the definitions it touches.", json!({}), &[]),
        spec("hot_kernels", "Where the time goes: the kernels that dominate real workloads on this device (from imported GPU profiles), with their share of Tree busy time, launches per step and microseconds per launch. The evaluator weights the score by these shares and compiles these exact expressions, so work on the top rows first. Without profiles every kernel counts equally.",
            json!({"limit": n("rows, default 15")}), &[]),
        spec("reference_kernels", "Given a Tree generator expression (e.g. \"(k:matvec 'q8_0 1)\"), list where ggml (Vulkan GLSL, CUDA, Metal) and the vLLM-derived paged-attention kernels implement the same operation: files with sizes, and kernel declarations with line numbers. Then read them with read_file offset/limit.",
            json!({"kernel": s("a generator expression or name, e.g. \"(k:flash-attn 'f16 'f16 #t #f 2048)\" or \"rms-norm\"")}), &["kernel"]),
        spec("reference_search", "Keyword search over the existing kernel sources (ggml Vulkan/CUDA/Metal, vLLM-derived paged attention, the native CUDA kernels), in windows of ~70 lines. Matches file names and code: try an instruction or technique (\"subgroupAdd\", \"__shfl_xor_sync\", \"simdgroup_reduce\", \"cooperative matrix\") plus an op.",
            json!({"query": s("keywords"), "backend": s("substring of a backend label: vulkan, cuda, metal, vllm, native (default: all)"), "limit": n("default 8")}), &["query"]),
        spec("forum_search", "Search the shared forum (ideas, passes with verdicts, negative results, replies) with full-text queries. Fields: headline, body, tags, code. Query syntax: words, \"phrases\", +must -not, field:term. Search before you start and before you post.",
            json!({"query": s("search query"), "field": s("restrict to headline, body, tags or code (default all)"), "kind": s("idea, pass, result, negative, note or post (a reply)"), "limit": n("default 8")}), &["query"]),
        spec("forum_grep", "Regex search over the raw text of all forum entries and replies (use for exact identifiers or code fragments full-text search tokenizes away).",
            json!({"pattern": s("regular expression"), "ignore_case": {"type": "boolean"}, "limit": n("default 20")}), &["pattern"]),
        spec("forum_read", "Read one forum entry in full: body, code, pipeline, verdict and replies.", json!({"id": s("entry id, e.g. e-1a2b3c4d")}), &["id"]),
        spec("forum_post", "Post to the forum. kind: idea (untested hypothesis), pass (a pass with a verdict: requires eval_id from evaluate, use \"last\" for your latest), result (a measurement or finding), negative (what you tried, that it failed and why: valuable, post these), note. The headline is one line stating the finding; the body has the reasoning. Entries with near-identical code to an existing one are refused unless you set parent to it or allow_duplicate.",
            json!({"kind": s("idea | pass | result | negative | note"), "headline": s("one line, what a search result shows"), "body": s("reasoning, what you observed, what to try next"),
                   "tags": {"type": "array", "items": {"type": "string"}}, "eval_id": s("evaluation id, or \"last\""), "code": s("code for an idea/negative without an evaluation"),
                   "parent": s("entry this builds on or refutes"), "allow_duplicate": {"type": "boolean"}}), &["kind", "headline", "body"]),
        spec("forum_reply", "Reply under an entry: confirm, refute with evidence, or report results on another device.", json!({"id": s("entry id"), "body": s("reply text")}), &["id", "body"]),
        spec("forum_leaderboard", "The best verified passes per target and device.", json!({"limit": n("per group, default 3")}), &[]),
        spec("similar_code", "Find existing code similar to a snippet (embeddings): forum entries and the repository's Scheme (tree_opt.ss, tree.ss, tree_kernels.ss). Use it before writing a pass to see if one exists.",
            json!({"code": s("code to look up"), "path": s("workspace file to look up instead"), "k": n("default 5"),
                   "scope": s("code (default: forum entries and the repository's Scheme), reference (ggml / vLLM-derived kernels: meaningful with an embedding model, see --embed-model), or all")}), &[]),
        spec("finish", "End the session with a summary of what you found and what the next agent should try.", json!({"summary": s("final summary")}), &["summary"]),
        spec("notes", "Your scratchpad. Whatever you put here is kept verbatim when older turns are compacted away (the rest of the history is summarized), so keep the current goal, the hypotheses you are testing, evaluation and entry ids that matter, files in your workspace, and the next steps. Keep it short (at most ~6000 characters); condense it as you go. Actions: read, set (replace), append.",
            json!({"action": s("read | set | append"), "text": s("the text for set or append")}), &["action"]),
    ];
    if rust {
        specs.push(spec("rust_map", "The Rust (and Scheme) files that decide how Tree kernels are used: how a graph node becomes a generator expression with its bindings, push constants and grid, what is fused before planning, how launches are grouped and synchronized, where the pass pipeline is hooked in. Each file with its own module documentation. Start here to understand what surrounds a kernel.", json!({}), &[]));
        specs.push(spec("op_lowering", "Which executors lower a graph op, and where: the match arms in the Tree lowering (shared by the Vulkan, CUDA and Metal Tree executors), the Vulkan ggml-shader lowering, the native CUDA executor (own NVRTC kernels) and the native Metal executor (ggml's .metal kernels). Without `op`, a matrix of every op against the four. Shows what the CUDA and Metal paths cannot do and where to read how they do it.", json!({"op": s("graph op, e.g. MUL_MAT, RMS_NORM, FLASH_ATTN_EXT (omit for the matrix)")}), &[]));
        specs.push(spec("rust_outline", "List the items (fn, struct, enum, impl, trait, mod; or define in Scheme) of a .rs or .ss file with line numbers, to find what to read_file next.", json!({"path": s("file path")}), &["path"]));
        specs.push(spec("launch_sites", "For a Tree generator (expression or name, e.g. \"(k:matvec 'q8_0 1)\" or \"flash-attn-part\"): where it is defined in tree_kernels.ss and which Rust functions emit it, with the line. Read the function to see the push constants, grid, bindings and fusion that surround the kernel, which a pass must not break.", json!({"kernel": s("generator expression or name")}), &["kernel"]));
    }
    specs
}

impl ToolCtx {
    pub fn new(forum: Forum, search: SearchIndex, code: CodeIndex, eval_cfg: EvalConfig, repo: PathBuf, workspace: PathBuf, author: &str) -> Result<Self> {
        std::fs::create_dir_all(&workspace)?;
        let repo = repo.canonicalize()?;
        let workspace = workspace.canonicalize()?;
        let saved_notes = std::fs::read_to_string(workspace_notes(&workspace)).unwrap_or_default();
        let repo_files = ["crates/tensorlisp/src/scheme/stdlib/tree_opt.ss", "crates/tensorlisp/src/scheme/stdlib/tree.ss", "crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss"].iter().map(|p| repo.join(p)).filter(|p| p.exists()).collect();
        Ok(ToolCtx {
            forum,
            search: Mutex::new(search),
            code: Mutex::new(code),
            eval_cfg,
            repo,
            workspace,
            repo_files,
            author: author.into(),
            session: author.into(),
            rust_tools: false,
            notes: Mutex::new(saved_notes),
            last_eval: Mutex::new(None),
            repo_indexed: Mutex::new(false),
            ref_embedded: Mutex::new(false),
            ref_index: Mutex::new(None),
            finished: Mutex::new(None),
        })
    }

    /// The tool definitions this context offers.
    pub fn specs(&self) -> Vec<Value> {
        specs(self.rust_tools)
    }

    pub fn finished(&self) -> Option<String> {
        self.finished.lock().unwrap().clone()
    }

    /// Runs a tool; the result is always text.
    pub fn call(&self, name: &str, args: &Value) -> String {
        if let Value::String(raw) = args {
            return format!("error: the arguments were not valid JSON ({}). Send a JSON object.", clip(raw.clone(), 200));
        }
        let out = match name {
            "read_file" => self.read_file(args),
            "grep" => self.grep(args),
            "glob" => self.glob(args),
            "write_file" => self.write_file(args),
            "edit_file" => self.edit_file(args),
            "edit_lines" => self.edit_lines(args),
            "scheme" => self.scheme(args),
            "evaluate" => self.evaluate(args),
            "evaluate_batch" => self.evaluate_batch(args),
            "hot_kernels" => self.hot_kernels(args),
            "generator_workspace" => self.generator_workspace(),
            "generator_diff" => self.generator_diff(),
            "notes" => self.notes_tool(args),
            "rust_map" | "rust_outline" | "launch_sites" | "op_lowering" => self.rust_tool(name, args),
            "reference_kernels" => self.reference_kernels(args),
            "reference_search" => self.reference_search(args),
            "forum_search" => self.forum_search(args),
            "forum_grep" => self.forum_grep(args),
            "forum_read" => self.forum_read(args),
            "forum_post" => self.forum_post(args),
            "forum_reply" => self.forum_reply(args),
            "forum_leaderboard" => self.forum_leaderboard(args),
            "similar_code" => self.similar_code(args),
            "finish" => {
                let summary = args["summary"].as_str().unwrap_or("").to_string();
                *self.finished.lock().unwrap() = Some(summary);
                Ok("done".into())
            }
            other => Err(anyhow::anyhow!("no tool named {other}")),
        };
        match out {
            Ok(s) => clip(s, MAX_OUTPUT),
            Err(e) => format!("error: {e}"),
        }
    }

    // ---- paths -------------------------------------------------------------------------------------

    fn resolve(&self, path: &str, write: bool) -> Result<PathBuf> {
        if path.is_empty() {
            bail!("empty path");
        }
        let raw = Path::new(path);
        // Relative paths: writes go to the workspace (spelled relative to it or to the repository root);
        // reads look in the workspace first, then the repository, so a file you wrote is found by its short name.
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else if write {
            if self.repo.join(raw).starts_with(&self.workspace) { self.repo.join(raw) } else { self.workspace.join(raw) }
        } else if self.workspace.join(raw).exists() {
            self.workspace.join(raw)
        } else {
            self.repo.join(raw)
        };
        // canonicalize what exists; for a new file, its parent
        let canon = match joined.canonicalize() {
            Ok(p) => p,
            Err(_) if write => {
                let parent = joined.parent().ok_or_else(|| anyhow::anyhow!("bad path"))?;
                std::fs::create_dir_all(parent).ok();
                parent.canonicalize()?.join(joined.file_name().ok_or_else(|| anyhow::anyhow!("bad path"))?)
            }
            Err(e) => bail!("{path}: {e}"),
        };
        let allowed = if write { canon.starts_with(&self.workspace) } else { canon.starts_with(&self.repo) || canon.starts_with(&self.workspace) };
        if !allowed {
            bail!("{path} is outside {}", if write { format!("your workspace ({})", self.rel(&self.workspace)) } else { "the repository".into() });
        }
        if canon.components().any(|c| c.as_os_str() == ".git") || canon.file_name().is_some_and(|n| n.to_string_lossy().starts_with(".env")) {
            bail!("{path} is not readable");
        }
        Ok(canon)
    }

    fn rel(&self, p: &Path) -> String {
        p.strip_prefix(&self.repo).unwrap_or(p).display().to_string()
    }

    // ---- files -------------------------------------------------------------------------------------

    fn read_file(&self, a: &Value) -> Result<String> {
        let path = self.resolve(req_str(a, "path")?, false)?;
        let bytes = std::fs::read(&path)?;
        if bytes.iter().take(2048).any(|b| *b == 0) {
            bail!("{} looks binary", self.rel(&path));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let offset = a["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = a["limit"].as_u64().unwrap_or(200).clamp(1, 400) as usize;
        if offset > lines.len().max(1) {
            bail!("{} has only {} lines", self.rel(&path), lines.len());
        }
        let end = (offset - 1 + limit).min(lines.len());
        let mut out = String::new();
        for (i, l) in lines[offset - 1..end].iter().enumerate() {
            let l: String = if l.chars().count() > 400 { format!("{}…", l.chars().take(400).collect::<String>()) } else { l.to_string() };
            out.push_str(&format!("{:>6}\t{l}\n", offset + i));
        }
        out.push_str(&format!("[{} lines {}-{} of {}]", self.rel(&path), offset, end, lines.len()));
        Ok(out)
    }

    fn walk(&self, root: &Path, out: &mut Vec<PathBuf>) {
        if root.is_file() {
            out.push(root.to_path_buf());
            return;
        }
        let Ok(rd) = std::fs::read_dir(root) else { return };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            let path = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    self.walk(&path, out);
                }
            } else if !name.starts_with(".env") {
                out.push(path);
            }
        }
    }

    fn grep(&self, a: &Value) -> Result<String> {
        let re = RegexBuilder::new(req_str(a, "pattern")?).case_insensitive(a["ignore_case"].as_bool().unwrap_or(false)).build().map_err(|e| anyhow::anyhow!("bad regex: {e}"))?;
        let root = match a["path"].as_str() {
            Some(p) => self.resolve(p, false)?,
            None => self.repo.clone(),
        };
        let glob = a["glob"].as_str().map(glob_regex).transpose()?;
        let context = a["context"].as_u64().unwrap_or(0).min(10) as usize;
        let max = a["max_results"].as_u64().unwrap_or(60).clamp(1, 300) as usize;
        let mut files = Vec::new();
        self.walk(&root, &mut files);
        let (mut out, mut hits) = (String::new(), 0);
        'files: for f in files {
            let rel = self.rel(&f);
            if let Some(g) = &glob
                && !g.is_match(&rel)
                && !f.file_name().is_some_and(|n| g.is_match(&n.to_string_lossy()))
            {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&f) else { continue };
            if meta.len() > 2_000_000 {
                continue;
            }
            let Ok(bytes) = std::fs::read(&f) else { continue };
            if bytes.iter().take(1024).any(|b| *b == 0) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            let mut last_printed = 0usize;
            for (i, l) in lines.iter().enumerate() {
                if re.is_match(l) {
                    hits += 1;
                    let from = i.saturating_sub(context).max(last_printed);
                    let to = (i + context + 1).min(lines.len());
                    if context > 0 && last_printed > 0 && from > last_printed {
                        out.push_str("--\n");
                    }
                    for (j, line) in lines.iter().enumerate().take(to).skip(from) {
                        let sep = if j == i { ':' } else { '-' };
                        let line: String = line.chars().take(300).collect();
                        out.push_str(&format!("{rel}{sep}{}{sep}{line}\n", j + 1));
                    }
                    last_printed = to;
                    if hits >= max {
                        out.push_str(&format!("[stopped at {max} matches]\n"));
                        break 'files;
                    }
                }
            }
        }
        if hits == 0 {
            return Ok("no matches".into());
        }
        Ok(out)
    }

    fn glob(&self, a: &Value) -> Result<String> {
        let re = glob_regex(req_str(a, "pattern")?)?;
        let mut files = Vec::new();
        self.walk(&self.repo, &mut files);
        let found: Vec<String> = files.iter().map(|f| self.rel(f)).filter(|r| re.is_match(r)).take(300).collect();
        Ok(if found.is_empty() { "no files".into() } else { found.join("\n") })
    }

    fn write_file(&self, a: &Value) -> Result<String> {
        let path = self.resolve(req_str(a, "path")?, true)?;
        let content = req_str(a, "content")?;
        std::fs::write(&path, content)?;
        Ok(format!("wrote {} ({} bytes){}", self.rel(&path), content.len(), paren_warning(&path, content)))
    }

    fn edit_file(&self, a: &Value) -> Result<String> {
        let path = self.resolve(req_str(a, "path")?, true)?;
        let (mut old, mut new) = (req_str(a, "old")?.to_string(), req_str(a, "new")?.to_string());
        let text = std::fs::read_to_string(&path)?;
        let mut note = String::new();
        if old.is_empty() {
            bail!("`old` is empty");
        }
        // pasted from read_file with its line numbers: use the text without them
        if text.matches(old.as_str()).count() == 0
            && let Some(stripped) = strip_line_numbers(&old)
            && text.matches(stripped.as_str()).count() > 0
        {
            old = stripped;
            new = strip_line_numbers(&new).unwrap_or(new);
            note = " (the line-number prefixes from read_file were removed from `old`)".into();
        }
        let count = text.matches(old.as_str()).count();
        if count == 0 {
            let hint = match closest_region(&text, &old) {
                Some((from, to, score)) => format!(
                    " The closest text is at lines {from}-{to} ({:.0}% similar); this is what is really there:\n{}\nCopy it exactly, or replace those lines with edit_lines.",
                    score * 100.0,
                    numbered(&text, from, to.min(from + 40))
                ),
                None => " Nothing in the file resembles it: read the part you mean to change with read_file first, and copy from that. (edit_lines replaces a range of lines without needing the old text.)".to_string(),
            };
            bail!("`old` does not occur in {}.{hint}", self.rel(&path));
        }
        if count > 1 && !a["replace_all"].as_bool().unwrap_or(false) {
            bail!("`old` occurs {count} times in {}; add context to make it unique or set replace_all", self.rel(&path));
        }
        let edited = text.replace(old.as_str(), &new);
        std::fs::write(&path, &edited)?;
        let first_line = if count == 1 { text.find(old.as_str()).map(|i| text[..i].matches('\n').count() + 1) } else { None };
        let shown = first_line.map(|f| snapshot(&edited, f, f + new.matches('\n').count())).unwrap_or_default();
        Ok(format!("replaced {count} occurrence(s) in {}{note}{}{shown}", self.rel(&path), paren_warning(&path, &edited)))
    }

    fn edit_lines(&self, a: &Value) -> Result<String> {
        let path = self.resolve(req_str(a, "path")?, true)?;
        let (start, end) = (a["start"].as_u64().unwrap_or(0) as usize, a["end"].as_u64().unwrap_or(0) as usize);
        let text = std::fs::read_to_string(&path)?;
        let lines: Vec<&str> = text.split('\n').collect();
        let total = if text.ends_with('\n') { lines.len() - 1 } else { lines.len() };
        if start < 1 || end < start || end > total {
            bail!("lines {start}-{end} are out of range: {} has {total} lines", self.rel(&path));
        }
        let new = req_str(a, "new")?;
        let new = strip_line_numbers(new).unwrap_or_else(|| new.to_string());
        let mut out: Vec<String> = lines[..start - 1].iter().map(|l| l.to_string()).collect();
        if !new.is_empty() {
            out.extend(new.trim_end_matches('\n').split('\n').map(String::from));
        }
        out.extend(lines[end..].iter().map(|l| l.to_string()));
        let edited = out.join("\n");
        std::fs::write(&path, &edited)?;
        let added = if new.is_empty() { 0 } else { new.trim_end_matches('\n').split('\n').count() };
        let moved = added as i64 - (end + 1 - start) as i64;
        let shift = if moved == 0 { String::new() } else { format!(" Lines after line {end} moved by {moved:+}: later line numbers you noted are stale.") };
        Ok(format!("replaced lines {start}-{end} of {} ({} lines removed, {added} added).{shift}{}{}", self.rel(&path), end + 1 - start, paren_warning(&path, &edited), snapshot(&edited, start, start + added.saturating_sub(1))))
    }

    // ---- scheme and evaluation ---------------------------------------------------------------------

    fn scheme(&self, a: &Value) -> Result<String> {
        let h = scheme::handle()?;
        let session = format!("{}:{}", self.session, a["session"].as_str().unwrap_or("main"));
        if a["reset"].as_bool().unwrap_or(false) {
            h.reset(&session)?;
        }
        let code = req_str(a, "code")?;
        let out = h.run(&session, code, a["timeout_ms"].as_u64().unwrap_or(10_000).clamp(100, 120_000))?;
        Ok(if out.is_empty() { "(no output)".into() } else { crate::lispcheck::explain(&out, code) })
    }

    fn evaluate(&self, a: &Value) -> Result<String> {
        let code = match (a["code"].as_str(), a["path"].as_str()) {
            (Some(c), _) => c.to_string(),
            (None, Some(p)) => std::fs::read_to_string(self.resolve(p, false)?)?,
            (None, None) => String::new(),
        };
        let pipeline: Vec<String> = a["pipeline"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let cfg = self.eval_cfg_now()?;
        let generators = self.generators_arg(&a["generators"])?;
        let focus: Vec<String> = a["focus"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let ev = eval::evaluate_with(&self.forum, &cfg, &code, &pipeline, generators.as_deref())?;
        *self.last_eval.lock().unwrap() = Some(ev.id.clone());
        let mut out = eval::render_focus(&ev, &focus);
        // asked to use the edited generators, but there is no edit: the result is the baseline, whatever the agent believes
        if generators.is_some() && !ev.verdict.metrics.contains_key("generators_edited") {
            out = format!("WARNING: generators=true, but your tree_kernels.ss in the workspace is IDENTICAL to the original, so nothing about the generators was changed and this result is just the baseline. Your edits are not in that file (did you write them to another file, or were they lost?). Check with generator_diff before you draw any conclusion or post.\n\n{out}");
        }
        Ok(out)
    }

    /// `generators`: true for the workspace copy, a path for another file of the workspace.
    fn generators_arg(&self, v: &Value) -> Result<Option<String>> {
        let path = match v {
            Value::Bool(true) => self.workspace.join(GENERATOR_COPY),
            Value::String(p) if !p.is_empty() => self.resolve(p, false)?,
            _ => return Ok(None),
        };
        std::fs::read_to_string(&path).map(Some).map_err(|_| anyhow::anyhow!("there is no generator copy at {}: call generator_workspace first, then edit it", path.display()))
    }

    fn evaluate_batch(&self, a: &Value) -> Result<String> {
        let list = a["candidates"].as_array().filter(|l| !l.is_empty()).ok_or_else(|| anyhow::anyhow!("candidates must be a non-empty list"))?;
        if list.len() > 12 {
            bail!("at most 12 candidates per batch, got {}", list.len());
        }
        let cfg = self.eval_cfg_now()?;
        let mut done: Vec<(String, std::result::Result<crate::forum::Eval, String>)> = Vec::new();
        for (i, c) in list.iter().enumerate() {
            let name = c["name"].as_str().map(String::from).unwrap_or_else(|| format!("#{}", i + 1));
            let run = || -> Result<crate::forum::Eval> {
                let code = match (c["code"].as_str(), c["path"].as_str()) {
                    (Some(c), _) => c.to_string(),
                    (None, Some(p)) => std::fs::read_to_string(self.resolve(p, false)?)?,
                    (None, None) => String::new(),
                };
                let pipeline: Vec<String> = c["pipeline"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
                let generators = match c["generator_edits"].as_array() {
                    Some(edits) if !edits.is_empty() => {
                        let mut text = eval::ORIGINAL_GENERATORS.to_string();
                        for (n, e) in edits.iter().enumerate() {
                            let (o, w) = (e["old"].as_str().unwrap_or(""), e["new"].as_str().unwrap_or(""));
                            match text.matches(o).count() {
                                1 if !o.is_empty() => text = text.replacen(o, w, 1),
                                0 => bail!("generator_edits[{n}]: `old` does not occur in the original tree_kernels.ss (copy it exactly from read_file of your workspace copy, which starts as the original)"),
                                k => bail!("generator_edits[{n}]: `old` must occur exactly once in the original tree_kernels.ss, it occurs {k} times: include more context"),
                            }
                        }
                        Some(text)
                    }
                    _ => self.generators_arg(&c["generators"])?,
                };
                eval::evaluate_with(&self.forum, &cfg, &code, &pipeline, generators.as_deref())
            };
            done.push((name, run().map_err(|e| format!("{e:#}"))));
        }
        let mut order: Vec<usize> = (0..done.len()).collect();
        let key = |d: &(String, std::result::Result<crate::forum::Eval, String>)| d.1.as_ref().map(|e| if e.verdict.ok { e.verdict.score } else { -1.0 }).unwrap_or(-2.0);
        order.sort_by(|&x, &y| key(&done[y]).total_cmp(&key(&done[x])));
        let mut out = format!("{} candidates, best first (score is the measured kernel speedup where the GPU could time them, else the size ratio; above 1 beats the reference):\n", done.len());
        for &i in &order {
            let (name, r) = &done[i];
            match r {
                Err(e) => out.push_str(&format!("  {name}: ERROR {e}\n")),
                Ok(ev) => {
                    let m = &ev.verdict.metrics;
                    let failed: Vec<&str> = ev.verdict.gates.iter().filter(|(_, ok)| !**ok).map(|(g, _)| g.as_str()).collect();
                    let timing = match (m.get("kernel_speedup"), m.get("kernel_noise")) {
                        (Some(s), Some(n)) => format!(", GPU speedup {s:.4} (noise {:.1}%)", n * 100.0),
                        _ => String::new(),
                    };
                    out.push_str(&format!("  {name}: eval {} score {:.4}{timing}, size ratio {:.4}{}\n", ev.id, ev.verdict.score, m.get("size_ratio").copied().unwrap_or(0.0), if failed.is_empty() { String::new() } else { format!(", FAILED gates: {}", failed.join(", ")) }));
                }
            }
        }
        if let Some(&best) = order.iter().find(|&&i| done[i].1.as_ref().is_ok_and(|e| e.verdict.ok)) {
            let ev = done[best].1.as_ref().unwrap();
            *self.last_eval.lock().unwrap() = Some(ev.id.clone());
            out.push_str(&format!("\nbest: {} (eval {}). Details of it:\n{}", done[best].0, ev.id, eval::render(ev)));
        } else {
            out.push_str("\nno candidate passed every gate.\n");
        }
        Ok(out)
    }

    /// The evaluator configuration with the device's current profile weights (other workers may have imported more).
    pub fn eval_cfg_now(&self) -> Result<EvalConfig> {
        let mut cfg = self.eval_cfg.clone();
        cfg.weights = weights::load(self.forum.kv().as_ref(), &cfg.device)?.weights;
        cfg.launches = weights::load_launches(self.forum.kv().as_ref(), &cfg.device)?;
        Ok(cfg)
    }

    fn rust_tool(&self, name: &str, a: &Value) -> Result<String> {
        if !self.rust_tools {
            bail!("the Rust implementation tools are not enabled for this run (start it with --rust)");
        }
        match name {
            "rust_map" => Ok(implementation::map(&self.repo)),
            "op_lowering" => implementation::op_lowering(&self.repo, a["op"].as_str()),
            "rust_outline" => {
                let path = self.resolve(req_str(a, "path")?, false)?;
                implementation::outline(&path.display().to_string(), &std::fs::read_to_string(&path)?)
            }
            _ => implementation::launch_sites(&self.repo, req_str(a, "kernel")?),
        }
    }

    fn generator_workspace(&self) -> Result<String> {
        let path = self.workspace.join(GENERATOR_COPY);
        std::fs::write(&path, eval::ORIGINAL_GENERATORS)?;
        let outline = implementation::outline("tree_kernels.ss", eval::ORIGINAL_GENERATORS)?;
        let defs: Vec<&str> = outline.lines().filter(|l| l.contains("(define k:") || l.contains("(define (k:")).collect();
        Ok(format!(
            "wrote a pristine copy to {} ({} lines). Edit it with read_file / edit_file, check with generator_diff, then evaluate with generators=true.\n\nGenerators (line: definition); the other definitions are helpers they share, see rust_outline or grep for the rest:\n{}",
            self.rel(&path),
            eval::ORIGINAL_GENERATORS.lines().count(),
            defs.join("\n")
        ))
    }

    fn generator_diff(&self) -> Result<String> {
        let text = std::fs::read_to_string(self.workspace.join(GENERATOR_COPY)).map_err(|_| anyhow::anyhow!("no generator copy yet: call generator_workspace"))?;
        match eval::generator_edit(&text) {
            None => Ok("no difference from the original generators".into()),
            Some((diff, names)) => Ok(format!("touches: {}\n{}", names.join(", "), diff)),
        }
    }

    /// The agent's notes, as kept across compactions.
    pub fn notes(&self) -> String {
        self.notes.lock().unwrap().clone()
    }

    fn notes_tool(&self, a: &Value) -> Result<String> {
        let mut notes = self.notes.lock().unwrap();
        match req_str(a, "action")? {
            "read" => return Ok(if notes.is_empty() { "(no notes yet)".into() } else { notes.clone() }),
            "set" => *notes = req_str(a, "text")?.trim_end().to_string(),
            "append" => {
                if !notes.is_empty() {
                    notes.push('\n');
                }
                notes.push_str(req_str(a, "text")?.trim_end());
            }
            other => bail!("action must be read, set or append, not {other:?}"),
        }
        if notes.chars().count() > MAX_NOTES {
            let too_long = notes.chars().count();
            // do not keep an over-long version: roll back by reloading the last saved one
            *notes = std::fs::read_to_string(workspace_notes(&self.workspace)).unwrap_or_default();
            bail!("the notes would be {too_long} characters; the limit is {MAX_NOTES} because they are re-sent after every compaction. Condense them and use `set`. (Nothing was changed.)");
        }
        std::fs::write(workspace_notes(&self.workspace), notes.as_bytes())?;
        Ok(format!("notes saved ({} characters)", notes.chars().count()))
    }

    fn hot_kernels(&self, a: &Value) -> Result<String> {
        let device = &self.eval_cfg.device;
        let profile = weights::load(self.forum.kv().as_ref(), device)?;
        let limit = a["limit"].as_u64().unwrap_or(15).clamp(1, 100) as usize;
        // which kernels no pass can change: only the table's rows need checking, and only for the weighted ones
        let locked: std::collections::HashSet<String> = if profile.weights.is_empty() {
            Default::default()
        } else {
            let exprs: Vec<String> = profile.weights.keys().cloned().collect();
            let flags = scheme::handle()?.locked(&exprs)?;
            exprs.into_iter().zip(flags).filter(|(_, f)| *f).map(|(e, _)| e).collect()
        };
        Ok(weights::render(&profile, device, limit, &|e| locked.contains(e)))
    }

    fn reference_kernels(&self, a: &Value) -> Result<String> {
        reference::kernels_for(&self.repo, req_str(a, "kernel")?)
    }

    fn reference_search(&self, a: &Value) -> Result<String> {
        let mut slot = self.ref_index.lock().unwrap();
        if slot.is_none() {
            *slot = Some(RefIndex::build(&self.repo)?);
        }
        let idx = slot.as_ref().unwrap();
        if idx.chunks == 0 {
            return Ok("no reference sources found (is vendor/ggml checked out?)".into());
        }
        let hits = idx.search(req_str(a, "query")?, a["backend"].as_str(), a["limit"].as_u64().unwrap_or(8).clamp(1, 30) as usize)?;
        if hits.is_empty() {
            return Ok(format!("no results ({} windows indexed)", idx.chunks));
        }
        let mut out = String::new();
        for h in hits {
            out.push_str(&format!("{}:{}-{}  [{}] score {:.2}\n    {}\n", h.path, h.first, h.last, h.backend, h.score, h.snippet.chars().take(200).collect::<String>()));
        }
        Ok(out)
    }

    // ---- forum ---------------------------------------------------------------------------------------

    fn forum_search(&self, a: &Value) -> Result<String> {
        let mut idx = self.search.lock().unwrap();
        idx.sync(&self.forum)?;
        let hits = idx.search(req_str(a, "query")?, a["field"].as_str(), a["kind"].as_str(), a["limit"].as_u64().unwrap_or(8).clamp(1, 30) as usize)?;
        if hits.is_empty() {
            return Ok(format!("no results ({} documents indexed)", idx.doc_count()));
        }
        let mut out = String::new();
        for h in hits {
            out.push_str(&format!("{} [{}] {}  (by {}, score {:.2}){}\n    {}\n", h.entry, h.kind, h.headline, h.author, h.score, if h.post.is_some() { " reply" } else { "" }, h.snippet.chars().take(220).collect::<String>()));
        }
        Ok(out)
    }

    fn forum_grep(&self, a: &Value) -> Result<String> {
        let re = RegexBuilder::new(req_str(a, "pattern")?).case_insensitive(a["ignore_case"].as_bool().unwrap_or(false)).build().map_err(|e| anyhow::anyhow!("bad regex: {e}"))?;
        let limit = a["limit"].as_u64().unwrap_or(20).clamp(1, 100) as usize;
        let kv = self.forum.kv();
        let mut out = String::new();
        let mut n = 0;
        for prefix in ["entries/", "posts/"] {
            for key in kv.list(prefix, None, 100_000)? {
                let Some(v) = kv.get_string(&key)? else { continue };
                // match on the decoded fields, not the JSON text, so newlines and quotes are real
                let doc: Value = serde_json::from_str(&v).unwrap_or(Value::Null);
                let text: String = ["headline", "body", "code"].iter().filter_map(|f| doc[*f].as_str()).collect::<Vec<_>>().join("\n");
                for (i, l) in text.lines().enumerate() {
                    if re.is_match(l) {
                        out.push_str(&format!("{key}:{}: {}\n", i + 1, l.chars().take(200).collect::<String>()));
                        n += 1;
                        if n >= limit {
                            out.push_str("[limit reached]\n");
                            return Ok(out);
                        }
                    }
                }
            }
        }
        Ok(if n == 0 { "no matches".into() } else { out })
    }

    fn forum_read(&self, a: &Value) -> Result<String> {
        let id = req_str(a, "id")?;
        let e = self.forum.get(id)?.ok_or_else(|| anyhow::anyhow!("no entry {id}"))?;
        let mut out = render_entry(&e);
        let posts = self.forum.posts(id)?;
        if !posts.is_empty() {
            out.push_str("\nreplies:\n");
            for p in posts {
                out.push_str(&format!("- {} says: {}\n", p.author, p.body));
            }
        }
        Ok(out)
    }

    fn forum_post(&self, a: &Value) -> Result<String> {
        let kind: Kind = req_str(a, "kind")?.parse()?;
        let eval_id = match a["eval_id"].as_str() {
            Some("last") => Some(self.last_eval.lock().unwrap().clone().ok_or_else(|| anyhow::anyhow!("you have not run evaluate yet"))?),
            Some(id) => Some(id.to_string()),
            None => None,
        };
        let code_for_check = match &eval_id {
            Some(id) => self.forum.get_eval(id)?.map(|e| e.code),
            None => a["code"].as_str().map(String::from),
        };
        let parent = a["parent"].as_str().map(String::from);
        let mut warning = String::new();
        if let Some(code) = code_for_check.filter(|c| !c.trim().is_empty()) {
            let mut idx = self.code.lock().unwrap();
            idx.sync(&self.forum)?;
            if let Some(top) = idx.nearest_where(&code, 1, |s| !s.starts_with("ref/"))?.first() {
                if top.similarity >= DUPLICATE_THRESHOLD && parent.is_none() && !a["allow_duplicate"].as_bool().unwrap_or(false) && top.source.starts_with("entries/") {
                    bail!("this code is {:.0}% similar to {} already in the forum. Read it with forum_read, reply there if you have new evidence, or set parent to it (or allow_duplicate) if this is a deliberate variation.", top.similarity * 100.0, top.source.trim_start_matches("entries/"));
                } else if top.similarity >= 0.85 {
                    warning = format!("\nnote: similar to {} ({:.0}%)", top.source, top.similarity * 100.0);
                }
            }
        }
        let entry = self.forum.submit(NewEntry {
            kind: Some(kind),
            author: self.author.clone(),
            parent,
            headline: req_str(a, "headline")?.to_string(),
            body: req_str(a, "body")?.to_string(),
            tags: a["tags"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
            code: a["code"].as_str().map(String::from),
            eval_id,
        })?;
        Ok(format!("posted {} [{}] {}{warning}", entry.id, entry.kind.as_str(), entry.headline))
    }

    fn forum_reply(&self, a: &Value) -> Result<String> {
        let p = self.forum.reply(req_str(a, "id")?, &self.author, req_str(a, "body")?)?;
        Ok(format!("replied under {}", p.entry))
    }

    fn forum_leaderboard(&self, a: &Value) -> Result<String> {
        let board = self.forum.leaderboard(a["limit"].as_u64().unwrap_or(3).clamp(1, 20) as usize)?;
        if board.is_empty() {
            return Ok("no verified pass has beaten the baseline yet (a pass must pass every gate and score above 1)".into());
        }
        let mut out = String::new();
        for (target, device, entries) in board {
            out.push_str(&format!("{target} / {device}\n"));
            for e in entries {
                let v = e.verdict.as_ref().unwrap();
                out.push_str(&format!("  {:.4}  {}  {}  pipeline {:?}  (by {})\n", v.score, e.id, e.headline, e.pipeline.clone().unwrap_or_default(), e.author));
            }
        }
        Ok(out)
    }

    fn similar_code(&self, a: &Value) -> Result<String> {
        let code = match (a["code"].as_str(), a["path"].as_str()) {
            (Some(c), _) => c.to_string(),
            (None, Some(p)) => std::fs::read_to_string(self.resolve(p, false)?)?,
            (None, None) => bail!("give code or path"),
        };
        let mut idx = self.code.lock().unwrap();
        {
            let mut done = self.repo_indexed.lock().unwrap();
            if !*done {
                for f in &self.repo_files {
                    let label = self.rel(f);
                    idx.add_file(&label, &std::fs::read_to_string(f)?)?;
                }
                *done = true;
            }
        }
        let scope = a["scope"].as_str().unwrap_or("code");
        if !matches!(scope, "code" | "reference" | "all") {
            bail!("scope must be code, reference or all");
        }
        if scope != "code" {
            let mut done = self.ref_embedded.lock().unwrap();
            if !*done {
                let mut budget = REFERENCE_BUDGET;
                for src in reference::catalog(&self.repo) {
                    if budget == 0 {
                        break;
                    }
                    if let Ok(text) = std::fs::read_to_string(&src.path) {
                        idx.add_reference(&src.rel, &text, &mut budget)?;
                    }
                }
                *done = true;
            }
        }
        idx.sync(&self.forum)?;
        let k = a["k"].as_u64().unwrap_or(5).clamp(1, 20) as usize;
        let mut out = String::new();
        let keep = |s: &str| match scope {
            "reference" => s.starts_with("ref/"),
            "all" => true,
            _ => !s.starts_with("ref/"),
        };
        for s in idx.nearest_where(&code, k, keep)? {
            let first = s.snippet.lines().next().unwrap_or("");
            out.push_str(&format!("{:.2}  {}\n      {}\n", s.similarity, s.source, first.chars().take(160).collect::<String>()));
        }
        Ok(if out.is_empty() { "nothing indexed yet".into() } else { out })
    }
}

fn render_entry(e: &Entry) -> String {
    let mut out = format!("{} [{}] {}\nby {}{}{}\n", e.id, e.kind.as_str(), e.headline, e.author, e.parent.as_ref().map(|p| format!(", builds on {p}")).unwrap_or_default(), if e.tags.is_empty() { String::new() } else { format!(", tags {}", e.tags.join(", ")) });
    if let Some(v) = &e.verdict {
        out.push_str(&format!("verdict: {} on {}/{}, score {:.4}, gates {:?}, metrics {:?}\n", if v.ok { "OK" } else { "FAILED" }, v.target, v.device, v.score, v.gates, v.metrics));
    }
    if let Some(p) = &e.pipeline {
        out.push_str(&format!("pipeline: {p:?}\n"));
    }
    out.push_str(&format!("\n{}\n", e.body));
    if let Some(c) = &e.code {
        out.push_str(&format!("\ncode:\n{}\n", c.chars().take(6000).collect::<String>()));
    }
    out
}

fn req_str<'a>(a: &'a Value, key: &str) -> Result<&'a str> {
    a[key].as_str().ok_or_else(|| anyhow::anyhow!("missing string argument `{key}`"))
}

/// `*` within a path segment, `**` across segments, `?` one character.
pub fn glob_regex(glob: &str) -> Result<Regex> {
    let mut re = String::from("^");
    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' if chars.get(i + 1) == Some(&'*') => {
                re.push_str(".*");
                i += 1;
                if chars.get(i + 1) == Some(&'/') {
                    i += 1;
                }
            }
            '*' => re.push_str("[^/]*"),
            '?' => re.push_str("[^/]"),
            c => re.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    re.push('$');
    Ok(Regex::new(&re)?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::embed::HashEmbedder;
    use crate::kv::{Kv, MemKv};

    fn ctx(name: &str) -> (ToolCtx, PathBuf) {
        let root = std::env::temp_dir().join(format!("evolve-tools-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.ss"), "(define (f x)\n  (+ x 1))\n(define (g y) (f (f y)))\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join(".env"), "SECRET=1\n").unwrap();
        let kv: Arc<dyn Kv> = Arc::new(MemKv::new());
        let forum = Forum::new(kv.clone());
        let cfg = {
            let mut c = EvalConfig::new("spirv", root.join("evalwork"));
            c.exprs = vec!["(k:binary 'add)".into(), "(k:rms-norm #f)".into()];
            c
        };
        let ctx = ToolCtx::new(forum, SearchIndex::in_ram().unwrap(), CodeIndex::new(Arc::new(HashEmbedder), kv), cfg, root.clone(), root.join("work/agent"), "agent").unwrap();
        (ctx, root)
    }

    #[test]
    fn a_missed_edit_shows_the_real_text_and_line_ranges_can_be_replaced() {
        let (c, root) = ctx("edits");
        let body = "(define (f x)\n  (let ([a (* x 2)]\n        [b (+ x 1)])\n    (+ a b)))\n\n(define (g y)\n  (f (f y)))\n";
        std::fs::create_dir_all(root.join("work/agent")).unwrap();
        std::fs::write(root.join("work/agent/m.ss"), body).unwrap();
        // wrong indentation and a renamed variable: the error shows what is really there
        let miss = c.call("edit_file", &json!({"path": "m.ss", "old": "(define (f x)\n  (let ([a (* x 3)]\n   [b (+ x 1)])", "new": "x"}));
        assert!(miss.starts_with("error:") && miss.contains("closest text is at lines 1-3") && miss.contains("     2\t  (let ([a (* x 2)]") && miss.contains("edit_lines"), "{miss}");
        // pasted with read_file's line numbers: accepted
        let numbered = "     6\t(define (g y)\n     7\t  (f (f y)))";
        let ok = c.call("edit_file", &json!({"path": "m.ss", "old": numbered, "new": "     6\t(define (g y)\n     7\t  (f y))"}));
        assert!(ok.starts_with("replaced 1") && ok.contains("line-number prefixes"), "{ok}");
        assert!(std::fs::read_to_string(root.join("work/agent/m.ss")).unwrap().ends_with("(define (g y)\n  (f y))\n"));
        // nothing resembles it: say so
        let none = c.call("edit_file", &json!({"path": "m.ss", "old": "zzz qqq www", "new": "x"}));
        assert!(none.contains("Nothing in the file resembles it") && none.contains("edit_lines"), "{none}");
        // replace a range
        let r = c.call("edit_lines", &json!({"path": "m.ss", "start": 2, "end": 4, "new": "  (+ x x))"}));
        assert!(r.starts_with("replaced lines 2-4") && r.contains("3 lines removed, 1 added") && r.contains("moved by -2") && r.contains("now (5 lines in the file)") && !r.contains("WARNING"), "{r}");
        assert_eq!(std::fs::read_to_string(root.join("work/agent/m.ss")).unwrap(), "(define (f x)\n  (+ x x))\n\n(define (g y)\n  (f y))\n");
        // out of range, deletion, and the paren warning
        assert!(c.call("edit_lines", &json!({"path": "m.ss", "start": 3, "end": 99, "new": ""})).contains("out of range"));
        assert!(c.call("edit_lines", &json!({"path": "m.ss", "start": 3, "end": 3, "new": ""})).starts_with("replaced lines 3-3"));
        let broken = c.call("edit_lines", &json!({"path": "m.ss", "start": 1, "end": 1, "new": "(define (f x"}));
        assert!(broken.contains("WARNING") && broken.contains("never closed"), "{broken}");
    }

    #[test]
    fn files_slice_grep_glob_and_the_sandbox() {
        let (c, _root) = ctx("files");
        let out = c.call("read_file", &json!({"path": "src/a.ss", "offset": 2, "limit": 1}));
        assert!(out.contains("     2\t  (+ x 1)") && out.contains("lines 2-2 of 3"), "{out}");
        assert!(c.call("read_file", &json!({"path": "src/a.ss", "offset": 9})).starts_with("error:"));
        let out = c.call("grep", &json!({"pattern": "\\(f ", "context": 1}));
        assert!(out.contains("src/a.ss:3:(define (g y) (f (f y)))") && out.contains("src/a.ss-2-"), "{out}");
        assert_eq!(c.call("grep", &json!({"pattern": "nothing-here"})), "no matches");
        assert!(c.call("grep", &json!({"pattern": "SECRET"})).contains("no matches"));
        assert!(c.call("read_file", &json!({"path": ".env"})).starts_with("error:"));
        assert!(c.call("grep", &json!({"pattern": "fn", "glob": "*.rs"})).contains("src/b.rs:1"));
        assert!(!c.call("grep", &json!({"pattern": "define", "glob": "*.rs"})).contains("a.ss"));
        assert_eq!(c.call("glob", &json!({"pattern": "src/*.ss"})), "src/a.ss");
        assert!(c.call("glob", &json!({"pattern": "**/*.rs"})).contains("src/b.rs"));
        // writes stay in the workspace
        assert!(c.call("write_file", &json!({"path": "../../src/evil.ss", "content": "x"})).starts_with("error:"));
        assert!(c.call("write_file", &json!({"path": "/etc/evil", "content": "x"})).starts_with("error:"));
        assert!(c.call("write_file", &json!({"path": "cand.ss", "content": "(define z 1)\n(define z 2)\n"})).starts_with("wrote work/agent/cand.ss"));
        assert!(c.call("edit_file", &json!({"path": "cand.ss", "old": "(define z", "new": "(define w"})).contains("occurs 2 times"));
        assert!(c.call("edit_file", &json!({"path": "cand.ss", "old": "(define z", "new": "(define w", "replace_all": true})).contains("2 occurrence"));
        assert!(c.call("read_file", &json!({"path": "work/agent/cand.ss"})).contains("(define w 1)"), "repository-relative spelling");
        assert!(c.call("read_file", &json!({"path": "cand.ss"})).contains("(define w 1)"), "workspace-relative spelling");
        assert!(c.call("write_file", &json!({"path": "work/agent/sub/two.ss", "content": "x"})).starts_with("wrote work/agent/sub/two.ss"));
        assert!(c.call("read_file", &Value::String("{broken".into())).contains("not valid JSON"));
        assert!(c.call("nope", &json!({})).starts_with("error:"));
    }

    #[test]
    fn the_forum_workflow_end_to_end() {
        let (c, _root) = ctx("forum");
        // evaluate, post with last, search, read, reply, leaderboard
        let out = c.call("evaluate", &json!({"code": "", "pipeline": ["fold", "cse", "dce"]}));
        assert!(out.contains("OK"), "{out}");
        assert!(c.call("forum_post", &json!({"kind": "pass", "headline": "no licm", "body": "b"})).starts_with("error:"), "a pass without an evaluation is refused");
        let out = c.call("forum_post", &json!({"kind": "pass", "headline": "drop licm from the default", "body": "licm never fires on these kernels", "tags": ["licm"], "eval_id": "last"}));
        assert!(out.starts_with("posted e-"), "{out}");
        let id = out.split_whitespace().nth(1).unwrap().to_string();
        let out = c.call("forum_search", &json!({"query": "licm", "field": "headline"}));
        assert!(out.contains(&id) && out.contains("[pass]"), "{out}");
        assert!(c.call("forum_search", &json!({"query": "licm", "kind": "idea"})).starts_with("no results"));
        let out = c.call("forum_read", &json!({"id": id}));
        assert!(out.contains("verdict: OK") && out.contains("score 1."), "{out}");
        assert!(c.call("forum_reply", &json!({"id": id, "body": "confirmed on metal"})).starts_with("replied"));
        assert!(c.call("forum_read", &json!({"id": id})).contains("confirmed on metal"));
        assert!(c.call("forum_search", &json!({"query": "confirmed", "kind": "post"})).contains(&id));
        assert!(c.call("forum_grep", &json!({"pattern": "never\\s+fires"})).contains("never fires"));
        // this pass is valid but only matches the baseline: not a winner
        assert!(c.call("forum_leaderboard", &json!({})).contains("no verified pass has beaten the baseline"));
        // a deliberate duplicate is refused, then allowed with parent
        let dup = c.call("forum_post", &json!({"kind": "idea", "headline": "same again", "body": "b", "code": "(register-pass! 'a (lambda (k) k))"}));
        assert!(dup.starts_with("posted"), "{dup}");
        let again = c.call("forum_post", &json!({"kind": "idea", "headline": "same code", "body": "b", "code": "(register-pass! 'a (lambda (k) k)) ; again"}));
        assert!(again.contains("similar to"), "{again}");
        let ok = c.call("forum_post", &json!({"kind": "idea", "headline": "same code", "body": "b", "code": "(register-pass! 'a (lambda (k) k)) ; again", "allow_duplicate": true}));
        assert!(ok.starts_with("posted"), "{ok}");
        // similar_code finds the repository file we point it at and the forum entry
        std::fs::write(c.repo.join("src/pass.ss"), "(define (pass-licm k) (map-body (lambda (stmts) (map licm-stmt stmts)) k))\n(define (pass-other k) (let loop ([x k] [acc 0]) (if (null? x) acc (loop (cdr x) (+ acc 1)))))\n").unwrap();
        let mut c = c;
        c.repo_files = vec![c.repo.join("src/pass.ss")];
        let out = c.call("similar_code", &json!({"code": "(define (pass-hoist kernel) (map-body (lambda (ss) (map hoist-stmt ss)) kernel))"}));
        assert!(out.lines().next().unwrap().contains("repo/src/pass.ss#0"), "{out}");
        // scheme and finish
        assert_eq!(c.call("scheme", &json!({"code": "(define q 5) (* q 2)"})), "=> 10");
        assert_eq!(c.call("scheme", &json!({"code": "q"})), "=> 5");
        assert_eq!(c.call("scheme", &json!({"code": "q", "reset": true})).starts_with("error"), true);
        assert!(c.finished().is_none());
        c.call("finish", &json!({"summary": "done"}));
        assert_eq!(c.finished().as_deref(), Some("done"));
    }

    #[test]
    fn profiles_pick_the_kernels_and_references_are_reachable() {
        let (mut c, root) = ctx("hot");
        for (p, t) in [
            ("vendor/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp", "void main() {\n  float s = inversesqrt(mean + eps);\n  subgroupAdd(s);\n}\n"),
            ("vendor/ggml/src/ggml-metal/ggml-metal.metal", "kernel void kernel_rms_norm_f32() { simd_sum(x); }\n"),
            ("vendor/chez/lib/huge.ss", "(define chez-internal 1)\n"),
        ] {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, t).unwrap();
        }
        // vendor/ggml is searchable, the Chez build is not
        assert!(c.call("grep", &json!({"pattern": "inversesqrt"})).contains("vendor/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp:2:"));
        assert_eq!(c.call("grep", &json!({"pattern": "chez-internal"})), "no matches");
        assert!(c.call("glob", &json!({"pattern": "vendor/ggml/**/*.comp"})).contains("rms_norm.comp"));

        // no profile yet: everything counts equally
        c.eval_cfg.device = "rtx3080".into();
        assert!(c.call("hot_kernels", &json!({})).contains("no profiles"));
        let profile = "kernel launches busy ms gap ms busy us/l\ntree:(k:rms-norm #f #t)   900   9.0   1.0   10.0\ntree:(k:argmax)   100   1.0   0.0   10.0\ntotal 1000 10.0 1.0\n";
        weights::store(c.forum.kv().as_ref(), &weights::parse_profile(profile, "gemma4-decode", "rtx3080", 1.0).unwrap()).unwrap();
        let out = c.call("hot_kernels", &json!({}));
        assert!(out.lines().nth(2).unwrap().contains("90.0%") && out.contains("(k:rms-norm #f #t)"), "{out}");

        // evaluate now compiles the profiled expressions, weights the score, and reports the focus
        let out = c.call("evaluate", &json!({"code": "", "pipeline": ["fold"], "focus": ["rms-norm"]}));
        assert!(out.contains("profiled kernels") && out.contains("90.0%  (k:rms-norm #f #t)") && out.contains("focus [\"rms-norm\"]"), "{out}");
        assert!(out.contains("weighted_ratio") && out.contains("hot_coverage=1.0000"), "{out}");

        // reference tools
        let out = c.call("reference_kernels", &json!({"kernel": "(k:rms-norm #t #f)"}));
        assert!(out.contains("ggml-vulkan:") && out.contains("rms_norm.comp") && out.contains("ggml-metal.metal:1:"), "{out}");
        let out = c.call("reference_search", &json!({"query": "subgroupAdd"}));
        assert!(out.contains("rms_norm.comp:1-4") && out.contains("[ggml-vulkan]"), "{out}");
        assert!(c.call("reference_search", &json!({"query": "simd_sum", "backend": "metal"})).contains("ggml-metal.metal"));
        assert!(c.call("reference_search", &json!({"query": "simd_sum", "backend": "cuda"})).starts_with("no results"));

        // similar_code: reference scope only returns reference windows, the default never does
        let q = json!({"code": "float s = inversesqrt(mean + eps);\nsubgroupAdd(s);", "scope": "reference"});
        let out = c.call("similar_code", &q);
        assert!(out.contains("ref/vendor/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp#1-4"), "{out}");
        assert!(!c.call("similar_code", &json!({"code": "(define x 1)"})).contains("ref/"));
        assert!(c.call("similar_code", &json!({"code": "x", "scope": "bogus"})).starts_with("error:"));
        c.repo_files = vec![];
    }

    #[test]
    fn writing_a_scheme_file_with_unbalanced_parens_warns_at_once() {
        let (c, _root) = ctx("parens-write");
        let bad = "(register-pass! 'p\n  (lambda (k)\n    k)))\n";
        let out = c.call("write_file", &json!({"path": "p.ss", "content": bad}));
        assert!(out.starts_with("wrote work/agent/p.ss") && out.contains("WARNING") && out.contains("an extra `)` at line 3"), "{out}");
        let ok = c.call("write_file", &json!({"path": "q.ss", "content": "(register-pass! 'q (lambda (k) k))\n"}));
        assert!(!ok.contains("WARNING"), "{ok}");
        // other file types are not checked, and edit_file reports the state after the edit
        assert!(!c.call("write_file", &json!({"path": "notes.txt", "content": "((("})).contains("WARNING"));
        let fixed = c.call("edit_file", &json!({"path": "p.ss", "old": "k)))", "new": "k))"}));
        assert!(fixed.starts_with("replaced 1") && !fixed.contains("WARNING"), "{fixed}");
        let broke = c.call("edit_file", &json!({"path": "q.ss", "old": "(lambda (k) k))", "new": "(lambda (k) k)"}));
        assert!(broke.contains("WARNING") && broke.contains("never closed"), "{broke}");
    }

    #[test]
    fn a_generator_change_goes_through_workspace_edit_diff_and_evaluate() {
        let (c, root) = ctx("generators");
        let out = c.call("evaluate", &json!({"pipeline": ["fold"], "generators": true}));
        assert!(out.starts_with("error:") && out.contains("generator_workspace"), "{out}");
        assert!(c.call("generator_diff", &json!({})).starts_with("error:"));
        let out = c.call("generator_workspace", &json!({}));
        assert!(out.contains("work/agent/tree_kernels.ss") && out.contains("(define (k:binary op)") && out.contains("(define k:unary"), "{out}");
        assert!(root.join("work/agent/tree_kernels.ss").exists());
        assert_eq!(c.call("generator_diff", &json!({})), "no difference from the original generators");
        // an unchanged copy evaluates like no copy
        let same = c.call("evaluate", &json!({"pipeline": ["fold", "licm", "cse", "dce"], "generators": true}));
        assert!(same.contains("score 1.0000") && !same.contains("generator edit touches"), "{same}");
        assert!(same.starts_with("WARNING: generators=true, but your tree_kernels.ss") && same.contains("IDENTICAL"), "{same}");
        // edit the silu definition through the normal file tool
        let edit = c.call("edit_file", &json!({"path": "tree_kernels.ss", "old": "[(silu) `(/ ,x (+ 1.0 (exp (- ,x))))]", "new": "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (- ,x)))))]"}));
        assert!(edit.starts_with("replaced 1") && !edit.contains("WARNING"), "{edit}");
        let diff = c.call("generator_diff", &json!({}));
        assert!(diff.starts_with("touches: unary-expr") && diff.contains("+      [(silu) `(* ,x"), "{diff}");
        let out = c.call("evaluate", &json!({"pipeline": ["fold", "licm", "cse", "dce"], "generators": true}));
        assert!(out.contains("OK") && out.contains("generator edit touches: unary-expr"), "{out}");
        // an edit that unbalances the file warns at once (the file is big, the paren check is cheap)
        let bad = c.call("edit_file", &json!({"path": "tree_kernels.ss", "old": "[(relu) `(max ,x 0.0)]", "new": "[(relu) `(max ,x 0.0)"}));
        assert!(bad.contains("WARNING") && bad.contains("not balance"), "{}", bad.chars().take(300).collect::<String>());
        // resetting discards the edits
        c.call("generator_workspace", &json!({}));
        assert_eq!(c.call("generator_diff", &json!({})), "no difference from the original generators");
    }

    #[test]
    fn reader_errors_come_back_with_lines_and_a_paren_check() {
        let (c, _root) = ctx("parens");
        let out = c.call("scheme", &json!({"code": "(define (f x)\n  (+ x 1)\n"}));
        assert!(out.contains("error:") && out.contains("1 parenthesis is never closed") && out.contains("line 1, column 1"), "{out}");
        let out = c.call("scheme", &json!({"code": "(display 1))"}));
        assert!(out.contains("an extra `)` at line 1"), "{out}");
        // evaluate: the load error in the report carries the same help
        let ev = c.call("evaluate", &json!({"code": "(register-pass! 'p\n  (lambda (k)\n    k)\n", "pipeline": ["fold"]}));
        assert!(ev.contains("did not load") && ev.contains("paren check") && ev.contains("outermost opens at line 1"), "{ev}");
        // a name that is exported by (tl tree opt) is bound for candidates: map-body, which fold and licm are built on
        let out = c.call("scheme", &json!({"code": "(map-body (lambda (stmts) stmts) (k:binary 'add))"}));
        assert!(out.starts_with("=> (kernel"), "{out}");
    }

    #[test]
    fn notes_persist_in_the_workspace_and_are_size_limited() {
        let (c, root) = ctx("notes");
        assert_eq!(c.call("notes", &json!({"action": "read"})), "(no notes yet)");
        assert!(c.call("notes", &json!({"action": "set", "text": "GOAL: x"})).starts_with("notes saved"));
        c.call("notes", &json!({"action": "append", "text": "TRIED: y\n"}));
        assert_eq!(c.call("notes", &json!({"action": "read"})), "GOAL: x\nTRIED: y");
        assert_eq!(c.notes(), "GOAL: x\nTRIED: y");
        assert_eq!(std::fs::read_to_string(root.join("work/agent/notes.md")).unwrap(), "GOAL: x\nTRIED: y");
        // too long: refused, and the saved notes are untouched
        let long = "z".repeat(MAX_NOTES + 1);
        let out = c.call("notes", &json!({"action": "append", "text": long}));
        assert!(out.starts_with("error:") && out.contains("Condense"), "{out}");
        assert_eq!(c.call("notes", &json!({"action": "read"})), "GOAL: x\nTRIED: y");
        assert!(c.call("notes", &json!({"action": "set", "text": "z".repeat(MAX_NOTES)})).starts_with("notes saved"));
        assert!(c.call("notes", &json!({"action": "frobnicate"})).starts_with("error:"));
        // a new context on the same workspace (a resumed run) starts with them
        let kv: Arc<dyn Kv> = Arc::new(MemKv::new());
        let again = ToolCtx::new(Forum::new(kv.clone()), SearchIndex::in_ram().unwrap(), CodeIndex::new(Arc::new(HashEmbedder), kv), EvalConfig::new("spirv", root.join("e")), root.clone(), root.join("work/agent"), "agent").unwrap();
        assert_eq!(again.notes().len(), MAX_NOTES);
    }

    #[test]
    fn the_rust_tools_exist_only_when_enabled() {
        let names = |rust: bool| -> Vec<String> { specs(rust).iter().map(|s| s["function"]["name"].as_str().unwrap().to_string()).collect() };
        assert!(names(false).contains(&"notes".to_string()) && !names(false).contains(&"rust_map".to_string()));
        for t in ["rust_map", "rust_outline", "launch_sites", "op_lowering"] {
            assert!(names(true).contains(&t.to_string()));
        }
        let (mut c, root) = ctx("rust");
        assert!(c.call("rust_map", &json!({})).contains("not enabled"));
        assert!(c.call("launch_sites", &json!({"kernel": "matvec"})).starts_with("error:"));
        assert_eq!(c.specs().len(), specs(false).len());
        c.rust_tools = true;
        assert_eq!(c.specs().len(), specs(true).len());
        assert!(c.call("rust_map", &json!({})).starts_with("none of the implementation files"));
        std::fs::create_dir_all(root.join("crates/kernels/src/vulkan/lower")).unwrap();
        std::fs::write(root.join("crates/kernels/src/vulkan/lower/tree_ops.rs"), "//! Lowering of graph nodes to Tree kernels.\nimpl Ctx {\n    fn tree_argmax(&self) {\n        let e = \"(k:argmax)\";\n    }\n}\n").unwrap();
        let map = c.call("rust_map", &json!({}));
        assert!(map.contains("tree_ops.rs (6 lines)") && map.contains("Lowering of graph nodes"), "{map}");
        assert!(c.call("rust_outline", &json!({"path": "crates/kernels/src/vulkan/lower/tree_ops.rs"})).contains("3:   fn tree_argmax"));
        assert!(c.call("launch_sites", &json!({"kernel": "(k:argmax)"})).contains("tree_ops.rs:4 in fn tree_argmax"));
        assert!(c.call("rust_outline", &json!({"path": "src/a.ss"})).contains("1: (define (f x)"));
        assert!(c.call("rust_outline", &json!({"path": "../../etc/passwd"})).starts_with("error:"));
    }

    #[test]
    fn a_batch_ranks_candidates_and_reports_errors_per_candidate() {
        let (c, root) = ctx("batch");
        std::fs::create_dir_all(root.join("work/agent")).unwrap();
        let out = c.call("evaluate_batch", &json!({"candidates": [
            {"name": "weak", "pipeline": ["fold"]},
            {"name": "full", "pipeline": ["fold", "licm", "cse", "dce"]},
            {"name": "typo", "pipeline": ["nonsense"]},
            {"name": "gen", "pipeline": ["fold"], "generators": "variants/missing.ss"},
            {"name": "edit", "pipeline": ["fold"], "generator_edits": [{"old": "[(relu) `(max ,x 0.0)]", "new": "[(relu) `(max 0.0 ,x)]"}]},
            {"name": "badedit", "pipeline": ["fold"], "generator_edits": [{"old": "no such text anywhere", "new": "x"}]},
        ]}));
        assert!(out.contains("6 candidates, best first"), "{out}");
        assert!(out.contains("  edit: eval") && out.contains("badedit: ERROR") && out.contains("does not occur"), "{out}");
        // the failing candidates come last
        let (typo, weak, full) = (out.find("  typo:").unwrap(), out.find("  weak:").unwrap(), out.find("  full:").unwrap());
        assert!(typo > weak && typo > full, "{out}");
        assert!(out.contains("typo: eval") && out.contains("FAILED gates: loads") && out.contains("gen: ERROR") && out.contains("missing.ss"), "{out}");
        assert!(out.contains("best: full") || out.contains("best: weak"), "{out}");
        assert!(c.call("evaluate_batch", &json!({"candidates": []})).starts_with("error:"));
    }

    #[test]
    fn glob_matching() {
        let g = |p: &str, s: &str| glob_regex(p).unwrap().is_match(s);
        assert!(g("*.ss", "a.ss") && !g("*.ss", "d/a.ss") && g("**/*.ss", "d/e/a.ss") && g("crates/**/tree*.ss", "crates/x/y/tree_opt.ss") && g("a?.rs", "ab.rs"));
    }
}
