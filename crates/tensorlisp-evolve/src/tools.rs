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
use crate::forum::{Entry, Forum, Kind, NewEntry};
use crate::scheme;
use crate::search::SearchIndex;

const MAX_OUTPUT: usize = 12_000;
const SKIP_DIRS: &[&str] = &[".git", "target", "vendor", "node_modules", ".venv", "venv", "__pycache__", "models", ".ggml-cuda-cache"];
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
    last_eval: Mutex<Option<String>>,
    repo_indexed: Mutex<bool>,
    finished: Mutex<Option<String>>,
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

pub fn specs() -> Vec<Value> {
    let s = |d: &str| json!({"type": "string", "description": d});
    let n = |d: &str| json!({"type": "integer", "description": d});
    vec![
        spec("read_file", "Read a slice of a text file with line numbers. Paths are relative to the repository root. Use offset/limit to page through large files (default the first 200 lines, at most 400 per call).",
            json!({"path": s("file path"), "offset": n("first line, 1-based (default 1)"), "limit": n("number of lines (default 200)")}), &["path"]),
        spec("grep", "Regex search over files (Rust regex syntax). Returns path:line:text, with optional context lines. Skips target/, vendor/, .git/.",
            json!({"pattern": s("regular expression"), "path": s("file or directory to search (default: the repository root)"), "glob": s("only files whose path matches, e.g. *.ss or crates/**/*.rs"),
                   "context": n("lines of context around each match (default 0)"), "ignore_case": {"type": "boolean"}, "max_results": n("default 60")}), &["pattern"]),
        spec("glob", "List files whose repository-relative path matches a glob (* within a directory, ** across directories, ?).", json!({"pattern": s("glob pattern")}), &["pattern"]),
        spec("write_file", "Create or overwrite a file in your workspace (the only place you can write).", json!({"path": s("path inside the workspace"), "content": s("file contents")}), &["path", "content"]),
        spec("edit_file", "Replace text in a workspace file. `old` must occur exactly once unless replace_all is set.",
            json!({"path": s("path inside the workspace"), "old": s("text to replace"), "new": s("replacement"), "replace_all": {"type": "boolean"}}), &["path", "old", "new"]),
        spec("scheme", "Evaluate Scheme in a persistent Chez session that has the Tree libraries ((tl tree), (tl tree opt), (tl tree kernels), ptx, msl) and helpers: (show x), (stats kernel), (compare kernel pipeline-a pipeline-b), (diff-stmts kernel-a kernel-b), (tree-size t), (spirv-words k). Definitions persist between calls. Prints output and `=> value`. No file access.",
            json!({"code": s("Scheme source, one or more forms"), "session": s("session name (default \"main\")"), "timeout_ms": n("default 10000"), "reset": {"type": "boolean", "description": "start the session from scratch first"}}), &["code"]),
        spec("evaluate", "Evaluate a candidate pass pipeline. `code` (or `path` to a workspace file) is Scheme that defines passes and calls (register-pass! 'name (lambda (kernel) ...)); `pipeline` lists pass names to run, built-ins included: fold, licm, cse, dce (the baseline is all four). Compiles every Tree kernel with the baseline and with your pipeline, checks the gates, and returns a score (above 1 beats the baseline). Only this tool produces verdicts; forum_post copies them by eval_id.",
            json!({"code": s("Scheme source defining passes"), "path": s("workspace file with the Scheme source (alternative to code)"), "pipeline": {"type": "array", "items": {"type": "string"}, "description": "pass names, in order"}}), &["pipeline"]),
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
            json!({"code": s("code to look up"), "path": s("workspace file to look up instead"), "k": n("default 5")}), &[]),
        spec("finish", "End the session with a summary of what you found and what the next agent should try.", json!({"summary": s("final summary")}), &["summary"]),
    ]
}

impl ToolCtx {
    pub fn new(forum: Forum, search: SearchIndex, code: CodeIndex, eval_cfg: EvalConfig, repo: PathBuf, workspace: PathBuf, author: &str) -> Result<Self> {
        std::fs::create_dir_all(&workspace)?;
        let repo = repo.canonicalize()?;
        let workspace = workspace.canonicalize()?;
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
            last_eval: Mutex::new(None),
            repo_indexed: Mutex::new(false),
            finished: Mutex::new(None),
        })
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
            "scheme" => self.scheme(args),
            "evaluate" => self.evaluate(args),
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
        Ok(format!("wrote {} ({} bytes)", self.rel(&path), content.len()))
    }

    fn edit_file(&self, a: &Value) -> Result<String> {
        let path = self.resolve(req_str(a, "path")?, true)?;
        let (old, new) = (req_str(a, "old")?, req_str(a, "new")?);
        let text = std::fs::read_to_string(&path)?;
        let count = text.matches(old).count();
        if old.is_empty() || count == 0 {
            bail!("`old` does not occur in {}", self.rel(&path));
        }
        if count > 1 && !a["replace_all"].as_bool().unwrap_or(false) {
            bail!("`old` occurs {count} times in {}; add context to make it unique or set replace_all", self.rel(&path));
        }
        std::fs::write(&path, text.replace(old, new))?;
        Ok(format!("replaced {count} occurrence(s) in {}", self.rel(&path)))
    }

    // ---- scheme and evaluation ---------------------------------------------------------------------

    fn scheme(&self, a: &Value) -> Result<String> {
        let h = scheme::handle()?;
        let session = format!("{}:{}", self.session, a["session"].as_str().unwrap_or("main"));
        if a["reset"].as_bool().unwrap_or(false) {
            h.reset(&session)?;
        }
        let out = h.run(&session, req_str(a, "code")?, a["timeout_ms"].as_u64().unwrap_or(10_000).clamp(100, 120_000))?;
        Ok(if out.is_empty() { "(no output)".into() } else { out })
    }

    fn evaluate(&self, a: &Value) -> Result<String> {
        let code = match (a["code"].as_str(), a["path"].as_str()) {
            (Some(c), _) => c.to_string(),
            (None, Some(p)) => std::fs::read_to_string(self.resolve(p, false)?)?,
            (None, None) => String::new(),
        };
        let pipeline: Vec<String> = a["pipeline"].as_array().map(|l| l.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let ev = eval::evaluate(&self.forum, &self.eval_cfg, &code, &pipeline)?;
        *self.last_eval.lock().unwrap() = Some(ev.id.clone());
        Ok(eval::render(&ev))
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
            if let Some(top) = idx.nearest(&code, 1, None)?.first() {
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
            return Ok("no verified passes yet".into());
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
        idx.sync(&self.forum)?;
        let k = a["k"].as_u64().unwrap_or(5).clamp(1, 20) as usize;
        let mut out = String::new();
        for s in idx.nearest(&code, k, None)? {
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
        assert!(c.call("forum_leaderboard", &json!({})).contains(&id));
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
    fn glob_matching() {
        let g = |p: &str, s: &str| glob_regex(p).unwrap().is_match(s);
        assert!(g("*.ss", "a.ss") && !g("*.ss", "d/a.ss") && g("**/*.ss", "d/e/a.ss") && g("crates/**/tree*.ss", "crates/x/y/tree_opt.ss") && g("a?.rs", "ab.rs"));
    }
}
