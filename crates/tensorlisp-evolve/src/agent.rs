//! The agent loop: chat with tools until the model calls `finish`, runs out of turns, or stops asking for tools.
//!
//! Long runs are kept inside the context window by [`crate::context`]: before each model call the
//! conversation is compacted if its token estimate passes the threshold (old tool output and arguments
//! are shrunk, then the old turns are summarized by a model call and replaced); a context-length error
//! from the endpoint triggers a harder compaction and a retry. Everything that happens is appended to the
//! transcript, which [`replay`] turns back into the live conversation for `--resume`.
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use crate::context::{self, Budget};
use crate::llm::{ContextOverflow, Llm, Reply, ToolCall};
use crate::tools::ToolCtx;

pub const SYSTEM_PROMPT: &str = include_str!("prompt.md");
const RUST_PROMPT: &str = include_str!("prompt_rust.md");

/// Anything that answers a chat with tools: [`Llm`] over HTTP, or a script in tests.
pub trait Chat {
    fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Reply>;
}

impl Chat for Llm {
    fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Reply> {
        Llm::chat(self, messages, tools)
    }
}

pub struct AgentConfig {
    pub max_turns: usize,
    pub system: String,
    pub task: String,
    /// JSON lines of every message and compaction, appended as the run goes.
    pub transcript: Option<PathBuf>,
    /// The model's context window in tokens (what to stay under).
    pub context_tokens: usize,
    /// Compact when the estimate passes this fraction of `context_tokens`.
    pub compact_at: f64,
    /// Assistant turns kept verbatim by a compaction.
    pub keep_turns: usize,
    /// Continue from these messages (see [`replay`]) instead of starting from the task.
    pub resume: Option<Vec<Value>>,
    /// Remind the agent of its turn budget: half way without an `evaluate`, and two turns before the end.
    pub remind: bool,
    pub verbose: bool,
}

impl AgentConfig {
    pub fn new(system: String, task: String) -> Self {
        AgentConfig { max_turns: 40, system, task, transcript: None, context_tokens: 120_000, compact_at: 0.75, keep_turns: 8, resume: None, remind: true, verbose: false }
    }
}

#[derive(Debug, Default)]
pub struct AgentResult {
    pub summary: Option<String>,
    pub turns: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub tool_calls: usize,
    /// Times the old turns were replaced by a summary.
    pub compactions: usize,
    /// Context-length errors recovered from.
    pub overflows: usize,
}

pub fn system_prompt(target: &str, device: &str, author: &str, workspace: &str, rust: bool) -> String {
    let mut p = SYSTEM_PROMPT.to_string();
    if rust {
        p.push('\n');
        p.push_str(RUST_PROMPT);
    }
    p.replace("{target}", target).replace("{device}", device).replace("{author}", author).replace("{workspace}", workspace)
}

struct Transcript(Option<std::fs::File>);

impl Transcript {
    fn open(path: &Option<PathBuf>, fresh: bool) -> Result<Self> {
        let Some(p) = path else { return Ok(Transcript(None)) };
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        // a new run must not be mistaken for the continuation of an old one
        if fresh && p.metadata().is_ok_and(|m| m.len() > 0) {
            let stamp = crate::forum::now_millis();
            let old = p.with_file_name(format!("transcript-{stamp}.jsonl"));
            std::fs::rename(p, old)?;
        }
        Ok(Transcript(Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)))
    }

    fn write(&mut self, v: &Value) {
        if let Some(f) = self.0.as_mut() {
            let _ = writeln!(f, "{v}");
        }
    }
}

/// Rebuilds the conversation from a transcript: messages in order, a `compaction` record replacing everything
/// but the system prompt and the last `keep` messages by its task-plus-summary message. A trailing assistant
/// message whose tool calls were not all answered (the run died mid-turn) is dropped.
pub fn replay(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut messages: Vec<Value> = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}: not JSON", path.display(), n + 1))?;
        if let Some(c) = v.get("compaction") {
            let head = c["head"].as_str().context("compaction record without a head")?.to_string();
            let keep = c["keep"].as_u64().unwrap_or(0) as usize;
            if messages.len() < 2 + keep {
                bail!("{}:{}: compaction keeps {keep} messages but only {} exist", path.display(), n + 1, messages.len());
            }
            let tail = messages.split_off(messages.len() - keep);
            messages.truncate(1);
            messages.push(json!({"role": "user", "content": head}));
            messages.extend(tail);
        } else if v.get("role").is_some() {
            messages.push(v);
        }
    }
    if messages.len() < 2 {
        bail!("{} has no conversation to resume", path.display());
    }
    if let Some(i) = messages.iter().rposition(|m| m["role"] == "assistant" && m["tool_calls"].as_array().is_some_and(|c| !c.is_empty())) {
        let ids: Vec<&str> = messages[i]["tool_calls"].as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
        let answered = messages[i + 1..].iter().filter(|m| m["role"] == "tool" && m["tool_call_id"].as_str().is_some_and(|t| ids.contains(&t))).count();
        if answered < ids.len() {
            messages.truncate(i);
        }
    }
    Ok(messages)
}

fn short(v: &Value, n: usize) -> String {
    let s = v.to_string();
    if s.chars().count() > n { format!("{}…", s.chars().take(n).collect::<String>()) } else { s }
}

enum Compacted {
    Nothing,
    Shrunk,
    Summarized,
}

struct Run<'a> {
    chat: &'a dyn Chat,
    summarizer: &'a dyn Chat,
    ctx: &'a ToolCtx,
    cfg: &'a AgentConfig,
    budget: Budget,
    transcript: Transcript,
    result: AgentResult,
    /// Turns since the last compaction attempt: a compaction is not repeated at once (a failing or
    /// ineffective one would otherwise run every turn).
    since_compaction: usize,
}

impl Run<'_> {
    fn say(&self, text: String) {
        if self.cfg.verbose {
            eprintln!("{text}");
        }
    }

    /// Makes room. `aggressive` (after a context-length error) keeps fewer turns and always summarizes.
    fn compact(&mut self, messages: &mut Vec<Value>, aggressive: bool) -> Result<Compacted> {
        self.since_compaction = 0;
        let max_keep = if aggressive { (self.cfg.keep_turns / 2).max(2) } else { self.cfg.keep_turns };
        // keep as many recent turns as fit in half of the room that is left (at least one)
        let mut cut = None;
        for keep in (1..=max_keep).rev() {
            let Some(c) = context::split_point(messages, keep) else { continue };
            cut = Some(c);
            if self.budget.tokens_of(&messages[c..]) <= self.budget.room() / 2 {
                break;
            }
        }
        let before = self.budget.estimate(messages);
        // what the summarizer will see, taken before anything is shrunk
        let rendered = cut.map(|c| {
            let head = messages[1]["content"].as_str().unwrap_or("");
            context::render_for_summary(context::summary_of(head), &messages[2..c], 120_000)
        });
        let upto = cut.unwrap_or_else(|| messages.len().saturating_sub(2)).max(2);
        let saved = context::elide_outputs(&mut messages[2..upto]) + context::shrink_args(&mut messages[2..upto]);
        if !aggressive && self.budget.estimate(messages) <= self.budget.settle() {
            self.say(format!("compacted: shrank old output, ~{before} -> ~{} tokens", self.budget.estimate(messages)));
            return Ok(if saved > 0 { Compacted::Shrunk } else { Compacted::Nothing });
        }
        let (Some(cut), Some(rendered)) = (cut, rendered) else {
            return Ok(if saved > 0 { Compacted::Shrunk } else { Compacted::Nothing });
        };
        let notes = self.ctx.notes();
        let mut input = rendered;
        if !notes.trim().is_empty() {
            input.push_str(&format!("\n\n# The agent's notes\n{notes}"));
        }
        let mut ask = vec![json!({"role": "system", "content": context::SUMMARIZER_PROMPT}), json!({"role": "user", "content": input})];
        // a usable summary, or fall back to shrinking: a copy of the history or an invention would mislead every later turn
        let mut summary = String::new();
        for attempt in 0..2 {
            let reply = match self.summarizer.chat(&ask, &[]) {
                Ok(r) => r,
                Err(e) => {
                    self.say(format!("summarizing failed ({e}); only shrinking"));
                    return Ok(if saved > 0 { Compacted::Shrunk } else { Compacted::Nothing });
                }
            };
            self.result.prompt_tokens += reply.prompt_tokens;
            self.result.completion_tokens += reply.completion_tokens;
            match context::check_summary(&reply.content) {
                Ok(()) => {
                    summary = reply.content.trim().to_string();
                    break;
                }
                Err(why) => {
                    self.say(format!("the summary was rejected ({why}){}", if attempt == 0 { "; asking again" } else { "; only shrinking" }));
                    ask.push(json!({"role": "assistant", "content": reply.content.chars().take(500).collect::<String>()}));
                    ask.push(json!({"role": "user", "content": context::SUMMARIZER_RETRY}));
                }
            }
        }
        if summary.is_empty() {
            return Ok(if saved > 0 { Compacted::Shrunk } else { Compacted::Nothing });
        }
        let task = context::task_of(messages[1]["content"].as_str().unwrap_or("")).to_string();
        let head = context::head_message(&task, &summary, &notes);
        let kept = messages.len() - cut;
        let tail = messages.split_off(cut);
        messages.truncate(1);
        messages.push(json!({"role": "user", "content": head}));
        messages.extend(tail);
        self.transcript.write(&json!({"compaction": {"head": messages[1]["content"].clone(), "keep": kept, "summary_chars": summary.len()}}));
        self.result.compactions += 1;
        // the turns kept verbatim can still be too big: their older outputs go too (the latest two groups stay)
        if self.budget.needs_compaction(messages) {
            let upto = messages.len().saturating_sub(4).max(2);
            context::elide_outputs(&mut messages[2..upto]);
            context::shrink_args(&mut messages[2..upto]);
        }
        self.say(format!("compacted: summarized {} messages, ~{before} -> ~{} tokens", cut - 2, self.budget.estimate(messages)));
        Ok(Compacted::Summarized)
    }

    /// One model call; a context-length error compacts harder and retries.
    fn ask(&mut self, messages: &mut Vec<Value>, specs: &[Value]) -> Result<Reply> {
        let mut overflows = 0;
        loop {
            let chars = self.budget.request_chars(messages);
            match self.chat.chat(messages, specs) {
                Ok(r) => {
                    self.budget.observe(chars, r.prompt_tokens);
                    return Ok(r);
                }
                Err(e) => {
                    let Some(o) = e.downcast_ref::<ContextOverflow>() else { return Err(e) };
                    overflows += 1;
                    self.result.overflows += 1;
                    if let Some(limit) = o.limit {
                        self.budget.learn_limit(limit);
                    } else {
                        // no limit in the message: assume the estimate was too optimistic
                        self.budget.max_tokens = (self.budget.estimate(messages) as f64 * 0.7) as usize;
                    }
                    self.say(format!("context window exceeded ({o}); compacting harder (attempt {overflows})"));
                    if overflows > 3 {
                        return Err(e.context("still over the context window after compacting three times"));
                    }
                    if let Compacted::Nothing = self.compact(messages, true)? {
                        return Err(e.context("the context window is exceeded and there is nothing left to compact"));
                    }
                }
            }
        }
    }
}

pub fn run(chat: &dyn Chat, ctx: &ToolCtx, cfg: &AgentConfig) -> Result<AgentResult> {
    run_with(chat, None, ctx, cfg)
}

/// Like [`run`], with a separate (typically cheaper) model for the compaction summaries.
pub fn run_with(chat: &dyn Chat, summarizer: Option<&dyn Chat>, ctx: &ToolCtx, cfg: &AgentConfig) -> Result<AgentResult> {
    let specs = ctx.specs();
    let resumed = cfg.resume.is_some();
    let mut messages = cfg.resume.clone().unwrap_or_else(|| vec![json!({"role": "system", "content": cfg.system}), json!({"role": "user", "content": cfg.task})]);
    let mut run = Run {
        chat,
        summarizer: summarizer.unwrap_or(chat),
        ctx,
        cfg,
        budget: Budget::new(cfg.context_tokens, cfg.compact_at, serde_json::to_string(&specs)?.len()),
        transcript: Transcript::open(&cfg.transcript, !resumed)?,
        result: AgentResult::default(),
        since_compaction: usize::MAX / 2,
    };
    if run.budget.overhead_tokens() * 5 > cfg.context_tokens * 2 {
        run.say(format!("warning: the tool definitions alone take ~{} tokens of a {}-token window; compaction cannot make much room", run.budget.overhead_tokens(), cfg.context_tokens));
    }
    if !resumed {
        for m in &messages {
            run.transcript.write(m);
        }
    }
    let mut idle = 0;
    // a resumed conversation may already contain an evaluation or a post
    let did = |tool: &str| messages.iter().any(|m| m["tool_calls"].as_array().is_some_and(|c| c.iter().any(|x| x["function"]["name"] == tool)));
    let (mut evaluated, mut posted) = (did("evaluate"), did("forum_post"));
    // what the agent has been doing, for noticing loops
    let mut recent: std::collections::VecDeque<(String, String)> = std::collections::VecDeque::new();
    let mut edits_since_check = 0usize;
    for turn in 1..=cfg.max_turns {
        run.result.turns = turn;
        run.since_compaction = run.since_compaction.saturating_add(1);
        if run.since_compaction >= 3 && run.budget.needs_compaction(&messages) {
            run.compact(&mut messages, false)?;
        }
        let reply = run.ask(&mut messages, &specs)?;
        run.result.prompt_tokens += reply.prompt_tokens;
        run.result.completion_tokens += reply.completion_tokens;
        // keep only the fields the API wants back
        let mut assistant = json!({"role": "assistant", "content": reply.message["content"].clone()});
        if !reply.tool_calls.is_empty() {
            assistant["tool_calls"] = reply.message["tool_calls"].clone();
        }
        messages.push(assistant.clone());
        run.transcript.write(&assistant);
        if cfg.verbose && !reply.content.trim().is_empty() {
            eprintln!("[{turn}] {}", reply.content.trim().lines().next().unwrap_or(""));
        }
        if reply.tool_calls.is_empty() {
            idle += 1;
            if idle >= 3 {
                run.result.summary = Some(reply.content.clone()).filter(|s| !s.trim().is_empty());
                break;
            }
            let nudge = json!({"role": "user", "content": "Continue with tool calls, or call `finish` with your summary if you are done."});
            run.transcript.write(&nudge);
            messages.push(nudge);
            continue;
        }
        idle = 0;
        for ToolCall { id, name, arguments } in &reply.tool_calls {
            run.result.tool_calls += 1;
            evaluated |= name == "evaluate";
            posted |= name == "forum_post";
            recent.push_back((name.clone(), arguments.to_string()));
            if recent.len() > 8 {
                recent.pop_front();
            }
            match name.as_str() {
                "edit_file" | "edit_lines" | "write_file" => edits_since_check += 1,
                "evaluate" | "generator_diff" | "read_file" | "scheme" => edits_since_check = 0,
                _ => {}
            }
            let out = ctx.call(name, arguments);
            if cfg.verbose {
                eprintln!("[{turn}] {name} {} -> {}", short(arguments, 100), out.lines().next().unwrap_or("").chars().take(100).collect::<String>());
            }
            let msg = json!({"role": "tool", "tool_call_id": id, "content": out});
            run.transcript.write(&msg);
            messages.push(msg);
        }
        if let Some(summary) = ctx.finished() {
            run.result.summary = Some(summary);
            break;
        }
        // loops: the same call again and again, or a run of edits that were never looked at
        if cfg.remind {
            let last = recent.back().cloned();
            let repeats = last.as_ref().map_or(0, |l| recent.iter().filter(|c| *c == l).count());
            let note = if repeats >= 3 {
                let (name, _) = last.unwrap();
                recent.clear();
                Some(format!("You have made the identical `{name}` call {repeats} times in your last turns. Repeating it will not give a different result (for an edit, the file already has it, and line numbers have moved since). Stop and look: read the file or run generator_diff, work out what is different from what you expect, then do something else."))
            } else if edits_since_check >= 4 {
                edits_since_check = 0;
                Some("You have made several edits in a row without looking at the result. Edits compound, and line numbers shift after each one. Run `generator_diff` (or read the region back), then `evaluate` the change now: a small change that evaluates is worth more than a large one that does not.".to_string())
            } else {
                None
            };
            if let Some(text) = note {
                let m = json!({"role": "user", "content": text});
                run.transcript.write(&m);
                messages.push(m);
            }
        }
        // a model that explores until the budget is gone never tests anything: say where it stands
        if cfg.remind {
            let left = cfg.max_turns - turn;
            let note = if turn == cfg.max_turns.div_ceil(2) && !evaluated {
                Some(format!("Turn {turn} of {}: you have not run `evaluate` yet. Stop reading; write a first candidate pass now, run `evaluate`, and refine it from the result.", cfg.max_turns))
            } else if left == 2 {
                Some(if posted {
                    format!("Two turns left. Call `finish` with a summary of what you found and what to try next.")
                } else {
                    format!("Two turns left. Post what you learned with `forum_post` (a negative result counts), then call `finish`.")
                })
            } else {
                None
            };
            if let Some(text) = note {
                let m = json!({"role": "user", "content": text});
                run.transcript.write(&m);
                messages.push(m);
            }
        }
    }
    Ok(run.result)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::Arc;

    use super::*;
    use crate::embed::{CodeIndex, HashEmbedder};
    use crate::eval::EvalConfig;
    use crate::forum::Forum;
    use crate::kv::{Kv, MemKv};
    use crate::llm::parse_reply;
    use crate::search::SearchIndex;

    enum Step {
        Say(Value),
        Overflow(Option<usize>),
    }

    struct Script {
        replies: RefCell<VecDeque<Step>>,
        seen: RefCell<Vec<Vec<Value>>>,
    }

    impl Script {
        fn new(steps: Vec<Step>) -> Self {
            Script { replies: RefCell::new(steps.into()), seen: RefCell::new(Vec::new()) }
        }
        fn msgs(msgs: Vec<Value>) -> Self {
            Self::new(msgs.into_iter().map(Step::Say).collect())
        }
    }

    impl Chat for Script {
        fn chat(&self, messages: &[Value], _tools: &[Value]) -> Result<Reply> {
            self.seen.borrow_mut().push(messages.to_vec());
            match self.replies.borrow_mut().pop_front().expect("script ran out") {
                Step::Say(next) => parse_reply(&json!({"choices": [{"message": next}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).to_string()),
                Step::Overflow(limit) => Err(anyhow::Error::new(ContextOverflow { message: "too long".into(), limit })),
            }
        }
    }

    fn call(id: &str, name: &str, args: Value) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}})
    }

    fn calling(id: &str, name: &str, args: Value) -> Value {
        json!({"role": "assistant", "content": format!("step {id}"), "tool_calls": [call(id, name, args)]})
    }

    /// A turn with ~1300 characters of reasoning (below what shrinking cuts) and a small tool result:
    /// only summarizing can make room for a conversation made of these.
    fn pondering(id: &str, file: &str) -> Value {
        json!({"role": "assistant", "content": format!("step {id}: {}", "weighing the options carefully; ".repeat(40)), "tool_calls": [call(id, "read_file", json!({"path": file, "limit": 3}))]})
    }

    fn ctx(name: &str) -> ToolCtx {
        let root = std::env::temp_dir().join(format!("evolve-agent-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let kv: Arc<dyn Kv> = Arc::new(MemKv::new());
        let mut cfg = EvalConfig::new("spirv", root.join("w"));
        cfg.exprs = vec!["(k:binary 'add)".into()];
        ToolCtx::new(Forum::new(kv.clone()), SearchIndex::in_ram().unwrap(), CodeIndex::new(Arc::new(HashEmbedder), kv), cfg, root.clone(), root.join("work/a"), "a").unwrap()
    }

    fn cfg(turns: usize) -> AgentConfig {
        AgentConfig { max_turns: turns, remind: false, ..AgentConfig::new("sys".into(), "go".into()) }
    }

    /// Every tool message answers a call made by the assistant message just before its group.
    fn assert_well_formed(messages: &[Value]) {
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        let mut open: Vec<String> = Vec::new();
        for m in &messages[2..] {
            match m["role"].as_str().unwrap() {
                "assistant" => {
                    assert!(open.is_empty(), "calls without results: {open:?}");
                    open = m["tool_calls"].as_array().into_iter().flatten().map(|c| c["id"].as_str().unwrap().to_string()).collect();
                }
                "tool" => {
                    let id = m["tool_call_id"].as_str().unwrap();
                    assert!(open.contains(&id.to_string()), "orphan result {id}");
                    open.retain(|x| x != id);
                }
                _ => assert!(open.is_empty()),
            }
        }
    }

    #[test]
    fn a_scripted_agent_searches_evaluates_posts_and_finishes() {
        let c = ctx("script");
        let script = Script::msgs(vec![
            json!({"role": "assistant", "content": "look around", "tool_calls": [call("1", "forum_search", json!({"query": "cse"})), call("2", "evaluate", json!({"code": "", "pipeline": ["fold", "dce"]}))]}),
            json!({"role": "assistant", "content": null, "tool_calls": [call("3", "forum_post", json!({"kind": "pass", "headline": "fold+dce only", "body": "licm and cse do nothing here", "eval_id": "last"}))]}),
            json!({"role": "assistant", "content": null, "tool_calls": [call("4", "finish", json!({"summary": "baseline is already tight"}))]}),
        ]);
        let r = run(&script, &c, &cfg(10)).unwrap();
        assert_eq!(r.summary.as_deref(), Some("baseline is already tight"));
        assert_eq!((r.turns, r.tool_calls, r.prompt_tokens, r.compactions), (3, 4, 30, 0));
        let last = script.seen.borrow().last().unwrap().clone();
        let tool_msgs: Vec<&Value> = last.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_msgs.len(), 3);
        assert!(tool_msgs[0]["content"].as_str().unwrap().starts_with("no results"));
        assert!(tool_msgs[1]["content"].as_str().unwrap().contains("evaluation"));
        assert!(tool_msgs[2]["content"].as_str().unwrap().starts_with("posted e-"));
        assert_eq!(c.forum.entries(None, 10).unwrap().len(), 1);
    }

    #[test]
    fn the_agent_is_told_when_half_the_turns_are_gone_without_a_test_and_when_two_are_left() {
        let c = ctx("remind");
        let read = |i: usize| calling(&format!("r{i}"), "forum_search", json!({"query": format!("x{i}")}));
        let script = Script::msgs((0..8).map(read).collect());
        let r = run(&script, &c, &AgentConfig { remind: true, ..cfg(8) }).unwrap();
        assert_eq!(r.turns, 8);
        let user_notes = |seen: &Vec<Value>| -> Vec<String> { seen.iter().skip(2).filter(|m| m["role"] == "user").map(|m| m["content"].as_str().unwrap().to_string()).collect() };
        let last = script.seen.borrow().last().unwrap().clone();
        let notes = user_notes(&last);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[0].starts_with("Turn 4 of 8") && notes[0].contains("evaluate"));
        assert!(notes[1].starts_with("Two turns left") && notes[1].contains("forum_post"));
        // an agent that has evaluated is not nagged at half way; one that has posted is told to finish
        let c = ctx("remind2");
        let script = Script::msgs(vec![
            calling("a", "evaluate", json!({"code": "", "pipeline": ["fold"]})),
            calling("b", "forum_search", json!({"query": "x1"})),
            calling("c", "forum_search", json!({"query": "x2"})),
            calling("d", "forum_post", json!({"kind": "note", "headline": "h", "body": "b"})),
            calling("e", "forum_search", json!({"query": "x3"})),
            calling("f", "forum_search", json!({"query": "x4"})),
        ]);
        run(&script, &c, &AgentConfig { remind: true, ..cfg(6) }).unwrap();
        let notes = user_notes(&script.seen.borrow().last().unwrap().clone());
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].starts_with("Two turns left") && notes[0].contains("`finish`") && !notes[0].contains("forum_post"));
        // switched off
        let c = ctx("remind3");
        let script = Script::msgs((0..4).map(read).collect());
        run(&script, &c, &cfg(4)).unwrap();
        assert!(user_notes(&script.seen.borrow().last().unwrap().clone()).is_empty());
    }

    #[test]
    fn repeated_calls_and_unchecked_runs_of_edits_are_called_out() {
        let c = ctx("loops");
        let mut steps: Vec<Value> = (0..4).map(|i| calling(&format!("r{i}"), "forum_search", json!({"query": "same"}))).collect();
        steps.push(calling("e1", "edit_lines", json!({"path": "a.ss", "start": 1, "end": 1, "new": "x"})));
        steps.push(calling("e2", "edit_lines", json!({"path": "a.ss", "start": 2, "end": 2, "new": "x"})));
        steps.push(calling("e3", "edit_lines", json!({"path": "a.ss", "start": 3, "end": 3, "new": "x"})));
        steps.push(calling("e4", "edit_lines", json!({"path": "a.ss", "start": 4, "end": 4, "new": "x"})));
        steps.push(calling("n", "forum_search", json!({"query": "different"})));
        let script = Script::msgs(steps);
        run(&script, &c, &AgentConfig { remind: true, ..cfg(9) }).unwrap();
        let last = script.seen.borrow().last().unwrap().clone();
        let notes: Vec<&str> = last.iter().skip(2).filter(|m| m["role"] == "user").map(|m| m["content"].as_str().unwrap()).collect();
        assert!(notes.iter().any(|n| n.starts_with("You have made the identical `forum_search` call 3 times")), "{notes:?}");
        assert!(notes.iter().any(|n| n.starts_with("You have made several edits in a row") && n.contains("generator_diff")), "{notes:?}");
        // an agent that reads in between is left alone
        let c = ctx("loops2");
        let steps: Vec<Value> = (0..6).map(|i| if i % 2 == 0 { calling(&format!("e{i}"), "edit_lines", json!({"path": "a.ss", "start": i + 1, "end": i + 1, "new": "x"})) } else { calling(&format!("r{i}"), "read_file", json!({"path": format!("f{i}")})) }).collect();
        let script = Script::msgs(steps);
        run(&script, &c, &AgentConfig { remind: true, ..cfg(6) }).unwrap();
        let last = script.seen.borrow().last().unwrap().clone();
        assert!(last.iter().skip(2).filter(|m| m["role"] == "user").all(|m| !m["content"].as_str().unwrap().contains("several edits")));
    }

    #[test]
    fn a_resumed_run_is_reminded_by_its_new_budget_and_remembers_what_it_did() {
        let c = ctx("remind-resume");
        let search = |i: usize| calling(&format!("s{i}"), "forum_search", json!({"query": format!("x{i}")}));
        // earlier: it evaluated already
        let mut history = vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": "go"})];
        history.push(calling("e0", "evaluate", json!({"code": "", "pipeline": ["fold"]})));
        history.push(json!({"role": "tool", "tool_call_id": "e0", "content": "evaluation ..."}));
        let script = Script::msgs((0..6).map(search).collect());
        run(&script, &c, &AgentConfig { remind: true, resume: Some(history), ..cfg(6) }).unwrap();
        let last = script.seen.borrow().last().unwrap().clone();
        let notes: Vec<&str> = last.iter().skip(2).filter(|m| m["role"] == "user").map(|m| m["content"].as_str().unwrap()).collect();
        // no "you have not evaluated" nag, but the end-of-budget note arrives
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].starts_with("Two turns left"));
    }

    #[test]
    fn an_agent_that_stops_calling_tools_is_nudged_then_ended() {
        let c = ctx("idle");
        let say = |t: &str| json!({"role": "assistant", "content": t});
        let script = Script::msgs(vec![say("hmm"), say("hmm"), say("I am done")]);
        let r = run(&script, &c, &cfg(10)).unwrap();
        assert_eq!(r.turns, 3);
        assert_eq!(r.summary.as_deref(), Some("I am done"));
        assert!(script.seen.borrow()[1].last().unwrap()["content"].as_str().unwrap().contains("finish"));
    }

    #[test]
    fn turns_are_capped_and_errors_go_back_to_the_model() {
        let c = ctx("cap");
        let bad = json!({"role": "assistant", "content": null, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": "{nope"}}]});
        let script = Script::msgs(vec![bad.clone(), bad]);
        let r = run(&script, &c, &cfg(2)).unwrap();
        assert_eq!((r.turns, r.summary), (2, None));
        assert!(script.seen.borrow()[1].last().unwrap()["content"].as_str().unwrap().contains("not valid JSON"));
    }

    #[test]
    fn the_prompt_has_its_placeholders_filled() {
        let p = system_prompt("ptx", "rtx3080", "agent7", "evolve/work/agent7", false);
        assert!(p.contains("Target: `ptx`") && p.contains("evolve/work/agent7") && !p.contains("{target}") && !p.contains("{author}"));
        assert!(!p.contains("rust_map") && system_prompt("ptx", "d", "a", "w", true).contains("rust_map"));
        assert!(p.contains("`notes`"));
    }

    /// A workspace file big enough that a few reads of it fill a small context window.
    fn big_file(c: &ToolCtx) -> String {
        let text: String = (1..=300).map(|i| format!("line {i} of the scratch file, padding padding padding padding\n")).collect();
        std::fs::write(c.workspace.join("big.txt"), text).unwrap();
        "big.txt".into()
    }

    #[test]
    fn old_turns_are_summarized_and_the_notes_survive() {
        let c = ctx("compact");
        let file = big_file(&c);
        c.call("notes", &json!({"action": "set", "text": "GOAL: beat baseline on matvec\nTRIED: licm hoist (e-1) no gain"}));
        let mut steps: Vec<Value> = (0..12).map(|i| pondering(&format!("r{i}"), &file)).collect();
        steps.push(calling("done", "finish", json!({"summary": "ok"})));
        let main = Script::msgs(steps);
        let summarizer = Script::msgs(vec![json!({"role": "assistant", "content": "SUMMARY: read the scratch file repeatedly; nothing learned"}); 8]);
        let cfg = AgentConfig { context_tokens: 10_000, compact_at: 0.6, keep_turns: 3, ..cfg(20) };
        let r = run_with(&main, Some(&summarizer), &c, &cfg).unwrap();
        assert_eq!(r.summary.as_deref(), Some("ok"));
        assert!(r.compactions >= 1, "{r:?}");
        // the summarizer saw the old turns and the notes
        let asked = summarizer.seen.borrow()[0].clone();
        assert_eq!(asked[0]["content"], context::SUMMARIZER_PROMPT);
        let input = asked[1]["content"].as_str().unwrap();
        assert!(input.contains("CALL read_file") && input.contains("RESULT:") && input.contains("weighing the options") && input.contains("GOAL: beat baseline on matvec"), "{input}");
        // later requests carry the summary and notes in the task message, keep recent turns, and stay well formed
        let seen = main.seen.borrow();
        let last = seen.last().unwrap();
        let head = last[1]["content"].as_str().unwrap();
        assert!(head.starts_with("go") && head.contains("SUMMARY: read the scratch file") && head.contains("TRIED: licm hoist (e-1)"), "{head}");
        assert_well_formed(last);
        assert!(last.len() < 2 + 2 * 12, "history was not replaced: {}", last.len());
        assert!(last.iter().filter(|m| m["role"] == "assistant").count() >= 1);
        // not one summary per turn
        assert!(summarizer.seen.borrow().len() <= 5, "{} summaries for 13 turns", summarizer.seen.borrow().len());
        // the original size would not have fit
        assert!(seen.iter().map(|m| m.len()).max().unwrap() >= last.len());
    }

    #[test]
    fn a_second_compaction_builds_on_the_first_summary() {
        let c = ctx("twice");
        let file = big_file(&c);
        let mut steps: Vec<Value> = (0..24).map(|i| pondering(&format!("r{i}"), &file)).collect();
        steps.push(calling("done", "finish", json!({"summary": "ok"})));
        let main = Script::msgs(steps);
        let sums: Vec<Value> = (1..=10).map(|i| json!({"role": "assistant", "content": format!("SUMMARY #{i}: explored the kernels and read the scratch file")})).collect();
        let summarizer = Script::msgs(sums);
        let cfg = AgentConfig { context_tokens: 9_000, compact_at: 0.6, keep_turns: 2, ..cfg(30) };
        let r = run_with(&main, Some(&summarizer), &c, &cfg).unwrap();
        assert!(r.compactions >= 2, "{r:?}");
        let second = summarizer.seen.borrow()[1][1]["content"].as_str().unwrap().to_string();
        assert!(second.contains("Summary written at the previous compaction") && second.contains("SUMMARY #1"), "{second}");
        let last = main.seen.borrow().last().unwrap().clone();
        assert!(last[1]["content"].as_str().unwrap().contains(&format!("SUMMARY #{}:", r.compactions)));
        assert!(!last[1]["content"].as_str().unwrap().contains("SUMMARY #1:"));
        assert_well_formed(&last);
    }

    #[test]
    fn a_context_length_error_compacts_harder_and_retries() {
        let c = ctx("overflow");
        let file = big_file(&c);
        let mut steps: Vec<Step> = (0..5).map(|i| Step::Say(pondering(&format!("r{i}"), &file))).collect();
        steps.push(Step::Overflow(Some(4000)));
        steps.push(Step::Say(calling("done", "finish", json!({"summary": "recovered"}))));
        let main = Script::new(steps);
        let summarizer = Script::msgs(vec![json!({"role": "assistant", "content": "SUMMARY: five reads of the scratch file, nothing learned yet"})]);
        // a generous window so only the endpoint's error triggers the compaction
        let cfg = AgentConfig { context_tokens: 1_000_000, keep_turns: 4, ..cfg(20) };
        let r = run_with(&main, Some(&summarizer), &c, &cfg).unwrap();
        assert_eq!((r.overflows, r.compactions), (1, 1));
        assert_eq!(r.summary.as_deref(), Some("recovered"));
        let seen = main.seen.borrow();
        let (failed, retried) = (&seen[5], &seen[6]);
        assert!(retried.len() < failed.len(), "{} vs {}", retried.len(), failed.len());
        assert!(retried[1]["content"].as_str().unwrap().contains("SUMMARY: five reads of the scratch file"));
        assert_well_formed(retried);
    }

    #[test]
    fn an_overflow_with_nothing_to_compact_is_an_error() {
        let c = ctx("stuck");
        let main = Script::new(vec![Step::Overflow(Some(100))]);
        let err = run(&main, &c, &cfg(5)).unwrap_err();
        assert!(format!("{err:#}").contains("nothing left to compact"), "{err:#}");
        // other errors pass through untouched
        struct Broken;
        impl Chat for Broken {
            fn chat(&self, _: &[Value], _: &[Value]) -> Result<Reply> {
                bail!("HTTP 500")
            }
        }
        assert!(format!("{:#}", run(&Broken, &ctx("broken"), &cfg(5)).unwrap_err()).contains("HTTP 500"));
    }

    #[test]
    fn a_summarizer_that_copies_the_history_is_asked_again_and_then_ignored() {
        let c = ctx("echo");
        let file = big_file(&c);
        let make_main = || {
            let mut steps: Vec<Value> = (0..8).map(|i| pondering(&format!("r{i}"), &file)).collect();
            steps.push(calling("done", "finish", json!({"summary": "ok"})));
            Script::msgs(steps)
        };
        let echo = json!({"role": "assistant", "content": "CALL read_file({\"path\":\"x\"})\nRESULT:     1\t(fabricated file contents that were never in the history)"});
        let good = json!({"role": "assistant", "content": "Goal: explore the kernels.\nTried: read the scratch file several times.\nNext: stop reading it and write a pass."});
        let cfg = AgentConfig { context_tokens: 10_000, compact_at: 0.6, keep_turns: 3, ..cfg(20) };

        // echo once, then a proper summary: the second attempt is used and nothing fabricated gets in
        let main = make_main();
        let summarizer = Script::msgs(vec![echo.clone(), good.clone(), good.clone(), good.clone()]);
        let r = run_with(&main, Some(&summarizer), &c, &cfg).unwrap();
        assert!(r.compactions >= 1, "{r:?}");
        let retried = summarizer.seen.borrow()[1].clone();
        assert_eq!(retried.last().unwrap()["content"], context::SUMMARIZER_RETRY);
        let last = main.seen.borrow().last().unwrap().clone();
        let head = last[1]["content"].as_str().unwrap();
        assert!(head.contains("Goal: explore the kernels.") && !head.contains("fabricated") && head.contains("can be wrong"), "{head}");

        // echo every time: no summary is used at all, the run goes on with shrinking
        let main = make_main();
        let summarizer = Script::msgs(vec![echo.clone(); 12]);
        let r = run_with(&main, Some(&summarizer), &c, &cfg).unwrap();
        assert_eq!((r.summary.as_deref(), r.compactions), (Some("ok"), 0));
        assert!(!main.seen.borrow().last().unwrap()[1]["content"].as_str().unwrap().contains("fabricated"));
    }

    #[test]
    fn a_failing_summarizer_degrades_to_shrinking() {
        struct Failing(std::cell::Cell<usize>);
        impl Chat for Failing {
            fn chat(&self, _: &[Value], _: &[Value]) -> Result<Reply> {
                self.0.set(self.0.get() + 1);
                bail!("summarizer is down")
            }
        }
        let c = ctx("nosummary");
        let file = big_file(&c);
        let mut steps: Vec<Value> = (0..14).map(|i| pondering(&format!("r{i}"), &file)).collect();
        steps.push(calling("done", "finish", json!({"summary": "ok"})));
        let main = Script::msgs(steps);
        let failing = Failing(Default::default());
        let cfg = AgentConfig { context_tokens: 10_000, compact_at: 0.6, keep_turns: 3, ..cfg(20) };
        let r = run_with(&main, Some(&failing), &c, &cfg).unwrap();
        assert_eq!((r.summary.as_deref(), r.compactions), (Some("ok"), 0));
        // it was tried, but not on every turn
        assert!(failing.0.get() >= 1 && failing.0.get() <= 5, "{} attempts", failing.0.get());
        assert_well_formed(main.seen.borrow().last().unwrap());
    }

    #[test]
    fn a_run_can_be_resumed_from_its_transcript() {
        let c = ctx("resume");
        let file = big_file(&c);
        let path = c.workspace.join("transcript.jsonl");
        let first = Script::msgs(vec![calling("a", "read_file", json!({"path": file, "limit": 3})), calling("b", "read_file", json!({"path": file, "limit": 3}))]);
        let mut cfg1 = cfg(2);
        cfg1.transcript = Some(path.clone());
        run(&first, &c, &cfg1).unwrap();
        // a second fresh run rotates the old transcript away instead of appending to it
        let again = Script::msgs(vec![json!({"role": "assistant", "content": "x"}); 3]);
        run(&again, &c, &AgentConfig { transcript: Some(path.clone()), ..cfg(3) }).unwrap();
        assert!(std::fs::read_dir(&c.workspace).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("transcript-")));
        assert_eq!(replay(&path).unwrap()[1]["content"], "go");
        assert!(replay(&path).unwrap().iter().all(|m| m["role"] != "tool"));

        // resume the interrupted first run: simulate by rebuilding one from scratch
        let path2 = c.workspace.join("t2.jsonl");
        let mut cfg2 = cfg(2);
        cfg2.transcript = Some(path2.clone());
        run(&Script::msgs(vec![calling("a", "read_file", json!({"path": file, "limit": 3})), calling("b", "read_file", json!({"path": file, "limit": 3}))]), &c, &cfg2).unwrap();
        let messages = replay(&path2).unwrap();
        assert_eq!(messages.len(), 2 + 2 * 2);
        assert_well_formed(&messages);
        let resumed = Script::msgs(vec![calling("z", "finish", json!({"summary": "continued"}))]);
        let mut cfg3 = cfg(5);
        cfg3.transcript = Some(path2.clone());
        cfg3.resume = Some(messages.clone());
        let r = run(&resumed, &c, &cfg3).unwrap();
        assert_eq!(r.summary.as_deref(), Some("continued"));
        assert_eq!(resumed.seen.borrow()[0].len(), messages.len());
        // and the transcript now replays to the longer conversation
        assert_eq!(replay(&path2).unwrap().len(), messages.len() + 2);
    }

    #[test]
    fn replay_applies_compactions_and_drops_a_half_finished_turn() {
        let dir = std::env::temp_dir().join(format!("evolve-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jsonl");
        let lines = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "task"}),
            calling("a", "x", json!({})),
            json!({"role": "tool", "tool_call_id": "a", "content": "ra"}),
            calling("b", "x", json!({})),
            json!({"role": "tool", "tool_call_id": "b", "content": "rb"}),
            json!({"compaction": {"head": "task + summary", "keep": 2}}),
            calling("c", "x", json!({})),
            json!({"role": "tool", "tool_call_id": "c", "content": "rc"}),
            // the run died here: a call with no result
            json!({"role": "assistant", "content": null, "tool_calls": [call("d", "x", json!({})), call("e", "x", json!({}))]}),
            json!({"role": "tool", "tool_call_id": "d", "content": "rd"}),
        ];
        std::fs::write(&path, lines.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("\n")).unwrap();
        let m = replay(&path).unwrap();
        let roles: Vec<&str> = m.iter().map(|x| x["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "assistant", "tool"]);
        assert_eq!(m[1]["content"], "task + summary");
        assert_eq!(m[2]["tool_calls"][0]["id"], "b");
        assert_well_formed(&m);
        std::fs::write(&path, "{\"compaction\": {\"head\": \"h\", \"keep\": 9}}\n").unwrap();
        assert!(replay(&path).is_err());
        assert!(replay(&dir.join("missing.jsonl")).is_err());
    }
}
