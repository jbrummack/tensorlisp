//! The agent loop: chat with tools until the model calls `finish`, runs out of turns, or stops asking for tools.
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use serde_json::{Value, json};

use crate::llm::{Llm, Reply, ToolCall};
use crate::tools::{self, ToolCtx};

pub const SYSTEM_PROMPT: &str = include_str!("prompt.md");

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
    /// JSON lines of every message, appended as the run goes.
    pub transcript: Option<PathBuf>,
    /// Old tool outputs are replaced by a stub once the conversation exceeds this many characters.
    pub max_context_chars: usize,
    pub verbose: bool,
}

#[derive(Debug, Default)]
pub struct AgentResult {
    pub summary: Option<String>,
    pub turns: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub tool_calls: usize,
}

pub fn system_prompt(target: &str, device: &str, author: &str, workspace: &str) -> String {
    SYSTEM_PROMPT.replace("{target}", target).replace("{device}", device).replace("{author}", author).replace("{workspace}", workspace)
}

fn message_chars(m: &Value) -> usize {
    m["content"].as_str().map_or(0, |s| s.len()) + m["tool_calls"].to_string().len()
}

/// Replaces the oldest large tool outputs with a stub until the conversation fits.
pub fn trim_context(messages: &mut [Value], max_chars: usize) {
    let mut total: usize = messages.iter().map(message_chars).sum();
    for m in messages.iter_mut().skip(2) {
        if total <= max_chars {
            break;
        }
        if m["role"] == "tool" {
            let len = m["content"].as_str().map_or(0, |s| s.len());
            if len > 300 {
                m["content"] = json!("[earlier output elided to save context; run the tool again if you need it]");
                total = total.saturating_sub(len);
            }
        }
    }
}

fn short(v: &Value, n: usize) -> String {
    let s = v.to_string();
    if s.chars().count() > n { format!("{}…", s.chars().take(n).collect::<String>()) } else { s }
}

pub fn run(chat: &dyn Chat, ctx: &ToolCtx, cfg: &AgentConfig) -> Result<AgentResult> {
    let specs = tools::specs();
    let mut messages = vec![json!({"role": "system", "content": cfg.system}), json!({"role": "user", "content": cfg.task})];
    let mut transcript = match &cfg.transcript {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
        }
        None => None,
    };
    let mut log = |m: &Value| {
        if let Some(f) = transcript.as_mut() {
            let _ = writeln!(f, "{m}");
        }
    };
    for m in &messages {
        log(m);
    }
    let mut result = AgentResult::default();
    let mut idle = 0;
    for turn in 1..=cfg.max_turns {
        result.turns = turn;
        trim_context(&mut messages, cfg.max_context_chars);
        let reply = chat.chat(&messages, &specs)?;
        result.prompt_tokens += reply.prompt_tokens;
        result.completion_tokens += reply.completion_tokens;
        // keep only the fields the API wants back
        let mut assistant = json!({"role": "assistant", "content": reply.message["content"].clone()});
        if !reply.tool_calls.is_empty() {
            assistant["tool_calls"] = reply.message["tool_calls"].clone();
        }
        messages.push(assistant.clone());
        log(&assistant);
        if cfg.verbose && !reply.content.trim().is_empty() {
            eprintln!("[{turn}] {}", reply.content.trim().lines().next().unwrap_or(""));
        }
        if reply.tool_calls.is_empty() {
            idle += 1;
            if idle >= 3 {
                result.summary = Some(reply.content.clone()).filter(|s| !s.trim().is_empty());
                break;
            }
            let nudge = json!({"role": "user", "content": "Continue with tool calls, or call `finish` with your summary if you are done."});
            log(&nudge);
            messages.push(nudge);
            continue;
        }
        idle = 0;
        for ToolCall { id, name, arguments } in &reply.tool_calls {
            result.tool_calls += 1;
            let out = ctx.call(name, arguments);
            if cfg.verbose {
                eprintln!("[{turn}] {name} {} -> {}", short(arguments, 100), out.lines().next().unwrap_or("").chars().take(100).collect::<String>());
            }
            let msg = json!({"role": "tool", "tool_call_id": id, "content": out});
            log(&msg);
            messages.push(msg);
        }
        if let Some(summary) = ctx.finished() {
            result.summary = Some(summary);
            break;
        }
    }
    Ok(result)
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

    struct Script {
        replies: RefCell<VecDeque<Value>>,
        seen: RefCell<Vec<Vec<Value>>>,
    }

    impl Chat for Script {
        fn chat(&self, messages: &[Value], _tools: &[Value]) -> Result<Reply> {
            self.seen.borrow_mut().push(messages.to_vec());
            let next = self.replies.borrow_mut().pop_front().expect("script ran out");
            parse_reply(&json!({"choices": [{"message": next}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).to_string())
        }
    }

    fn call(id: &str, name: &str, args: Value) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}})
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
        AgentConfig { max_turns: turns, system: "sys".into(), task: "go".into(), transcript: None, max_context_chars: 1_000_000, verbose: false }
    }

    #[test]
    fn a_scripted_agent_searches_evaluates_posts_and_finishes() {
        let c = ctx("script");
        let script = Script {
            replies: RefCell::new(VecDeque::from(vec![
                json!({"role": "assistant", "content": "look around", "tool_calls": [call("1", "forum_search", json!({"query": "cse"})), call("2", "evaluate", json!({"code": "", "pipeline": ["fold", "dce"]}))]}),
                json!({"role": "assistant", "content": null, "tool_calls": [call("3", "forum_post", json!({"kind": "pass", "headline": "fold+dce only", "body": "licm and cse do nothing here", "eval_id": "last"}))]}),
                json!({"role": "assistant", "content": null, "tool_calls": [call("4", "finish", json!({"summary": "baseline is already tight"}))]}),
            ])),
            seen: RefCell::new(Vec::new()),
        };
        let r = run(&script, &c, &cfg(10)).unwrap();
        assert_eq!(r.summary.as_deref(), Some("baseline is already tight"));
        assert_eq!((r.turns, r.tool_calls, r.prompt_tokens), (3, 4, 30));
        // the model saw the tool results in order
        let last = script.seen.borrow().last().unwrap().clone();
        let tool_msgs: Vec<&Value> = last.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_msgs.len(), 3);
        assert!(tool_msgs[0]["content"].as_str().unwrap().starts_with("no results"));
        assert!(tool_msgs[1]["content"].as_str().unwrap().contains("evaluation"));
        assert!(tool_msgs[2]["content"].as_str().unwrap().starts_with("posted e-"));
        assert_eq!(c.forum.entries(None, 10).unwrap().len(), 1);
    }

    #[test]
    fn an_agent_that_stops_calling_tools_is_nudged_then_ended() {
        let c = ctx("idle");
        let say = |t: &str| json!({"role": "assistant", "content": t});
        let script = Script { replies: RefCell::new(VecDeque::from(vec![say("hmm"), say("hmm"), say("I am done")])), seen: RefCell::new(Vec::new()) };
        let r = run(&script, &c, &cfg(10)).unwrap();
        assert_eq!(r.turns, 3);
        assert_eq!(r.summary.as_deref(), Some("I am done"));
        let turns = script.seen.borrow();
        assert!(turns[1].last().unwrap()["content"].as_str().unwrap().contains("finish"));
    }

    #[test]
    fn turns_are_capped_and_errors_go_back_to_the_model() {
        let c = ctx("cap");
        let bad = json!({"role": "assistant", "content": null, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "read_file", "arguments": "{nope"}}]});
        let script = Script { replies: RefCell::new(VecDeque::from(vec![bad.clone(), bad])), seen: RefCell::new(Vec::new()) };
        let r = run(&script, &c, &cfg(2)).unwrap();
        assert_eq!((r.turns, r.summary), (2, None));
        let seen = script.seen.borrow();
        assert!(seen[1].last().unwrap()["content"].as_str().unwrap().contains("not valid JSON"));
    }

    #[test]
    fn old_tool_output_is_elided_when_the_context_grows() {
        let big = "x".repeat(5000);
        let mut msgs = vec![json!({"role": "system", "content": "s"}), json!({"role": "user", "content": "u"}), json!({"role": "tool", "tool_call_id": "1", "content": big.clone()}), json!({"role": "tool", "tool_call_id": "2", "content": big})];
        trim_context(&mut msgs, 6000);
        assert!(msgs[2]["content"].as_str().unwrap().contains("elided"));
        assert_eq!(msgs[3]["content"].as_str().unwrap().len(), 5000);
    }

    #[test]
    fn the_prompt_has_its_placeholders_filled() {
        let p = system_prompt("ptx", "rtx3080", "agent7", "evolve/work/agent7");
        assert!(p.contains("Target: `ptx`") && p.contains("evolve/work/agent7") && !p.contains("{target}") && !p.contains("{author}"));
    }
}
