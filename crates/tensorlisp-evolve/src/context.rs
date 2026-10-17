//! Keeping a long agent run inside the model's context window.
//!
//! Three steps, cheapest first (the loop in `agent.rs` decides when):
//!
//! 1. *Shrink* the old part of the conversation: large tool outputs become a stub, large tool-call
//!    arguments (a whole `write_file` body, which is on disk anyway) and long assistant messages are cut.
//! 2. *Summarize* it: the old turns, plus the previous summary and the agent's notes, are condensed by a
//!    model call into one message that replaces them; the last few turns stay verbatim.
//! 3. On a context-length error from the endpoint, do both harder and retry.
//!
//! Cuts are always placed at the start of an assistant message, so a tool call and its results are
//! never separated (endpoints reject orphaned tool messages). The agent's notes (`notes` tool) are
//! re-injected into the summary message, so they survive every compaction.
use serde_json::{Value, json};

pub const STUB: &str = "[earlier output elided to save context; run the tool again if you need it]";

/// Token accounting. Estimates come from characters; every reply's reported `prompt_tokens`
/// recalibrates the characters-per-token ratio for this model and content.
#[derive(Debug, Clone)]
pub struct Budget {
    pub max_tokens: usize,
    /// Compact when the estimate passes this fraction of `max_tokens`.
    pub compact_at: f64,
    pub chars_per_token: f64,
    /// Characters sent with every request besides the messages (the tool definitions).
    pub overhead_chars: usize,
}

impl Budget {
    pub fn new(max_tokens: usize, compact_at: f64, overhead_chars: usize) -> Self {
        Budget { max_tokens, compact_at: compact_at.clamp(0.2, 0.95), chars_per_token: 3.5, overhead_chars }
    }

    pub fn request_chars(&self, messages: &[Value]) -> usize {
        messages.iter().map(message_chars).sum::<usize>() + self.overhead_chars
    }

    pub fn estimate(&self, messages: &[Value]) -> usize {
        (self.request_chars(messages) as f64 / self.chars_per_token) as usize
    }

    /// Learns from a real measurement: this request of `chars` characters cost `prompt_tokens`.
    pub fn observe(&mut self, request_chars: usize, prompt_tokens: u64) {
        if prompt_tokens >= 500 {
            let seen = (request_chars as f64 / prompt_tokens as f64).clamp(1.5, 8.0);
            self.chars_per_token = 0.5 * self.chars_per_token + 0.5 * seen;
        }
    }

    pub fn threshold(&self) -> usize {
        (self.max_tokens as f64 * self.compact_at) as usize
    }

    /// Shrinking alone is enough when it gets the estimate under this.
    pub fn settle(&self) -> usize {
        (self.max_tokens as f64 * (self.compact_at - 0.15).max(0.25)) as usize
    }

    pub fn overhead_tokens(&self) -> usize {
        (self.overhead_chars as f64 / self.chars_per_token) as usize
    }

    /// Tokens the messages may take once a compaction has settled (the window's settle point minus the
    /// tool definitions that ride along with every request).
    pub fn room(&self) -> usize {
        self.settle().saturating_sub(self.overhead_tokens())
    }

    /// Tokens of the messages alone.
    pub fn tokens_of(&self, messages: &[Value]) -> usize {
        (messages.iter().map(message_chars).sum::<usize>() as f64 / self.chars_per_token) as usize
    }

    pub fn needs_compaction(&self, messages: &[Value]) -> bool {
        self.estimate(messages) > self.threshold()
    }

    /// The endpoint told us its limit: stay under it from now on.
    pub fn learn_limit(&mut self, limit: usize) {
        self.max_tokens = self.max_tokens.min((limit as f64 * 0.9) as usize).max(2000);
    }
}

/// The window to budget against. `spec` is `auto` (what the endpoint reports, capped by `cap`: a 1M-token
/// window is real but quality and cost fall long before it is full) or a number of tokens, which is taken
/// as given. Returns the window and a line saying where it came from.
pub fn resolve_window(spec: &str, cap: usize, detected: Option<usize>) -> anyhow::Result<(usize, String)> {
    const FALLBACK: usize = 120_000;
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(match detected {
            Some(d) if d > cap => (cap, format!("the model's window is {d} tokens; using {cap} (--context-cap)")),
            Some(d) => (d, format!("the model's window is {d} tokens")),
            None => (FALLBACK.min(cap), format!("the endpoint does not report a window; assuming {}", FALLBACK.min(cap))),
        });
    }
    let n: usize = spec.replace(['_', ','], "").trim_end_matches(['k', 'K']).parse::<usize>().map(|n| if spec.ends_with(['k', 'K']) { n * 1000 } else { n }).map_err(|_| anyhow::anyhow!("--context-tokens is `auto` or a number of tokens (e.g. 200000 or 200k), not {spec:?}"))?;
    if n < 4000 {
        anyhow::bail!("a {n}-token window is smaller than the tool definitions alone");
    }
    Ok((n, format!("window {n} tokens (given)")))
}

/// `0.75`, `75` and `75%` all mean 75%.
pub fn parse_fraction(s: &str) -> Result<f64, String> {
    let t = s.trim().trim_end_matches('%').trim();
    let x: f64 = t.parse().map_err(|_| format!("expected a fraction or a percentage, not {s:?}"))?;
    let x = if x > 1.0 || s.trim().ends_with('%') { x / 100.0 } else { x };
    if (0.2..=0.95).contains(&x) { Ok(x) } else { Err(format!("compaction must start between 20% and 95% of the window, not {s:?}")) }
}

pub fn message_chars(m: &Value) -> usize {
    m["content"].as_str().map_or(0, |s| s.len()) + if m["tool_calls"].is_null() { 0 } else { m["tool_calls"].to_string().len() } + 16
}

/// The index where the last `keep_turns` assistant turns begin, or `None` when there are not more turns
/// than that to compact. Always the index of an assistant message (>= 2: the system prompt and the task
/// message are never part of the old section).
pub fn split_point(messages: &[Value], keep_turns: usize) -> Option<usize> {
    let assistants: Vec<usize> = messages.iter().enumerate().skip(2).filter(|(_, m)| m["role"] == "assistant").map(|(i, _)| i).collect();
    if assistants.len() <= keep_turns.max(1) {
        return None;
    }
    let cut = assistants[assistants.len() - keep_turns.max(1)];
    // nothing before it to compact except what is already a stub
    (cut > 2).then_some(cut)
}

/// Replaces tool outputs longer than a stub by the stub. Returns how many characters were saved.
pub fn elide_outputs(msgs: &mut [Value]) -> usize {
    let mut saved = 0;
    for m in msgs.iter_mut().filter(|m| m["role"] == "tool") {
        let len = m["content"].as_str().map_or(0, |s| s.len());
        if len > STUB.len() + 40 {
            saved += len - STUB.len();
            m["content"] = json!(STUB);
        }
    }
    saved
}

fn cut_str(s: &str, keep: usize) -> String {
    if s.chars().count() <= keep + 40 {
        return s.to_string();
    }
    format!("{}…[{} characters omitted]", s.chars().take(keep).collect::<String>(), s.chars().count() - keep)
}

/// Cuts big string arguments in old tool calls and long old assistant messages. Returns characters saved.
pub fn shrink_args(msgs: &mut [Value]) -> usize {
    let mut saved = 0;
    for m in msgs.iter_mut().filter(|m| m["role"] == "assistant") {
        if let Some(c) = m["content"].as_str()
            && c.chars().count() > 1500
        {
            let cut = cut_str(c, 1000);
            saved += c.len().saturating_sub(cut.len());
            m["content"] = json!(cut);
        }
        let Some(calls) = m["tool_calls"].as_array_mut() else { continue };
        for call in calls {
            let Some(raw) = call["function"]["arguments"].as_str() else { continue };
            if raw.len() < 600 {
                continue;
            }
            let Ok(mut args) = serde_json::from_str::<Value>(raw) else { continue };
            if let Some(obj) = args.as_object_mut() {
                for v in obj.values_mut() {
                    if let Some(s) = v.as_str() {
                        *v = json!(cut_str(s, 200));
                    }
                }
            }
            let new = args.to_string();
            saved += raw.len().saturating_sub(new.len());
            call["function"]["arguments"] = json!(new);
        }
    }
    saved
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n).collect::<String>()) }
}

/// The old turns as text for the summarizer, within `cap` characters (the oldest are dropped first).
pub fn render_for_summary(previous: Option<&str>, old: &[Value], cap: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for m in old {
        match m["role"].as_str().unwrap_or("") {
            "assistant" => {
                if let Some(c) = m["content"].as_str().filter(|c| !c.trim().is_empty()) {
                    lines.push(format!("ASSISTANT: {}", clip(c, 1500)));
                }
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    lines.push(format!("CALL {}({})", call["function"]["name"].as_str().unwrap_or("?"), clip(call["function"]["arguments"].as_str().unwrap_or(""), 700)));
                }
            }
            "tool" => {
                let c = m["content"].as_str().unwrap_or("");
                if c != STUB {
                    lines.push(format!("RESULT: {}", clip(c, 900)));
                } else {
                    lines.push("RESULT: (elided)".into());
                }
            }
            "user" => lines.push(format!("USER: {}", clip(m["content"].as_str().unwrap_or(""), 600))),
            _ => {}
        }
    }
    let mut total: usize = lines.iter().map(|l| l.len() + 1).sum();
    let mut dropped = 0;
    while total > cap && lines.len() > 1 {
        total -= lines[0].len() + 1;
        lines.remove(0);
        dropped += 1;
    }
    let mut out = String::new();
    if let Some(p) = previous {
        out.push_str(&format!("# Summary written at the previous compaction\n{p}\n\n"));
    }
    out.push_str("# Turns since then, oldest first\n");
    if dropped > 0 {
        out.push_str(&format!("[{dropped} older lines omitted for length]\n"));
    }
    out.push_str(&lines.join("\n"));
    out
}

pub const SUMMARIZER_PROMPT: &str = "You condense the working history of an agent that is searching for compiler optimization passes, so that it can continue after its older context is removed. The history you are given is formatted with the markers `CALL`, `RESULT:`, `ASSISTANT:` and `USER:`; your answer must NOT use or reproduce them. Write, in your own words, under exactly these headings:\n\
Goal: the goal and the approach it settled on.\n\
Tried: each hypothesis tried, with the result (cite evaluation ids, forum entry ids, file names and numbers exactly as they appeared) and why it failed or worked.\n\
Files: the files in its workspace and what each is for; the repository files it read and the line ranges that mattered.\n\
Facts: things learned about the kernels, the passes and the evaluator that it would otherwise have to rediscover. Only facts that appear in the history.\n\
Next: what it was doing last and the next steps, in order.\n\
Never copy tool output or file contents (refer to a file and its line numbers instead), and never state anything that is not in the history: if you are unsure, leave it out. Be terse. At most 600 words.";

/// Added when the first summary was unusable (it copied the history, or was empty).
pub const SUMMARIZER_RETRY: &str = "That was not a summary: it repeated the history or was empty. Write a summary in your own words under the headings Goal, Tried, Files, Facts, Next. Do not use the markers CALL, RESULT:, ASSISTANT: or USER:, do not copy any tool output or file contents, and do not include anything that is not in the history.";

/// Is this a usable summary? A model that echoes its input (or invents content) poisons every later turn, so
/// the harness checks what it can: the text must not contain the markers the history was rendered with, and
/// must be short enough to be a summary. `Err` says why not.
pub fn check_summary(text: &str) -> Result<(), &'static str> {
    let t = text.trim();
    if t.chars().count() < 40 {
        return Err("empty or too short");
    }
    if t.chars().count() > 6000 {
        return Err("too long to be a summary");
    }
    let echoed = t.lines().filter(|l| {
        let l = l.trim_start();
        l.starts_with("CALL ") || l.starts_with("RESULT:") || l.starts_with("ASSISTANT:") || l.starts_with("USER:") || l.starts_with("# Turns since then") || l.starts_with("# Summary written at the previous")
    });
    if echoed.count() > 0 {
        return Err("it repeats the history");
    }
    Ok(())
}

const COMPACTED: &str = "\n\n---\nYour earlier turns were compacted to fit the context window. What you had done so far:\n\n";
const NOTES_MARK: &str = "\nYour notes (kept across compaction; edit them with the `notes` tool):\n";
const CONTINUE_MARK: &str = "\nContinue from where you stopped. The most recent turns follow verbatim.";

/// The task message after a compaction: the original task, the summary, and the agent's notes.
pub fn head_message(task: &str, summary: &str, notes: &str) -> String {
    let mut out = format!("{task}{COMPACTED}{summary}{FALLIBLE}\n");
    if !notes.trim().is_empty() {
        out.push_str(&format!("{NOTES_MARK}{notes}\n"));
    }
    out.push_str(CONTINUE_MARK);
    out
}

/// Shown right after a summary so the agent treats it as a lead, not as ground truth.
pub const FALLIBLE: &str = "\n(This summary was written by a model and can be wrong: check specific names, ids and numbers with the tools before relying on them.)";

/// The original task out of a task message, compacted or not.
pub fn task_of(head: &str) -> &str {
    head.split(COMPACTED).next().unwrap_or(head)
}

/// The summary out of a compacted task message (so the next compaction can build on it).
pub fn summary_of(head: &str) -> Option<&str> {
    let rest = head.split_once(COMPACTED)?.1;
    let end = rest.find(FALLIBLE).or_else(|| rest.find(NOTES_MARK)).or_else(|| rest.rfind(CONTINUE_MARK)).unwrap_or(rest.len());
    Some(rest[..end].trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, name: &str, args: &str) -> Value {
        json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}})
    }

    /// system, task, then `turns` rounds of (assistant with one tool call, tool result).
    pub(crate) fn conversation(turns: usize, result_len: usize) -> Vec<Value> {
        let mut m = vec![json!({"role": "system", "content": "sys"}), json!({"role": "user", "content": "task"})];
        for i in 0..turns {
            m.push(json!({"role": "assistant", "content": format!("thinking {i}"), "tool_calls": [call(&format!("c{i}"), "read_file", &json!({"path": format!("f{i}")}).to_string())]}));
            m.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": "x".repeat(result_len)}));
        }
        m
    }

    #[test]
    fn estimates_calibrate_to_the_reported_prompt_tokens() {
        let msgs = conversation(10, 3000);
        let mut b = Budget::new(10_000, 0.75, 1000);
        let chars = b.request_chars(&msgs);
        assert!(chars > 30_000);
        let before = b.estimate(&msgs);
        // the model reports twice as many tokens as we guessed: the ratio moves toward chars/tokens
        b.observe(chars, (before * 2) as u64);
        assert!(b.estimate(&msgs) > before && b.estimate(&msgs) < before * 2);
        for _ in 0..12 {
            b.observe(chars, (before * 2) as u64);
        }
        assert!((b.estimate(&msgs) as f64 / (before * 2) as f64 - 1.0).abs() < 0.02);
        // tiny measurements are ignored
        let cpt = b.chars_per_token;
        b.observe(100, 10);
        assert_eq!(b.chars_per_token, cpt);
        assert!(b.needs_compaction(&msgs) && !b.needs_compaction(&msgs[..2]));
        assert_eq!(b.room(), b.settle().saturating_sub(b.overhead_tokens()));
        assert!(b.tokens_of(&msgs) < b.estimate(&msgs));
        let mut b = Budget::new(100_000, 0.8, 0);
        b.learn_limit(32_000);
        assert_eq!(b.max_tokens, 28_800);
        b.learn_limit(1);
        assert_eq!(b.max_tokens, 2000);
    }

    #[test]
    fn summaries_that_copy_the_history_are_rejected() {
        let good = "Goal: find a pass that beats the baseline.\nTried: licm hoist (e-1) gave no gain.\nFiles: work/a.ss.\nNext: try cse on matvec.";
        assert!(check_summary(good).is_ok());
        assert_eq!(check_summary("   "), Err("empty or too short"));
        assert_eq!(check_summary("ok"), Err("empty or too short"));
        assert_eq!(check_summary(&"x".repeat(7000)), Err("too long to be a summary"));
        // what the free model did: echoed the rendered input
        let echo = "CALL read_file({\"path\":\"x\"})\nRESULT:     1\t;; (tl tree opt)\nThe file shows there is a pass.";
        assert_eq!(check_summary(echo), Err("it repeats the history"));
        assert!(check_summary("Tried: read tree_opt.ss.\n  RESULT: nothing, the file is long enough to count as text here.").is_err());
        // the markers are only suspicious at the start of a line
        assert!(check_summary("Tried: a CALL to read_file returned RESULT: 3 lines, and that matched expectations well.").is_ok());
    }

    #[test]
    fn the_window_is_detected_capped_or_given() {
        assert_eq!(resolve_window("auto", 200_000, Some(1_048_576)).unwrap().0, 200_000);
        assert_eq!(resolve_window("auto", 200_000, Some(131_072)).unwrap().0, 131_072);
        assert_eq!(resolve_window("AUTO", 100_000, None).unwrap().0, 100_000);
        assert_eq!(resolve_window("auto", 500_000, None).unwrap().0, 120_000);
        assert!(resolve_window("auto", 200_000, Some(1_048_576)).unwrap().1.contains("--context-cap"));
        // an explicit number is not capped
        assert_eq!(resolve_window("500000", 200_000, Some(1_000_000)).unwrap().0, 500_000);
        assert_eq!(resolve_window("64k", 1, None).unwrap().0, 64_000);
        assert_eq!(resolve_window("32_768", 1, None).unwrap().0, 32_768);
        assert!(resolve_window("lots", 1, None).is_err() && resolve_window("100", 1, None).is_err());
        for (s, want) in [("0.75", 0.75), ("75", 0.75), ("75%", 0.75), ("60 %", 0.6), ("0.5", 0.5), ("1%", f64::NAN)] {
            match parse_fraction(s) {
                Ok(x) => assert!((x - want).abs() < 1e-9, "{s}"),
                Err(_) => assert!(want.is_nan(), "{s}"),
            }
        }
        assert!(parse_fraction("99%").is_err() && parse_fraction("x").is_err());
    }

    #[test]
    fn the_split_point_never_separates_a_call_from_its_results() {
        let msgs = conversation(10, 100);
        assert_eq!(split_point(&msgs, 10), None);
        assert_eq!(split_point(&msgs, 20), None);
        for keep in 1..10 {
            let cut = split_point(&msgs, keep).unwrap();
            assert_eq!(msgs[cut]["role"], "assistant");
            assert_eq!(msgs[cut..].iter().filter(|m| m["role"] == "assistant").count(), keep);
            // every tool message after the cut answers a call after the cut
            let ids: Vec<&str> = msgs[cut..].iter().filter_map(|m| m["tool_calls"][0]["id"].as_str()).collect();
            assert!(msgs[cut..].iter().filter(|m| m["role"] == "tool").all(|m| ids.contains(&m["tool_call_id"].as_str().unwrap())));
        }
        assert_eq!(split_point(&msgs, 0), split_point(&msgs, 1));
        // nothing but the task before the first assistant: nothing to compact
        assert_eq!(split_point(&conversation(1, 10), 1), None);
    }

    #[test]
    fn shrinking_keeps_structure_and_removes_the_bulk() {
        let mut msgs = conversation(3, 5000);
        let big = json!({"path": "work/x.ss", "content": "(define x 1)\n".repeat(300)}).to_string();
        msgs[2]["tool_calls"][0]["function"]["arguments"] = json!(big);
        msgs[2]["content"] = json!("reasoning ".repeat(400));
        let before: usize = msgs.iter().map(message_chars).sum();
        let saved = elide_outputs(&mut msgs) + shrink_args(&mut msgs);
        let after: usize = msgs.iter().map(message_chars).sum();
        assert!(saved > 15_000 && before - after >= saved - 100, "{saved} {before} {after}");
        assert_eq!(msgs[3]["content"], STUB);
        assert_eq!(msgs[3]["tool_call_id"], "c0");
        // the arguments are still valid JSON, the path survives, the body is cut
        let args: Value = serde_json::from_str(msgs[2]["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["path"], "work/x.ss");
        assert!(args["content"].as_str().unwrap().contains("characters omitted"));
        assert!(msgs[2]["content"].as_str().unwrap().contains("characters omitted"));
        // idempotent
        assert_eq!(elide_outputs(&mut msgs) + shrink_args(&mut msgs), 0);
    }

    #[test]
    fn the_summarizer_input_is_bounded_and_carries_the_previous_summary() {
        let msgs = conversation(40, 2000);
        let text = render_for_summary(Some("earlier: tried licm, no gain (e-1)"), &msgs[2..], 6000);
        assert!(text.len() < 6600, "{}", text.len());
        assert!(text.contains("tried licm") && text.contains("older lines omitted"));
        // the newest turns survive the cut
        assert!(text.contains("thinking 39") && !text.contains("thinking 0\n"));
        let head = head_message("find passes", "SUMMARY", "- keep x");
        assert!(head.starts_with("find passes") && head.contains("SUMMARY") && head.contains("- keep x"));
        assert!(!head_message("t", "s", "  ").contains("Your notes"));
        assert_eq!(task_of(&head), "find passes");
        assert_eq!(task_of("plain task"), "plain task");
        assert_eq!(summary_of(&head), Some("SUMMARY"));
        assert!(head.contains("can be wrong"));
        assert_eq!(summary_of(&head_message("t", "multi\nline", "")), Some("multi\nline"));
        assert_eq!(summary_of("plain task"), None);
    }
}
