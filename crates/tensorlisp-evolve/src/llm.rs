//! A chat client for any OpenAI-compatible endpoint (`/chat/completions` with tools): OpenRouter,
//! a local server, or OpenAI itself. Configured by environment:
//!
//! * `TL_EVOLVE_BASE_URL` (default `https://openrouter.ai/api/v1`; `OPENAI_BASE_URL` also works)
//! * `TL_EVOLVE_API_KEY` (`OPENROUTER_KEY`, `OPENROUTER_API_KEY` or `OPENAI_API_KEY`)
//! * `TL_EVOLVE_MODEL`, e.g. `anthropic/claude-sonnet-4.5` or `thinkingmachines/inkling:free`
//!
//! Each is read from the process environment, else from the repository's `.env` (see `config.rs`).
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// The endpoint refused the request because the conversation is longer than the model's context window.
/// `limit` is the window in tokens when the message says it.
#[derive(Debug, Clone)]
pub struct ContextOverflow {
    pub message: String,
    pub limit: Option<usize>,
}

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "context window exceeded{}: {}", self.limit.map(|l| format!(" (limit {l} tokens)")).unwrap_or_default(), self.message)
    }
}

impl std::error::Error for ContextOverflow {}

/// Recognizes the context-length errors of OpenAI-style, OpenRouter and Anthropic-style endpoints.
pub fn context_overflow(status: u16, body: &str) -> Option<ContextOverflow> {
    if !matches!(status, 400 | 413 | 422) {
        return None;
    }
    let lower = body.to_lowercase();
    const MARKERS: &[&str] = &["context length", "context_length", "maximum context", "context window", "prompt is too long", "too many tokens", "input is too long", "reduce the length", "exceeds the limit of", "request too large", "token limit"];
    if !MARKERS.iter().any(|m| lower.contains(m)) {
        return None;
    }
    let limit = [r"maximum context length is (\d+)", r"> (\d+) maximum", r"context window of (\d+)", r"limit of (\d+) tokens", r"(\d+) token limit", r"max(?:imum)? (?:of )?(\d+) tokens"]
        .iter()
        .find_map(|p| regex::Regex::new(p).ok()?.captures(&lower)?.get(1)?.as_str().parse::<usize>().ok());
    Some(ContextOverflow { message: body.chars().take(300).collect(), limit })
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed arguments; a string when the model produced invalid JSON (the tool reports it back).
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// The assistant message exactly as returned, to append to the conversation.
    pub message: Value,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

pub struct Llm {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    /// App attribution for OpenRouter (`X-Title`, `HTTP-Referer`): it identifies *this* program, as the
    /// service documents; it does not claim to be any other app.
    pub headers: Vec<(String, String)>,
    agent: ureq::Agent,
}

fn first_env(names: &[&str]) -> Option<String> {
    crate::config::first_var(names)
}

/// What a `/models` listing says about a model (OpenRouter's schema; other servers may not have the endpoint).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: Option<usize>,
    pub free: bool,
    pub tools: bool,
}

pub fn parse_models(text: &str) -> Result<Vec<ModelInfo>> {
    let v: Value = serde_json::from_str(text).context("the models listing is not JSON")?;
    let data = v["data"].as_array().context("the models listing has no `data`")?;
    Ok(data
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?.to_string();
            let ctx = [m["context_length"].as_u64(), m["top_provider"]["context_length"].as_u64()].into_iter().flatten().min().map(|c| c as usize);
            let zero = |p: &Value| p.as_str().is_some_and(|s| s.parse::<f64>().is_ok_and(|x| x == 0.0));
            let free = id.ends_with(":free") || (zero(&m["pricing"]["prompt"]) && zero(&m["pricing"]["completion"]));
            let tools = m["supported_parameters"].as_array().is_some_and(|p| p.iter().any(|x| x == "tools"));
            Some(ModelInfo { id, context_length: ctx, free, tools })
        })
        .collect())
}

impl Llm {
    pub fn new(base_url: &str, api_key: Option<String>, model: &str) -> Self {
        let agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_connect(Some(Duration::from_secs(30))).timeout_global(Some(Duration::from_secs(600))).build().into();
        let base_url: String = base_url.trim_end_matches('/').into();
        let mut headers = Vec::new();
        if base_url.contains("openrouter.ai") {
            let title = first_env(&["TL_EVOLVE_TITLE"]).unwrap_or_else(|| "tl-evolve".into());
            headers.push(("X-Title".to_string(), title.clone()));
            headers.push(("X-OpenRouter-Title".to_string(), title));
            if let Some(r) = first_env(&["TL_EVOLVE_REFERER"]) {
                headers.push(("HTTP-Referer".to_string(), r));
            }
        }
        Llm { base_url, api_key, model: model.into(), temperature: None, max_tokens: None, headers, agent }
    }

    pub fn from_env(model: Option<&str>) -> Result<Self> {
        let base = first_env(&["TL_EVOLVE_BASE_URL", "OPENAI_BASE_URL"]).unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
        let key = first_env(&["TL_EVOLVE_API_KEY", "OPENROUTER_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]);
        let model = model.map(String::from).or_else(|| first_env(&["TL_EVOLVE_MODEL"])).context("no model: pass --model or set TL_EVOLVE_MODEL")?;
        Ok(Self::new(&base, key, &model))
    }

    /// The endpoint's model listing (`GET /models`).
    pub fn models(&self) -> Result<Vec<ModelInfo>> {
        let mut req = self.agent.get(format!("{}/models", self.base_url));
        if let Some(k) = &self.api_key {
            req = req.header("authorization", format!("Bearer {k}"));
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let mut resp = req.call().context("models request")?;
        let status = resp.status();
        let body = resp.body_mut().read_to_string()?;
        if !status.is_success() {
            bail!("models endpoint returned {status}");
        }
        parse_models(&body)
    }

    /// This model's context window in tokens, if the endpoint says (OpenRouter does; a local server may not).
    pub fn context_window(&self) -> Option<usize> {
        self.models().ok()?.into_iter().find(|m| m.id == self.model)?.context_length
    }

    pub fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Reply> {
        let mut body = json!({"model": self.model, "messages": messages});
        if !tools.is_empty() {
            // no `tool_choice`: "auto" is the default with tools, and some endpoints (e.g. inkling) list it as
            // unsupported, which can make OpenRouter route the request nowhere
            body["tools"] = json!(tools);
        }
        if let Some(t) = self.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(m) = self.max_tokens {
            body["max_tokens"] = json!(m);
        }
        let body = body.to_string();
        let mut delay = Duration::from_secs(2);
        for attempt in 0..6 {
            let mut req = self.agent.post(format!("{}/chat/completions", self.base_url)).header("content-type", "application/json");
            if let Some(k) = &self.api_key {
                req = req.header("authorization", format!("Bearer {k}"));
            }
            for (k, v) in &self.headers {
                req = req.header(k.as_str(), v.as_str());
            }
            let sent = req.send(body.clone());
            let (status, text) = match sent {
                Ok(mut r) => (r.status().as_u16(), r.body_mut().read_to_string().unwrap_or_default()),
                Err(e) if attempt < 5 => {
                    eprintln!("llm: {e}; retrying");
                    std::thread::sleep(delay);
                    delay *= 2;
                    continue;
                }
                Err(e) => return Err(e).context("chat request"),
            };
            // some free tiers answer 200 with nothing: that is a transient failure, not a reply
            if status < 300 && text.trim().is_empty() && attempt < 5 {
                eprintln!("llm: empty response; retrying in {delay:?}");
                std::thread::sleep(delay);
                delay *= 2;
                continue;
            }
            // OpenRouter reports upstream trouble as an error object inside a 200: retry the transient ones
            if status < 300 && attempt < 5 && transient_error_body(&text) {
                eprintln!("llm: upstream error in the response body; retrying in {delay:?}");
                std::thread::sleep(delay);
                delay *= 2;
                continue;
            }
            if (status == 429 || status >= 500) && attempt < 5 {
                eprintln!("llm: HTTP {status}; retrying in {delay:?}");
                std::thread::sleep(delay);
                delay *= 2;
                continue;
            }
            if let Some(overflow) = context_overflow(status, &text) {
                return Err(anyhow::Error::new(overflow));
            }
            if status >= 300 {
                bail!("chat endpoint returned {status}: {}", text.chars().take(500).collect::<String>());
            }
            return parse_reply(&text);
        }
        bail!("chat endpoint kept failing")
    }
}

/// An `{"error": {...}}` body that says the service is busy, not that the request is wrong.
pub fn transient_error_body(text: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return false };
    let Some(e) = v.get("error") else { return false };
    let code = e["code"].as_u64().or_else(|| e["code"].as_str().and_then(|c| c.parse().ok())).unwrap_or(0);
    let kind = format!("{} {}", e["metadata"]["error_type"].as_str().unwrap_or(""), e["message"].as_str().unwrap_or("")).to_lowercase();
    code == 429 || (500..600).contains(&code) || kind.contains("overloaded") || kind.contains("rate limit") || kind.contains("temporarily")
}

pub fn parse_reply(text: &str) -> Result<Reply> {
    let v: Value = serde_json::from_str(text).with_context(|| format!("not JSON: {}", text.chars().take(200).collect::<String>()))?;
    if let Some(err) = v.get("error") {
        bail!("endpoint error: {err}");
    }
    let message = v["choices"][0]["message"].clone();
    if message.is_null() {
        bail!("no message in response: {}", text.chars().take(300).collect::<String>());
    }
    let content = message["content"].as_str().unwrap_or("").to_string();
    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let raw = &c["function"]["arguments"];
                    let arguments = match raw {
                        Value::String(s) if s.trim().is_empty() => json!({}),
                        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
                        Value::Null => json!({}),
                        other => other.clone(),
                    };
                    ToolCall { id: c["id"].as_str().map(String::from).unwrap_or_else(|| format!("call_{i}")), name: c["function"]["name"].as_str().unwrap_or("").to_string(), arguments }
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Reply { content, tool_calls, message, prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0), completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_calls_with_string_and_broken_arguments() {
        let r = parse_reply(
            r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[
                {"id":"a","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"x\"}"}},
                {"id":"b","type":"function","function":{"name":"ls","arguments":"{oops"}},
                {"id":"c","type":"function","function":{"name":"finish","arguments":""}}]}}],
                "usage":{"prompt_tokens":5,"completion_tokens":7}}"#,
        )
        .unwrap();
        assert_eq!(r.tool_calls.len(), 3);
        assert_eq!(r.tool_calls[0].arguments["pattern"], "x");
        assert!(r.tool_calls[1].arguments.is_string());
        assert_eq!(r.tool_calls[2].arguments, json!({}));
        assert_eq!((r.prompt_tokens, r.completion_tokens), (5, 7));
        assert!(parse_reply(r#"{"error":{"message":"no"}}"#).is_err());
    }

    /// A one-connection-at-a-time HTTP server answering with the given (status, body) in turn.
    fn serve(responses: Vec<(u16, String)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (mut head_end, mut want) = (None, 0usize);
                loop {
                    let n = sock.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if head_end.is_none()
                        && let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n")
                    {
                        head_end = Some(i + 4);
                        let head = String::from_utf8_lossy(&buf[..i]).to_lowercase();
                        want = head.lines().find_map(|l| l.strip_prefix("content-length:")).map(|v| v.trim().parse().unwrap()).unwrap_or(0);
                    }
                    if let Some(h) = head_end
                        && buf.len() >= h + want
                    {
                        break;
                    }
                    if n == 0 {
                        break;
                    }
                }
                requests.push(String::from_utf8_lossy(&buf).into_owned());
                let reply = format!("HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
                sock.write_all(reply.as_bytes()).unwrap();
            }
            requests
        });
        (url, handle)
    }

    #[test]
    fn talks_to_an_openai_compatible_endpoint_with_tools_and_retries_on_429() {
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"hi","tool_calls":[{"id":"t1","type":"function","function":{"name":"glob","arguments":"{\"pattern\":\"*.ss\"}"}}]}}],"usage":{"prompt_tokens":3,"completion_tokens":4}}"#;
        let (url, server) = serve(vec![(429, "{\"error\":\"slow down\"}".into()), (200, ok.into())]);
        let llm = Llm::new(&url, Some("sk-test".into()), "vendor/some-model");
        let tools = vec![json!({"type": "function", "function": {"name": "glob", "description": "d", "parameters": {"type": "object", "properties": {}}}})];
        let reply = llm.chat(&[json!({"role": "user", "content": "hello"})], &tools).unwrap();
        assert_eq!(reply.content, "hi");
        assert_eq!(reply.tool_calls[0].name, "glob");
        assert_eq!(reply.tool_calls[0].arguments["pattern"], "*.ss");
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), 2);
        let r = requests[1].to_lowercase();
        assert!(r.starts_with("post /v1/chat/completions"), "{r}");
        assert!(r.contains("authorization: bearer sk-test"));
        let body: Value = serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["model"], "vendor/some-model");
        assert_eq!(body["tools"][0]["function"]["name"], "glob");
        assert!(body.get("tool_choice").is_none(), "tool_choice is left to the default");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn model_listings_give_context_windows_and_free_tiers() {
        let list = r#"{"data":[
            {"id":"thinkingmachines/inkling:free","context_length":1048576,"top_provider":{"context_length":1000000},"pricing":{"prompt":"0","completion":"0"},"supported_parameters":["tools","temperature"]},
            {"id":"vendor/paid","context_length":32000,"pricing":{"prompt":"0.000001","completion":"0.000002"},"supported_parameters":["temperature"]},
            {"id":"vendor/zero-priced","context_length":8000,"pricing":{"prompt":"0.0","completion":"0.0"}},
            {"nonsense": true}]}"#;
        let m = parse_models(list).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!((m[0].context_length, m[0].free, m[0].tools), (Some(1_000_000), true, true));
        assert_eq!((m[1].context_length, m[1].free, m[1].tools), (Some(32_000), false, false));
        assert!(m[2].free);
        assert!(parse_models("[]").is_err());
        let (url, server) = serve(vec![(200, list.into())]);
        assert_eq!(Llm::new(&url, Some("k".into()), "vendor/paid").context_window(), Some(32_000));
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("GET /v1/models") && requests[0].to_lowercase().contains("authorization: bearer k"));
        // a server without the endpoint: no window, no error
        let (url, server) = serve(vec![(404, "{}".into())]);
        assert_eq!(Llm::new(&url, None, "vendor/paid").context_window(), None);
        server.join().unwrap();
    }

    #[test]
    fn context_length_errors_are_recognized_with_their_limit() {
        let openai = r#"{"error":{"message":"This model's maximum context length is 32768 tokens. However, your messages resulted in 40000 tokens.","code":"context_length_exceeded"}}"#;
        let o = context_overflow(400, openai).unwrap();
        assert_eq!(o.limit, Some(32768));
        let anthropic = r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#;
        assert_eq!(context_overflow(400, anthropic).unwrap().limit, Some(200000));
        let router = r#"{"error":{"message":"This endpoint's maximum context length is 131072 tokens. However, you requested about 140000 tokens"}}"#;
        assert_eq!(context_overflow(400, router).unwrap().limit, Some(131072));
        assert!(context_overflow(400, r#"{"error":"Input is too long for requested model"}"#).unwrap().limit.is_none());
        // other client errors, and server errors, are not overflows
        assert!(context_overflow(401, openai).is_none());
        assert!(context_overflow(400, r#"{"error":{"message":"invalid tool schema"}}"#).is_none());
        assert!(context_overflow(500, openai).is_none());
    }

    #[test]
    fn the_client_surfaces_an_overflow_as_a_typed_error() {
        let (url, server) = serve(vec![(400, r#"{"error":{"message":"maximum context length is 8000 tokens"}}"#.into())]);
        let err = Llm::new(&url, None, "m").chat(&[], &[]).unwrap_err();
        let o = err.downcast_ref::<ContextOverflow>().expect("typed");
        assert_eq!(o.limit, Some(8000));
        server.join().unwrap();
    }

    #[test]
    fn upstream_errors_inside_a_200_are_retried_but_request_errors_are_not() {
        let busy = r#"{"error":{"code":503,"message":"Upstream error from Nvidia: Service temporarily overloaded","metadata":{"error_type":"provider_overloaded"}}}"#;
        assert!(transient_error_body(busy));
        assert!(transient_error_body(r#"{"error":{"code":"429","message":"x"}}"#));
        assert!(!transient_error_body(r#"{"error":{"code":400,"message":"invalid tool schema"}}"#));
        assert!(!transient_error_body(r#"{"choices":[]}"#) && !transient_error_body("not json"));
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"back"}}]}"#;
        let (url, server) = serve(vec![(200, busy.into()), (200, ok.into())]);
        assert_eq!(Llm::new(&url, None, "m").chat(&[], &[]).unwrap().content, "back");
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn an_empty_200_is_retried() {
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"hello"}}]}"#;
        let (url, server) = serve(vec![(200, String::new()), (200, ok.into())]);
        assert_eq!(Llm::new(&url, None, "m").chat(&[], &[]).unwrap().content, "hello");
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn openrouter_gets_our_own_app_title_and_other_servers_get_nothing_extra() {
        let or = Llm::new("https://openrouter.ai/api/v1", None, "m");
        assert!(or.headers.iter().any(|(k, v)| k == "X-Title" && v == "tl-evolve"));
        assert!(Llm::new("http://localhost:8080/v1", None, "m").headers.is_empty());
        let (url, server) = serve(vec![(200, r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#.into())]);
        let mut llm = Llm::new(&url, None, "m");
        llm.headers.push(("X-Title".into(), "tl-evolve".into()));
        llm.chat(&[], &[]).unwrap();
        assert!(server.join().unwrap()[0].to_lowercase().contains("x-title: tl-evolve"));
    }

    #[test]
    fn a_client_error_is_reported_with_the_body() {
        let (url, server) = serve(vec![(401, "{\"error\":{\"message\":\"bad key\"}}".into())]);
        let err = Llm::new(&url, None, "m").chat(&[], &[]).unwrap_err().to_string();
        assert!(err.contains("401") && err.contains("bad key"), "{err}");
        server.join().unwrap();
    }
}
