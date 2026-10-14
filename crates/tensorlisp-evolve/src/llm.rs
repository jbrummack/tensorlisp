//! A chat client for any OpenAI-compatible endpoint (`/chat/completions` with tools): OpenRouter,
//! a local server, or OpenAI itself. Configured by environment:
//!
//! * `TL_EVOLVE_BASE_URL` (default `https://openrouter.ai/api/v1`; `OPENAI_BASE_URL` also works)
//! * `TL_EVOLVE_API_KEY` (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`)
//! * `TL_EVOLVE_MODEL`, e.g. `anthropic/claude-sonnet-4.5` or `qwen/qwen3-coder`
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

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
    agent: ureq::Agent,
}

fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()))
}

impl Llm {
    pub fn new(base_url: &str, api_key: Option<String>, model: &str) -> Self {
        let agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_connect(Some(Duration::from_secs(30))).timeout_global(Some(Duration::from_secs(600))).build().into();
        Llm { base_url: base_url.trim_end_matches('/').into(), api_key, model: model.into(), temperature: None, max_tokens: None, agent }
    }

    pub fn from_env(model: Option<&str>) -> Result<Self> {
        let base = first_env(&["TL_EVOLVE_BASE_URL", "OPENAI_BASE_URL"]).unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
        let key = first_env(&["TL_EVOLVE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]);
        let model = model.map(String::from).or_else(|| first_env(&["TL_EVOLVE_MODEL"])).context("no model: pass --model or set TL_EVOLVE_MODEL")?;
        Ok(Self::new(&base, key, &model))
    }

    pub fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Reply> {
        let mut body = json!({"model": self.model, "messages": messages});
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("auto");
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
            if (status == 429 || status >= 500) && attempt < 5 {
                eprintln!("llm: HTTP {status}; retrying in {delay:?}");
                std::thread::sleep(delay);
                delay *= 2;
                continue;
            }
            if status >= 300 {
                bail!("chat endpoint returned {status}: {}", text.chars().take(500).collect::<String>());
            }
            return parse_reply(&text);
        }
        bail!("chat endpoint kept failing")
    }
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
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn a_client_error_is_reported_with_the_body() {
        let (url, server) = serve(vec![(401, "{\"error\":{\"message\":\"bad key\"}}".into())]);
        let err = Llm::new(&url, None, "m").chat(&[], &[]).unwrap_err().to_string();
        assert!(err.contains("401") && err.contains("bad key"), "{err}");
        server.join().unwrap();
    }
}
