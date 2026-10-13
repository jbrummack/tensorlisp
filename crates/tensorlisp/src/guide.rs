//! Constrained decoding with `outlines-core`: a regular expression compiled against a byte-level BPE tokenizer's
//! vocabulary gives, at every step, the token ids that keep the output a prefix of a match. VoiceChat uses it for the
//! function channel: after `<SPECIAL_20>` the tokens must spell `<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>`
//! for the tools the system prompt lists, and the end token is allowed only once that is complete.
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use outlines_core::{index::Index, json_schema, vocabulary::Vocabulary};

pub struct Guide {
    index: Arc<Index>,
    state: u32,
    vocab: usize,
}

/// GPT-2's byte <-> printable-character table, inverted.
fn byte_of_char() -> HashMap<char, u8> {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = bytes.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    bytes.iter().zip(&chars).map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

/// The vocabulary of a Hugging Face `tokenizer.json` (byte-level BPE), without its added (special) tokens; `eos`
/// is the id the guide allows when the match is complete.
pub fn vocabulary(tokenizer_json: &[u8], eos: u32) -> Result<(Vocabulary, usize), String> {
    let json: serde_json::Value = serde_json::from_slice(tokenizer_json).map_err(|e| e.to_string())?;
    let vocab = json["model"]["vocab"].as_object().ok_or("tokenizer.json has no model.vocab")?;
    let special: std::collections::HashSet<u32> = json["added_tokens"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t["id"].as_u64().map(|i| i as u32)).collect())
        .unwrap_or_default();
    let table = byte_of_char();
    let mut v = Vocabulary::new(eos);
    let mut size = 0usize;
    for (token, id) in vocab {
        let id = id.as_u64().ok_or("token id is not a number")? as u32;
        size = size.max(id as usize + 1);
        if special.contains(&id) || id == eos {
            continue;
        }
        let Some(bytes) = token.chars().map(|c| table.get(&c).copied()).collect::<Option<Vec<u8>>>() else { continue };
        v.try_insert(bytes, id).map_err(|e| e.to_string())?;
    }
    Ok((v, size))
}

type Cache = Mutex<HashMap<(String, usize), Arc<Index>>>;

impl Guide {
    pub fn new(tokenizer_json: &[u8], regex: &str, eos: u32) -> Result<Guide, String> {
        static CACHE: OnceLock<Cache> = OnceLock::new();
        let key = (regex.to_string(), tokenizer_json.len() * 31 + eos as usize);
        let cache = CACHE.get_or_init(Default::default);
        let (index, vocab) = {
            let hit = cache.lock().unwrap().get(&key).cloned();
            match hit {
                Some(i) => {
                    let vocab = i.vocab_size();
                    (i, vocab)
                }
                None => {
                    let (v, size) = vocabulary(tokenizer_json, eos)?;
                    let index = Arc::new(Index::new(regex, &v).map_err(|e| e.to_string())?);
                    cache.lock().unwrap().insert(key, index.clone());
                    (index, size)
                }
            }
        };
        let state = index.initial_state();
        Ok(Guide { index, state, vocab: vocab.max(131072) })
    }

    /// 0 for the tokens that may come next, -1e30 for the others: `vocab` values.
    pub fn mask(&self) -> Vec<f32> {
        let mut m = vec![-1e30f32; self.vocab];
        if let Some(allowed) = self.index.allowed_tokens(&self.state) {
            for t in allowed {
                if (t as usize) < m.len() {
                    m[t as usize] = 0.0;
                }
            }
        }
        m
    }

    pub fn advance(&mut self, token: u32) -> bool {
        match self.index.next_state(&self.state, &token) {
            Some(s) => {
                self.state = s;
                true
            }
            None => false,
        }
    }

    pub fn is_final(&self) -> bool {
        self.index.is_final_state(&self.state)
    }

    pub fn reset(&mut self) {
        self.state = self.index.initial_state();
    }
}

/// The regular expression of a tool call for the given tools (a JSON array of
/// `{"name", "description", "parameters": <JSON schema>}`): `<TOOLCALL>[call(, call)*]</TOOLCALL>` where a call is
/// `{"name": "<one of the tools>", "arguments": <that tool's parameters>}`.
pub fn tool_call_regex(tools_json: &str, max_calls: usize) -> Result<String, String> {
    let tools: serde_json::Value = serde_json::from_str(tools_json).map_err(|e| format!("tools: {e}"))?;
    let tools = tools.as_array().ok_or("tools must be a JSON array")?;
    if tools.is_empty() {
        return Err("no tools".into());
    }
    let mut calls = Vec::new();
    for tool in tools {
        let tool = tool.get("function").unwrap_or(tool);
        let name = tool["name"].as_str().ok_or("a tool has no name")?;
        let params = tool.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"const": name}, "arguments": params},
            "required": ["name", "arguments"],
            "additionalProperties": false,
        });
        calls.push(json_schema::regex_from_value(&schema, None, None).map_err(|e| format!("tool {name}: {e}"))?);
    }
    let call = format!("({})", calls.join("|"));
    Ok(format!("<TOOLCALL>\\[{call}(, {call}){{0,{}}}\\]</TOOLCALL>", max_calls.saturating_sub(1)))
}

/// Python's `json.dumps(value, sort_keys=True)` as Jinja's `tojson` renders it: keys sorted at every level, ", " and
/// ": " separators, non-ASCII as \uXXXX, and `<`, `>`, `&`, `'` escaped.
fn dumps_sorted(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dumps_string(k, out);
                out.push_str(": ");
                dumps_sorted(&m[k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dumps_sorted(x, out);
            }
            out.push(']');
        }
        Value::String(s) => dumps_string(s, out),
        other => out.push_str(&other.to_string()),
    }
}

fn dumps_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' | '>' | '&' | '\'' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The tools as the system prompt shows them (NVIDIA's template.jinja): a JSON array of each tool's `function` (or the
/// tool itself) without its `type`, keys sorted.
pub fn tools_prompt_json(tools_json: &str) -> Result<String, String> {
    let tools: serde_json::Value = serde_json::from_str(tools_json).map_err(|e| format!("tools: {e}"))?;
    let tools = tools.as_array().ok_or("tools must be a JSON array")?;
    let mut out = String::from("[");
    for (i, tool) in tools.iter().enumerate() {
        let t = tool.get("function").unwrap_or(tool);
        let mut d = serde_json::Map::new();
        for (k, v) in t.as_object().ok_or("a tool must be an object")? {
            if k != "type" {
                d.insert(k.clone(), v.clone());
            }
        }
        if i > 0 {
            out.push_str(", ");
        }
        dumps_sorted(&serde_json::Value::Object(d), &mut out);
    }
    out.push(']');
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn prompt_json_is_jinjas() {
        let tools = r#"[{"function": {"name": "f", "description": "a < b, é", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}}}}]"#;
        assert_eq!(
            super::tools_prompt_json(tools).unwrap(),
            r#"[{"description": "a \u003c b, \u00e9", "name": "f", "parameters": {"properties": {"max": {"type": "integer"}, "min": {"type": "integer"}}, "type": "object"}}]"#
        );
    }

    use super::*;

    #[test]
    fn regex_accepts_a_call() {
        let tools = r#"[{"name": "get_weather", "description": "x", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]"#;
        let re = tool_call_regex(tools, 2).unwrap();
        let r = regex::Regex::new(&format!("^{re}$")).unwrap();
        assert!(r.is_match(r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</TOOLCALL>"#), "{re}");
        assert!(!r.is_match(r#"<TOOLCALL>[{"name": "other", "arguments": {"city": "Paris"}}]</TOOLCALL>"#));
    }

    /// The real Nemotron tokenizer: the tokens of a valid call are accepted one by one, a wrong tool name is not, and
    /// the end token is allowed exactly when the call is complete.
    #[test]
    fn guide_follows_a_call_through_the_real_vocabulary() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../ports/voicechat/tokenizer.json");
        let Ok(bytes) = std::fs::read(path) else { return };
        let tools = r#"[{"name": "get_weather", "description": "x", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
                        {"name": "get_stock_price", "description": "y", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}]"#;
        let re = tool_call_regex(tools, 2).unwrap();
        let t = std::time::Instant::now();
        let mut g = Guide::new(&bytes, &re, 21).unwrap();
        eprintln!("index built in {:?}", t.elapsed());
        let tok = autopro::text::Tokenizer::from_bytes(&bytes).unwrap();
        let ids = |text: &str| -> Vec<u32> {
            let o = autopro::text::TextOptions { lowercase: false, add_special_tokens: false, max_length: None, padding: autopro::text::Padding::None, pad_id: None };
            o.encode_batch(&tok, &[text]).unwrap().ids.row(0).iter().map(|&v| v as u32).collect()
        };
        let call = r#"<TOOLCALL>[{"name": "get_stock_price", "arguments": {"symbol": "NVDA"}}]</TOOLCALL>"#;
        for (i, id) in ids(call).into_iter().enumerate() {
            assert!(g.mask()[id as usize] == 0.0, "token {i} ({id}) of the call is refused");
            assert!(g.advance(id));
            if i == 0 {
                assert!(!g.is_final());
            }
        }
        assert!(g.is_final());
        assert_eq!(g.mask()[21], 0.0, "the end token is allowed after a complete call");
        let mut g2 = Guide::new(&bytes, &re, 21).unwrap();
        let bad = ids(r#"<TOOLCALL>[{"name": "get_"#);
        let ok = bad.iter().all(|&id| g2.advance(id));
        assert!(ok);
        let w = ids("x");
        assert!(!g2.advance(w[0]), "'x' cannot continue get_weather / get_stock_price");
    }
}
