//! Code similarity: "does something like this already exist?".
//!
//! An [`Embedder`] turns code into a vector; [`CodeIndex`] keeps vectors for forum entries (kept in
//! the KV under `emb/<model>/`, so workers share them) and for chunks of the repository's Scheme, and
//! answers nearest-neighbour queries by cosine similarity. Two embedders: [`HashEmbedder`], a
//! deterministic structural embedding (token n-grams plus a skeleton of the form's shape, no network),
//! and [`HttpEmbedder`], any OpenAI-compatible `/embeddings` endpoint.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::forum::{Forum, sha_hex};
use crate::kv::Kv;

pub trait Embedder: Send + Sync {
    /// Names the vector space: vectors from different models are never compared.
    fn model(&self) -> String;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// ---- a structural embedding that needs no network -------------------------------------------------

const DIMS: usize = 512;

/// Special forms and Tree/Scheme heads that stay as themselves in the skeleton; every other symbol
/// becomes `_`, so renamed variables still look alike.
const KEYWORDS: &[&str] = &[
    "define", "lambda", "let", "let*", "letrec", "if", "cond", "else", "when", "unless", "begin", "set!", "for", "while", "var", "load", "store", "select", "guard", "map", "for-each", "fold-left", "filter", "car", "cdr", "cons", "list", "append", "memq", "assq", "case", "and", "or", "not", "quote", "kernel", "register-pass!", "map-stmt", "expr-map", "pure?", "block-sync", "shared", "cast", "rsqrt", "sqrt", "exp",
];

fn tokens(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_comment = false;
    let mut in_string = false;
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(std::mem::take(cur));
        }
    };
    for c in code.chars() {
        if in_comment {
            in_comment = c != '\n';
        } else if in_string {
            if c == '"' {
                in_string = false;
                out.push("\"\"".into());
            }
        } else if c == ';' {
            flush(&mut cur, &mut out);
            in_comment = true;
        } else if c == '"' {
            flush(&mut cur, &mut out);
            in_string = true;
        } else if c == '(' || c == '[' {
            flush(&mut cur, &mut out);
            out.push("(".into());
        } else if c == ')' || c == ']' {
            flush(&mut cur, &mut out);
            out.push(")".into());
        } else if c.is_whitespace() || c == '\'' || c == '`' || c == ',' {
            flush(&mut cur, &mut out);
        } else {
            cur.push(c);
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn fnv(s: &str, salt: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ salt.wrapping_mul(0x9e3779b97f4a7c15);
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn add_feature(v: &mut [f32], feature: &str, weight: f32) {
    let h = fnv(feature, 1);
    let sign = if fnv(feature, 2) & 1 == 0 { 1.0 } else { -1.0 };
    v[(h % DIMS as u64) as usize] += sign * weight;
}

pub struct HashEmbedder;

impl HashEmbedder {
    pub fn embed_one(code: &str) -> Vec<f32> {
        let toks = tokens(code);
        let norm: Vec<String> = toks
            .iter()
            .map(|t| if t.parse::<f64>().is_ok() { "#n".to_string() } else { t.to_lowercase() })
            .collect();
        let skeleton: Vec<&str> = norm.iter().map(|t| if t == "(" || t == ")" || t == "#n" || KEYWORDS.contains(&t.as_str()) || !t.chars().any(|c| c.is_alphabetic()) { t.as_str() } else { "_" }).collect();
        let mut v = vec![0f32; DIMS];
        for n in 1..=3 {
            for w in norm.windows(n) {
                add_feature(&mut v, &format!("t{n}:{}", w.join(" ")), 1.0 / n as f32);
            }
        }
        for n in 2..=4 {
            for w in skeleton.windows(n) {
                add_feature(&mut v, &format!("s{n}:{}", w.join(" ")), 0.7);
            }
        }
        // sublinear term frequency, then unit length
        for x in v.iter_mut() {
            *x = x.signum() * x.abs().sqrt();
        }
        normalize(&mut v);
        v
    }
}

impl Embedder for HashEmbedder {
    fn model(&self) -> String {
        "hash-v1".into()
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::embed_one(t)).collect())
    }
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---- an OpenAI-compatible embeddings endpoint -------------------------------------------------------

pub struct HttpEmbedder {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    agent: ureq::Agent,
}

impl HttpEmbedder {
    pub fn new(base_url: &str, api_key: Option<String>, model: &str) -> Self {
        let agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(120))).build().into();
        HttpEmbedder { base_url: base_url.trim_end_matches('/').into(), api_key, model: model.into(), agent }
    }
}

impl Embedder for HttpEmbedder {
    fn model(&self) -> String {
        format!("http-{}", self.model.replace(['/', ':'], "_"))
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        for batch in texts.chunks(64) {
            let mut req = self.agent.post(format!("{}/embeddings", self.base_url)).header("content-type", "application/json");
            if let Some(k) = &self.api_key {
                req = req.header("authorization", format!("Bearer {k}"));
            }
            let mut resp = req.send(json!({"model": self.model, "input": batch}).to_string()).context("embeddings request")?;
            let status = resp.status();
            let body = resp.body_mut().read_to_string()?;
            if !status.is_success() {
                bail!("embeddings endpoint returned {status}: {}", body.chars().take(300).collect::<String>());
            }
            let v: Value = serde_json::from_str(&body).context("embeddings response is not JSON")?;
            let data = v["data"].as_array().context("embeddings response has no data")?;
            let mut rows: Vec<(usize, Vec<f32>)> = data
                .iter()
                .map(|d| {
                    let idx = d["index"].as_u64().unwrap_or(0) as usize;
                    let e = d["embedding"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
                    (idx, e)
                })
                .collect();
            rows.sort_by_key(|r| r.0);
            if rows.len() != batch.len() {
                bail!("asked for {} embeddings, got {}", batch.len(), rows.len());
            }
            for (_, mut e) in rows {
                normalize(&mut e);
                out.push(e);
            }
        }
        Ok(out)
    }
}

// ---- the index -------------------------------------------------------------------------------------------

/// Splits Scheme source into top-level forms (comments before a form stay with it).
pub fn top_level_forms(src: &str) -> Vec<String> {
    let b: Vec<char> = src.chars().collect();
    let mut forms = Vec::new();
    let (mut i, mut start, mut depth) = (0, 0, 0i32);
    let mut started = false;
    while i < b.len() {
        match b[i] {
            ';' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            '"' => {
                i += 1;
                while i < b.len() && b[i] != '"' {
                    if b[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            '#' if i + 1 < b.len() && b[i + 1] == '\\' => {
                i += 3; // #\x, including #\( and #\)
                while i < b.len() && b[i].is_alphanumeric() {
                    i += 1;
                }
                continue;
            }
            '(' | '[' => {
                depth += 1;
                started = true;
            }
            ')' | ']' => {
                depth -= 1;
                if depth == 0 && started {
                    let text: String = b[start..=i].iter().collect();
                    forms.push(text.trim().to_string());
                    start = i + 1;
                    started = false;
                }
            }
            _ => {}
        }
        i += 1;
    }
    forms
}

#[derive(Debug, Clone)]
pub struct Similar {
    /// `entries/<id>` or `repo/<path>#<n>` (the n-th top-level form of the file).
    pub source: String,
    pub similarity: f32,
    pub snippet: String,
}

pub struct CodeIndex {
    embedder: Arc<dyn Embedder>,
    kv: Arc<dyn Kv>,
    prefix: String,
    /// source -> (vector, text for the snippet)
    items: HashMap<String, (Vec<f32>, String)>,
    cursor: Option<String>,
}

impl CodeIndex {
    pub fn new(embedder: Arc<dyn Embedder>, kv: Arc<dyn Kv>) -> Self {
        let prefix = format!("emb/{}/", embedder.model());
        CodeIndex { embedder, kv, prefix, items: HashMap::new(), cursor: None }
    }

    fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// The vector for `code`, from the shared cache when some worker already computed it.
    fn vectors(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let keys: Vec<String> = texts.iter().map(|t| format!("{}c/{}", self.prefix, &sha_hex(&[t])[..32])).collect();
        let mut out: Vec<Option<Vec<f32>>> = Vec::new();
        for k in &keys {
            out.push(self.kv.get(k)?.map(|b| Self::bytes_to_vec(&b)));
        }
        let missing: Vec<usize> = (0..texts.len()).filter(|i| out[*i].is_none()).collect();
        if !missing.is_empty() {
            let todo: Vec<String> = missing.iter().map(|i| texts[*i].clone()).collect();
            let got = self.embedder.embed(&todo)?;
            for (i, v) in missing.iter().zip(got) {
                self.kv.put(&keys[*i], &Self::vec_to_bytes(&v))?;
                out[*i] = Some(v);
            }
        }
        Ok(out.into_iter().map(|v| v.unwrap()).collect())
    }

    pub fn add(&mut self, source: &str, code: &str) -> Result<()> {
        let v = self.vectors(&[code.to_string()])?.remove(0);
        self.items.insert(source.to_string(), (v, code.to_string()));
        Ok(())
    }

    /// Adds every top-level form of `text` (a Scheme file) as `repo/<label>#<n>`.
    pub fn add_file(&mut self, label: &str, text: &str) -> Result<usize> {
        let forms: Vec<(String, String)> = top_level_forms(text).into_iter().enumerate().filter(|(_, f)| f.len() > 60).map(|(n, f)| (format!("repo/{label}#{n}"), f)).collect();
        let texts: Vec<String> = forms.iter().map(|f| f.1.clone()).collect();
        let vs = self.vectors(&texts)?;
        for ((source, text), v) in forms.into_iter().zip(vs) {
            self.items.insert(source, (v, text));
        }
        Ok(texts.len())
    }

    /// Embeds the code of entries added to the forum since the last sync.
    pub fn sync(&mut self, forum: &Forum) -> Result<usize> {
        let mut added = 0;
        let mut after = self.cursor.clone();
        loop {
            let batch = forum.changes(after.as_deref(), 200)?;
            if batch.is_empty() {
                break;
            }
            for (log_key, change) in &batch {
                after = Some(log_key.clone());
                if let Some(id) = change.key.strip_prefix("entries/")
                    && change.op == "put"
                    && let Some(e) = forum.get(id)?
                    && let Some(code) = e.code.filter(|c| !c.trim().is_empty())
                    && !self.items.contains_key(&change.key)
                {
                    self.add(&change.key, &code)?;
                    added += 1;
                }
            }
            self.cursor = after.clone();
        }
        Ok(added)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The `k` nearest sources to `code`, best first.
    pub fn nearest(&self, code: &str, k: usize, exclude: Option<&str>) -> Result<Vec<Similar>> {
        let q = self.vectors(&[code.to_string()])?.remove(0);
        let mut scored: Vec<(f32, &String, &String)> = self.items.iter().filter(|(s, _)| Some(s.as_str()) != exclude).map(|(s, (v, t))| (cosine(&q, v), s, t)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
        Ok(scored.into_iter().take(k).map(|(similarity, source, text)| Similar { source: source.clone(), similarity, snippet: text.chars().take(400).collect() }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forum::{Kind, NewEntry};
    use crate::kv::MemKv;

    const LICM: &str = "(define (pass-licm k) (map-body (lambda (stmts) (map licm-stmt stmts)) k)) ; hoist\n";
    const RENAMED: &str = "(define (pass-hoist kernel) (map-body (lambda (ss) (map hoist-stmt ss)) kernel))";
    const OTHER: &str = "(define (count-stores t) (cond [(pair? t) (+ (count-stores (car t)) (count-stores (cdr t)))] [(eq? t 'store) 1] [else 0]))";

    #[test]
    fn renamed_code_is_closer_than_different_code() {
        let a = HashEmbedder::embed_one(LICM);
        let b = HashEmbedder::embed_one(RENAMED);
        let c = HashEmbedder::embed_one(OTHER);
        assert!(cosine(&a, &b) > cosine(&a, &c) + 0.15, "{} vs {}", cosine(&a, &b), cosine(&a, &c));
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-4);
        // comments and numbers do not matter
        assert!(cosine(&HashEmbedder::embed_one("(+ x 1) ; one"), &HashEmbedder::embed_one("(+ x 2)")) > 0.99);
    }

    #[test]
    fn splits_top_level_forms_around_strings_chars_and_comments() {
        let src = "; header (\n(define a \"(\") ; (\n(define b #\\( ) (define c '(1 2))";
        let forms = top_level_forms(src);
        // a comment travels with the form that follows it; parens in strings, chars and comments do not count
        assert_eq!(forms.len(), 3);
        assert!(forms[0].ends_with("(define a \"(\")") && forms[1].ends_with("(define b #\\( )"));
        assert_eq!(forms[2], "(define c '(1 2))");
    }

    #[test]
    fn index_finds_forum_code_and_repo_forms_and_shares_vectors() {
        let kv: Arc<dyn Kv> = Arc::new(MemKv::new());
        let forum = Forum::new(kv.clone());
        let mut idx = CodeIndex::new(Arc::new(HashEmbedder), kv.clone());
        let e = forum.submit(NewEntry { kind: Some(Kind::Idea), author: "a".into(), headline: "licm".into(), code: Some(LICM.into()), ..Default::default() }).unwrap();
        forum.submit(NewEntry { kind: Some(Kind::Idea), author: "a".into(), headline: "stores".into(), code: Some(OTHER.into()), ..Default::default() }).unwrap();
        assert_eq!(idx.sync(&forum).unwrap(), 2);
        assert_eq!(idx.sync(&forum).unwrap(), 0);
        let near = idx.nearest(RENAMED, 2, None).unwrap();
        assert_eq!(near[0].source, format!("entries/{}", e.id));
        assert!(near[0].similarity > near[1].similarity);
        // a second worker finds the vectors already in the KV
        let before = kv.list("emb/", None, 100).unwrap().len();
        let mut other = CodeIndex::new(Arc::new(HashEmbedder), kv.clone());
        other.sync(&forum).unwrap();
        assert_eq!(kv.list("emb/", None, 100).unwrap().len(), before);
        other.add_file("x.ss", &format!("{LICM}\n{OTHER}")).unwrap();
        assert!(other.len() >= 3);
    }
}
