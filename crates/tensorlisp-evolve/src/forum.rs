//! The forum: what agents learned, on top of a [`Kv`].
//!
//! Layout (every key is written with `put`/`create` only, so any ordered KV can back it):
//!
//! ```text
//! entries/<id>          Entry   an idea, a pass, a result or a negative result
//! posts/<id>/<post>     Post    a reply under an entry
//! evals/<hash>          Eval    the evaluator's verdict on one candidate (written only by the evaluator)
//! log/<nanos>-<rand>    Change  one record per write: the change feed that indexes tail
//! ```
//!
//! A verdict can only come from an [`Eval`] record, which the evaluator writes: an entry that names an
//! `eval_id` gets its code, pipeline and verdict copied from that record, so an agent cannot post
//! numbers it did not measure.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::kv::Kv;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// A hypothesis not (yet) implemented.
    Idea,
    /// A pass or pipeline with a verdict.
    Pass,
    /// A measurement or finding that is not a pass.
    Result,
    /// Something that was tried and does not work, and why.
    Negative,
    Note,
}

impl std::str::FromStr for Kind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "idea" => Kind::Idea,
            "pass" => Kind::Pass,
            "result" => Kind::Result,
            "negative" => Kind::Negative,
            "note" => Kind::Note,
            _ => bail!("kind must be idea, pass, result, negative or note, not {s:?}"),
        })
    }
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Idea => "idea",
            Kind::Pass => "pass",
            Kind::Result => "result",
            Kind::Negative => "negative",
            Kind::Note => "note",
        }
    }
}

/// What the evaluator measured. `score` is a ratio where above 1 is better than the baseline pipeline
/// (static: baseline code size over candidate code size; a gate command can report its own `speedup`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Verdict {
    pub eval_id: String,
    pub target: String,
    pub device: String,
    /// Every gate passed.
    pub ok: bool,
    pub score: f64,
    pub gates: BTreeMap<String, bool>,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eval {
    pub id: String,
    pub code: String,
    pub pipeline: Vec<String>,
    pub verdict: Verdict,
    /// The full report (per-kernel rows, notes, gate output) for `forum read`.
    pub report: serde_json::Value,
    pub created: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub kind: Kind,
    pub author: String,
    pub parent: Option<String>,
    /// One line, what a search result shows.
    pub headline: String,
    /// The reasoning and findings.
    pub body: String,
    pub tags: Vec<String>,
    pub code: Option<String>,
    pub pipeline: Option<Vec<String>>,
    pub verdict: Option<Verdict>,
    pub created: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Post {
    pub id: String,
    pub entry: String,
    pub author: String,
    pub body: String,
    pub created: u64,
}

#[derive(Debug, Clone, Default)]
pub struct NewEntry {
    pub kind: Option<Kind>,
    pub author: String,
    pub parent: Option<String>,
    pub headline: String,
    pub body: String,
    pub tags: Vec<String>,
    /// Code without a verdict (an idea sketch). Ignored when `eval_id` is set.
    pub code: Option<String>,
    pub eval_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub op: String,
    pub key: String,
}

pub fn now_millis() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub fn now_nanos() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

pub fn sha_hex(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn unique_suffix(seed: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed).to_string();
    sha_hex(&[seed, &n, &std::process::id().to_string(), &now_nanos().to_string()])[..8].to_string()
}

#[derive(Clone)]
pub struct Forum {
    kv: Arc<dyn Kv>,
}

impl Forum {
    pub fn new(kv: Arc<dyn Kv>) -> Self {
        Forum { kv }
    }

    pub fn kv(&self) -> &Arc<dyn Kv> {
        &self.kv
    }

    fn put_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        self.kv.put(key, &serde_json::to_vec_pretty(value)?)?;
        self.log("put", key)
    }

    fn log(&self, op: &str, key: &str) -> Result<()> {
        let record = serde_json::to_vec(&Change { op: op.into(), key: key.into() })?;
        for _ in 0..8 {
            let log_key = format!("log/{:020}-{}", now_nanos(), unique_suffix(key));
            if self.kv.create(&log_key, &record)? {
                return Ok(());
            }
        }
        bail!("could not append to the change log")
    }

    fn get_json<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        match self.kv.get(key)? {
            Some(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("decoding {key}"))?)),
            None => Ok(None),
        }
    }

    /// Stores an evaluator verdict. Only the evaluator calls this.
    pub fn put_eval(&self, eval: &Eval) -> Result<()> {
        self.put_json(&format!("evals/{}", eval.id), eval)
    }

    pub fn get_eval(&self, id: &str) -> Result<Option<Eval>> {
        self.get_json(&format!("evals/{id}"))
    }

    pub fn submit(&self, new: NewEntry) -> Result<Entry> {
        let kind = new.kind.ok_or_else(|| anyhow!("kind is required"))?;
        if new.headline.trim().is_empty() {
            bail!("a headline is required: one line saying what this is, it is what searches show");
        }
        if new.headline.contains('\n') || new.headline.len() > 200 {
            bail!("the headline must be one line of at most 200 characters");
        }
        if let Some(p) = &new.parent
            && self.get(p)?.is_none()
        {
            bail!("parent {p} does not exist");
        }
        let (code, pipeline, verdict) = match &new.eval_id {
            Some(id) => {
                let eval = self.get_eval(id)?.ok_or_else(|| anyhow!("no evaluation {id}: only the evaluate tool creates them"))?;
                (Some(eval.code), Some(eval.pipeline), Some(eval.verdict))
            }
            None => (new.code, None, None),
        };
        if kind == Kind::Pass && verdict.is_none() {
            bail!("a pass entry needs the eval_id of an evaluation: run evaluate first (use kind idea for an untested sketch)");
        }
        for attempt in 0..8 {
            let id = format!("e-{}", &unique_suffix(&format!("{}{}{attempt}", new.author, new.headline)));
            let entry = Entry {
                id: id.clone(),
                kind,
                author: new.author.clone(),
                parent: new.parent.clone(),
                headline: new.headline.trim().to_string(),
                body: new.body.clone(),
                tags: new.tags.clone(),
                code: code.clone(),
                pipeline: pipeline.clone(),
                verdict: verdict.clone(),
                created: now_millis(),
            };
            if self.kv.create(&format!("entries/{id}"), &serde_json::to_vec_pretty(&entry)?)? {
                self.log("put", &format!("entries/{id}"))?;
                return Ok(entry);
            }
        }
        bail!("could not allocate an entry id")
    }

    pub fn get(&self, id: &str) -> Result<Option<Entry>> {
        if !id.starts_with("e-") || id.contains('/') {
            return Ok(None);
        }
        self.get_json(&format!("entries/{id}"))
    }

    pub fn reply(&self, entry: &str, author: &str, body: &str) -> Result<Post> {
        if self.get(entry)?.is_none() {
            bail!("no entry {entry}");
        }
        if body.trim().is_empty() {
            bail!("empty reply");
        }
        let id = format!("{:020}-{}", now_nanos(), unique_suffix(body));
        let post = Post { id: id.clone(), entry: entry.into(), author: author.into(), body: body.into(), created: now_millis() };
        self.put_json(&format!("posts/{entry}/{id}"), &post)?;
        Ok(post)
    }

    pub fn posts(&self, entry: &str) -> Result<Vec<Post>> {
        let mut out = Vec::new();
        for key in self.kv.list(&format!("posts/{entry}/"), None, 10_000)? {
            if let Some(p) = self.get_json::<Post>(&key)? {
                out.push(p);
            }
        }
        Ok(out)
    }

    /// Entries in id order after `after`.
    pub fn entries(&self, after: Option<&str>, limit: usize) -> Result<Vec<Entry>> {
        let after = after.map(|a| format!("entries/{a}"));
        let mut out = Vec::new();
        for key in self.kv.list("entries/", after.as_deref(), limit)? {
            if let Some(e) = self.get_json::<Entry>(&key)? {
                out.push(e);
            }
        }
        Ok(out)
    }

    /// The best verified entry per (target, device): the highest score among entries whose gates all passed
    /// and that beat the baseline (score above 1): a pass that is merely valid is not a winner.
    pub fn leaderboard(&self, per_group: usize) -> Result<Vec<(String, String, Vec<Entry>)>> {
        let mut groups: BTreeMap<(String, String), Vec<Entry>> = BTreeMap::new();
        for e in self.entries(None, 100_000)? {
            if let Some(v) = &e.verdict
                && v.ok
                && v.score > 1.0
            {
                groups.entry((v.target.clone(), v.device.clone())).or_default().push(e);
            }
        }
        Ok(groups
            .into_iter()
            .map(|((t, d), mut es)| {
                es.sort_by(|a, b| b.verdict.as_ref().unwrap().score.total_cmp(&a.verdict.as_ref().unwrap().score));
                es.truncate(per_group);
                (t, d, es)
            })
            .collect())
    }

    /// Change records with log keys strictly after `after`.
    pub fn changes(&self, after: Option<&str>, limit: usize) -> Result<Vec<(String, Change)>> {
        let mut out = Vec::new();
        for key in self.kv.list("log/", after, limit)? {
            if let Some(b) = self.kv.get(&key)?
                && let Ok(c) = serde_json::from_slice::<Change>(&b)
            {
                out.push((key, c));
            }
        }
        Ok(out)
    }
}

/// A log key from `lookback_ms` before `key`'s timestamp: tailers restart from here so a record
/// that landed late (clock skew between writers) is still seen. Applying a change twice is harmless.
pub fn log_key_before(key: &str, lookback_ms: u64) -> Option<String> {
    let nanos: u128 = key.strip_prefix("log/")?.split('-').next()?.parse().ok()?;
    let back = nanos.saturating_sub(lookback_ms as u128 * 1_000_000);
    Some(format!("log/{back:020}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::MemKv;

    fn forum() -> Forum {
        Forum::new(Arc::new(MemKv::new()))
    }

    fn eval(f: &Forum, ok: bool, score: f64, target: &str) -> String {
        let id = sha_hex(&[&score.to_string(), target]);
        let verdict = Verdict { eval_id: id.clone(), target: target.into(), device: "cpu".into(), ok, score, gates: BTreeMap::new(), metrics: BTreeMap::new() };
        f.put_eval(&Eval { id: id.clone(), code: "(define x 1)".into(), pipeline: vec!["fold".into()], verdict, report: serde_json::json!({}), created: 0 }).unwrap();
        id
    }

    #[test]
    fn entries_posts_and_the_change_log() {
        let f = forum();
        let e = f.submit(NewEntry { kind: Some(Kind::Idea), author: "a".into(), headline: "hoist rsqrt".into(), body: "b".into(), ..Default::default() }).unwrap();
        assert!(e.id.starts_with("e-"));
        let p = f.reply(&e.id, "b", "tried it, no gain").unwrap();
        assert_eq!(f.posts(&e.id).unwrap()[0].id, p.id);
        assert_eq!(f.get(&e.id).unwrap().unwrap().headline, "hoist rsqrt");
        let changes = f.changes(None, 100).unwrap();
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].1.key, format!("entries/{}", e.id));
        assert!(changes[0].0 < changes[1].0);
        assert!(f.changes(Some(&changes[0].0), 100).unwrap().len() == 1);
    }

    #[test]
    fn a_pass_needs_an_evaluation_and_copies_its_verdict() {
        let f = forum();
        let base = NewEntry { kind: Some(Kind::Pass), author: "a".into(), headline: "h".into(), ..Default::default() };
        assert!(f.submit(base.clone()).is_err());
        assert!(f.submit(NewEntry { eval_id: Some("nope".into()), ..base.clone() }).is_err());
        let id = eval(&f, true, 1.2, "spirv");
        let e = f.submit(NewEntry { eval_id: Some(id.clone()), code: Some("ignored".into()), ..base }).unwrap();
        assert_eq!(e.code.as_deref(), Some("(define x 1)"));
        assert_eq!(e.verdict.unwrap().score, 1.2);
    }

    #[test]
    fn leaderboard_keeps_only_verified_winners_per_target() {
        let f = forum();
        for (ok, score, target) in [(true, 1.1, "spirv"), (true, 1.3, "spirv"), (false, 9.0, "spirv"), (true, 1.05, "ptx"), (true, 0.999, "msl"), (true, 1.0, "msl")] {
            let id = eval(&f, ok, score, target);
            f.submit(NewEntry { kind: Some(Kind::Pass), author: "a".into(), headline: format!("{score}"), eval_id: Some(id), ..Default::default() }).unwrap();
        }
        let board = f.leaderboard(1).unwrap();
        // valid but not better than the baseline (0.999, 1.0) does not make the board
        assert_eq!(board.len(), 2);
        let spirv = board.iter().find(|g| g.0 == "spirv").unwrap();
        assert_eq!(spirv.2[0].verdict.as_ref().unwrap().score, 1.3);
    }

    #[test]
    fn headline_rules() {
        let f = forum();
        let mk = |h: &str| NewEntry { kind: Some(Kind::Note), author: "a".into(), headline: h.into(), ..Default::default() };
        assert!(f.submit(mk("")).is_err());
        assert!(f.submit(mk("two\nlines")).is_err());
        assert!(f.submit(NewEntry { parent: Some("e-missing".into()), ..mk("x") }).is_err());
    }

    #[test]
    fn log_lookback() {
        assert_eq!(log_key_before("log/00000000002000000000-abc", 1).unwrap(), "log/00000000001999000000");
        assert!(log_key_before("bogus", 1).is_none());
    }
}
