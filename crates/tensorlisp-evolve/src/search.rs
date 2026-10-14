//! Full-text search over the forum with a Tantivy index that follows the forum's change log (CDC).
//!
//! The index is derived state: it can live in RAM or in a directory next to an agent, it is
//! rebuilt by replaying `log/`, and it tails the log on every [`SearchIndex::sync`]. Entries are
//! indexed with separate `headline`, `body`, `tags` and `code` fields so a query can target one
//! (`headline:rsqrt`, or `field: "code"` in the tool).
use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value};
use tantivy::snippet::SnippetGenerator;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term, doc};

use crate::forum::{Entry, Forum, Post, log_key_before};

/// Records that landed within this long before the cursor are re-applied on sync.
const LOOKBACK_MS: u64 = 60_000;

struct Fields {
    key: Field,
    entry: Field,
    kind: Field,
    author: Field,
    headline: Field,
    body: Field,
    tags: Field,
    code: Field,
}

#[derive(Debug, Clone)]
pub struct Hit {
    /// The entry, whether the match was the entry or one of its replies.
    pub entry: String,
    /// Set when a reply matched.
    pub post: Option<String>,
    pub kind: String,
    pub author: String,
    pub headline: String,
    pub score: f32,
    pub snippet: String,
}

pub struct SearchIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    f: Fields,
    cursor: Option<String>,
    cursor_file: Option<PathBuf>,
    recent: BTreeSet<String>,
}

impl SearchIndex {
    pub fn in_ram() -> Result<Self> {
        Self::build(None)
    }

    /// An index that survives restarts: it resumes from the cursor it saved.
    pub fn in_dir(dir: impl Into<PathBuf>) -> Result<Self> {
        Self::build(Some(dir.into()))
    }

    fn build(dir: Option<PathBuf>) -> Result<Self> {
        let mut sb = Schema::builder();
        let key = sb.add_text_field("key", STRING | STORED);
        let entry = sb.add_text_field("entry", STRING | STORED);
        let kind = sb.add_text_field("kind", STRING | STORED);
        let author = sb.add_text_field("author", STRING | STORED);
        let headline = sb.add_text_field("headline", TEXT | STORED);
        let body = sb.add_text_field("body", TEXT | STORED);
        let tags = sb.add_text_field("tags", TEXT | STORED);
        let code = sb.add_text_field("code", TEXT | STORED);
        let schema = sb.build();
        let (index, cursor_file) = match dir {
            Some(dir) => {
                std::fs::create_dir_all(&dir)?;
                let index = Index::open_or_create(tantivy::directory::MmapDirectory::open(&dir)?, schema)
                    .with_context(|| format!("opening the index in {}", dir.display()))?;
                (index, Some(dir.join("cursor")))
            }
            None => (Index::create_in_ram(schema), None),
        };
        let writer = index.writer(50_000_000)?;
        let reader = index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        let cursor = cursor_file.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        Ok(SearchIndex { index, writer, reader, f: Fields { key, entry, kind, author, headline, body, tags, code }, cursor, cursor_file, recent: BTreeSet::new() })
    }

    fn upsert_entry(&mut self, e: &Entry) {
        let key = format!("entries/{}", e.id);
        self.writer.delete_term(Term::from_field_text(self.f.key, &key));
        let _ = self.writer.add_document(doc!(
            self.f.key => key,
            self.f.entry => e.id.clone(),
            self.f.kind => e.kind.as_str(),
            self.f.author => e.author.clone(),
            self.f.headline => e.headline.clone(),
            self.f.body => e.body.clone(),
            self.f.tags => e.tags.join(" "),
            self.f.code => e.code.clone().unwrap_or_default(),
        ));
    }

    fn upsert_post(&mut self, p: &Post) {
        let key = format!("posts/{}/{}", p.entry, p.id);
        self.writer.delete_term(Term::from_field_text(self.f.key, &key));
        let _ = self.writer.add_document(doc!(
            self.f.key => key,
            self.f.entry => p.entry.clone(),
            self.f.kind => "post",
            self.f.author => p.author.clone(),
            self.f.body => p.body.clone(),
        ));
    }

    /// Applies the forum's changes since the last sync. Returns how many records were new.
    pub fn sync(&mut self, forum: &Forum) -> Result<usize> {
        let mut applied = 0;
        let start = self.cursor.as_deref().and_then(|c| log_key_before(c, LOOKBACK_MS));
        let mut after = start;
        loop {
            let batch = forum.changes(after.as_deref(), 500)?;
            if batch.is_empty() {
                break;
            }
            for (log_key, change) in &batch {
                after = Some(log_key.clone());
                if !self.recent.insert(log_key.clone()) {
                    continue;
                }
                applied += 1;
                if let Some(id) = change.key.strip_prefix("entries/") {
                    match (change.op.as_str(), forum.get(id)?) {
                        ("put", Some(e)) => self.upsert_entry(&e),
                        _ => self.writer.delete_term(Term::from_field_text(self.f.key, &change.key)).pipe_unit(),
                    }
                } else if change.key.starts_with("posts/") {
                    let post = if change.op == "put" { forum.kv().get(&change.key)?.and_then(|b| serde_json::from_slice::<Post>(&b).ok()) } else { None };
                    match post {
                        Some(p) => self.upsert_post(&p),
                        None => self.writer.delete_term(Term::from_field_text(self.f.key, &change.key)).pipe_unit(),
                    }
                }
            }
            self.cursor = after.clone();
        }
        if let Some(c) = &self.cursor {
            // Forget log keys older than the window: they will not be re-read.
            if let Some(cut) = log_key_before(c, LOOKBACK_MS) {
                self.recent = self.recent.split_off(&cut);
            }
        }
        if applied > 0 {
            self.writer.commit()?;
            self.reader.reload()?;
            if let (Some(file), Some(c)) = (&self.cursor_file, &self.cursor) {
                std::fs::write(file, c)?;
            }
        }
        Ok(applied)
    }

    /// `field` narrows the query to one of headline, body, tags, code; `kind` filters on the entry kind
    /// (or `post` for replies).
    pub fn search(&self, query: &str, field: Option<&str>, kind: Option<&str>, limit: usize) -> Result<Vec<Hit>> {
        let f = &self.f;
        let fields = match field.unwrap_or("all") {
            "all" => vec![f.headline, f.body, f.tags, f.code],
            "headline" => vec![f.headline],
            "body" => vec![f.body],
            "tags" => vec![f.tags],
            "code" => vec![f.code],
            other => anyhow::bail!("field must be all, headline, body, tags or code, not {other:?}"),
        };
        let mut parser = QueryParser::for_index(&self.index, fields.clone());
        parser.set_field_boost(f.headline, 3.0);
        parser.set_field_boost(f.tags, 2.0);
        let (parsed, _errors) = parser.parse_query_lenient(query);
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, parsed)];
        if let Some(kind) = kind {
            clauses.push((Occur::Must, Box::new(TermQuery::new(Term::from_field_text(f.kind, kind), IndexRecordOption::Basic))));
        }
        let full = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let top = searcher.search(&full, &TopDocs::with_limit(limit.max(1)).order_by_score())?;
        let snippet_field = fields.iter().copied().find(|x| *x == f.code).filter(|_| field == Some("code")).unwrap_or(f.body);
        let snip = SnippetGenerator::create(&searcher, &full, snippet_field).ok();
        let text = |d: &TantivyDocument, field: Field| d.get_first(field).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let mut hits = Vec::new();
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            let key = text(&d, f.key);
            let body = text(&d, snippet_field);
            let mut snippet = snip.as_ref().map(|g| g.snippet(&body).fragment().to_string()).unwrap_or_default();
            if snippet.trim().is_empty() {
                snippet = body.chars().take(160).collect();
            }
            let post = key.starts_with("posts/").then(|| key.clone());
            let mut headline = text(&d, f.headline);
            if headline.is_empty() {
                headline = format!("(reply) {}", text(&d, f.body).lines().next().unwrap_or("").chars().take(80).collect::<String>());
            }
            hits.push(Hit { entry: text(&d, f.entry), post, kind: text(&d, f.kind), author: text(&d, f.author), headline, score, snippet: snippet.replace('\n', " ") });
        }
        Ok(hits)
    }

    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

trait PipeUnit {
    fn pipe_unit(self);
}
impl<T> PipeUnit for T {
    fn pipe_unit(self) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::forum::{Kind, NewEntry};
    use crate::kv::MemKv;

    fn add(f: &Forum, headline: &str, body: &str, code: &str) -> Entry {
        f.submit(NewEntry { kind: Some(Kind::Idea), author: "a".into(), headline: headline.into(), body: body.into(), code: Some(code.into()), tags: vec!["licm".into()], ..Default::default() }).unwrap()
    }

    #[test]
    fn headline_and_code_fields_are_separate_and_replies_are_found() {
        let forum = Forum::new(Arc::new(MemKv::new()));
        let mut idx = SearchIndex::in_ram().unwrap();
        let a = add(&forum, "hoist rsqrt out of the row loop", "The reciprocal square root is loop invariant.", "(define (pass-x k) k)");
        let _b = add(&forum, "unroll the dot loop", "Unrolling by four helps matvec.", "(for i 0 n (rsqrt x))");
        assert_eq!(idx.sync(&forum).unwrap(), 2);
        let hits = idx.search("rsqrt", Some("headline"), None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry, a.id);
        let hits = idx.search("rsqrt", Some("code"), None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_ne!(hits[0].entry, a.id);
        assert_eq!(idx.search("rsqrt", None, None, 10).unwrap().len(), 2);
        // the headline match ranks above the code match
        assert_eq!(idx.search("rsqrt", None, None, 10).unwrap()[0].entry, a.id);

        forum.reply(&a.id, "b", "this regressed on metal because of register pressure").unwrap();
        assert_eq!(idx.sync(&forum).unwrap(), 1);
        let hits = idx.search("register pressure", None, None, 10).unwrap();
        assert_eq!(hits[0].entry, a.id);
        assert!(hits[0].post.is_some());
        assert_eq!(idx.search("regressed", None, Some("post"), 10).unwrap().len(), 1);
        assert_eq!(idx.search("regressed", None, Some("idea"), 10).unwrap().len(), 0);
        // nothing new
        assert_eq!(idx.sync(&forum).unwrap(), 0);
    }

    #[test]
    fn the_index_resumes_from_its_cursor() {
        let dir = std::env::temp_dir().join(format!("evolve-idx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let forum = Forum::new(Arc::new(MemKv::new()));
        {
            let mut idx = SearchIndex::in_dir(&dir).unwrap();
            add(&forum, "first idea", "alpha", "");
            idx.sync(&forum).unwrap();
        }
        add(&forum, "second idea", "beta", "");
        let mut idx = SearchIndex::in_dir(&dir).unwrap();
        assert_eq!(idx.doc_count(), 1);
        idx.sync(&forum).unwrap();
        assert_eq!(idx.doc_count(), 2);
        assert_eq!(idx.search("beta", None, None, 5).unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
