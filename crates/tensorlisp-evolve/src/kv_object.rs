//! [`Kv`] over an `object_store`: S3, GCS, Azure Blob, a local directory, or memory.
//!
//! This is how agents on several machines share one forum: point every worker at the same bucket
//! prefix. `create` is a conditional put (`PutMode::Create`), which S3, GCS and Azure implement, so two
//! workers cannot both take an id. Listings are sorted here because the trait does not promise an order.
//! The calls are blocking: a private current-thread runtime drives the async store.
//!
//! ```ignore
//! let store = object_store::aws::AmazonS3Builder::from_env().with_bucket_name("forum").build()?;
//! let kv = ObjectStoreKv::new(Arc::new(store), "tensorlisp/evolve")?;
//! ```
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{Error, ObjectStore, ObjectStoreExt, PutMode, PutPayload};

use crate::kv::{Kv, check_key};

pub struct ObjectStoreKv {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    rt: tokio::runtime::Runtime,
}

impl ObjectStoreKv {
    /// Keys live under `prefix` inside the store ("" for the root).
    pub fn new(store: Arc<dyn ObjectStore>, prefix: &str) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(ObjectStoreKv { store, prefix: prefix.trim_matches('/').to_string(), rt })
    }

    fn path(&self, key: &str) -> Result<Path> {
        check_key(key)?;
        Ok(if self.prefix.is_empty() { Path::from(key) } else { Path::from(format!("{}/{key}", self.prefix)) })
    }

    fn strip<'a>(&self, location: &'a str) -> &'a str {
        if self.prefix.is_empty() { location } else { location.strip_prefix(&self.prefix).and_then(|r| r.strip_prefix('/')).unwrap_or(location) }
    }
}

impl Kv for ObjectStoreKv {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.path(key)?;
        self.rt.block_on(async {
            match self.store.get(&path).await {
                Ok(r) => Ok(Some(r.bytes().await?.to_vec())),
                Err(Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(anyhow::Error::from(e)),
            }
        })
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.path(key)?;
        self.rt.block_on(self.store.put(&path, PutPayload::from(value.to_vec()))).map(|_| ()).with_context(|| format!("put {key}"))
    }

    fn create(&self, key: &str, value: &[u8]) -> Result<bool> {
        let path = self.path(key)?;
        self.rt.block_on(async {
            match self.store.put_opts(&path, PutPayload::from(value.to_vec()), PutMode::Create.into()).await {
                Ok(_) => Ok(true),
                Err(Error::AlreadyExists { .. }) => Ok(false),
                Err(e) => Err(anyhow::Error::from(e)),
            }
        })
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.path(key)?;
        self.rt.block_on(async {
            match self.store.delete(&path).await {
                Ok(()) | Err(Error::NotFound { .. }) => Ok(()),
                Err(e) => Err(anyhow::Error::from(e)),
            }
        })
    }

    fn list(&self, prefix: &str, after: Option<&str>, limit: usize) -> Result<Vec<String>> {
        // object stores list by whole path segments: list the directory the prefix names, filter on the rest
        let dir = match prefix.rfind('/') {
            Some(i) => &prefix[..i],
            None => "",
        };
        let root = if self.prefix.is_empty() { dir.to_string() } else if dir.is_empty() { self.prefix.clone() } else { format!("{}/{dir}", self.prefix) };
        let root = (!root.is_empty()).then(|| Path::from(root));
        let metas = match after.filter(|a| a.starts_with(dir)) {
            Some(a) => {
                let offset = self.path(a)?;
                self.rt.block_on(self.store.list_with_offset(root.as_ref(), &offset).try_collect::<Vec<_>>())?
            }
            None => self.rt.block_on(self.store.list(root.as_ref()).try_collect::<Vec<_>>())?,
        };
        let mut keys: Vec<String> = metas.iter().map(|m| self.strip(m.location.as_ref()).to_string()).filter(|k| k.starts_with(prefix) && after.is_none_or(|a| k.as_str() > a)).collect();
        keys.sort();
        keys.truncate(limit);
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn behaves_like_the_other_backends() {
        crate::kv::tests::exercise(&ObjectStoreKv::new(Arc::new(InMemory::new()), "").unwrap());
        crate::kv::tests::exercise(&ObjectStoreKv::new(Arc::new(InMemory::new()), "some/prefix").unwrap());
    }

    #[test]
    fn two_stores_on_one_bucket_share_a_forum() {
        use crate::forum::{Forum, Kind, NewEntry};
        let bucket = Arc::new(InMemory::new());
        let a = Forum::new(Arc::new(ObjectStoreKv::new(bucket.clone(), "forum").unwrap()));
        let b = Forum::new(Arc::new(ObjectStoreKv::new(bucket, "forum").unwrap()));
        let e = a.submit(NewEntry { kind: Some(Kind::Note), author: "a".into(), headline: "from a".into(), ..Default::default() }).unwrap();
        b.reply(&e.id, "b", "seen from b").unwrap();
        assert_eq!(b.get(&e.id).unwrap().unwrap().headline, "from a");
        assert_eq!(a.posts(&e.id).unwrap().len(), 1);
        assert_eq!(b.changes(None, 10).unwrap().len(), 2);
    }
}
