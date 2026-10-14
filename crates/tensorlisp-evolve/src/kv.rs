//! The storage the forum sits on: an ordered key-value store with prefix listing.
//!
//! Keys are `/`-separated paths (`entries/e-1a2b3c4d`). The trait is the part an object store or
//! SlateDB implements (`get`/`put` map to the object API, `create` to a conditional put, `list` to a
//! ranged scan), so a shared backend only has to provide these five calls. [`FsKv`] keeps one file per
//! key and is what a single machine, or a network filesystem, needs; [`MemKv`] is for tests.
//!
//! Rule for keys: no key may be a directory prefix of another (`a/b` and `a/b/c` do not coexist).
//! The forum's layout never does that.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

pub trait Kv: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    /// Replaces the value (atomically: readers see the old or the new one).
    fn put(&self, key: &str, value: &[u8]) -> Result<()>;
    /// Writes only if the key does not exist; false if it did.
    fn create(&self, key: &str, value: &[u8]) -> Result<bool>;
    fn delete(&self, key: &str) -> Result<()>;
    /// Keys starting with `prefix`, sorted, strictly after `after`, at most `limit`.
    fn list(&self, prefix: &str, after: Option<&str>, limit: usize) -> Result<Vec<String>>;

    fn get_string(&self, key: &str) -> Result<Option<String>> {
        Ok(self.get(key)?.map(|b| String::from_utf8_lossy(&b).into_owned()))
    }
}

pub fn check_key(key: &str) -> Result<()> {
    if key.is_empty() || key.starts_with('/') || key.ends_with('/') {
        bail!("bad key {key:?}");
    }
    if !key.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.' | ':' | '=')) {
        bail!("bad key {key:?}: only [A-Za-z0-9/_.:=-]");
    }
    if key.split('/').any(|s| s.is_empty() || s == "." || s == ".." || s.starts_with(".tmp-")) {
        bail!("bad key {key:?}");
    }
    Ok(())
}

/// One file per key under a root directory.
pub struct FsKv {
    root: PathBuf,
}

impl FsKv {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(FsKv { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, key: &str) -> Result<PathBuf> {
        check_key(key)?;
        Ok(self.root.join(key))
    }

    fn temp_for(path: &Path) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let name = format!(".tmp-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
        path.with_file_name(name)
    }

    fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) -> Result<()> {
        let Ok(read) = std::fs::read_dir(dir) else { return Ok(()) };
        for entry in read {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                continue;
            }
            let key = if rel.is_empty() { name } else { format!("{rel}/{name}") };
            if entry.file_type()?.is_dir() {
                Self::walk(&entry.path(), &key, out)?;
            } else {
                out.push(key);
            }
        }
        Ok(())
    }
}

impl Kv for FsKv {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path(key)?) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.path(key)?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = Self::temp_for(&path);
        std::fs::write(&tmp, value)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn create(&self, key: &str, value: &[u8]) -> Result<bool> {
        let path = self.path(key)?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = Self::temp_for(&path);
        std::fs::write(&tmp, value)?;
        // hard_link fails if the target exists and makes the file appear complete: a create-if-absent
        // that never exposes a half-written value.
        let made = match std::fs::hard_link(&tmp, &path) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e.into());
            }
        };
        let _ = std::fs::remove_file(&tmp);
        Ok(made)
    }

    fn delete(&self, key: &str) -> Result<()> {
        match std::fs::remove_file(self.path(key)?) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn list(&self, prefix: &str, after: Option<&str>, limit: usize) -> Result<Vec<String>> {
        if !prefix.is_empty() {
            check_key(prefix.trim_end_matches('/')).or_else(|_| check_key(&format!("{prefix}x")))?;
        }
        // Walk the deepest directory the prefix names, then filter on the full prefix.
        let dir_part = match prefix.rfind('/') {
            Some(i) => &prefix[..i],
            None => "",
        };
        let mut keys = Vec::new();
        Self::walk(&self.root.join(dir_part), dir_part, &mut keys)?;
        keys.retain(|k| k.starts_with(prefix) && after.is_none_or(|a| k.as_str() > a));
        keys.sort();
        keys.truncate(limit);
        Ok(keys)
    }
}

/// In memory, for tests and for building an index from a snapshot.
#[derive(Default)]
pub struct MemKv {
    map: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl MemKv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Kv for MemKv {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        check_key(key)?;
        Ok(self.map.lock().unwrap().get(key).cloned())
    }
    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        check_key(key)?;
        self.map.lock().unwrap().insert(key.into(), value.to_vec());
        Ok(())
    }
    fn create(&self, key: &str, value: &[u8]) -> Result<bool> {
        check_key(key)?;
        let mut map = self.map.lock().unwrap();
        if map.contains_key(key) {
            return Ok(false);
        }
        map.insert(key.into(), value.to_vec());
        Ok(true)
    }
    fn delete(&self, key: &str) -> Result<()> {
        check_key(key)?;
        self.map.lock().unwrap().remove(key);
        Ok(())
    }
    fn list(&self, prefix: &str, after: Option<&str>, limit: usize) -> Result<Vec<String>> {
        let map = self.map.lock().unwrap();
        Ok(map
            .range::<str, _>((std::ops::Bound::Included(prefix), std::ops::Bound::Unbounded))
            .map(|(k, _)| k)
            .take_while(|k| k.starts_with(prefix))
            .filter(|k| after.is_none_or(|a| k.as_str() > a))
            .take(limit)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn exercise(kv: &dyn Kv) {
        assert_eq!(kv.get("a/b").unwrap(), None);
        kv.put("a/b", b"1").unwrap();
        kv.put("a/c", b"2").unwrap();
        kv.put("z", b"3").unwrap();
        assert_eq!(kv.get("a/b").unwrap().unwrap(), b"1");
        assert!(!kv.create("a/b", b"x").unwrap());
        assert!(kv.create("a/d", b"4").unwrap());
        assert_eq!(kv.get("a/d").unwrap().unwrap(), b"4");
        assert_eq!(kv.list("a/", None, 10).unwrap(), ["a/b", "a/c", "a/d"]);
        assert_eq!(kv.list("a/", Some("a/b"), 10).unwrap(), ["a/c", "a/d"]);
        assert_eq!(kv.list("a/", None, 2).unwrap(), ["a/b", "a/c"]);
        assert_eq!(kv.list("", None, 10).unwrap().len(), 4);
        assert_eq!(kv.list("a/c", None, 10).unwrap(), ["a/c"]);
        kv.delete("a/b").unwrap();
        kv.delete("a/b").unwrap();
        assert_eq!(kv.get("a/b").unwrap(), None);
        assert!(kv.put("../x", b"").is_err());
        assert!(kv.put("/x", b"").is_err());
        assert!(kv.put("a//b", b"").is_err());
    }

    #[test]
    fn mem() {
        exercise(&MemKv::new());
    }

    #[test]
    fn fs() {
        let dir = std::env::temp_dir().join(format!("evolve-kv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        exercise(&FsKv::open(&dir).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
