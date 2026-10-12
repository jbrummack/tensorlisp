//! The model store: GGUF files named by the SHA-256 of their bytes, plus a JSON alias index.
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    /// alias -> hash
    aliases: BTreeMap<String, String>,
}

/// A file being written into the store; `finish` moves it to its content address.
pub struct Incoming {
    tmp: PathBuf,
    file: File,
    hasher: Sha256,
    pub bytes: u64,
}

impl Incoming {
    pub fn write(&mut self, chunk: &[u8]) -> Result<()> {
        self.hasher.update(chunk);
        self.file.write_all(chunk).context("writing the upload")?;
        self.bytes += chunk.len() as u64;
        Ok(())
    }

    pub fn abort(self) {
        let _ = std::fs::remove_file(&self.tmp);
    }
}

impl Store {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Store> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).with_context(|| format!("creating the model store {}", dir.display()))?;
        let index = dir.join("index.json");
        let aliases = match std::fs::read_to_string(&index) {
            Ok(text) => serde_json::from_str(&text).with_context(|| format!("reading {}", index.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", index.display())),
        };
        Ok(Store { dir, aliases })
    }

    pub fn path(&self, hash: &str) -> PathBuf {
        self.dir.join(format!("{hash}.gguf"))
    }

    fn save(&self) -> Result<()> {
        let tmp = self.dir.join("index.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.aliases)?)?;
        std::fs::rename(tmp, self.dir.join("index.json"))?;
        Ok(())
    }

    /// Hashes of the stored models, sorted.
    pub fn hashes(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.strip_suffix(".gguf").map(str::to_owned))
            .collect();
        out.sort();
        out
    }

    /// The alias of `hash`, or its first 12 digits.
    pub fn name_of(&self, hash: &str) -> String {
        self.aliases.iter().find(|(_, h)| *h == hash).map(|(a, _)| a.clone()).unwrap_or_else(|| hash[..12.min(hash.len())].to_owned())
    }

    /// An alias, a full hash or a unique hash prefix of at least 8 digits.
    pub fn resolve(&self, name: &str) -> Result<String> {
        if let Some(hash) = self.aliases.get(name) {
            return Ok(hash.clone());
        }
        let hashes = self.hashes();
        if hashes.iter().any(|h| h == name) {
            return Ok(name.to_owned());
        }
        if name.len() >= 8 {
            let mut hits = hashes.iter().filter(|h| h.starts_with(name));
            if let (Some(hit), None) = (hits.next(), hits.next()) {
                return Ok(hit.clone());
            }
        }
        bail!("no model {name:?} (installed: {})", self.aliases.keys().cloned().collect::<Vec<_>>().join(", "))
    }

    pub fn begin(&self) -> Result<Incoming> {
        let tmp = self.dir.join(format!("incoming-{}-{}.part", std::process::id(), next_id()));
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        Ok(Incoming { tmp, file, hasher: Sha256::new(), bytes: 0 })
    }

    /// Moves a finished upload to `<hash>.gguf` (dropping it if the model is already there).
    /// Returns the hash and whether the model was new.
    pub fn commit(&self, incoming: Incoming) -> Result<(String, bool)> {
        let Incoming { tmp, mut file, hasher, .. } = incoming;
        file.flush()?;
        drop(file);
        let hash = hex(&hasher.finalize());
        let dest = self.path(&hash);
        if dest.exists() {
            std::fs::remove_file(&tmp)?;
            Ok((hash, false))
        } else {
            std::fs::rename(&tmp, &dest)?;
            Ok((hash, true))
        }
    }

    /// Copies a local file into the store.
    pub fn import(&self, source: &Path) -> Result<(String, bool)> {
        let mut input = File::open(source).with_context(|| format!("opening {}", source.display()))?;
        let mut incoming = self.begin()?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = input.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if let Err(e) = incoming.write(&buf[..n]) {
                incoming.abort();
                return Err(e);
            }
        }
        self.commit(incoming)
    }

    pub fn alias(&mut self, name: &str, hash: &str, replace: bool) -> Result<()> {
        if let Some(old) = self.aliases.get(name)
            && old != hash
            && !replace
        {
            bail!("the name {name:?} already refers to model {}; set `replace` to repoint it", &old[..12.min(old.len())]);
        }
        self.aliases.insert(name.to_owned(), hash.to_owned());
        self.save()
    }

    /// Removes an alias, or a hash with all its aliases. The file goes once no alias refers to
    /// it. Returns the hash and whether the file was deleted.
    pub fn remove(&mut self, name: &str) -> Result<(String, bool)> {
        let hash = self.resolve(name)?;
        if self.aliases.remove(name).is_none() {
            self.aliases.retain(|_, h| *h != hash);
        }
        let delete = !self.aliases.values().any(|h| *h == hash);
        if delete {
            let _ = std::fs::remove_file(self.path(&hash));
        }
        self.save()?;
        Ok((hash, delete))
    }
}

fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_addressing_and_aliases() {
        let dir = std::env::temp_dir().join(format!("tl-store-test-{}", std::process::id()));
        let mut store = Store::open(&dir).unwrap();
        let mut a = store.begin().unwrap();
        a.write(b"hello ").unwrap();
        a.write(b"world").unwrap();
        let (hash, new) = store.commit(a).unwrap();
        assert!(new);
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
        let mut b = store.begin().unwrap();
        b.write(b"hello world").unwrap();
        assert!(!store.commit(b).unwrap().1);

        store.alias("greeter", &hash, false).unwrap();
        assert_eq!(store.resolve("greeter").unwrap(), hash);
        assert_eq!(store.resolve(&hash[..10]).unwrap(), hash);
        assert!(store.alias("greeter", &"0".repeat(64), false).is_err());
        assert!(store.resolve("nope").is_err());
        // The index survives a restart.
        assert_eq!(Store::open(&dir).unwrap().resolve("greeter").unwrap(), hash);

        assert_eq!(store.remove("greeter").unwrap(), (hash.clone(), true));
        assert!(!store.path(&hash).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
