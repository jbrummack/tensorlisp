//! A persistent cache for compiled Tree kernels.
//!
//! The Tree compiler (Scheme) turns a kernel expression like `(k:matmul-cm 'q8_0 64 64 16 4 2 #f #t)`
//! into SPIR-V, PTX or MSL; the result depends only on the expression, the optimization passes and
//! the compiler's own source. It is stored under a key made of those, in one file per kernel in
//! `$TL_KERNEL_CACHE` (default `$XDG_CACHE_HOME/tensorlisp/kernels`, else `~/.cache/tensorlisp/kernels`;
//! `TL_KERNEL_CACHE=off` disables it), so a new process skips running the compiler for kernels any
//! earlier process built. Driver-level caches (Metal's, the CUDA JIT's, Vulkan pipeline caches) are
//! separate and keep working on top of this.
//!
//! Files are `TLKC1`, the key (so a hash collision or a stale file is detected rather than trusted)
//! and the encoded value; they are written to a temporary name and renamed, so concurrent processes
//! can share the directory.
use std::{
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use chez::Value;

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static HIT_NANOS: AtomicU64 = AtomicU64::new(0);
static MISS_NANOS: AtomicU64 = AtomicU64::new(0);

/// What the cache did in this process: kernels found, kernels compiled, and the time spent on each.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KernelCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Reading and decoding the hits.
    pub hit_ms: f64,
    /// Running the Tree compiler for the misses (and storing the result).
    pub miss_ms: f64,
}

pub fn kernel_cache_stats() -> KernelCacheStats {
    KernelCacheStats {
        hits: HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        hit_ms: HIT_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        miss_ms: MISS_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
    }
}

/// The cache directory, or `None` when disabled.
fn directory() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| match std::env::var("TL_KERNEL_CACHE") {
        Ok(v) if matches!(v.as_str(), "" | "0" | "off" | "none") => None,
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => std::env::var_os("XDG_CACHE_HOME")
            .map(|d| PathBuf::from(d).join("tensorlisp/kernels"))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/tensorlisp/kernels"))),
    })
    .as_deref()
}

/// Identifies the compiler: the Tree stdlib sources and this crate's version.
fn fingerprint() -> &'static str {
    static FP: OnceLock<String> = OnceLock::new();
    FP.get_or_init(|| {
        let mut h = Fnv::new();
        h.write(env!("CARGO_PKG_VERSION").as_bytes());
        for (name, source) in crate::scheme::STDLIB {
            if name.starts_with("tree") {
                h.write(name.as_bytes());
                h.write(source.as_bytes());
            }
        }
        h.hex()
    })
}

/// Returns the cached value for `src` compiled for `target` ("msl", "ptx", "spirv"), or runs `compile`
/// and stores its result. Values that can't be stored are returned uncached.
pub(crate) fn cached<E>(target: &str, src: &str, compile: impl FnOnce() -> Result<Value, E>) -> Result<Value, E> {
    let Some(dir) = directory() else { return compile() };
    cached_in(dir, &format!("{target}\n{}\n{src}", fingerprint()), compile)
}

fn cached_in<E>(dir: &Path, key: &str, compile: impl FnOnce() -> Result<Value, E>) -> Result<Value, E> {
    let t0 = std::time::Instant::now();
    let mut h = Fnv::new();
    h.write(key.as_bytes());
    let path = dir.join(format!("{}.kc", h.hex()));
    if let Some(value) = load(&path, key) {
        HITS.fetch_add(1, Ordering::Relaxed);
        HIT_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(value);
    }
    let value = compile()?;
    store(dir, &path, key, &value);
    MISSES.fetch_add(1, Ordering::Relaxed);
    MISS_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(value)
}

fn load(path: &Path, key: &str) -> Option<Value> {
    let bytes = std::fs::read(path).ok()?;
    let rest = bytes.strip_prefix(b"TLKC1")?;
    let (len, rest) = rest.split_first_chunk::<4>()?;
    let len = u32::from_le_bytes(*len) as usize;
    if rest.len() < len || &rest[..len] != key.as_bytes() {
        return None;
    }
    let mut r = &rest[len..];
    let value = decode(&mut r)?;
    r.is_empty().then_some(value)
}

fn store(dir: &Path, path: &Path, key: &str, value: &Value) {
    let mut body = Vec::new();
    if !encode(value, &mut body) || std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut file = b"TLKC1".to_vec();
    file.extend((key.len() as u32).to_le_bytes());
    file.extend(key.as_bytes());
    file.extend(body);
    let tmp = dir.join(format!(".{}.{}.tmp", std::process::id(), path.file_name().and_then(|n| n.to_str()).unwrap_or("kc")));
    if std::fs::write(&tmp, file).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn put_len(out: &mut Vec<u8>, n: usize) {
    out.extend((n as u64).to_le_bytes());
}

/// Appends `v`; false if it holds something without a representation (a procedure, ...).
fn encode(v: &Value, out: &mut Vec<u8>) -> bool {
    match v {
        Value::Void => out.push(0),
        Value::Nil => out.push(1),
        Value::Eof => out.push(2),
        Value::Bool(b) => out.extend([3, *b as u8]),
        Value::Int(i) => {
            out.push(4);
            out.extend(i.to_le_bytes());
        }
        Value::Float(f) => {
            out.push(5);
            out.extend(f.to_bits().to_le_bytes());
        }
        Value::Char(c) => {
            out.push(6);
            out.extend((*c as u32).to_le_bytes());
        }
        Value::String(s) | Value::Symbol(s) => {
            out.push(if matches!(v, Value::String(_)) { 7 } else { 8 });
            put_len(out, s.len());
            out.extend(s.as_bytes());
        }
        Value::List(items) | Value::Vector(items) => {
            out.push(if matches!(v, Value::List(_)) { 9 } else { 11 });
            put_len(out, items.len());
            return items.iter().all(|i| encode(i, out));
        }
        Value::Pair(a, b) => {
            out.push(10);
            return encode(a, out) && encode(b, out);
        }
        Value::Bytevector(b) => {
            out.push(12);
            put_len(out, b.len());
            out.extend(b);
        }
        Value::Opaque(_) => return false,
    }
    true
}

fn take<'a>(r: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    (r.len() >= n).then(|| {
        let (head, tail) = r.split_at(n);
        *r = tail;
        head
    })
}

fn take_len(r: &mut &[u8]) -> Option<usize> {
    let n = u64::from_le_bytes(take(r, 8)?.try_into().ok()?) as usize;
    // A length can't exceed the bytes left (each element takes at least one).
    (n <= r.len()).then_some(n)
}

fn decode(r: &mut &[u8]) -> Option<Value> {
    let tag = take(r, 1)?[0];
    Some(match tag {
        0 => Value::Void,
        1 => Value::Nil,
        2 => Value::Eof,
        3 => Value::Bool(take(r, 1)?[0] != 0),
        4 => Value::Int(i64::from_le_bytes(take(r, 8)?.try_into().ok()?)),
        5 => Value::Float(f64::from_bits(u64::from_le_bytes(take(r, 8)?.try_into().ok()?))),
        6 => Value::Char(char::from_u32(u32::from_le_bytes(take(r, 4)?.try_into().ok()?))?),
        7 | 8 => {
            let n = take_len(r)?;
            let s = String::from_utf8(take(r, n)?.to_vec()).ok()?;
            if tag == 7 { Value::String(s) } else { Value::Symbol(s) }
        }
        9 | 11 => {
            let n = take_len(r)?;
            let items = (0..n).map(|_| decode(r)).collect::<Option<Vec<_>>>()?;
            if tag == 9 { Value::List(items) } else { Value::Vector(items) }
        }
        10 => Value::Pair(Box::new(decode(r)?), Box::new(decode(r)?)),
        12 => {
            let n = take_len(r)?;
            Value::Bytevector(take(r, n)?.to_vec())
        }
        _ => return None,
    })
}

/// 128-bit FNV-1a: stable across Rust versions (unlike `DefaultHasher`), good enough for file names
/// because the key is stored in the file and compared on load.
struct Fnv(u128);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0x6c62272e07bb014262b821756295c58d)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u128;
            self.0 = self.0.wrapping_mul(0x0000000001000000000000000000013b);
        }
    }

    fn hex(&self) -> String {
        format!("{:032x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        Value::List(vec![
            Value::String("#include <metal_stdlib>\nkernel void k() {}".into()),
            Value::String("k".into()),
            Value::List(vec![Value::Symbol("ro".into()), Value::Symbol("wo".into())]),
            Value::Int(16),
            Value::Vector(vec![Value::Int(32), Value::Float(1.5), Value::Bool(true)]),
            Value::Pair(Box::new(Value::Int(1)), Box::new(Value::Char('x'))),
            Value::Bytevector(vec![1, 2, 3]),
            Value::Nil,
            Value::Void,
        ])
    }

    #[test]
    fn values_round_trip() {
        let mut bytes = Vec::new();
        assert!(encode(&sample(), &mut bytes));
        let mut r = &bytes[..];
        assert_eq!(decode(&mut r), Some(sample()));
        assert!(r.is_empty());
        // Truncated or garbled data is rejected, not misread.
        for cut in [1, bytes.len() / 2, bytes.len() - 1] {
            assert_eq!(decode(&mut &bytes[..cut]), None, "cut at {cut}");
        }
        assert!(!encode(&Value::List(vec![Value::Opaque("procedure")]), &mut Vec::new()));
    }

    #[test]
    fn second_lookup_is_a_hit_and_stale_files_are_ignored() {
        let dir = std::env::temp_dir().join(format!("tl-kernel-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut compiled = 0;
        let mut get = |key: &str| {
            cached_in::<()>(&dir, key, || {
                compiled += 1;
                Ok(sample())
            })
            .unwrap()
        };
        assert_eq!(get("k1"), sample());
        assert_eq!(get("k1"), sample());
        assert_eq!(get("k2"), sample());
        drop(get);
        assert_eq!(compiled, 2, "k1 was compiled once, k2 once");

        // A file with the wrong key (a collision, or an older compiler) counts as a miss and is replaced.
        let mut h = Fnv::new();
        h.write(b"k1");
        let path = dir.join(format!("{}.kc", h.hex()));
        let mut bad = b"TLKC1".to_vec();
        bad.extend(2u32.to_le_bytes());
        bad.extend(b"zz");
        std::fs::write(&path, bad).unwrap();
        assert!(load(&path, "k1").is_none());
        let mut again = 0;
        cached_in::<()>(&dir, "k1", || {
            again += 1;
            Ok(sample())
        })
        .unwrap();
        assert_eq!(again, 1);
        assert_eq!(load(&path, "k1"), Some(sample()));

        // Errors from the compiler are returned and nothing is stored.
        assert_eq!(cached_in(&dir, "bad", || Err("boom")), Err("boom"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
