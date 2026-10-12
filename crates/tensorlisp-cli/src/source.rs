//! Where safetensors bytes come from: a local (mmap'd) file, or an HTTP(S)
//! URL read with Range requests, so a checkpoint can be converted while it
//! downloads and never exists on disk. `hf:ORG/REPO[@REV]` expands to every
//! `.safetensors` file of a Hugging Face repo.

use std::{borrow::Cow, fs::File, io::Read, path::Path, time::Duration};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::Dtype;
use serde::Deserialize;

const HF: &str = "https://huggingface.co";
const RETRIES: u32 = 6;

pub enum Source {
    Local(Mmap),
    Remote { agent: ureq::Agent, url: String, token: Option<String> },
}

/// One tensor of a safetensors file; `start..end` are absolute byte offsets in the file.
pub struct TensorEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

pub fn is_remote(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://") || input.starts_with("hf:")
}

fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        return Some(t.trim().to_string()).filter(|t| !t.is_empty());
    }
    let home = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache/huggingface")))?;
    std::fs::read_to_string(home.join("token")).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_recv_body(Some(Duration::from_secs(120))).build().into()
}

/// Expands `hf:ORG/REPO[@REV]` into resolve URLs of its .safetensors files.
fn expand_hf(spec: &str, token: Option<&str>) -> Result<Vec<String>> {
    let spec = spec.trim_start_matches("hf:").trim_start_matches("//");
    let (repo, rev) = spec.split_once('@').unwrap_or((spec, "main"));
    let api = format!("{HF}/api/models/{repo}/revision/{rev}");
    let mut req = agent().get(&api);
    if let Some(t) = token {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let body = req
        .call()
        .with_context(|| format!("listing {repo}@{rev} (gated models need HF_TOKEN)"))?
        .body_mut()
        .read_to_string()?;
    let info: serde_json::Value = serde_json::from_str(&body)?;
    let mut files: Vec<String> = info["siblings"]
        .as_array()
        .map(|s| s.iter().filter_map(|f| f["rfilename"].as_str()).filter(|f| f.ends_with(".safetensors")).map(String::from).collect())
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors files in {repo}@{rev}");
    }
    Ok(files.into_iter().map(|f| format!("{HF}/{repo}/resolve/{rev}/{f}")).collect())
}

/// Resolves command-line inputs to sources: local files and directories, URLs and `hf:` repos.
pub fn open_all(inputs: &[String]) -> Result<Vec<(String, Source, Vec<TensorEntry>)>> {
    let token = hf_token();
    let mut out = Vec::new();
    for input in inputs {
        if is_remote(input) {
            let urls = if input.starts_with("hf:") { expand_hf(input, token.as_deref())? } else { vec![input.clone()] };
            for url in urls {
                // The token only goes to Hugging Face; ureq does not forward it across redirects.
                let token = url.starts_with(HF).then(|| token.clone()).flatten();
                let source = Source::Remote { agent: agent(), url: url.clone(), token };
                let entries = source.header().with_context(|| format!("reading the header of {url}"))?;
                out.push((url, source, entries));
            }
            continue;
        }
        let path = Path::new(input);
        let files = if path.is_dir() {
            let mut found: Vec<_> = std::fs::read_dir(path)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            found.sort();
            if found.is_empty() {
                bail!("no .safetensors files in {}", path.display());
            }
            found
        } else {
            vec![path.to_path_buf()]
        };
        for f in files {
            let file = File::open(&f).with_context(|| format!("opening {}", f.display()))?;
            let source = Source::Local(unsafe { Mmap::map(&file)? });
            let entries = source.header().with_context(|| format!("reading {}", f.display()))?;
            out.push((f.display().to_string(), source, entries));
        }
    }
    Ok(out)
}

impl Source {
    /// Reads the file's tensor index (the first 8 + N bytes), sorted by data offset.
    fn header(&self) -> Result<Vec<TensorEntry>> {
        let n = u64::from_le_bytes(self.read(0, 8)?.as_ref().try_into().unwrap());
        if n > 100 << 20 {
            bail!("not a safetensors file (header of {n} bytes)");
        }
        let json = self.read(8, 8 + n)?;
        let mut raw: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&json).context("invalid safetensors header")?;
        raw.remove("__metadata__");
        let base = 8 + n;
        let mut entries = Vec::with_capacity(raw.len());
        for (name, value) in raw {
            let t: RawTensor = serde_json::from_value(value).with_context(|| format!("tensor {name}"))?;
            if t.data_offsets.0 > t.data_offsets.1 {
                bail!("tensor {name}: bad data offsets");
            }
            entries.push(TensorEntry { name, dtype: t.dtype, shape: t.shape, start: base + t.data_offsets.0, end: base + t.data_offsets.1 });
        }
        entries.sort_by_key(|e| e.start);
        Ok(entries)
    }

    /// Bytes `start..end` of the file. Remote reads retry with a fresh Range request.
    pub fn read(&self, start: u64, end: u64) -> Result<Cow<'_, [u8]>> {
        match self {
            Source::Local(map) => map
                .get(start as usize..end as usize)
                .map(Cow::Borrowed)
                .with_context(|| format!("bytes {start}..{end} are past the end of the file ({} bytes)", map.len())),
            Source::Remote { agent, url, token } => {
                let mut last = None;
                for attempt in 0..RETRIES {
                    if attempt > 0 {
                        std::thread::sleep(Duration::from_secs(1 << attempt.min(5)));
                    }
                    match fetch(agent, url, token.as_deref(), start, end) {
                        Ok(bytes) => return Ok(Cow::Owned(bytes)),
                        Err(e) => {
                            eprintln!("warning: {url} bytes {start}..{end}: {e:#} (attempt {}/{RETRIES})", attempt + 1);
                            last = Some(e);
                        }
                    }
                }
                Err(last.unwrap()).with_context(|| format!("downloading {url}"))
            }
        }
    }
}

fn fetch(agent: &ureq::Agent, url: &str, token: Option<&str>, start: u64, end: u64) -> Result<Vec<u8>> {
    let len = (end - start) as usize;
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut req = agent.get(url).header("Range", &format!("bytes={start}-{}", end - 1));
    if let Some(t) = token {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let mut resp = req.call()?;
    if resp.status() != 206 {
        bail!("server answered {} to a Range request (it must support byte ranges)", resp.status());
    }
    let mut buf = vec![0u8; len];
    resp.body_mut().as_reader().read_exact(&mut buf).context("connection ended early")?;
    Ok(buf)
}
