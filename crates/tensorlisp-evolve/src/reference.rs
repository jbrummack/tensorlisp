//! Kernels that already exist elsewhere in the repository, for agents to learn from.
//!
//! * ggml's Vulkan (GLSL), CUDA and Metal kernels, vendored in `vendor/ggml`,
//! * the paged-attention / reshape-and-cache kernels from vLLM (via mistral.rs) in `crates/kernels`,
//! * this project's own native CUDA kernels.
//!
//! [`kernels_for`] maps a Tree generator expression to the files that implement the same operation,
//! [`RefIndex`] is a keyword index over those sources in windows of lines (built on first use, in RAM).
//! Embedding search over the same windows is `CodeIndex::add_reference` (see `embed.rs`).
use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::Regex;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
use tantivy::schema::{IndexRecordOption, STORED, STRING, Schema, TEXT, Value};
use tantivy::snippet::SnippetGenerator;
use tantivy::{Index, TantivyDocument, Term, doc};

/// (backend label, directory, extensions, recurse)
pub const ROOTS: &[(&str, &str, &[&str], bool)] = &[
    ("ggml-vulkan", "vendor/ggml/src/ggml-vulkan/vulkan-shaders", &["comp", "glsl", "glslh"], false),
    ("ggml-vulkan-host", "vendor/ggml/src/ggml-vulkan", &["cpp"], false),
    ("ggml-cuda", "vendor/ggml/src/ggml-cuda", &["cu", "cuh"], true),
    ("ggml-metal", "vendor/ggml/src/ggml-metal", &["metal", "cpp"], false),
    ("vllm-paged-attn-cuda", "crates/kernels/src/cuda/shaders", &["cu", "cuh"], false),
    ("vllm-paged-attn-metal", "crates/kernels/src/metal/shaders", &["metal"], false),
    ("native-cuda", "crates/kernels/src/cuda/native", &["cu"], false),
];

/// Directories that are generated instances of the same kernels: indexing them would drown the originals.
const SKIP: &[&str] = &["template-instances", "vendors"];

#[derive(Debug, Clone)]
pub struct Source {
    pub backend: &'static str,
    /// Relative to the repository root.
    pub rel: String,
    pub path: PathBuf,
}

pub fn catalog(repo: &Path) -> Vec<Source> {
    fn walk(dir: &Path, exts: &[&str], recurse: bool, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if recurse && !SKIP.contains(&name.as_str()) {
                    walk(&path, exts, recurse, out);
                }
            } else if path.extension().is_some_and(|x| exts.contains(&x.to_string_lossy().as_ref())) {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    for (backend, dir, exts, recurse) in ROOTS {
        let mut files = Vec::new();
        walk(&repo.join(dir), exts, *recurse, &mut files);
        for path in files {
            let rel = path.strip_prefix(repo).unwrap_or(&path).display().to_string();
            out.push(Source { backend, rel, path });
        }
    }
    out
}

/// Names a Tree generator is known by elsewhere: file-name stems and kernel-name fragments.
fn stems(expr: &str) -> Vec<String> {
    let name = Regex::new(r"\(k:([a-z0-9-]+)").unwrap().captures(expr).map(|c| c[1].to_string()).unwrap_or_else(|| expr.trim_matches(|c| c == '(' || c == ')').trim_start_matches("k:").to_string());
    let first_sym = Regex::new(r"'([a-z0-9_]+)").unwrap().captures(expr).map(|c| c[1].to_string());
    let table: &[(&str, &[&str])] = &[
        ("matvec-q8r", &["mul_mat_vecq", "mmvq", "quantize"]),
        ("matvec-q8", &["mul_mat_vecq", "mmvq", "quantize"]),
        ("matvec", &["mul_mat_vec", "mmvq", "mmv", "mul_mv"]),
        ("gateup-q8", &["mul_mat_vecq", "mmvq", "swiglu", "glu"]),
        ("matmul-cm", &["mul_mm_cm", "mul_mm", "mma"]),
        ("matmul", &["mul_mm", "mmq", "mma"]),
        ("flash-attn", &["flash_attn", "fattn", "pagedattention", "paged_attention"]),
        ("rms-norm", &["rms_norm", "norm"]),
        ("norm-rope", &["rms_norm", "rope", "norm"]),
        ("rowchain", &["rms_norm", "norm", "add"]),
        ("norm", &["norm"]),
        ("soft-max", &["soft_max", "softmax"]),
        ("rope", &["rope"]),
        ("get-rows", &["get_rows", "getrows"]),
        ("set-rows", &["set_rows", "set-rows"]),
        ("sum-rows", &["sum_rows", "sumrows"]),
        ("quantize-q8", &["quantize"]),
        ("repack-q8", &["quantize", "repack"]),
        ("concat", &["concat"]),
        ("arange", &["arange"]),
        ("argmax", &["argmax"]),
        ("ew-q8", &["unary", "add", "mul"]),
        ("ew", &["unary", "add", "mul"]),
    ];
    let mut out: Vec<String> = Vec::new();
    if matches!(name.as_str(), "binary" | "unary") {
        if let Some(s) = &first_sym {
            out.push(s.clone());
            if s == "copy" {
                out.extend(["cpy".into(), "copy".into(), "dup".into()]);
            }
        }
        out.push(if name == "binary" { "add".into() } else { "unary".into() });
    }
    // longest table key that prefixes the generator name
    if let Some((_, v)) = table.iter().filter(|(k, _)| name.starts_with(k)).max_by_key(|(k, _)| k.len()) {
        out.extend(v.iter().map(|s| s.to_string()));
    }
    if out.is_empty() {
        out.push(name.replace('-', "_"));
    }
    out.dedup();
    out
}

/// Where the same operation is implemented in ggml and in the vLLM-derived kernels.
pub fn kernels_for(repo: &Path, expr: &str) -> Result<String> {
    let stems = stems(expr);
    let sources = catalog(repo);
    if sources.is_empty() {
        return Ok("no reference sources found (is vendor/ggml checked out?)".into());
    }
    // data types named in the expression (`'q8_0`, `'f16`): declarations and files mentioning them come first
    let types: Vec<String> = Regex::new(r"'(q[0-9]_[0-9a-z]+|f16|f32|bf16|i32)\b").unwrap().captures_iter(expr).map(|c| c[1].to_lowercase()).collect();
    let mentions = |t: &str| types.iter().any(|ty| t.to_lowercase().contains(ty.as_str()));
    let mut out = format!("reference kernels for {expr} (looked for: {}{})\n", stems.join(", "), if types.is_empty() { String::new() } else { format!("; types first: {}", types.join(", ")) });
    let mut any = false;
    for (backend, ..) in ROOTS {
        let mut files: Vec<&Source> = sources.iter().filter(|s| s.backend == *backend).filter(|s| {
            let stem = Path::new(&s.rel).file_stem().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
            stems.iter().any(|st| stem.contains(&st.to_lowercase()))
        }).collect();
        files.sort_by_key(|s| (!mentions(&s.rel), s.rel.len()));
        let mut lines = Vec::new();
        for s in files.iter().take(8) {
            let n = std::fs::read_to_string(&s.path).map(|t| t.lines().count()).unwrap_or(0);
            lines.push(format!("  {} ({n} lines)", s.rel));
        }
        // big files that hold many kernels: list the declarations that mention the operation
        let pattern = format!(r"(?i)\b(kernel_|ggml_vk_|ggml_metal_op_|k_)\w*({})", stems.iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|"));
        let re = Regex::new(&pattern)?;
        for s in sources.iter().filter(|s| s.backend == *backend && (s.rel.ends_with(".metal") || s.rel.ends_with(".cpp"))) {
            let Ok(text) = std::fs::read_to_string(&s.path) else { continue };
            let mut found: Vec<(usize, &str)> = text.lines().enumerate().filter(|(_, l)| re.is_match(l) && (l.contains("kernel") || l.contains("static") || l.contains("void") || l.contains("__global__"))).collect();
            found.sort_by_key(|(i, l)| (!mentions(l), *i));
            for (i, l) in found.into_iter().take(6) {
                lines.push(format!("  {}:{}: {}", s.rel, i + 1, l.trim().chars().take(110).collect::<String>()));
            }
        }
        if !lines.is_empty() {
            any = true;
            out.push_str(&format!("{backend}:\n{}\n", lines.join("\n")));
        }
    }
    if !any {
        out.push_str("no file or declaration matched; try reference_search with keywords\n");
    }
    Ok(out)
}

const WINDOW: usize = 70;
const STRIDE: usize = 55;

/// Windows of lines: (first line, last line, text), 1-based inclusive.
pub fn windows(text: &str) -> Vec<(usize, usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + WINDOW).min(lines.len());
        out.push((start + 1, end, lines[start..end].join("\n")));
        if end == lines.len() {
            break;
        }
        start += STRIDE;
    }
    out
}

#[derive(Debug, Clone)]
pub struct RefHit {
    pub backend: String,
    pub path: String,
    pub first: u64,
    pub last: u64,
    pub score: f32,
    pub snippet: String,
}

/// Keyword search over the reference sources, in RAM, built from the catalog.
pub struct RefIndex {
    index: Index,
    reader: tantivy::IndexReader,
    path: tantivy::schema::Field,
    backend: tantivy::schema::Field,
    first: tantivy::schema::Field,
    last: tantivy::schema::Field,
    body: tantivy::schema::Field,
    name: tantivy::schema::Field,
    pub chunks: usize,
}

impl RefIndex {
    pub fn build(repo: &Path) -> Result<Self> {
        let mut sb = Schema::builder();
        let path = sb.add_text_field("path", STRING | STORED);
        let backend = sb.add_text_field("backend", STRING | STORED);
        let first = sb.add_u64_field("first", STORED);
        let last = sb.add_u64_field("last", STORED);
        let name = sb.add_text_field("name", TEXT);
        let body = sb.add_text_field("body", TEXT | STORED);
        let index = Index::create_in_ram(sb.build());
        let mut writer = index.writer(50_000_000)?;
        let mut chunks = 0;
        for s in catalog(repo) {
            let Ok(text) = std::fs::read_to_string(&s.path) else { continue };
            let file_name = Path::new(&s.rel).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
            for (a, b, w) in windows(&text) {
                writer.add_document(doc!(path => s.rel.clone(), backend => s.backend, first => a as u64, last => b as u64, name => file_name.clone(), body => w))?;
                chunks += 1;
            }
        }
        writer.commit()?;
        let reader = index.reader()?;
        Ok(RefIndex { index, reader, path, backend, first, last, body, name, chunks })
    }

    /// `backend` restricts to labels containing it (`vulkan`, `cuda`, `metal`, `vllm`, ...).
    pub fn search(&self, query: &str, backend: Option<&str>, limit: usize) -> Result<Vec<RefHit>> {
        let mut parser = QueryParser::for_index(&self.index, vec![self.name, self.body]);
        parser.set_field_boost(self.name, 3.0);
        let (q, _) = parser.parse_query_lenient(query);
        let q: Box<dyn Query> = match backend {
            Some(b) => {
                let labels: Vec<&str> = ROOTS.iter().map(|r| r.0).filter(|l| l.contains(b)).collect();
                let any: Vec<(Occur, Box<dyn Query>)> = labels.iter().map(|l| (Occur::Should, Box::new(TermQuery::new(Term::from_field_text(self.backend, l), IndexRecordOption::Basic)) as Box<dyn Query>)).collect();
                if any.is_empty() {
                    anyhow::bail!("no backend matches {b:?}; backends: {}", ROOTS.iter().map(|r| r.0).collect::<Vec<_>>().join(", "));
                }
                Box::new(BooleanQuery::new(vec![(Occur::Must, q), (Occur::Must, Box::new(BooleanQuery::new(any)))]))
            }
            None => q,
        };
        let searcher = self.reader.searcher();
        let top = searcher.search(&q, &TopDocs::with_limit(limit.max(1)).order_by_score())?;
        let snip = SnippetGenerator::create(&searcher, &*q, self.body).ok();
        let mut out = Vec::new();
        for (score, addr) in top {
            let d: TantivyDocument = searcher.doc(addr)?;
            let text = |f| d.get_first(f).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let num = |f| d.get_first(f).and_then(|v| v.as_u64()).unwrap_or(0);
            let body = text(self.body);
            let mut snippet = snip.as_ref().map(|g| g.snippet(&body).fragment().to_string()).unwrap_or_default();
            if snippet.trim().is_empty() {
                snippet = body.chars().take(160).collect();
            }
            out.push(RefHit { backend: text(self.backend), path: text(self.path), first: num(self.first), last: num(self.last), score, snippet: snippet.replace('\n', " ⏎ ") });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_repo(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("evolve-ref-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let w = |p: &str, t: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, t).unwrap();
        };
        w("vendor/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp", "#version 450\nlayout(local_size_x = 32) in;\nvoid main() { float acc = subgroupAdd(acc); }\n");
        w("vendor/ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp", "void main() { float s = inversesqrt(mean + eps); }\n");
        w("vendor/ggml/src/ggml-cuda/mmvq.cu", "template <int ncols> __global__ void mul_mat_vec_q(const void * vx) { __shfl_xor_sync(0xffffffff, sum, 16); }\n");
        w("vendor/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0.cu", "// generated instance\n");
        w("vendor/ggml/src/ggml-metal/ggml-metal.metal", "kernel void kernel_mul_mv_q8_0_f32(device const char * src0) {}\nkernel void kernel_rms_norm_f32() {}\n");
        w("crates/kernels/src/cuda/shaders/pagedattention.cuh", "// vLLM paged attention\n__device__ void paged_attention_kernel() { __syncthreads(); }\n");
        root
    }

    #[test]
    fn maps_generators_to_the_files_that_implement_them() {
        let repo = fake_repo("map");
        let out = kernels_for(&repo, "(k:matvec 'q8_0 1)").unwrap();
        assert!(out.contains("ggml-vulkan:") && out.contains("vulkan-shaders/mul_mat_vec.comp (3 lines)"), "{out}");
        assert!(out.contains("ggml-cuda:") && out.contains("mmvq.cu"), "{out}");
        assert!(out.contains("ggml-metal:") && out.contains("ggml-metal.metal:1: kernel void kernel_mul_mv_q8_0_f32"), "{out}");
        assert!(!out.contains("template-instances"));
        let out = kernels_for(&repo, "(k:rms-norm #t #f)").unwrap();
        assert!(out.contains("rms_norm.comp") && out.contains("kernel_rms_norm_f32"), "{out}");
        // flash attention also points at the vLLM-derived paged attention
        assert!(kernels_for(&repo, "(k:flash-attn-part 'f16 'f16 #t #f 128 256 256)").unwrap().contains("vllm-paged-attn-cuda"));
        assert!(kernels_for(&repo, "(k:binary 'zzz)").unwrap().contains("no file or declaration matched"));
        assert!(kernels_for(Path::new("/nonexistent"), "(k:binary 'add)").unwrap().contains("no reference sources"));
    }

    #[test]
    fn the_data_type_in_the_expression_ranks_declarations_first() {
        let repo = fake_repo("types");
        let f = repo.join("vendor/ggml/src/ggml-metal/ggml-metal.metal");
        std::fs::write(&f, "kernel void kernel_mul_mv_q1_0_f32() {}\nkernel void kernel_mul_mv_q2_0_f32() {}\nkernel void kernel_mul_mv_q8_0_f32() {}\n").unwrap();
        let out = kernels_for(&repo, "(k:matvec 'q8_0 1)").unwrap();
        let first_decl = out.lines().find(|l| l.contains("ggml-metal.metal:")).unwrap();
        assert!(first_decl.contains("kernel_mul_mv_q8_0_f32"), "{out}");
        assert!(out.contains("types first: q8_0"));
    }

    #[test]
    fn keyword_search_with_backend_filter() {
        let repo = fake_repo("search");
        let idx = RefIndex::build(&repo).unwrap();
        assert!(idx.chunks >= 5);
        let hits = idx.search("subgroupAdd", None, 5).unwrap();
        assert_eq!(hits[0].path, "vendor/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp");
        assert_eq!((hits[0].first, hits[0].last), (1, 3));
        let hits = idx.search("sync", Some("cuda"), 5).unwrap();
        assert!(hits.is_empty() || hits.iter().all(|h| h.backend.contains("cuda")));
        let hits = idx.search("__syncthreads", Some("vllm"), 5).unwrap();
        assert!(hits.iter().all(|h| h.backend == "vllm-paged-attn-cuda"), "{hits:?}");
        assert!(idx.search("anything", Some("opencl"), 5).is_err());
        // file names count: searching the kernel name finds the file
        assert!(idx.search("mul_mat_vec", None, 3).unwrap().iter().any(|h| h.path.ends_with("mul_mat_vec.comp")));
    }

    #[test]
    fn windows_overlap_and_cover() {
        let text: String = (1..=200).map(|i| format!("line {i}\n")).collect();
        let w = windows(&text);
        assert_eq!((w[0].0, w[0].1), (1, 70));
        assert_eq!(w[1].0, 56);
        assert_eq!(w.last().unwrap().1, 200);
        assert!(windows("").is_empty());
    }

    #[test]
    fn the_real_repository_has_the_references() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !repo.join("vendor/ggml/src/ggml-vulkan").exists() {
            return;
        }
        let out = kernels_for(&repo, "(k:matvec 'q8_0 1)").unwrap();
        assert!(out.contains("mul_mat_vec") && out.contains("ggml-cuda:") && out.contains("ggml-metal:"), "{out}");
        let out = kernels_for(&repo, "(k:flash-attn 'f16 'f16 #t #f 2048)").unwrap();
        assert!(out.contains("flash_attn") && out.contains("vllm-paged-attn"), "{out}");
    }
}
