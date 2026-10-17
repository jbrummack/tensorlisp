//! The code around a kernel, for agents that want to know what is going on (`--rust`).
//!
//! The Tree compiler and the passes are Scheme (`stdlib/tree*.ss`). What decides how a kernel is *used*
//! is Rust: which graph nodes become which generator expression, the bindings, push constants and grid
//! each launch gets, what is fused before planning, how launches are grouped between barriers. A pass
//! author needs that (a kernel's push layout and workgroup are not free to change; its inputs may be
//! quantized activations; neighbours may have been fused into it). Three read-only helpers:
//!
//! * [`map`]: the relevant files, grouped, each with its own module documentation (read from the source,
//!   so it cannot go stale),
//! * [`outline`]: the items of a file with line numbers, to slice a large file with `read_file`,
//! * [`launch_sites`]: the Rust functions that emit a given generator expression, and where it is defined.
use std::path::Path;

use anyhow::{Result, bail};
use regex::Regex;

const GROUPS: &[(&str, &[&str])] = &[
    (
        "The Tree compiler and passes (Scheme)",
        &[
            "crates/tensorlisp/src/scheme/stdlib/tree.ss",
            "crates/tensorlisp/src/scheme/stdlib/tree_opt.ss",
            "crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss",
            "crates/tensorlisp/src/scheme/stdlib/tree_spirv.ss",
            "crates/tensorlisp/src/scheme/stdlib/tree_ptx.ss",
            "crates/tensorlisp/src/scheme/stdlib/tree_msl.ss",
        ],
    ),
    (
        "Graph node -> generator expression, bindings, push constants, grid (shared by Vulkan, CUDA and Metal Tree executors)",
        &["crates/kernels/src/vulkan/lower/tree_ops.rs", "crates/kernels/src/vulkan/tree.rs"],
    ),
    ("Graph rewrites before planning (fusion, CSE)", &["crates/kernels/src/vulkan/fuse_tree.rs", "crates/kernels/src/vulkan/fuse.rs"]),
    (
        "Vulkan: executor, device, ggml's precompiled shaders",
        &["crates/kernels/src/vulkan/exec.rs", "crates/kernels/src/vulkan/mod.rs", "crates/kernels/src/vulkan/lower.rs", "crates/kernels/src/vulkan/ggml.rs"],
    ),
    (
        "CUDA: Tree executor (PTX), then the older native executor (own NVRTC kernels), paged attention",
        &[
            "crates/kernels/src/cuda/tree.rs",
            "crates/kernels/src/cuda/backend.rs",
            "crates/kernels/src/cuda/mod.rs",
            "crates/kernels/src/cuda/exec.rs",
            "crates/kernels/src/cuda/lower.rs",
            "crates/kernels/src/cuda/native/core.cu",
            "crates/kernels/src/cuda/native/fa.cu",
            "crates/kernels/src/cuda/native/mm.cu",
            "crates/kernels/src/cuda/paged_attn.rs",
            "crates/kernels/src/cuda/shaders/pagedattention.cuh",
        ],
    ),
    (
        "Metal: Tree executor (MSL), then the older native executor (ggml's .metal kernels), paged attention",
        &[
            "crates/kernels/src/metal/tree.rs",
            "crates/kernels/src/metal/backend.rs",
            "crates/kernels/src/metal/mod.rs",
            "crates/kernels/src/metal/exec.rs",
            "crates/kernels/src/metal/lower.rs",
            "crates/kernels/src/metal/kargs.rs",
            "crates/kernels/src/metal/ggml.rs",
            "crates/kernels/src/metal/paged_attn.rs",
            "crates/kernels/src/metal/shaders/pagedattention.metal",
        ],
    ),
    (
        "Scheduler: hazards, barriers, memory, submission (all backends)",
        &["crates/kernels/src/sched/engine.rs", "crates/kernels/src/sched/hazard.rs", "crates/kernels/src/sched/backend.rs", "crates/kernels/src/sched/spirv.rs", "crates/kernels/src/plan.rs", "crates/kernels/src/graph.rs"],
    ),
    (
        "Runtime glue: pass pipeline hook (TL_TREE_PASSES), kernel cache, one file per device",
        &[
            "crates/tensorlisp/src/scheme/mod.rs",
            "crates/tensorlisp/src/kernel_cache.rs",
            "crates/tensorlisp/src/native_ir.rs",
            "crates/tensorlisp/src/native_vulkan.rs",
            "crates/tensorlisp/src/native_cuda.rs",
            "crates/tensorlisp/src/native_cuda_tree.rs",
            "crates/tensorlisp/src/native_metal.rs",
            "crates/tensorlisp/src/native_metal_tree.rs",
        ],
    ),
];

const MAP_INTRO: &str = "Tree kernels are compiled by Scheme (first group). Rust decides what each kernel is launched with: the generator expression, bindings, push constants and grid for a graph node, what is fused into it beforehand, and how launches are ordered and synchronized.\n\
Which executor runs what: the Tree executors (features native-vulkan, native-cuda-tree, native-metal-tree) share ONE lowering, vulkan/lower/tree_ops.rs, and differ only in the backend that runs the compiled SPIR-V / PTX / MSL (vulkan/mod.rs, cuda/backend.rs, metal/backend.rs). The older executors (native-cuda: own NVRTC kernels in cuda/native/*.cu via cuda/lower.rs; native-metal: ggml's .metal kernels via metal/lower.rs) do not use Tree at all; passes do not affect them. `op_lowering` shows which executor lowers which graph op.\n\
Profiles (for `tl-evolve weights capture`): TL_VULKAN_PROFILE=1, TL_CUDA_PROFILE=1, TL_METAL_PROFILE=1 print per-kernel GPU time; TL_NATIVE_TRACE / TL_NATIVE_DUMP=<file> show the graph and fusion; TL_TREE_PASSES / TL_TREE_PASS_FILE select and load passes; TL_KERNEL_CACHE=off disables the compiled-kernel cache.\n";

/// The file's own header documentation (`//!` in Rust, the leading `;;` block in Scheme): the title line
/// and the first prose line after it, skipping code examples.
fn header_doc(text: &str, path: &str, n: usize) -> String {
    let prefix = if path.ends_with(".rs") { "//!" } else { ";;" };
    let block: Vec<&str> = text.lines().take_while(|l| l.starts_with(prefix) || l.trim().is_empty()).filter(|l| l.starts_with(prefix)).map(|l| l[prefix.len()..].trim()).collect();
    let mut lines: Vec<&str> = Vec::new();
    for (i, l) in block.iter().enumerate() {
        if l.is_empty() {
            continue;
        }
        // the first line is the title even when it starts with a library name in parentheses
        if i == 0 || !(l.starts_with('(') || l.starts_with('\'') || l.starts_with("--") || l.starts_with('`') || l.starts_with('*')) {
            lines.push(l);
        }
        if lines.len() >= n {
            break;
        }
    }
    lines.join(" ").chars().take(260).collect()
}

pub fn map(repo: &Path) -> String {
    let mut out = String::from(MAP_INTRO);
    out.push_str("Read with read_file offset/limit; rust_outline lists a file's items; launch_sites finds who emits a generator; op_lowering shows per graph op which executors lower it.\n");
    let mut any = false;
    for (title, paths) in GROUPS {
        let mut lines = Vec::new();
        for p in *paths {
            let Ok(text) = std::fs::read_to_string(repo.join(p)) else { continue };
            any = true;
            lines.push(format!("  {p} ({} lines)\n      {}", text.lines().count(), header_doc(&text, p, 2)));
        }
        if !lines.is_empty() {
            out.push_str(&format!("\n{title}:\n{}\n", lines.join("\n")));
        }
    }
    if !any {
        return "none of the implementation files exist under this repository root".into();
    }
    out
}

/// Items of a Rust or Scheme file with line numbers.
pub fn outline(path: &str, text: &str) -> Result<String> {
    let re = if path.ends_with(".rs") {
        Regex::new(r"^(\s*)((?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:fn|struct|enum|trait|impl|mod|static|type|macro_rules!)\b.*)").unwrap()
    } else if path.ends_with(".ss") {
        Regex::new(r"^(\s{0,4})(\((?:define(?:-syntax|-record-type)?|library)\s+.*)").unwrap()
    } else {
        bail!("outlines are for .rs and .ss files; use read_file or grep for {path}");
    };
    let mut out = String::new();
    let mut n = 0;
    for (i, l) in text.lines().enumerate() {
        let Some(c) = re.captures(l) else { continue };
        let item = c[2].trim_end_matches([' ', '{']).trim_end();
        if item.starts_with("const ") && !item.contains("fn ") && !path.ends_with(".rs") {
            continue;
        }
        let depth = (c[1].len() / 4).min(3);
        out.push_str(&format!("{:>6}: {}{}\n", i + 1, "  ".repeat(depth), item.chars().take(120).collect::<String>()));
        n += 1;
        if n >= 400 {
            out.push_str("[stopped at 400 items]\n");
            break;
        }
    }
    if n == 0 {
        return Ok(format!("no items found in {path}"));
    }
    Ok(out)
}

fn walk(dir: &Path, exts: &[&str], out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            walk(&p, exts, out);
        } else if p.extension().is_some_and(|x| exts.contains(&x.to_string_lossy().as_ref())) {
            out.push(p);
        }
    }
}

const LOWERINGS: &[(&str, &str)] = &[
    ("Tree kernels (native-vulkan, native-cuda-tree, native-metal-tree)", "crates/kernels/src/vulkan/lower/tree_ops.rs"),
    ("Vulkan, ggml's precompiled shaders (native-vulkan; the fallback when Tree has no kernel)", "crates/kernels/src/vulkan/lower.rs"),
    ("CUDA, own NVRTC kernels (native-cuda)", "crates/kernels/src/cuda/lower.rs"),
    ("Metal, ggml's .metal kernels (native-metal)", "crates/kernels/src/metal/lower.rs"),
];

/// `(op, line, arm text)` for every match arm in `text` that names ops in quotes: `"A" | "B" if cond => handler`.
fn arms(text: &str) -> Vec<(String, usize, String)> {
    let arm = Regex::new(r#"^\s*("[A-Z][A-Z0-9_]*"(?:\s*\|\s*"[A-Z][A-Z0-9_]*")*)(?:\s+if\s.+?)?\s*=>\s*(.*)$"#).unwrap();
    let name = Regex::new(r#""([A-Z][A-Z0-9_]*)""#).unwrap();
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if let Some(c) = arm.captures(l) {
            for n in name.captures_iter(&c[1]) {
                out.push((n[1].to_string(), i + 1, l.trim().chars().take(140).collect::<String>()));
            }
        }
    }
    out
}

/// Which executors lower a graph op, and where: the arms of their dispatch `match`es. Without an op, a matrix
/// of every op named in any of them.
pub fn op_lowering(repo: &Path, op: Option<&str>) -> Result<String> {
    let sources: Vec<(&str, &str, Option<String>)> = LOWERINGS.iter().map(|(l, p)| (*l, *p, std::fs::read_to_string(repo.join(p)).ok())).collect();
    if sources.iter().all(|s| s.2.is_none()) {
        return Ok("none of the lowering files exist under this repository root".into());
    }
    match op.map(|o| o.trim().to_uppercase()).filter(|o| !o.is_empty()) {
        Some(op) => {
            if !op.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                bail!("an op name is like MUL_MAT or RMS_NORM");
            }
            let mut out = format!("{op}:\n");
            for (label, path, text) in &sources {
                let Some(text) = text else { continue };
                let found: Vec<(usize, String)> = arms(text).into_iter().filter(|a| a.0 == op).map(|a| (a.1, a.2)).collect();
                if found.is_empty() {
                    out.push_str(&format!("  {label}: no match arm in {path}\n"));
                }
                for (line, text) in found.iter().take(3) {
                    out.push_str(&format!("  {label}: {path}:{line}: {text}\n"));
                }
            }
            out.push_str("(Tree falls back to ggml's shaders for an op or shape it has no kernel for, unless TL_NATIVE_TREE=only, which fails at compile time naming the op. A match arm can still return `None` for shapes or types it does not handle: read the handler.)\n");
            Ok(out)
        }
        None => {
            let per: Vec<std::collections::BTreeSet<String>> = sources.iter().map(|s| s.2.as_deref().map(|t| arms(t).into_iter().map(|a| a.0).collect()).unwrap_or_default()).collect();
            let all: std::collections::BTreeSet<&String> = per.iter().flatten().collect();
            let mut out = String::from("ops named in each lowering's match arms (T = Tree kernels, V = Vulkan ggml shaders, C = CUDA native, M = Metal native); the handler can still decline some shapes or types\n");
            out.push_str("   T V C M  op\n");
            for op in all {
                let mark = |i: usize| if per[i].contains(op) { "x" } else { "." };
                out.push_str(&format!("   {} {} {} {}  {op}\n", mark(0), mark(1), mark(2), mark(3)));
            }
            Ok(out)
        }
    }
}

/// Where a generator is defined (Scheme) and which Rust functions emit its expression.
pub fn launch_sites(repo: &Path, kernel: &str) -> Result<String> {
    let name = Regex::new(r"k:([a-z0-9-]+)").unwrap().captures(kernel).map(|c| c[1].to_string()).unwrap_or_else(|| kernel.trim().trim_matches(|c| c == '(' || c == ')').to_string());
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        bail!("give a generator expression like (k:matvec 'q8_0 1) or a name like matvec");
    }
    let mut out = format!("generator {name}:\n");
    // `(define (k:name args) ...)` or `(define k:name (case-lambda ...))`
    let def = Regex::new(&format!(r"^\s*\(define \(?k:{}(?:[\s)]|$)", regex::escape(&name)))?;
    let mut found_def = false;
    if let Ok(text) = std::fs::read_to_string(repo.join("crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss")) {
        for (i, l) in text.lines().enumerate() {
            if def.is_match(l) {
                out.push_str(&format!("  defined at crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss:{}: {}\n", i + 1, l.trim()));
                found_def = true;
            }
        }
    }
    if !found_def {
        out.push_str("  (no generator with that name in tree_kernels.ss)\n");
    }
    let emit = Regex::new(&format!(r#"\(k:{}(?:[\s)"'{{]|$)"#, regex::escape(&name)))?;
    let func = Regex::new(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)").unwrap();
    let mut files = Vec::new();
    walk(&repo.join("crates/kernels/src"), &["rs"], &mut files);
    walk(&repo.join("crates/tensorlisp/src"), &["rs"], &mut files);
    let mut sites = 0;
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let lines: Vec<&str> = text.lines().collect();
        for (i, l) in lines.iter().enumerate() {
            if !l.contains("(k:") || !emit.is_match(l) || l.trim_start().starts_with("//") {
                continue;
            }
            let in_fn = lines[..=i].iter().rev().find_map(|x| func.captures(x).map(|c| c[1].to_string())).unwrap_or_else(|| "?".into());
            let rel = f.strip_prefix(repo).unwrap_or(&f).display();
            out.push_str(&format!("  emitted at {rel}:{} in fn {in_fn}: {}\n", i + 1, l.trim().chars().take(130).collect::<String>()));
            sites += 1;
            if sites >= 14 {
                out.push_str("  [stopped at 14 sites]\n");
                return Ok(out);
            }
        }
    }
    if sites == 0 {
        out.push_str("  (no Rust code emits this generator directly; it may be built from another, or used only by tests)\n");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlines_rust_and_scheme_items_with_line_numbers() {
        let rs = "//! doc\nuse x;\npub struct A {\n  f: u32,\n}\nimpl A {\n    pub(crate) fn go(&self) -> u32 {\n        let fn_like = 1;\n        1\n    }\n}\nfn free() {}\n";
        let o = outline("a.rs", rs).unwrap();
        assert!(o.contains("     3: pub struct A") && o.contains("     6: impl A") && o.contains("     7:   pub(crate) fn go(&self) -> u32") && o.contains("    12: fn free()"), "{o}");
        assert!(!o.contains("fn_like"));
        let ss = "(library (tl x)\n  (export a)\n  (define (helper x) x)\n  (define-syntax m (syntax-rules ()))\n  (define k 1))\n";
        let o = outline("x.ss", ss).unwrap();
        assert!(o.contains("1: (library (tl x)") && o.contains("3: (define (helper x) x)") && o.contains("4: (define-syntax m"), "{o}");
        assert!(outline("x.py", "").is_err());
        assert!(outline("e.rs", "// nothing\n").unwrap().contains("no items"));
    }

    #[test]
    fn finds_who_emits_a_generator_and_where_it_is_defined() {
        let root = std::env::temp_dir().join(format!("evolve-impl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let w = |p: &str, t: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, t).unwrap();
        };
        w("crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss", "(library (tl tree kernels)\n  (define k:matvec\n    (case-lambda [(a) 1]))\n  (define (k:matvec-q8 n) 2))\n");
        w("crates/kernels/src/vulkan/lower/tree_ops.rs", "impl Ctx {\n    fn tree_matvec(&self) {\n        let a = format!(\"(k:matvec '{atype} {n})\");\n        // (k:matvec 'commented)\n        let b = \"(k:matvec-q8 1 2)\";\n    }\n}\n");
        let out = launch_sites(&root, "(k:matvec 'q8_0 1)").unwrap();
        assert!(out.contains("defined at crates/tensorlisp/src/scheme/stdlib/tree_kernels.ss:2: (define k:matvec"), "{out}");
        assert!(out.contains("tree_ops.rs:3 in fn tree_matvec") && !out.contains("commented") && !out.contains("matvec-q8 1 2"), "{out}");
        let out = launch_sites(&root, "matvec-q8").unwrap();
        assert!(out.contains("tree_ops.rs:5 in fn tree_matvec") && out.contains("tree_kernels.ss:4"), "{out}");
        assert!(launch_sites(&root, "(k:nothing)").unwrap().contains("no generator with that name"));
        assert!(launch_sites(&root, "bad name!").is_err());
        let m = map(&root);
        assert!(m.contains("tree_kernels.ss (4 lines)") && !m.contains("tree_opt.ss"), "{m}");
    }

    #[test]
    fn match_arms_show_which_executor_lowers_which_op() {
        let root = std::env::temp_dir().join(format!("evolve-ops-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let w = |p: &str, t: &str| {
            let f = root.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, t).unwrap();
        };
        w("crates/kernels/src/vulkan/lower/tree_ops.rs", "match op {\n    \"MUL_MAT\" => self.tree_mul_mat(tree, id),\n    \"RMS_NORM\" | \"RMS_NORM_MUL\" => self.tree_rms_norm(tree, id),\n    _ => None,\n}\n");
        w("crates/kernels/src/cuda/lower.rs", "match op {\n            \"NORM\" | \"RMS_NORM\" => self.norm(id),\n            \"MUL_MAT\" if self.fast => self.mul_mat(id),\n            \"PAD_REFLECT_1D\" => self.pad(id),\n }\n");
        let one = op_lowering(&root, Some("rms_norm")).unwrap();
        assert!(one.contains("Tree kernels") && one.contains("tree_ops.rs:3:") && one.contains("CUDA, own NVRTC kernels (native-cuda): crates/kernels/src/cuda/lower.rs:2:"), "{one}");
        assert!(one.contains("Metal, ggml's .metal kernels (native-metal)") || !root.join("crates/kernels/src/metal/lower.rs").exists());
        let none = op_lowering(&root, Some("PAD_REFLECT_1D")).unwrap();
        assert!(none.contains("Tree kernels (native-vulkan, native-cuda-tree, native-metal-tree): no match arm") && none.contains("cuda/lower.rs:4:"), "{none}");
        let matrix = op_lowering(&root, None).unwrap();
        assert!(matrix.contains("x . x .  MUL_MAT") && matrix.contains(". . x .  PAD_REFLECT_1D") && matrix.contains("x . x .  RMS_NORM"), "{matrix}");
        assert!(op_lowering(&root, Some("bad op!")).is_err());
        assert!(op_lowering(Path::new("/nonexistent"), None).unwrap().contains("none of the lowering files"));
    }

    #[test]
    fn the_real_repository_map_is_generated_from_the_sources() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if !repo.join("crates/kernels/src/vulkan/lower/tree_ops.rs").exists() {
            return;
        }
        let m = map(&repo);
        assert!(m.contains("Lowering of graph nodes to Tree kernels") && m.contains("hazard") && m.contains("TL_TREE_PASSES"), "{m}");
        // CUDA and Metal are covered: executors, backends, glue, native kernels, profiler switches
        for needle in ["cuda/backend.rs", "cuda/lower.rs", "cuda/native/mm.cu", "metal/backend.rs", "metal/lower.rs", "metal/kargs.rs", "native_metal_tree.rs", "native_cuda.rs", "TL_CUDA_PROFILE", "TL_METAL_PROFILE"] {
            assert!(m.contains(needle), "{needle} missing from the map:\n{m}");
        }
        let ops = op_lowering(&repo, Some("MUL_MAT")).unwrap();
        assert!(ops.contains("tree_ops.rs") && ops.contains("cuda/lower.rs") && ops.contains("metal/lower.rs"), "{ops}");
        let matrix = op_lowering(&repo, None).unwrap();
        assert!(matrix.lines().count() > 20 && matrix.contains("MUL_MAT"), "{matrix}");
        let sites = launch_sites(&repo, "(k:matvec 'q8_0 1)").unwrap();
        assert!(sites.contains("defined at") && sites.contains("emitted at crates/kernels/src/vulkan/lower/tree_ops.rs"), "{sites}");
        let o = outline("tree_ops.rs", &std::fs::read_to_string(repo.join("crates/kernels/src/vulkan/lower/tree_ops.rs")).unwrap()).unwrap();
        assert!(o.lines().count() > 20);
    }
}
