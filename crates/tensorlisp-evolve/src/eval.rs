//! The evaluator: the only thing that produces verdicts.
//!
//! A candidate is Scheme source that registers passes (`register-pass!`) and a pipeline naming them.
//! Gates, in order (any failure means `ok = false` and the entry cannot reach the leaderboard):
//!
//! * `loads`      the code evaluates (in a sandboxed environment, with a time limit)
//! * `compiles`   every Tree generator in `tests/tree_exprs.txt` still compiles for the target
//! * `structure`  the pipeline kept each kernel's interface (buffers, push, workgroup, shared) and
//!                effects (the buffers it stores to, whether it synchronizes). This catches passes that
//!                "win" by deleting work; it is a heuristic, not a proof of equivalence.
//! * `spirv-val`  the SPIR-V validates (SPIR-V target, when `spirv-val` is installed)
//! * external gates: shell commands run with `TL_TREE_PASS_FILE` and `TL_TREE_PASSES` set (and the kernel
//!   cache off), so
//!   `cargo test --test tree_gemma` and friends exercise the real models with the candidate. Exit status
//!   is the verdict; lines `TL_EVOLVE_METRIC name=value` on stdout are recorded, and a `speedup` metric
//!   replaces the static score.
//!
//! The score is the geometric mean over the kernels of baseline size / candidate size (SPIR-V words,
//! or characters of PTX / MSL text), so above 1 is better. It is a proxy: it says the pass makes
//! smaller code, not faster code; speed comes from an external gate on a device.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use serde_json::json;

use crate::forum::{Eval, Forum, Verdict, now_millis, sha_hex};
use crate::scheme::{self, Outcome};

pub const DEFAULT_EXPRS: &str = include_str!("../../tensorlisp/tests/tree_exprs.txt");

#[derive(Debug, Clone)]
pub struct GateCommand {
    pub name: String,
    pub command: String,
    pub timeout: Duration,
}

impl std::str::FromStr for GateCommand {
    type Err = anyhow::Error;
    /// `NAME=COMMAND`
    fn from_str(s: &str) -> Result<Self> {
        let Some((name, command)) = s.split_once('=') else { bail!("a gate is NAME=COMMAND, got {s:?}") };
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            bail!("gate names are [A-Za-z0-9_-]+, got {name:?}");
        }
        Ok(GateCommand { name: name.into(), command: command.into(), timeout: Duration::from_secs(1800) })
    }
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub exprs: Vec<String>,
    /// `spirv`, `ptx` or `msl`.
    pub target: String,
    /// What the dynamic gates measured on ("static" when there are none).
    pub device: String,
    pub timeout_ms: u64,
    pub gates: Vec<GateCommand>,
    pub spirv_val: Option<PathBuf>,
    /// Candidate files and gate logs go here.
    pub work_dir: PathBuf,
    /// Kernel expression -> weight from profiles (see `weights.rs`); empty weighs every kernel equally.
    /// The profiled expressions are compiled in addition to `exprs`.
    pub weights: BTreeMap<String, f64>,
    /// Launches captured from real runs on the device (see `kbench.rs`); with them, a candidate that passed the
    /// built-in gates is timed against the reference kernels on the GPU (MSL only for now).
    pub launches: Vec<crate::weights::Launch>,
    pub bench: crate::kbench::Options,
}

impl EvalConfig {
    pub fn new(target: &str, work_dir: impl Into<PathBuf>) -> Self {
        EvalConfig {
            exprs: DEFAULT_EXPRS.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect(),
            target: target.into(),
            device: "static".into(),
            timeout_ms: 20_000,
            gates: Vec::new(),
            spirv_val: find_spirv_val(),
            work_dir: work_dir.into(),
            weights: BTreeMap::new(),
            launches: Vec::new(),
            bench: crate::kbench::Options::default(),
        }
    }
}

pub fn find_spirv_val() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("VULKAN_SDK") {
        let d = PathBuf::from(d);
        candidates.push(d.join("bin").join("spirv-val"));
        candidates.push(d.join("Bin").join("spirv-val"));
    }
    candidates.push("spirv-val".into());
    candidates.into_iter().find(|p| Command::new(p).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success()))
}

pub fn check_pipeline(pipeline: &[String]) -> Result<()> {
    if pipeline.is_empty() {
        bail!("the pipeline is empty: name at least one pass, e.g. [\"fold\", \"my-pass\", \"dce\"]");
    }
    for p in pipeline {
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_alphanumeric() || "-_!?*<>=/+".contains(c)) {
            bail!("bad pass name {p:?}");
        }
    }
    Ok(())
}

struct GateRun {
    passed: bool,
    metrics: BTreeMap<String, f64>,
    tail: String,
}

fn run_gate(gate: &GateCommand, cfg: &EvalConfig, file: &Path, kernels_file: Option<&Path>, pipeline: &[String]) -> Result<GateRun> {
    let log = cfg.work_dir.join(format!("gate-{}-{}.log", gate.name, std::process::id()));
    let out = std::fs::File::create(&log)?;
    let err = out.try_clone()?;
    let mut command = Command::new("sh");
    command.arg("-c").arg(&gate.command);
    match kernels_file {
        Some(f) => command.env("TL_TREE_KERNELS_FILE", f),
        None => command.env_remove("TL_TREE_KERNELS_FILE"),
    };
    let mut child = command
        .env("TL_TREE_PASS_FILE", file)
        .env("TL_TREE_PASSES", pipeline.join(","))
        // the kernel cache is keyed by the pass names, not by what the passes do: a candidate must be compiled afresh
        .env("TL_KERNEL_CACHE", "off")
        .env("TL_EVOLVE_TARGET", &cfg.target)
        .env("TL_EVOLVE_DEVICE", &cfg.device)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()?;
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if start.elapsed() > gate.timeout {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let mut metrics = BTreeMap::new();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("TL_EVOLVE_METRIC ")
            && let Some((k, v)) = rest.split_once('=')
            && let Ok(v) = v.trim().parse::<f64>()
        {
            metrics.insert(k.trim().to_string(), v);
        }
    }
    let tail: String = {
        let chars: Vec<char> = text.chars().collect();
        chars[chars.len().saturating_sub(1500)..].iter().collect()
    };
    Ok(GateRun { passed: status.is_some_and(|s| s.success()), metrics, tail: if status.is_none() { format!("timed out after {:?}\n{tail}", gate.timeout) } else { tail } })
}

/// The generator library as this binary has it: the baseline a candidate generator change is compared with.
pub const ORIGINAL_GENERATORS: &str = include_str!("../../tensorlisp/src/scheme/stdlib/tree_kernels.ss");
const LIBRARY_HEADER: &str = "(library (tl tree kernels)";

/// A candidate edit of `tree_kernels.ss`: the unified diff against the original and the names of the
/// top-level definitions the edit touches. `None` when the text is identical.
pub fn generator_edit(candidate: &str) -> Option<(String, Vec<String>)> {
    if candidate == ORIGINAL_GENERATORS {
        return None;
    }
    let diff = similar::TextDiff::from_lines(ORIGINAL_GENERATORS, candidate);
    let unified = diff.unified_diff().context_radius(2).header("tree_kernels.ss", "candidate").to_string();
    let define = regex::Regex::new(r"^\s{0,3}\(define(?:-syntax|-record-type)?\s+\(?([^\s()]+)").unwrap();
    let owner = |text: &str, line: usize| -> Option<String> { text.lines().take(line + 1).collect::<Vec<_>>().iter().rev().find_map(|l| define.captures(l).map(|c| c[1].to_string())) };
    let mut names: Vec<String> = Vec::new();
    for op in diff.ops() {
        let (old, new) = (op.old_range(), op.new_range());
        // inserted or replaced lines belong to the definition above them in the new text; a pure deletion, to the old
        let name = if !new.is_empty() { owner(candidate, new.start) } else { owner(ORIGINAL_GENERATORS, old.start) };
        if let Some(n) = name
            && !matches!(op.tag(), similar::DiffTag::Equal)
            && !names.contains(&n)
        {
            names.push(n);
        }
    }
    Some((unified, names))
}

/// The candidate's generator text as a library the evaluator can load next to the original.
fn renamed_library(candidate: &str) -> Result<String> {
    if !candidate.contains(LIBRARY_HEADER) {
        bail!("the generator file must keep its first form, `{LIBRARY_HEADER}`: copy it with generator_workspace and edit the copy");
    }
    Ok(candidate.replacen(LIBRARY_HEADER, "(library (tl tree kernels-candidate)", 1))
}

/// Runs every gate on a candidate and stores the result in the forum as an [`Eval`].
///
/// Compiles `cfg.exprs` plus every profiled expression. The score is the weighted geometric mean of
/// baseline size / candidate size over the profiled kernels when the device has profiles, else the plain
/// geometric mean over all kernels.
pub fn evaluate(forum: &Forum, cfg: &EvalConfig, code: &str, pipeline: &[String]) -> Result<Eval> {
    evaluate_with(forum, cfg, code, pipeline, None)
}

/// [`evaluate`], optionally with a candidate edit of the kernel generators (the full text of a modified
/// `tree_kernels.ss`). The baseline is built by the original generators and the default passes; the candidate
/// by the edited generators and `pipeline`. Everything else is the same: the interface of each kernel
/// (buffers, push constants, workgroup) must be unchanged, because the Rust launch sites rely on it.
pub fn evaluate_with(forum: &Forum, cfg: &EvalConfig, code: &str, pipeline: &[String], generators: Option<&str>) -> Result<Eval> {
    check_pipeline(pipeline)?;
    std::fs::create_dir_all(&cfg.work_dir)?;
    let mut edit = generators.and_then(|g| generator_edit(g).map(|(d, n)| (g.to_string(), d, n)));
    let gen_library = match &edit {
        Some((text, _, _)) => renamed_library(text)?,
        None => String::new(),
    };
    let gen_print = edit.as_ref().map(|e| sha_hex(&[&e.0])[..16].to_string()).unwrap_or_default();
    let weights_print: String = cfg.weights.iter().map(|(k, v)| format!("{k}={v:.6}")).collect::<Vec<_>>().join(";");
    let id = sha_hex(&[code, &pipeline.join(","), &cfg.target, &cfg.device, &cfg.gates.iter().map(|g| format!("{}={}", g.name, g.command)).collect::<Vec<_>>().join(";"), &weights_print, &gen_print])[..16].to_string();

    let mut exprs = cfg.exprs.clone();
    for e in cfg.weights.keys() {
        if !exprs.contains(e) {
            exprs.push(e.clone());
        }
    }
    let outcome = scheme::handle()?.evaluate(code, pipeline, &cfg.target, &exprs, cfg.timeout_ms, &gen_library)?;
    let mut gates: BTreeMap<String, bool> = BTreeMap::new();
    let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut rows_json = Vec::new();
    let mut logsum = 0.0f64;
    let mut counted = 0usize;
    let (mut wlog, mut wsum) = (0.0f64, 0.0f64);
    let mut locked_weight = 0.0f64;
    let mut compiled_rows: Option<&Vec<crate::scheme::Row>> = None;

    match &outcome {
        Outcome::LoadError(msg) => {
            gates.insert("loads".into(), false);
            match (msg.strip_prefix("[generators] "), &edit) {
                (Some(m), Some((text, _, _))) => notes.push(format!("the edited generator library did not load: {}", crate::lispcheck::explain(m, text))),
                _ => notes.push(format!("did not load: {}", crate::lispcheck::explain(msg, code))),
            }
        }
        Outcome::Rows(rows) => {
            gates.insert("loads".into(), true);
            compiled_rows = Some(rows);
            let (mut smaller, mut larger, mut skipped) = (0, 0, 0);
            for r in rows {
                let weight = cfg.weights.get(&r.expr).copied().unwrap_or(0.0);
                // the baseline pipeline leaves kernels with cooperative matrices alone, and so does every pipeline
                let locked = r.status == "ok" && r.note == "cmat";
                if locked {
                    locked_weight += weight;
                }
                rows_json.push(json!({"expr": r.expr, "status": r.status, "note": if locked { "" } else { r.note.as_str() }, "baseline": r.base_size, "candidate": r.cand_size, "weight": weight, "locked": locked}));
                match r.status.as_str() {
                    "ok" if r.base_size > 0 && r.cand_size > 0 => {
                        let ln = (r.base_size as f64 / r.cand_size as f64).ln();
                        logsum += ln;
                        wlog += weight * ln;
                        wsum += weight;
                        counted += 1;
                        if r.cand_size < r.base_size {
                            smaller += 1;
                        } else if r.cand_size > r.base_size {
                            larger += 1;
                        }
                    }
                    "skip" => skipped += 1,
                    "error" | "structure" => notes.push(format!("{}: {} ({})", r.expr, r.status, r.note)),
                    _ => {}
                }
            }
            gates.insert("compiles".into(), !rows.iter().any(|r| r.status == "error"));
            gates.insert("structure".into(), !rows.iter().any(|r| r.status == "structure"));
            metrics.insert("kernels".into(), counted as f64);
            metrics.insert("smaller".into(), smaller as f64);
            metrics.insert("larger".into(), larger as f64);
            if skipped > 0 {
                notes.push(format!("{skipped} kernels do not build for {} even with the baseline pipeline and were skipped", cfg.target));
            }
            if counted == 0 && gates["compiles"] && gates["structure"] {
                gates.insert("compiles".into(), false);
                notes.push("no kernel could be compared".into());
            }
            if cfg.target == "spirv" {
                match &cfg.spirv_val {
                    Some(exe) => {
                        let validate = |bytes: &[u8], tag: &str| -> Result<Option<String>> {
                            let path = cfg.work_dir.join(format!("val-{}-{tag}.spv", std::process::id()));
                            std::fs::write(&path, bytes)?;
                            let out = Command::new(exe).args(["--target-env", "vulkan1.2"]).arg(&path).output()?;
                            let _ = std::fs::remove_file(&path);
                            Ok((!out.status.success()).then(|| String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or("").to_string()))
                        };
                        let (mut ok, mut baseline_rejected) = (true, 0);
                        for (i, r) in rows.iter().enumerate().filter(|(_, r)| r.status == "ok") {
                            let Some(bytes) = &r.bytes else { continue };
                            // identical to the baseline's output: nothing the candidate could have broken
                            if r.base_bytes.as_deref() == Some(bytes.as_slice()) {
                                continue;
                            }
                            if let Some(why) = validate(bytes, &format!("c{i}"))? {
                                // the validator may be older than the kernel (cooperative matrices): only blame
                                // the candidate when the baseline's SPIR-V validates
                                let base_fine = match &r.base_bytes {
                                    Some(b) => validate(b, &format!("b{i}"))?.is_none(),
                                    None => true,
                                };
                                if base_fine {
                                    ok = false;
                                    notes.push(format!("spirv-val rejected {}: {why}", r.expr));
                                } else {
                                    baseline_rejected += 1;
                                }
                            }
                        }
                        if baseline_rejected > 0 {
                            notes.push(format!("spirv-val also rejects the baseline SPIR-V of {baseline_rejected} kernels (validator too old for them?): not held against the candidate"));
                        }
                        gates.insert("spirv-val".into(), ok);
                    }
                    None => notes.push("spirv-val is not installed: SPIR-V was not validated".into()),
                }
            }
        }
    }

    if let Some((_, _, names)) = &edit {
        metrics.insert("generators_edited".into(), names.len() as f64);
        notes.insert(0, format!("generator edit touches: {}", names.join(", ")));
    }
    // time the kernels the candidate changed against the reference ones, on the real GPU
    let mut bench_report = None;
    if let Some(rows) = compiled_rows
        && cfg.target == "msl"
        && !cfg.launches.is_empty()
        && gates.values().all(|ok| *ok)
    {
        match crate::kbench::run(rows, &cfg.launches, &cfg.weights, &cfg.bench) {
            Ok(r) => {
                gates.insert("kernel-bench".into(), r.ok);
                metrics.insert("kernel_speedup".into(), r.speedup);
                metrics.insert("kernel_noise".into(), r.noise);
                metrics.insert("kernel_coverage".into(), r.coverage);
                let e = metrics.entry("speedup".into()).or_insert(r.speedup);
                *e = e.min(r.speedup);
                bench_report = Some(r);
            }
            Err(e) => notes.push(format!("kernel benchmark did not run: {e}")),
        }
    }
    let mut gate_logs = serde_json::Map::new();
    if !cfg.gates.is_empty() {
        if gates.values().all(|ok| *ok) {
            let dir = cfg.work_dir.join("candidates");
            std::fs::create_dir_all(&dir)?;
            let file = dir.join(format!("{id}.ss"));
            std::fs::write(&file, code)?;
            // the full edited generator file, for runtimes started by a gate (TL_TREE_KERNELS_FILE)
            let kernels_file = match &edit {
                Some((text, _, _)) => {
                    let f = dir.join(format!("{id}.kernels.ss"));
                    std::fs::write(&f, text)?;
                    Some(f)
                }
                None => None,
            };
            for g in &cfg.gates {
                let run = run_gate(g, cfg, &file, kernels_file.as_deref(), pipeline)?;
                gates.insert(g.name.clone(), run.passed);
                for (k, v) in run.metrics {
                    metrics.insert(format!("{}.{k}", g.name), v);
                    if k == "speedup" {
                        let e = metrics.entry("speedup".into()).or_insert(v);
                        *e = e.min(v);
                    }
                }
                gate_logs.insert(g.name.clone(), json!(run.tail));
            }
        } else {
            notes.push("external gates were not run: a built-in gate failed".into());
        }
    }

    let uniform = if counted > 0 { (logsum / counted as f64).exp() } else { 0.0 };
    metrics.insert("size_ratio".into(), uniform);
    let static_score = if !cfg.weights.is_empty() && wsum > 0.0 {
        let weighted = (wlog / wsum).exp();
        metrics.insert("weighted_ratio".into(), weighted);
        // how much of the profiled time the compared kernels account for
        metrics.insert("hot_coverage".into(), wsum / cfg.weights.values().sum::<f64>().max(1e-12));
        // the share of profiled time in kernels no pass can change: an upper bound on what passes can win
        metrics.insert("hot_locked".into(), locked_weight / cfg.weights.values().sum::<f64>().max(1e-12));
        weighted
    } else {
        uniform
    };
    let score = metrics.get("speedup").copied().unwrap_or(static_score);
    let verdict = Verdict { eval_id: id.clone(), target: cfg.target.clone(), device: cfg.device.clone(), ok: gates.values().all(|g| *g), score, gates, metrics };
    let shown_code = match &edit {
        Some((_, diff, _)) => {
            let commented: String = diff.lines().take(400).map(|l| format!(";; {l}\n")).collect();
            format!("{code}\n\n;;; generator edit (diff of tree_kernels.ss)\n{commented}")
        }
        None => code.to_string(),
    };
    edit.take();
    let eval = Eval {
        id,
        code: shown_code,
        pipeline: pipeline.to_vec(),
        verdict,
        report: json!({"rows": rows_json, "notes": notes, "gate_output": gate_logs, "bench": bench_report}),
        created: now_millis(),
    };
    forum.put_eval(&eval)?;
    Ok(eval)
}

/// A short text report for the agent (and for `tl-evolve eval`).
pub fn render(eval: &Eval) -> String {
    render_focus(eval, &[])
}

/// Like [`render`], plus the rows whose expression contains any of `focus` (substrings such as `matvec`)
/// with their own geometric mean. The verdict does not change: focus only chooses what to look at.
pub fn render_focus(eval: &Eval, focus: &[String]) -> String {
    let v = &eval.verdict;
    let mut out = format!("evaluation {} on {} / {}: {}\nscore {:.4} (above 1 is better than the baseline pipeline)\ngates:", eval.id, v.target, v.device, if v.ok { "OK" } else { "FAILED" }, v.score);
    for (g, ok) in &v.gates {
        out.push_str(&format!(" {g}={}", if *ok { "pass" } else { "FAIL" }));
    }
    out.push('\n');
    if !v.metrics.is_empty() {
        out.push_str("metrics:");
        for (k, x) in &v.metrics {
            out.push_str(&format!(" {k}={x:.4}"));
        }
        out.push('\n');
    }
    if let Some(notes) = eval.report["notes"].as_array() {
        for n in notes.iter().take(12) {
            out.push_str(&format!("- {}\n", n.as_str().unwrap_or("")));
        }
    }
    if let Some(rows) = eval.report["rows"].as_array() {
        if !focus.is_empty() {
            let picked: Vec<&serde_json::Value> = rows.iter().filter(|r| focus.iter().any(|f| r["expr"].as_str().is_some_and(|e| e.contains(f.as_str())))).collect();
            let ratios: Vec<f64> = picked.iter().filter(|r| r["status"] == "ok").map(|r| r["baseline"].as_f64().unwrap_or(1.0) / r["candidate"].as_f64().unwrap_or(1.0).max(1.0)).collect();
            if ratios.is_empty() {
                out.push_str(&format!("focus {focus:?}: no kernel matched (or none compiled)\n"));
            } else {
                let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
                out.push_str(&format!("focus {focus:?}: {} kernels, geometric-mean size ratio {g:.4}\n", ratios.len()));
                for r in picked.iter().take(12) {
                    out.push_str(&format!("  {}: {} -> {}{}\n", r["expr"].as_str().unwrap_or(""), r["baseline"], r["candidate"], if r["status"] == "ok" { String::new() } else { format!(" [{}]", r["status"].as_str().unwrap_or("")) }));
                }
            }
        }
        // profiled kernels first, heaviest first: that is where a change pays
        let mut hot: Vec<(&str, f64, i64, i64, bool)> = rows.iter().filter(|r| r["status"] == "ok" && r["weight"].as_f64().unwrap_or(0.0) > 0.0).map(|r| (r["expr"].as_str().unwrap_or(""), r["weight"].as_f64().unwrap_or(0.0), r["baseline"].as_i64().unwrap_or(0), r["candidate"].as_i64().unwrap_or(0), r["locked"] == true)).collect();
        if !hot.is_empty() {
            hot.sort_by(|a, b| b.1.total_cmp(&a.1));
            out.push_str("profiled kernels (share of Tree busy time, baseline -> candidate size):\n");
            for (e, w, b, c, locked) in hot.iter().take(8) {
                out.push_str(&format!("  {:>5.1}%  {e}: {b} -> {c}{}\n", w * 100.0, if *locked { "  [locked: cooperative matrices, no pass changes it]" } else { "" }));
            }
            let locked_share: f64 = hot.iter().filter(|h| h.4).map(|h| h.1).sum::<f64>() / hot.iter().map(|h| h.1).sum::<f64>().max(1e-12);
            if locked_share > 0.05 {
                out.push_str(&format!("{:.0}% of the profiled time is in kernels passes cannot change (they contain cooperative matrices; `optimize` skips them), so a pass can win at most on the rest.\n", locked_share * 100.0));
            }
        }
        let mut moved: Vec<(&str, i64, i64)> = rows
            .iter()
            .filter(|r| r["status"] == "ok" && r["baseline"] != r["candidate"])
            .map(|r| (r["expr"].as_str().unwrap_or(""), r["baseline"].as_i64().unwrap_or(0), r["candidate"].as_i64().unwrap_or(0)))
            .collect();
        moved.sort_by_key(|m| m.2 - m.1);
        if !moved.is_empty() {
            out.push_str("kernels whose size changed (baseline -> candidate):\n");
            for (e, b, c) in moved.iter().take(6).chain(moved.iter().rev().take(3)) {
                out.push_str(&format!("  {e}: {b} -> {c}\n"));
            }
        }
    }
    if let Ok(b) = serde_json::from_value::<crate::kbench::Report>(eval.report["bench"].clone()) {
        out.push_str(&crate::kbench::render(&b, 10));
    }
    if let Some(logs) = eval.report["gate_output"].as_object() {
        for (name, tail) in logs {
            if eval.verdict.gates.get(name) == Some(&false) {
                out.push_str(&format!("gate {name} output (tail):\n{}\n", tail.as_str().unwrap_or("")));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::kv::MemKv;

    fn setup(name: &str) -> (Forum, EvalConfig) {
        let dir = std::env::temp_dir().join(format!("evolve-eval-{name}-{}", std::process::id()));
        let mut cfg = EvalConfig::new("spirv", dir);
        // a representative subset keeps the test quick
        cfg.exprs = cfg.exprs.iter().filter(|e| ["k:binary 'add", "k:rms-norm #f)", "k:soft-max #t", "k:rope 'norm", "k:matvec 'q8_0 1"].iter().any(|k| e.contains(k))).cloned().collect();
        assert!(cfg.exprs.len() >= 4);
        (Forum::new(Arc::new(MemKv::new())), cfg)
    }

    fn p(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_default_pipeline_is_neutral_and_a_weaker_one_is_not_better() {
        let (forum, cfg) = setup("neutral");
        let same = evaluate(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"])).unwrap();
        assert!(same.verdict.ok, "{}", render(&same));
        assert!((same.verdict.score - 1.0).abs() < 1e-9);
        let weaker = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert!(weaker.verdict.ok, "{}", render(&weaker));
        assert!(weaker.verdict.score <= 1.0 + 1e-9, "{}", render(&weaker));
        assert_ne!(same.id, weaker.id);
        assert_eq!(forum.get_eval(&weaker.id).unwrap().unwrap().verdict, weaker.verdict);
    }

    #[test]
    fn failing_candidates_fail_the_right_gate() {
        let (forum, cfg) = setup("gates");
        let strip = "(define (strip t) (cond [(and (pair? t) (eq? (car t) 'store)) '(begin)] [(pair? t) (map strip t)] [else t]))\n(register-pass! 'strip strip)";
        let e = evaluate(&forum, &cfg, strip, &p(&["strip"])).unwrap();
        assert!(!e.verdict.ok && e.verdict.gates["structure"] == false, "{}", render(&e));
        let e = evaluate(&forum, &cfg, "(this is broken", &p(&["fold"])).unwrap();
        assert!(!e.verdict.ok && e.verdict.gates["loads"] == false);
        let e = evaluate(&forum, &cfg, "", &p(&["no-such-pass"])).unwrap();
        assert!(!e.verdict.ok, "{}", render(&e));
        assert!(evaluate(&forum, &cfg, "", &[]).is_err());
        // a pass that breaks a kernel so it no longer compiles
        let bad = "(register-pass! 'bad (lambda (k) (append k '((store nowhere 0 1.0)))))";
        let e = evaluate(&forum, &cfg, bad, &p(&["bad"])).unwrap();
        assert!(!e.verdict.ok, "{}", render(&e));
    }

    #[test]
    fn external_gates_run_with_the_candidate_and_report_metrics() {
        let (forum, mut cfg) = setup("external");
        cfg.gates = vec![
            "pass=echo \"candidate=$TL_TREE_PASS_FILE passes=$TL_TREE_PASSES\"; echo TL_EVOLVE_METRIC speedup=1.25".parse().unwrap(),
            "fail=echo boom; exit 3".parse().unwrap(),
        ];
        let e = evaluate(&forum, &cfg, "(register-pass! 'noop (lambda (k) k))", &p(&["fold", "noop"])).unwrap();
        assert_eq!(e.verdict.gates["pass"], true);
        assert_eq!(e.verdict.gates["fail"], false);
        assert!(!e.verdict.ok);
        assert_eq!(e.verdict.score, 1.25);
        let out = render(&e);
        assert!(out.contains("boom"), "{out}");
        // gates are skipped when a built-in gate fails
        let e = evaluate(&forum, &cfg, "(oops", &p(&["fold"])).unwrap();
        assert!(!e.verdict.gates.contains_key("pass"));
    }

    #[test]
    fn profiled_kernels_are_compiled_weighted_and_listed_first() {
        let (forum, mut cfg) = setup("weighted");
        // neither expression is in the representative subset: the profile brings them in
        cfg.weights = BTreeMap::from([("(k:rms-norm #f #t)".to_string(), 0.7), ("(k:argmax)".to_string(), 0.3)]);
        let e = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert!(e.verdict.ok, "{}", render(&e));
        let m = &e.verdict.metrics;
        assert!((m["hot_coverage"] - 1.0).abs() < 1e-9, "{m:?}");
        assert_eq!(e.verdict.score, m["weighted_ratio"]);
        assert_ne!(m["weighted_ratio"], m["size_ratio"]);
        assert_eq!(m["hot_locked"], 0.0);
        let rows = e.report["rows"].as_array().unwrap();
        assert!(rows.iter().any(|r| r["expr"] == "(k:argmax)" && r["weight"] == 0.3));
        let text = render_focus(&e, &["rms-norm".to_string()]);
        assert!(text.contains("profiled kernels") && text.contains("70.0%  (k:rms-norm #f #t)"), "{text}");
        assert!(text.contains("focus [\"rms-norm\"]: ") && text.contains("geometric-mean size ratio"), "{text}");
        assert!(render_focus(&e, &["nothing".to_string()]).contains("no kernel matched"));
        // weights are part of the verdict's identity: the same candidate without them is a different evaluation
        cfg.weights.clear();
        let plain = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert_ne!(plain.id, e.id);
        assert!(!plain.verdict.metrics.contains_key("weighted_ratio"));
        // an expression the baseline cannot build is skipped, and lowers the coverage
        cfg.weights = BTreeMap::from([("(k:argmax)".to_string(), 0.5), ("(k:no-such-generator)".to_string(), 0.5)]);
        let e = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert!(e.verdict.ok && (e.verdict.metrics["hot_coverage"] - 0.5).abs() < 1e-9, "{}", render(&e));
    }

    #[test]
    fn profiled_kernels_with_cooperative_matrices_are_flagged_as_locked() {
        let (forum, mut cfg) = setup("locked");
        cfg.weights = BTreeMap::from([("(k:matmul-cm 'f16 32 32 16 1 1)".to_string(), 0.6), ("(k:argmax)".to_string(), 0.4)]);
        let e = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert!(e.verdict.ok, "{}", render(&e));
        assert!((e.verdict.metrics["hot_locked"] - 0.6).abs() < 1e-9, "{:?}", e.verdict.metrics);
        let text = render(&e);
        assert!(text.contains("60.0%  (k:matmul-cm") && text.contains("[locked: cooperative matrices") && text.contains("60% of the profiled time is in kernels passes cannot change"), "{text}");
        let rows = e.report["rows"].as_array().unwrap();
        assert!(rows.iter().any(|r| r["expr"] == "(k:matmul-cm 'f16 32 32 16 1 1)" && r["locked"] == true && r["note"] == ""));
    }

    const SILU: &str = "[(silu) `(/ ,x (+ 1.0 (exp (- ,x))))]";

    fn edited(old: &str, new: &str) -> String {
        assert!(ORIGINAL_GENERATORS.matches(old).count() == 1, "`{old}` must occur exactly once in tree_kernels.ss");
        ORIGINAL_GENERATORS.replacen(old, new, 1)
    }

    fn gen_setup(name: &str) -> (Forum, EvalConfig) {
        let (forum, mut cfg) = setup(name);
        cfg.exprs = vec!["(k:unary 'silu 'same)".into(), "(k:binary 'add)".into(), "(k:rms-norm #f)".into()];
        (forum, cfg)
    }

    #[test]
    fn the_diff_names_the_definitions_an_edit_touches() {
        assert!(generator_edit(ORIGINAL_GENERATORS).is_none());
        let (diff, names) = generator_edit(&edited(SILU, "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (- ,x)))))]")).unwrap();
        assert_eq!(names, ["unary-expr"]);
        assert!(diff.contains("-      [(silu) `(/ ,x") && diff.contains("+      [(silu) `(* ,x (/ 1.0"), "{diff}");
        // two edits in different definitions; a pure deletion is attributed to the definition it was in
        let two = edited(SILU, "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (- ,x)))))]").replacen("(define wg 256)", "(define wg 256)\n  (define extra-helper 1)", 1);
        let (_, names) = generator_edit(&two).unwrap();
        assert!(names.contains(&"unary-expr".to_string()) && names.contains(&"extra-helper".to_string()) || names.contains(&"wg".to_string()), "{names:?}");
    }

    #[test]
    fn an_edited_generator_is_compared_with_the_original_one() {
        let (forum, cfg) = gen_setup("genedit");
        let base = evaluate(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"])).unwrap();
        assert!((base.verdict.score - 1.0).abs() < 1e-9);
        // the same text as the original is no edit at all
        let same = evaluate_with(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"]), Some(ORIGINAL_GENERATORS)).unwrap();
        assert_eq!(same.id, base.id);
        assert!(same.report["notes"].as_array().unwrap().iter().all(|n| !n.as_str().unwrap().contains("generator edit")));

        // a different way to write silu: same interface and meaning, different code
        let text = edited(SILU, "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (- ,x)))))]");
        let e = evaluate_with(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"]), Some(&text)).unwrap();
        assert!(e.verdict.ok, "{}", render(&e));
        assert_eq!(e.verdict.metrics["generators_edited"], 1.0);
        assert!(render(&e).contains("generator edit touches: unary-expr"), "{}", render(&e));
        let silu_row = e.report["rows"].as_array().unwrap().iter().find(|r| r["expr"] == "(k:unary 'silu 'same)").unwrap().clone();
        assert_ne!(silu_row["baseline"], silu_row["candidate"], "{silu_row}");
        // other kernels are untouched by this edit
        let add_row = e.report["rows"].as_array().unwrap().iter().find(|r| r["expr"] == "(k:binary 'add)").unwrap().clone();
        assert_eq!(add_row["baseline"], add_row["candidate"]);
        // the stored record carries the diff, so the forum shows what changed
        assert!(e.code.contains(";;; generator edit") && e.code.contains(";; +      [(silu)"), "{}", e.code);
        let stored = forum.get_eval(&e.id).unwrap().unwrap().verdict;
        assert!(stored.ok == e.verdict.ok && (stored.score - e.verdict.score).abs() < 1e-12 && stored.gates == e.verdict.gates);
        assert_ne!(e.id, base.id);

        // a second, different edit: the candidate library can be loaded again (no stale state from the first)
        let text2 = edited(SILU, "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (* -1.0 ,x)))))]");
        let e2 = evaluate_with(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"]), Some(&text2)).unwrap();
        assert!(e2.verdict.ok && e2.id != e.id, "{}", render(&e2));
        // and a plain evaluation afterwards is back to the original generators
        let after = evaluate(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"])).unwrap();
        assert!((after.verdict.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn generator_edits_that_change_the_interface_or_do_not_load_fail_their_gate() {
        let (forum, cfg) = gen_setup("genbad");
        let pipe = p(&["fold", "licm", "cse", "dce"]);
        // the workgroup size is part of the launch contract with the Rust side
        let e = evaluate_with(&forum, &cfg, "", &pipe, Some(&edited("(define wg 256)", "(define wg 128)"))).unwrap();
        assert!(!e.verdict.ok && !e.verdict.gates["structure"], "{}", render(&e));
        assert!(render(&e).contains("clauses changed"), "{}", render(&e));
        // a syntax error in the generator file is reported against that file
        let broken = edited(SILU, "[(silu) `(/ ,x (+ 1.0 (exp (- ,x))))");
        let e = evaluate_with(&forum, &cfg, "", &pipe, Some(&broken)).unwrap();
        assert!(!e.verdict.ok && !e.verdict.gates["loads"], "{}", render(&e));
        let text = render(&e);
        assert!(text.contains("the edited generator library did not load") && text.contains("paren check"), "{text}");
        // a file without the library header is refused up front
        assert!(evaluate_with(&forum, &cfg, "", &pipe, Some("(define x 1)")).unwrap_err().to_string().contains("keep its first form"));
        // an edit that makes the baseline-valid expression fail to build is a compile failure
        let e = evaluate_with(&forum, &cfg, "", &pipe, Some(&edited(SILU, "[(silu) `(this-is-not-an-operator ,x)]"))).unwrap();
        assert!(!e.verdict.ok && !e.verdict.gates["compiles"], "{}", render(&e));
    }

    #[test]
    fn a_semantic_change_passes_the_built_in_gates_which_is_why_the_device_gate_exists() {
        let (forum, cfg) = gen_setup("gensem");
        let wrong = edited(SILU, "[(silu) `(/ ,x (+ 2.0 (exp (- ,x))))]");
        let e = evaluate_with(&forum, &cfg, "", &p(&["fold", "licm", "cse", "dce"]), Some(&wrong)).unwrap();
        assert!(e.verdict.ok, "the structure check cannot see this: {}", render(&e));
    }

    #[test]
    fn device_gates_get_the_edited_generator_file() {
        let (forum, mut cfg) = gen_setup("gengate");
        cfg.gates = vec!["kern=test -s \"$TL_TREE_KERNELS_FILE\" && grep -q 'define wg 256' \"$TL_TREE_KERNELS_FILE\" && grep -q '(\\* ,x (/ 1.0' \"$TL_TREE_KERNELS_FILE\"".parse().unwrap()];
        let text = edited(SILU, "[(silu) `(* ,x (/ 1.0 (+ 1.0 (exp (- ,x)))))]");
        let e = evaluate_with(&forum, &cfg, "", &p(&["fold"]), Some(&text)).unwrap();
        assert_eq!(e.verdict.gates["kern"], true, "{}", render(&e));
        // without an edit the variable is not set for the gate
        cfg.gates = vec!["none=test -z \"$TL_TREE_KERNELS_FILE\"".parse().unwrap()];
        let e = evaluate(&forum, &cfg, "", &p(&["fold"])).unwrap();
        assert_eq!(e.verdict.gates["none"], true, "{}", render(&e));
    }

    #[test]
    fn gate_syntax() {
        assert!("a=b".parse::<GateCommand>().is_ok());
        assert!("noequals".parse::<GateCommand>().is_err());
        assert!("bad name=x".parse::<GateCommand>().is_err());
    }
}
