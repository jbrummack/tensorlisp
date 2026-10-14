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
//! * external gates: shell commands run with `TL_TREE_PASS_FILE` and `TL_TREE_PASSES` set, so
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

fn run_gate(gate: &GateCommand, cfg: &EvalConfig, file: &Path, pipeline: &[String]) -> Result<GateRun> {
    let log = cfg.work_dir.join(format!("gate-{}-{}.log", gate.name, std::process::id()));
    let out = std::fs::File::create(&log)?;
    let err = out.try_clone()?;
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(&gate.command)
        .env("TL_TREE_PASS_FILE", file)
        .env("TL_TREE_PASSES", pipeline.join(","))
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

/// Runs every gate on a candidate and stores the result in the forum as an [`Eval`].
pub fn evaluate(forum: &Forum, cfg: &EvalConfig, code: &str, pipeline: &[String]) -> Result<Eval> {
    check_pipeline(pipeline)?;
    std::fs::create_dir_all(&cfg.work_dir)?;
    let id = sha_hex(&[code, &pipeline.join(","), &cfg.target, &cfg.device, &cfg.gates.iter().map(|g| format!("{}={}", g.name, g.command)).collect::<Vec<_>>().join(";")])[..16].to_string();

    let outcome = scheme::handle()?.evaluate(code, pipeline, &cfg.target, &cfg.exprs, cfg.timeout_ms)?;
    let mut gates: BTreeMap<String, bool> = BTreeMap::new();
    let mut metrics: BTreeMap<String, f64> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut rows_json = Vec::new();
    let mut logsum = 0.0f64;
    let mut counted = 0usize;

    match &outcome {
        Outcome::LoadError(msg) => {
            gates.insert("loads".into(), false);
            notes.push(format!("did not load: {msg}"));
        }
        Outcome::Rows(rows) => {
            gates.insert("loads".into(), true);
            let (mut smaller, mut larger, mut skipped) = (0, 0, 0);
            for r in rows {
                rows_json.push(json!({"expr": r.expr, "status": r.status, "note": r.note, "baseline": r.base_size, "candidate": r.cand_size}));
                match r.status.as_str() {
                    "ok" if r.base_size > 0 && r.cand_size > 0 => {
                        logsum += (r.base_size as f64 / r.cand_size as f64).ln();
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

    let mut gate_logs = serde_json::Map::new();
    if !cfg.gates.is_empty() {
        if gates.values().all(|ok| *ok) {
            let dir = cfg.work_dir.join("candidates");
            std::fs::create_dir_all(&dir)?;
            let file = dir.join(format!("{id}.ss"));
            std::fs::write(&file, code)?;
            for g in &cfg.gates {
                let run = run_gate(g, cfg, &file, pipeline)?;
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

    let static_score = if counted > 0 { (logsum / counted as f64).exp() } else { 0.0 };
    metrics.insert("size_ratio".into(), static_score);
    let score = metrics.get("speedup").copied().unwrap_or(static_score);
    let verdict = Verdict { eval_id: id.clone(), target: cfg.target.clone(), device: cfg.device.clone(), ok: gates.values().all(|g| *g), score, gates, metrics };
    let eval = Eval {
        id,
        code: code.to_string(),
        pipeline: pipeline.to_vec(),
        verdict,
        report: json!({"rows": rows_json, "notes": notes, "gate_output": gate_logs}),
        created: now_millis(),
    };
    forum.put_eval(&eval)?;
    Ok(eval)
}

/// A short text report for the agent (and for `tl-evolve eval`).
pub fn render(eval: &Eval) -> String {
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
    fn gate_syntax() {
        assert!("a=b".parse::<GateCommand>().is_ok());
        assert!("noequals".parse::<GateCommand>().is_err());
        assert!("bad name=x".parse::<GateCommand>().is_err());
    }
}
