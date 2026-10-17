//! `tl-evolve bench`: how much faster is the candidate, measured on the device.
//!
//! Code size says little about speed, and nothing about a generator change that unrolls more. The profilers
//! (`TL_VULKAN_PROFILE`, `TL_CUDA_PROFILE`, `TL_METAL_PROFILE`) give GPU time per kernel for a command. This
//! runs the command with the profiler on several times *without* the candidate (the environment variables a
//! gate sets are removed, the kernel cache stays on) and several times *with* it (the environment as the gate
//! gave it), takes the fastest time per launch of each kernel over the runs (the minimum is the standard
//! robust statistic for GPU timings: noise only ever adds time), leaves out kernels launched fewer than
//! `min_launches` times (a single cold launch can be ten times slower than a warm one), and combines the
//! ratios as a weighted geometric mean (profile weights from the forum for the device, else the baseline's own time shares). Disagreement
//! between the baseline runs themselves is reported as noise: a speedup inside it is not a result.
//!
//! The runs alternate (baseline, candidate, baseline, ...) so that thermal drift and background load affect both
//! sides alike.
//!
//! Use it as a gate: `--gate 'metal-time=tl-evolve bench --backend metal --device m1pro -- cargo test ...'`.
//! It prints `TL_EVOLVE_METRIC speedup=` and `noise=`, which the harness records (`speedup` replaces the size
//! score in the verdict).
use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::weights;

/// expr -> (launches, busy ms) of one profiled run.
pub type Run = BTreeMap<String, (u64, f64)>;

/// Variables a gate sets for a candidate: removed for the baseline runs.
pub const CANDIDATE_VARS: &[&str] = &["TL_TREE_PASS_FILE", "TL_TREE_PASSES", "TL_TREE_KERNELS_FILE", "TL_KERNEL_CACHE"];

pub fn profiled_run(backend: &str, command: &str, baseline: bool, extra_env: &[(&str, &str)]) -> Result<Run> {
    let text = if baseline { weights::capture_env(backend, command, CANDIDATE_VARS, &[])? } else { weights::capture_env(backend, command, &[], extra_env)? };
    let w = weights::parse_profile(&text, "bench", "bench", 1.0)?;
    Ok(w.kernels.into_iter().map(|k| (k.expr, (k.launches, k.busy_ms))).collect())
}

fn fastest(v: Vec<f64>) -> f64 {
    v.into_iter().fold(f64::INFINITY, f64::min)
}

fn per_launch_us(run: &Run, expr: &str, min_launches: u64) -> Option<f64> {
    run.get(expr).filter(|(l, ms)| *l >= min_launches.max(1) && *ms > 0.0).map(|(l, ms)| ms * 1000.0 / *l as f64)
}

#[derive(Debug)]
pub struct Row {
    pub expr: String,
    pub weight: f64,
    pub base_us: f64,
    pub cand_us: f64,
}

#[derive(Debug)]
pub struct Comparison {
    /// Weighted geometric mean of baseline time / candidate time: above 1 is faster.
    pub speedup: f64,
    /// Typical relative disagreement between the baseline runs (0.02 = 2%); `None` with a single baseline run.
    pub noise: Option<f64>,
    pub rows: Vec<Row>,
}

/// `weights`: kernel weights (empty: each kernel by its share of the baseline's time).
pub fn compare(base: &[Run], cand: &[Run], weights: &BTreeMap<String, f64>, min_launches: u64) -> Result<Comparison> {
    if base.is_empty() || cand.is_empty() {
        bail!("need at least one baseline and one candidate run");
    }
    let exprs: Vec<&String> = base[0].keys().filter(|e| base.iter().all(|r| per_launch_us(r, e, min_launches).is_some()) && cand.iter().all(|r| per_launch_us(r, e, min_launches).is_some())).collect();
    if exprs.is_empty() {
        bail!("no Tree kernel with at least {min_launches} launches appears in every run: the runs profiled different things, or the command launches each kernel too few times for timing (lower --min-launches, or use a longer workload)");
    }
    let total_base: f64 = exprs.iter().map(|e| base[0][*e].1).sum();
    let mut rows: Vec<Row> = exprs
        .iter()
        .map(|e| {
            let w = if weights.is_empty() { base[0][*e].1 / total_base.max(1e-12) } else { weights.get(*e).copied().unwrap_or(0.0) };
            Row { expr: (*e).clone(), weight: w, base_us: fastest(base.iter().filter_map(|r| per_launch_us(r, e, min_launches)).collect()), cand_us: fastest(cand.iter().filter_map(|r| per_launch_us(r, e, min_launches)).collect()) }
        })
        .collect();
    if rows.iter().all(|r| r.weight == 0.0) {
        // the profile weights name other kernels: fall back to time shares
        for r in rows.iter_mut() {
            r.weight = base[0][&r.expr].1 / total_base.max(1e-12);
        }
    }
    let wsum: f64 = rows.iter().map(|r| r.weight).sum();
    let speedup = (rows.iter().map(|r| r.weight * (r.base_us / r.cand_us).ln()).sum::<f64>() / wsum).exp();
    // spread of the baseline runs: per kernel the ratio of its slowest to its fastest run, weighted
    let noise = (base.len() >= 2).then(|| {
        let spread = |e: &str| {
            let t: Vec<f64> = base.iter().filter_map(|r| per_launch_us(r, e, min_launches)).collect();
            (t.iter().cloned().fold(0.0, f64::max) / t.iter().cloned().fold(f64::INFINITY, f64::min)).ln()
        };
        (rows.iter().map(|r| r.weight * spread(&r.expr)).sum::<f64>() / wsum).exp() - 1.0
    });
    rows.sort_by(|a, b| (b.weight * (b.base_us / b.cand_us).ln().abs()).total_cmp(&(a.weight * (a.base_us / a.cand_us).ln().abs())));
    Ok(Comparison { speedup, noise, rows })
}

pub fn report(c: &Comparison) -> String {
    let mut out = format!("bench: weighted speedup {:.4} over {} kernels", c.speedup, c.rows.len());
    match c.noise {
        Some(n) => {
            out.push_str(&format!(" (run-to-run noise {:.1}%{})\n", n * 100.0, if (c.speedup - 1.0).abs() <= n { ": the difference is within noise" } else { "" }));
        }
        None => out.push_str(" (one baseline run: noise unknown)\n"),
    }
    out.push_str("biggest movers (weight, baseline us/launch -> candidate):\n");
    for r in c.rows.iter().take(8) {
        out.push_str(&format!("  {:>5.1}%  {:>9.1} -> {:>9.1}  x{:.3}  {}\n", r.weight * 100.0, r.base_us, r.cand_us, r.base_us / r.cand_us, r.expr));
    }
    out.push_str(&format!("TL_EVOLVE_METRIC speedup={:.5}\n", c.speedup));
    if let Some(n) = c.noise {
        out.push_str(&format!("TL_EVOLVE_METRIC noise={n:.5}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(rows: &[(&str, u64, f64)]) -> Run {
        rows.iter().map(|(e, l, ms)| (e.to_string(), (*l, *ms))).collect()
    }

    #[test]
    fn the_speedup_is_a_weighted_geometric_mean_of_per_launch_ratios() {
        let base = [run(&[("(k:a)", 10, 1.0), ("(k:b)", 10, 1.0)]), run(&[("(k:a)", 10, 1.02), ("(k:b)", 10, 0.98)])];
        // a is twice as fast, b unchanged; launches differ in count only by workload size
        let cand = [run(&[("(k:a)", 10, 0.5), ("(k:b)", 10, 1.0)]), run(&[("(k:a)", 10, 0.5), ("(k:b)", 10, 1.0)])];
        let w = BTreeMap::from([("(k:a)".to_string(), 0.75), ("(k:b)".to_string(), 0.25)]);
        let c = compare(&base, &cand, &w, 1).unwrap();
        assert!((c.speedup - 2f64.powf(0.75)).abs() < 0.03, "{}", c.speedup);
        assert!(c.noise.unwrap() > 0.015 && c.noise.unwrap() < 0.03, "{:?}", c.noise);
        assert_eq!(c.rows[0].expr, "(k:a)");
        // without weights, kernels count by their share of the baseline's time (equal here)
        let c = compare(&base, &cand, &BTreeMap::new(), 1).unwrap();
        assert!((c.speedup - 2f64.sqrt()).abs() < 0.03, "{}", c.speedup);
        // weights that name other kernels fall back to time shares
        let c = compare(&base, &cand, &BTreeMap::from([("(k:zzz)".to_string(), 1.0)]), 1).unwrap();
        assert!((c.speedup - 2f64.sqrt()).abs() < 0.03);
        let text = report(&c);
        assert!(text.contains("TL_EVOLVE_METRIC speedup=1.4") && text.contains("TL_EVOLVE_METRIC noise="), "{text}");
    }

    #[test]
    fn noise_is_reported_and_a_difference_inside_it_is_called_noise() {
        let base = [run(&[("(k:a)", 1, 1.0)]), run(&[("(k:a)", 1, 1.10)])];
        let cand = [run(&[("(k:a)", 1, 0.97)]), run(&[("(k:a)", 1, 0.99)])];
        let c = compare(&base, &cand, &BTreeMap::new(), 1).unwrap();
        assert!(report(&c).contains("within noise"), "{}", report(&c));
        let one = compare(&base[..1], &cand[..1], &BTreeMap::new(), 1).unwrap();
        assert!(one.noise.is_none() && report(&one).contains("noise unknown") && !report(&one).contains("TL_EVOLVE_METRIC noise"));
    }

    #[test]
    fn the_fastest_run_counts_and_kernels_launched_too_rarely_are_left_out() {
        // a is launched often and stable; b once, with a cold first launch ten times slower in the baseline
        let base = [run(&[("(k:a)", 100, 10.0), ("(k:b)", 1, 5.0)]), run(&[("(k:a)", 100, 12.0), ("(k:b)", 1, 0.5)])];
        let cand = [run(&[("(k:a)", 100, 5.0), ("(k:b)", 1, 0.5)]), run(&[("(k:a)", 100, 9.0), ("(k:b)", 1, 0.5)])];
        let c = compare(&base, &cand, &BTreeMap::new(), 1).unwrap();
        assert_eq!(c.rows.len(), 2);
        // the minimum over runs: a 100us -> 50us per launch
        let a = c.rows.iter().find(|r| r.expr == "(k:a)").unwrap();
        assert_eq!((a.base_us, a.cand_us), (100.0, 50.0));
        // with a launch threshold the single-launch kernel does not take part
        let c = compare(&base, &cand, &BTreeMap::new(), 3).unwrap();
        assert_eq!(c.rows.len(), 1);
        assert!((c.speedup - 2.0).abs() < 1e-9);
        assert!(compare(&base, &cand, &BTreeMap::new(), 1000).unwrap_err().to_string().contains("at least 1000 launches"));
    }

    #[test]
    fn runs_that_profiled_different_things_are_an_error() {
        let a = [run(&[("(k:a)", 1, 1.0)])];
        let b = [run(&[("(k:b)", 1, 1.0)])];
        assert!(compare(&a, &b, &BTreeMap::new(), 1).unwrap_err().to_string().contains("launches appears in every run"));
        assert!(compare(&[], &b, &BTreeMap::new(), 1).is_err());
        // a kernel that only one side launched is left out of the comparison
        let c = compare(&[run(&[("(k:a)", 1, 1.0), ("(k:x)", 1, 5.0)])], &[run(&[("(k:a)", 1, 1.0)])], &BTreeMap::new(), 1).unwrap();
        assert_eq!(c.rows.len(), 1);
    }

    #[test]
    fn candidate_variables_are_removed_for_the_baseline_runs_only() {
        // a command that prints a table whose time depends on whether the candidate's variable is set
        let cmd = r#"if [ -n "$TL_TREE_PASS_FILE" ]; then ms=0.5; else ms=1.0; fi; echo "metal profile: $ms ms over 1 kernels"; echo "$ms ms 10 x 100.0 us tree:(k:argmax) grid=[1, 1, 1]""#;
        let base = profiled_run("metal", cmd, true, &[("TL_TREE_PASS_FILE", "/tmp/x.ss")]).unwrap();
        let cand = profiled_run("metal", cmd, false, &[("TL_TREE_PASS_FILE", "/tmp/x.ss")]).unwrap();
        assert_eq!(base["(k:argmax)"], (10, 1.0));
        assert_eq!(cand["(k:argmax)"], (10, 0.5));
        let c = compare(&[base], &[cand], &BTreeMap::new(), 1).unwrap();
        assert!((c.speedup - 2.0).abs() < 1e-9);
    }
}
