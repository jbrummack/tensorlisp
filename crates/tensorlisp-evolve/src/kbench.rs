//! Kernel-level benchmarks: what a candidate does to the time of the kernels a real run launches.
//!
//! Code size says little about speed. This replays the launches captured from a real run
//! (`TL_METAL_LAUNCHES`, stored by `weights capture`: expression, grid, push constants, buffer sizes) on
//! random data with the reference kernel (original generators, default passes) and with the candidate's,
//! on the real GPU, and reports per kernel
//!
//! * whether the outputs agree (the candidate is run on the same inputs; bit-identical, or within a
//!   tolerance relative to the output's scale),
//! * the GPU time per launch of both, as the median of interleaved trials (each trial is one command buffer
//!   of enough serial launches to last a few milliseconds, timed by the GPU itself),
//! * the ratio of the fast-state (lower quartile) times, and the noise: the same reference timed twice.
//!
//! The model-level number is the geometric mean of the per-kernel ratios weighted by each kernel's share of
//! the profiled time. Kernels whose code did not change are not run: ratio 1 by construction. Metal only for
//! now; the other backends need their own `submit` timing, the rest is generic.
use std::sync::Mutex;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::scheme::{Compiled, Row};
use crate::weights::Launch;

/// One benchmarked kernel (expression), over the launches measured for it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelResult {
    pub expr: String,
    /// `identical` (same code as the reference, not run), `ok`, `mismatch` (outputs differ), `error`, `unmeasured`.
    pub status: String,
    pub note: String,
    /// Microseconds per launch, reference and candidate.
    pub ref_us: f64,
    pub cand_us: f64,
    /// reference / candidate: above 1 is faster.
    pub ratio: f64,
    /// How far the reference drifts from itself on the same kernel (|ratio - 1| of two runs of identical code):
    /// a measured ratio inside this is noise.
    pub spread: f64,
    pub max_err: f64,
    pub launches: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Report {
    pub kernels: Vec<KernelResult>,
    /// Weighted geometric mean of the ratios over the kernels with a weight (1.0 when nothing changed).
    pub speedup: f64,
    /// Weighted mean of the spreads of the kernels that were run.
    pub noise: f64,
    /// Share of the profiled time covered by kernels that were compared.
    pub coverage: f64,
    pub ok: bool,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub trials: usize,
    /// Wanted duration of one timed command buffer.
    pub target_ms: f64,
    pub samples_per_kernel: usize,
    /// Largest accepted difference relative to the reference output's largest magnitude.
    pub tolerance: f64,
    /// Upper bound on the bytes bound by one launch (both sides are allocated).
    pub max_bytes: u64,
    /// Distinct input copies are made until they add up to this many bytes (more than the GPU's caches hold).
    pub cold_bytes: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options { trials: 15, target_ms: 3.0, samples_per_kernel: 2, tolerance: 2e-3, max_bytes: 3 << 30, cold_bytes: 128 << 20 }
    }
}

/// Generators whose inputs include index buffers: random data would read out of bounds, so these are only
/// replayed with the contents of the small buffers a real launch had (`Launch::data`).
pub fn reads_indices(expr: &str) -> bool {
    ["get-rows", "set-rows", "rope", "argmax", "paged", "kv-", "gather", "scatter", "top-k", "argsort"].iter().any(|k| expr.contains(k))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The 25th percentile.
fn lower_quartile(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    if v.is_empty() { 0.0 } else { v[(v.len() - 1) / 4] }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Finite half-precision bits for `x` in (-2, 2); tiny values flush to zero.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign as u16;
    }
    (sign | ((exp.min(30) as u32) << 10) | ((b >> 13) & 0x3ff)) as u16
}

/// Deterministic test data: every 16-bit word is a finite half in [-1, 1], so f16 data, f32 data (two
/// halves make a small finite float) and quantized blocks (a half scale first) are all well-behaved.
pub fn fill(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        // four halves per random word
        let r = xorshift(&mut s);
        for k in 0..4 {
            let x = ((r >> (16 * k)) & 0xffff) as f32 / 32768.0 - 1.0;
            out.extend_from_slice(&f16_bits(x).to_le_bytes());
        }
    }
    out.truncate(len);
    out
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Largest difference between two buffers relative to the reference's largest finite magnitude, reading
/// the bytes as f32 and as f16 and taking the smaller (the element type is not known here). 0 when the
/// bytes are identical; infinite when one side has a NaN or infinity the other does not.
pub fn max_error(reference: &[u8], candidate: &[u8]) -> f64 {
    if reference == candidate {
        return 0.0;
    }
    if reference.len() != candidate.len() {
        return f64::INFINITY;
    }
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let f16s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect() };
    let err = |a: Vec<f32>, b: Vec<f32>| -> f64 {
        let scale = a.iter().filter(|x| x.is_finite()).fold(0.0f32, |m, x| m.max(x.abs())) as f64;
        let mut worst = 0.0f64;
        for (x, y) in a.iter().zip(&b) {
            if x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()) {
                continue;
            }
            if !x.is_finite() || !y.is_finite() {
                return f64::INFINITY;
            }
            worst = worst.max((*x as f64 - *y as f64).abs());
        }
        worst / scale.max(1e-6)
    };
    let e32 = if reference.len() % 4 == 0 { err(f32s(reference), f32s(candidate)) } else { f64::INFINITY };
    let e16 = if reference.len() % 2 == 0 { err(f16s(reference), f16s(candidate)) } else { f64::INFINITY };
    e32.min(e16)
}

/// The weighted geometric mean of `(ratio, weight)`, 1.0 for no weight.
pub fn weighted_geomean(items: &[(f64, f64)]) -> f64 {
    let w: f64 = items.iter().map(|i| i.1).sum();
    if w <= 0.0 {
        return 1.0;
    }
    (items.iter().map(|(r, w)| w * r.max(1e-9).ln()).sum::<f64>() / w).exp()
}

/// Compares candidate and reference kernels on the captured launches. `weights`: expression -> share of
/// profiled time. Needs a GPU; one benchmark runs at a time (the GPU is shared by every agent of the process).
pub fn run(rows: &[Row], launches: &[Launch], weights: &std::collections::BTreeMap<String, f64>, opts: &Options) -> Result<Report> {
    static GPU: Mutex<()> = Mutex::new(());
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut results = Vec::new();
    for row in rows.iter().filter(|r| r.status == "ok") {
        let mine: Vec<&Launch> = launches.iter().filter(|l| l.expr == row.expr).collect();
        if mine.is_empty() {
            continue;
        }
        let count: u64 = mine.iter().map(|l| l.count).sum();
        let mut res = KernelResult { expr: row.expr.clone(), status: "ok".into(), note: String::new(), ref_us: 0.0, cand_us: 0.0, ratio: 1.0, spread: 0.0, max_err: 0.0, launches: count };
        match (&row.base_code, &row.cand_code) {
            (Some(b), Some(c)) if b == c => res.status = "identical".into(),
            (Some(b), Some(c)) => {
                // launches that carry the contents of their small buffers can replay index readers
                let usable: Vec<&Launch> = if reads_indices(&row.expr) { mine.iter().copied().filter(|l| l.data.iter().any(|d| !d.is_empty())).collect() } else { mine.clone() };
                if usable.is_empty() {
                    res.status = "unmeasured".into();
                    res.note = "reads index buffers and the captured launches have no copy of them (capture again with `weights capture`): random data would read out of bounds".into();
                } else {
                    let mut mine = usable;
                    mine.sort_by_key(|l| std::cmp::Reverse(l.count * l.bind.iter().sum::<u64>().max(1)));
                    match compare(&row.expr, b, c, &mine[..mine.len().min(opts.samples_per_kernel)], opts) {
                        Ok(m) => {
                            res.ref_us = m.ref_us;
                            res.cand_us = m.cand_us;
                            res.ratio = m.ratio;
                            res.spread = m.spread;
                            res.max_err = m.max_err;
                            if m.max_err > opts.tolerance {
                                res.status = "mismatch".into();
                                res.note = format!("outputs differ from the reference by {:.3e} of their scale (tolerance {:.0e})", m.max_err, opts.tolerance);
                            }
                        }
                        Err(e) => {
                            res.status = "error".into();
                            res.note = format!("{e}");
                        }
                    }
                }
            }
            _ => {
                res.status = "unmeasured".into();
                res.note = "no compiled code for this target".into();
            }
        }
        results.push(res);
    }
    let total: f64 = weights.values().sum::<f64>().max(1e-12);
    let mut items = Vec::new();
    let mut covered = 0.0;
    let (mut noise, mut noise_w) = (0.0, 0.0);
    for r in &results {
        let w = weights.get(&r.expr).copied().unwrap_or(0.0);
        if matches!(r.status.as_str(), "ok" | "identical") {
            items.push((r.ratio, w));
            covered += w;
        }
        if r.status == "ok" {
            noise += w * r.spread;
            noise_w += w;
        }
    }
    let ok = !results.iter().any(|r| matches!(r.status.as_str(), "mismatch" | "error"));
    Ok(Report { kernels: results, speedup: weighted_geomean(&items), noise: if noise_w > 0.0 { noise / noise_w } else { 0.0 }, coverage: covered / total, ok })
}

struct Measured {
    ref_us: f64,
    cand_us: f64,
    ratio: f64,
    spread: f64,
    max_err: f64,
}

#[cfg(not(target_os = "macos"))]
fn compare(_: &str, _: &Compiled, _: &Compiled, _: &[&Launch], _: &Options) -> Result<Measured> {
    bail!("kernel benchmarks run on Metal only for now")
}

#[cfg(target_os = "macos")]
fn blob(label: &str, c: &Compiled) -> tensorlisp_kernels::sched::KernelBlob {
    use tensorlisp_kernels::sched::{Access, Code, KernelBlob, OffsetMode, Signature, Target};
    KernelBlob {
        target: Target::Metal,
        code: Code::Msl(c.text.clone()),
        entry: c.entry.clone(),
        label: format!("tree:{label}"),
        sig: Signature {
            bindings: c.accesses.iter().map(|a| match a.as_str() { "ro" => Access::Read, "wo" => Access::Write, _ => Access::ReadWrite }).collect(),
            push_bytes: c.push_bytes,
            workgroup: c.workgroup,
            offsets: OffsetMode::Prefix,
        },
        spec: vec![],
    }
}

#[cfg(target_os = "macos")]
fn metal() -> Result<&'static tensorlisp_kernels::metal::backend::Metal> {
    use tensorlisp_kernels::metal::backend::Metal;
    static M: std::sync::OnceLock<std::result::Result<Metal, String>> = std::sync::OnceLock::new();
    M.get_or_init(|| Metal::system_default().map_err(|e| e.to_string())).as_ref().map_err(|e| anyhow::anyhow!("no Metal device: {e}"))
}

#[cfg(target_os = "macos")]
fn compare(expr: &str, base: &Compiled, cand: &Compiled, samples: &[&Launch], opts: &Options) -> Result<Measured> {
    use tensorlisp_kernels::sched::Backend;
    use tensorlisp_kernels::sched::backend::{Bound, Launch as Dispatch};
    let m = metal()?;
    let (pb, pc) = (m.pipeline(&blob(expr, base)).map_err(|e| anyhow::anyhow!("reference: {e}"))?, m.pipeline(&blob(expr, cand)).map_err(|e| anyhow::anyhow!("candidate: {e}"))?);
    if base.accesses != cand.accesses || base.push_bytes != cand.push_bytes || base.workgroup != cand.workgroup {
        bail!("the candidate's interface differs from the reference");
    }
    let (mut ref_t, mut cand_t, mut ratios, mut worst) = (Vec::new(), Vec::new(), Vec::new(), 0.0f64);
    let mut weight_sum = 0.0;
    let mut weighted: Vec<(f64, f64)> = Vec::new();
    for (si, s) in samples.iter().enumerate() {
        if s.bind.len() != base.accesses.len() {
            bail!("the captured launch binds {} buffers, the kernel takes {}", s.bind.len(), base.accesses.len());
        }
        let push = s.push_bytes();
        // A real run streams each weight matrix from memory once per step, while replaying one launch over and
        // over would keep its buffers in the GPU's caches and measure arithmetic alone. So every side gets
        // enough distinct copies of the inputs to overflow the caches, and consecutive launches rotate through them.
        let bytes: u64 = s.bind.iter().sum::<u64>().max(1);
        let copies = (opts.cold_bytes / bytes).clamp(1, 64) as usize;
        // buffers the kernel only reads are shared by both sides (same inputs by construction, and half the memory:
        // embedding tables are gigabytes); the ones it writes are the side's own. `sets[side][copy][binding]` indexes `store`.
        let ro_bytes: u64 = base.accesses.iter().zip(&s.bind).filter(|(a, _)| a.as_str() == "ro").map(|(_, b)| *b).sum();
        if (ro_bytes + 2 * (bytes - ro_bytes)) * copies as u64 > opts.max_bytes {
            bail!("a launch binding {} MiB is too large to replay", bytes >> 20);
        }
        let mut store = Vec::new();
        let mut sets: Vec<Vec<Vec<usize>>> = vec![Vec::new(), Vec::new()];
        for c in 0..copies {
            let mut shared: Vec<Option<usize>> = vec![None; s.bind.len()];
            for side in 0..2 {
                let mut row = Vec::new();
                for (i, &len) in s.bind.iter().enumerate() {
                    let ro = base.accesses[i] == "ro";
                    if ro && let Some(k) = shared[i] {
                        row.push(k);
                        continue;
                    }
                    let data = s.binding_data(i).unwrap_or_else(|| fill(len as usize, 1 + si as u64 * 31 + i as u64 + c as u64 * 977));
                    let b = m.alloc(len + 64, tensorlisp_kernels::sched::MemKind::Step)?;
                    m.write(&b, 0, &data)?;
                    store.push(b);
                    row.push(store.len() - 1);
                    if ro {
                        shared[i] = Some(store.len() - 1);
                    }
                }
                sets[side].push(row);
            }
        }
        let dispatch = |p: &'_ <tensorlisp_kernels::metal::backend::Metal as Backend>::Pipeline, side: &'_ Vec<Vec<usize>>, n: usize| -> Result<f64> {
            let groups: Vec<Vec<Dispatch<'_, tensorlisp_kernels::metal::backend::Metal>>> = (0..n)
                .map(|k| vec![Dispatch { pipeline: p, bindings: side[k % side.len()].iter().zip(&s.bind).map(|(&b, &len)| Bound { buffer: &store[b], offset: 0, len }).collect(), push: &push, grid: s.grid }])
                .collect();
            let pending = m.submit(&groups)?;
            let secs = m.gpu_seconds(&pending);
            m.wait(&pending)?;
            Ok(secs)
        };
        // outputs after one launch on identical inputs
        dispatch(&pb, &sets[0], 1)?;
        dispatch(&pc, &sets[1], 1)?;
        for (i, acc) in base.accesses.iter().enumerate().filter(|(_, a)| a.as_str() != "ro") {
            let _ = acc;
            let (mut a, mut b) = (vec![0u8; s.bind[i] as usize], vec![0u8; s.bind[i] as usize]);
            m.read(&store[sets[0][0][i]], 0, &mut a)?;
            m.read(&store[sets[1][0][i]], 0, &mut b)?;
            worst = worst.max(max_error(&a, &b));
        }
        // how many serial launches make a timed submission long enough
        let t1 = dispatch(&pb, &sets[0], 1)?.max(1e-6);
        let reps = ((opts.target_ms * 1e-3 / t1).ceil() as usize).clamp(1, 400).max(copies);
        // keep the GPU busy for a while first: it clocks down when idle, and the first trials would be slow
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 40 {
            dispatch(&pb, &sets[0], reps)?;
            dispatch(&pc, &sets[1], reps)?;
        }
        // three timed sides per trial in rotating order: the reference, the candidate and the reference again
        // (the same code twice measures how far identical kernels drift apart, i.e. the noise)
        let (mut rt, mut ct, mut nt) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..opts.trials {
            let mut got = [0.0f64; 3];
            for k in 0..3 {
                let side = (t + k) % 3;
                let (p, set) = if side == 1 { (&pc, &sets[1]) } else { (&pb, &sets[0]) };
                got[side] = dispatch(p, set, reps)? / reps as f64 * 1e6;
            }
            rt.push(got[0]);
            ct.push(got[1]);
            nt.push(got[2]);
        }
        if std::env::var_os("TL_KBENCH_DEBUG").is_some() {
            eprintln!("kbench {expr}: reps {reps}, ref us {rt:.1?}, cand us {ct:.1?}, ref again us {nt:.1?}");
        }
        // the lower quartile: the GPU alternates between power states, and the fast state is the stable one
        let (r, c, n) = (lower_quartile(&rt), lower_quartile(&ct), lower_quartile(&nt));
        let w = s.count as f64 * r;
        weighted.push((r / c.max(1e-12), w));
        weight_sum += w;
        ref_t.push(r);
        cand_t.push(c);
        ratios.push(((r / n.max(1e-12)) - 1.0).abs());
    }
    let _ = weight_sum;
    let ratio = weighted_geomean(&weighted);
    Ok(Measured { ref_us: median(&mut ref_t), cand_us: median(&mut cand_t), ratio, spread: ratios.iter().cloned().fold(0.0, f64::max), max_err: worst })
}

/// A table for the agent.
pub fn render(r: &Report, limit: usize) -> String {
    let mut out = format!("kernel benchmark: weighted speedup {:.4} (typical noise {:.1}%), covers {:.0}% of the profiled time\n", r.speedup, r.noise * 100.0, r.coverage * 100.0);
    let mut rows: Vec<&KernelResult> = r.kernels.iter().filter(|k| k.status != "identical").collect();
    rows.sort_by(|a, b| (b.launches as f64 * b.ref_us).total_cmp(&(a.launches as f64 * a.ref_us)));
    let same = r.kernels.len() - rows.len();
    for k in rows.iter().take(limit) {
        if k.status == "ok" {
            let verdict = if (k.ratio - 1.0).abs() <= 2.0 * k.spread + 0.01 { "within noise" } else if k.ratio > 1.0 { "faster" } else { "SLOWER" };
            out.push_str(&format!("  {:.3}x {verdict} ({:.1} -> {:.1} us, noise {:.1}%, {} launches)  {}\n", k.ratio, k.ref_us, k.cand_us, k.spread * 100.0, k.launches, k.expr));
        } else {
            out.push_str(&format!("  [{}] {}  {}\n", k.status, k.expr, k.note));
        }
    }
    if same > 0 {
        out.push_str(&format!("  {same} kernels compile to the same code as the reference (not run)\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_is_finite_deterministic_and_half_sized() {
        let a = fill(1001, 3);
        assert_eq!(a, fill(1001, 3));
        assert_ne!(a, fill(1001, 4));
        assert_eq!(a.len(), 1001);
        for c in a.chunks_exact(2) {
            let h = half_to_f32(u16::from_le_bytes([c[0], c[1]]));
            assert!(h.is_finite() && h.abs() <= 1.01, "{h}");
        }
        for c in a.chunks_exact(4) {
            assert!(f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite());
        }
    }

    #[test]
    fn errors_are_relative_to_the_scale_and_read_both_element_types() {
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        assert_eq!(max_error(&f32s(&[1.0, 2.0]), &f32s(&[1.0, 2.0])), 0.0);
        let e = max_error(&f32s(&[1.0, 100.0]), &f32s(&[1.0, 100.1]));
        assert!(e > 9e-4 && e < 1.1e-3, "{e}");
        assert!(max_error(&f32s(&[1.0]), &f32s(&[f32::NAN])).is_infinite());
        // halves: 1.0 vs the next half up differ by 2^-10 relative
        let h = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let e = max_error(&h(&[0x3c00, 0x3c00]), &h(&[0x3c00, 0x3c01]));
        assert!(e < 1.1e-3 && e > 0.0, "{e}");
        assert!(max_error(&[0; 4], &[0; 8]).is_infinite());
    }

    #[test]
    fn the_speedup_is_the_weighted_geometric_mean() {
        assert_eq!(weighted_geomean(&[]), 1.0);
        let g = weighted_geomean(&[(2.0, 1.0), (0.5, 1.0)]);
        assert!((g - 1.0).abs() < 1e-12);
        let g = weighted_geomean(&[(2.0, 3.0), (1.0, 1.0)]);
        assert!((g - 2f64.powf(0.75)).abs() < 1e-12);
    }

    #[test]
    fn index_readers_are_not_replayed_on_random_data() {
        assert!(reads_indices("(k:get-rows 'q4_0)"));
        assert!(!reads_indices("(k:matvec 'q4_0 1 4)"));
    }

    fn row(expr: &str, base: Option<Compiled>, cand: Option<Compiled>) -> Row {
        Row { expr: expr.into(), status: "ok".into(), note: String::new(), base_size: 1, cand_size: 1, bytes: None, base_bytes: None, cand_code: cand, base_code: base }
    }

    fn code(text: &str) -> Compiled {
        Compiled { text: text.into(), entry: "k".into(), accesses: vec!["ro".into(), "wo".into()], push_bytes: 4, workgroup: [64, 1, 1] }
    }

    fn launch(expr: &str) -> Launch {
        Launch { expr: expr.into(), grid: [1, 1, 1], push: "00000000".into(), bind: vec![256, 256], data: vec![], count: 5 }
    }

    #[test]
    fn unchanged_kernels_are_not_run_and_count_as_one() {
        let rows = [row("(k:a)", Some(code("x")), Some(code("x")))];
        let w = [("(k:a)".to_string(), 0.5)].into_iter().collect();
        let r = run(&rows, &[launch("(k:a)")], &w, &Options::default()).unwrap();
        assert_eq!((r.kernels[0].status.as_str(), r.speedup, r.ok), ("identical", 1.0, true));
        assert!((r.coverage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kernels_without_a_launch_record_are_left_out() {
        let rows = [row("(k:b)", Some(code("x")), Some(code("y")))];
        let r = run(&rows, &[launch("(k:a)")], &Default::default(), &Options::default()).unwrap();
        assert!(r.kernels.is_empty());
        assert_eq!(r.speedup, 1.0);
    }

    #[cfg(target_os = "macos")]
    mod gpu {
        use super::*;

        fn msl(body: &str) -> Compiled {
            Compiled {
                text: format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {{ {body} }}\n"),
                entry: "k".into(),
                accesses: vec!["ro".into(), "wo".into()],
                push_bytes: 0,
                workgroup: [64, 1, 1],
            }
        }

        fn big(expr: &str) -> Launch {
            Launch { expr: expr.into(), grid: [1024, 1, 1], push: String::new(), bind: vec![1 << 18, 1 << 18], data: vec![], count: 10 }
        }

        #[test]
        fn a_faster_equal_kernel_a_slower_one_and_a_wrong_one_are_told_apart() {
            let base = "float x = a[i]; for (int j = 0; j < 400; j++) { x = fma(x, 0.999f, 0.001f); } o[i] = x;";
            let wrong = "o[i] = a[i] + 1.0f;";
            let launches = [big("(k:fast)"), big("(k:wrong)"), big("(k:same)")];
            // `fast` computes the same values with a trivially different expression
            let rows = [row("(k:fast)", Some(msl(base)), Some(msl(&base.replace("o[i] = x;", "o[i] = x + 0.0f;")))), row("(k:wrong)", Some(msl(base)), Some(msl(wrong))), row("(k:same)", Some(msl(base)), Some(msl(base)))];
            let w = [("(k:fast)".to_string(), 0.5), ("(k:wrong)".to_string(), 0.3), ("(k:same)".to_string(), 0.2)].into_iter().collect();
            let r = run(&rows, &launches, &w, &Options::default()).unwrap();
            let by = |e: &str| r.kernels.iter().find(|k| k.expr == e).unwrap();
            assert_eq!(by("(k:same)").status, "identical");
            assert_eq!(by("(k:wrong)").status, "mismatch", "{:?}", by("(k:wrong)"));
            assert!(by("(k:fast)").status == "ok" && (0.8..1.25).contains(&by("(k:fast)").ratio), "{:?}", by("(k:fast)"));
            assert!(!r.ok);
        }

        fn gather(body: &str) -> Compiled {
            Compiled {
                text: format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device const float* a [[buffer(0)]], device const uint* idx [[buffer(1)]], device float* o [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {{ {body} }}\n"),
                entry: "k".into(),
                accesses: vec!["ro".into(), "ro".into(), "wo".into()],
                push_bytes: 0,
                workgroup: [64, 1, 1],
            }
        }

        #[test]
        fn index_readers_run_with_the_captured_indices_and_not_without() {
            let base = "o[i] = a[idx[i % 64]] * 2.0f;";
            let cand = "o[i] = a[idx[i % 64]] + a[idx[i % 64]];";
            let rows = [row("(k:get-rows 'x)", Some(gather(base)), Some(gather(cand)))];
            // 64 valid row numbers into a 1024-float table; random words would index far outside it
            let idx: Vec<u8> = (0..64u32).flat_map(|k| ((k * 7) % 1024).to_le_bytes()).collect();
            let hex: String = idx.iter().map(|b| format!("{b:02x}")).collect();
            let mut with = Launch { expr: "(k:get-rows 'x)".into(), grid: [16, 1, 1], push: String::new(), bind: vec![4096, 256, 4096], data: vec![String::new(), hex, String::new()], count: 4 };
            let w = [("(k:get-rows 'x)".to_string(), 1.0)].into_iter().collect();
            let r = run(&rows, &[with.clone()], &w, &Options::default()).unwrap();
            assert_eq!(r.kernels[0].status, "ok", "{:?}", r.kernels[0]);
            assert!(r.ok && r.kernels[0].max_err == 0.0);
            with.data.clear();
            let r = run(&rows, &[with], &w, &Options::default()).unwrap();
            assert_eq!(r.kernels[0].status, "unmeasured");
            assert!(r.kernels[0].note.contains("capture again"), "{:?}", r.kernels[0]);
        }

        #[test]
        fn a_kernel_doing_a_quarter_of_the_work_is_measured_faster() {
            let slow = "float x = a[i]; for (int j = 0; j < 800; j++) { x = fma(x, 0.999f, 0.001f); } o[i] = x;";
            // the same values; the reference additionally spends time in a loop whose result never reaches the output
            let slower = "float x = a[i]; for (int j = 0; j < 800; j++) { x = fma(x, 0.999f, 0.001f); } float d = x; for (int j = 0; j < 3200; j++) { d = fma(d, 0.5f, 0.0f) + 1e-30f; } o[i] = x + (d < -1e30f ? 1.0f : 0.0f);";
            let rows = [row("(k:t)", Some(msl(slower)), Some(msl(slow)))];
            let w = [("(k:t)".to_string(), 1.0)].into_iter().collect();
            let r = run(&rows, &[big("(k:t)")], &w, &Options::default()).unwrap();
            let k = &r.kernels[0];
            assert_eq!(k.status, "ok", "{k:?}");
            assert!(k.ratio > 1.5, "{k:?}");
            assert!(r.speedup > 1.5);
        }
    }
}
