//! Which kernels matter: weights taken from real runs.
//!
//! `TL_VULKAN_PROFILE=1`, `TL_CUDA_PROFILE=1` and `TL_METAL_PROFILE=1` print a per-kernel GPU time table
//! (formats below) whose labels are `tree:<generator expression>`, exactly the expressions the evaluator
//! compiles. A profile of a workload (a Gemma 4
//! decode step, TIPSv2, ...) imported with `tl-evolve weights add` becomes `weights/<device>/<workload>`
//! in the forum's KV, so every worker sees it. A kernel's weight is its share of the Tree kernels' busy
//! time, averaged over the device's workloads (each workload counts equally, times its own `weight`).
//! The evaluator then scores the weighted geometric mean of size ratios and also compiles the profiled
//! expressions themselves, which are the exact arguments the model uses, not the representative ones in
//! `tree_exprs.txt`.
use std::collections::BTreeMap;

use anyhow::{Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::forum::now_millis;
use crate::kv::Kv;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelTime {
    pub expr: String,
    pub launches: u64,
    pub busy_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub workload: String,
    pub device: String,
    /// Relative importance of this workload among the device's (default 1).
    pub weight: f64,
    /// Everything the profile accounted for, Tree kernels or not.
    pub total_busy_ms: f64,
    pub kernels: Vec<KernelTime>,
    pub imported: u64,
}

/// What the evaluator and the `hot_kernels` tool use.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub workloads: Vec<Workload>,
    /// expression -> weight, summing to 1 (empty when the device has no profiles).
    pub weights: BTreeMap<String, f64>,
}

/// The generator expression in a profile label: `tree:` prefix and Metal's ` grid=[..]` suffix removed.
fn tree_expr(label: &str) -> Option<&str> {
    let mut e = label.strip_prefix("tree:").unwrap_or(label).trim();
    if let Some(i) = e.rfind(" grid=[") {
        e = e[..i].trim_end();
    }
    (e.starts_with("(k:") && e.ends_with(')')).then_some(e)
}

/// Parses what the profilers print, anywhere in `text` (a log with other output, or several tables from one
/// run: the rows add up):
///
/// * Vulkan: `kernel launches busy-ms gap-ms busy-us/launch`, then a `total` row,
/// * CUDA and Metal: `metal profile: 12.3 ms over 20 kernels`, then rows `5.000 ms  120 x  41.7 us  label`
///   (Metal labels end in ` grid=[..]`: one row per grid, summed here),
/// * plain lines `expr launches busy-ms`.
///
/// Rows that are not Tree kernels only count toward the total.
pub fn parse_profile(text: &str, workload: &str, device: &str, weight: f64) -> Result<Workload> {
    let rows_ms = Regex::new(r"^(?P<busy>\d+(?:\.\d+)?) ms\s+(?P<l>\d+) x\s+\d+(?:\.\d+)? us\s+(?P<name>.+?)\s*$")?;
    let header = Regex::new(r"^(?:cuda|metal) profile: (?P<total>\d+(?:\.\d+)?) ms over \d+ kernels")?;
    let four = Regex::new(r"^(?P<name>.+?)\s+(?P<l>\d+)\s+(?P<busy>\d+(?:\.\d+)?)\s+(?P<gap>\d+(?:\.\d+)?)\s+(?P<us>\d+(?:\.\d+)?)\s*$")?;
    let three = Regex::new(r"^(?P<name>\(k:.*\))\s+(?P<l>\d+)\s+(?P<busy>\d+(?:\.\d+)?)\s*$")?;
    let total = Regex::new(r"^total\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")?;
    let mut kernels: BTreeMap<String, KernelTime> = BTreeMap::new();
    let (mut other_ms, mut total_ms) = (0.0, None::<f64>);
    for line in text.lines() {
        let line = line.trim();
        if let Some(c) = total.captures(line) {
            total_ms = Some(total_ms.unwrap_or(0.0) + c[2].parse::<f64>().unwrap_or(0.0));
            continue;
        }
        if let Some(c) = header.captures(line) {
            total_ms = Some(total_ms.unwrap_or(0.0) + c["total"].parse::<f64>().unwrap_or(0.0));
            continue;
        }
        let Some(c) = rows_ms.captures(line).or_else(|| three.captures(line)).or_else(|| four.captures(line)) else { continue };
        let (name, launches, busy): (&str, u64, f64) = (c.name("name").unwrap().as_str(), c["l"].parse()?, c["busy"].parse()?);
        match tree_expr(name) {
            Some(expr) => {
                let k = kernels.entry(expr.to_string()).or_insert(KernelTime { expr: expr.into(), launches: 0, busy_ms: 0.0 });
                k.launches += launches;
                k.busy_ms += busy;
            }
            None => other_ms += busy,
        }
    }
    if kernels.is_empty() {
        bail!("no Tree kernel rows (labels like `tree:(k:matvec 'q8_0 1)`) found: run a Tree executor with TL_VULKAN_PROFILE=1, TL_CUDA_PROFILE=1 or TL_METAL_PROFILE=1 (native-vulkan with TL_NATIVE_TREE=1|only, native-cuda-tree, native-metal-tree)");
    }
    let tree_ms: f64 = kernels.values().map(|k| k.busy_ms).sum();
    Ok(Workload {
        workload: workload.into(),
        device: device.into(),
        weight,
        total_busy_ms: total_ms.unwrap_or(tree_ms + other_ms),
        kernels: kernels.into_values().collect(),
        imported: now_millis(),
    })
}

/// One distinct launch seen in a real run (`TL_METAL_LAUNCHES`): the generator expression and the exact
/// arguments it was launched with, so the kernel can be replayed on random data and timed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Launch {
    pub expr: String,
    pub grid: [u32; 3],
    /// Push-constant bytes, hex.
    pub push: String,
    /// Length in bytes of every bound buffer range.
    pub bind: Vec<u64>,
    /// Contents (hex) of the small bound buffers (index, position and block-table buffers), "" for the others.
    /// The first launch seen with these arguments decides.
    #[serde(default)]
    pub data: Vec<String>,
    pub count: u64,
}

impl Launch {
    /// The captured bytes of binding `i`, when there are some of the right length.
    pub fn binding_data(&self, i: usize) -> Option<Vec<u8>> {
        let hex = self.data.get(i).filter(|h| !h.is_empty())?;
        let bytes: Vec<u8> = (0..hex.len() / 2).filter_map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).ok()).collect();
        (bytes.len() as u64 == *self.bind.get(i)?).then_some(bytes)
    }

    pub fn push_bytes(&self) -> Vec<u8> {
        (0..self.push.len() / 2).filter_map(|i| u8::from_str_radix(&self.push[2 * i..2 * i + 2], 16).ok()).collect()
    }
}

/// Reads the JSON lines the Metal backend writes (equal launches add up); launches of kernels that are not Tree kernels are dropped.
pub fn parse_launches(text: &str) -> Vec<Launch> {
    #[derive(Deserialize)]
    struct Line {
        label: String,
        grid: [u32; 3],
        push: String,
        bind: Vec<u64>,
        #[serde(default)]
        data: Vec<Option<String>>,
        count: u64,
    }
    let mut out = Vec::new();
    for l in text.lines().filter(|l| l.trim_start().starts_with('{')) {
        if let Ok(x) = serde_json::from_str::<Line>(l)
            && let Some(expr) = tree_expr(&x.label)
        {
            let launch = Launch { expr: expr.to_string(), grid: x.grid, push: x.push, bind: x.bind, data: x.data.into_iter().map(Option::unwrap_or_default).collect(), count: x.count };
            match out.iter_mut().find(|y: &&mut Launch| y.expr == launch.expr && y.grid == launch.grid && y.push == launch.push && y.bind == launch.bind) {
                Some(y) => {
                    y.count += launch.count;
                    if y.data.iter().all(String::is_empty) {
                        y.data = launch.data;
                    }
                }
                None => out.push(launch),
            }
        }
    }
    out
}

fn launches_key(device: &str, workload: &str) -> Result<String> {
    Ok(key(device, workload)?.replacen("weights/", "launches/", 1))
}

pub fn store_launches(kv: &dyn Kv, device: &str, workload: &str, launches: &[Launch]) -> Result<()> {
    kv.put(&launches_key(device, workload)?, &serde_json::to_vec(launches)?)
}

/// Every launch recorded for the device, over all its workloads.
pub fn load_launches(kv: &dyn Kv, device: &str) -> Result<Vec<Launch>> {
    let mut out: Vec<Launch> = Vec::new();
    if device.is_empty() || !device.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.') {
        return Ok(out);
    }
    for k in kv.list(&format!("launches/{device}/"), None, 1000)? {
        if let Some(b) = kv.get(&k)?
            && let Ok(l) = serde_json::from_slice::<Vec<Launch>>(&b)
        {
            for x in l {
                match out.iter_mut().find(|y| y.expr == x.expr && y.grid == x.grid && y.push == x.push && y.bind == x.bind) {
                    Some(y) => y.count += x.count,
                    None => out.push(x),
                }
            }
        }
    }
    Ok(out)
}

/// Which environment variable switches a backend's profiler on.
pub fn profile_env(backend: &str) -> Result<&'static str> {
    Ok(match backend {
        "vulkan" => "TL_VULKAN_PROFILE",
        "cuda" => "TL_CUDA_PROFILE",
        "metal" => "TL_METAL_PROFILE",
        other => bail!("backend must be vulkan, cuda or metal, not {other:?}"),
    })
}

/// Runs `command` (through `sh -c`) with the backend's profiler on and returns everything it printed.
/// Profiling times launches one by one, so the command runs slower than usual.
pub fn capture(backend: &str, command: &str) -> Result<String> {
    capture_env(backend, command, &[], &[])
}

/// [`capture`] with environment variables removed and set for the command.
pub fn capture_env(backend: &str, command: &str, remove: &[&str], set: &[(&str, &str)]) -> Result<String> {
    let var = profile_env(backend)?;
    let mut cmd = std::process::Command::new("sh");
    cmd.arg("-c").arg(command).env(var, "1").stdin(std::process::Stdio::null());
    for r in remove {
        cmd.env_remove(r);
    }
    for (k, v) in set {
        cmd.env(k, v);
    }
    let out = cmd.output()?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        let tail: Vec<&str> = text.lines().rev().take(8).collect();
        bail!("the command failed ({}); its last output:\n{}", out.status, tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
    }
    Ok(text)
}

fn key(device: &str, workload: &str) -> Result<String> {
    for part in [device, workload] {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.') {
            bail!("device and workload names are [A-Za-z0-9._-]+, got {part:?}");
        }
    }
    Ok(format!("weights/{device}/{workload}"))
}

pub fn store(kv: &dyn Kv, w: &Workload) -> Result<()> {
    kv.put(&key(&w.device, &w.workload)?, &serde_json::to_vec_pretty(w)?)
}

pub fn remove(kv: &dyn Kv, device: &str, workload: &str) -> Result<()> {
    kv.delete(&key(device, workload)?)
}

pub fn load(kv: &dyn Kv, device: &str) -> Result<Profile> {
    let mut workloads = Vec::new();
    if device.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.') && !device.is_empty() {
        for k in kv.list(&format!("weights/{device}/"), None, 1000)? {
            if let Some(b) = kv.get(&k)?
                && let Ok(w) = serde_json::from_slice::<Workload>(&b)
            {
                workloads.push(w);
            }
        }
    }
    let weight_sum: f64 = workloads.iter().map(|w| w.weight.max(0.0)).sum();
    let mut weights: BTreeMap<String, f64> = BTreeMap::new();
    for w in &workloads {
        let tree_ms: f64 = w.kernels.iter().map(|k| k.busy_ms).sum();
        if tree_ms <= 0.0 || weight_sum <= 0.0 {
            continue;
        }
        for k in &w.kernels {
            *weights.entry(k.expr.clone()).or_default() += (w.weight.max(0.0) / weight_sum) * (k.busy_ms / tree_ms);
        }
    }
    Ok(Profile { workloads, weights })
}

/// A table for agents and humans: where the time goes, heaviest first. `locked` says which kernels no pass
/// can change (they contain cooperative matrices).
pub fn render(p: &Profile, device: &str, limit: usize, locked: &dyn Fn(&str) -> bool) -> String {
    if p.workloads.is_empty() {
        return format!("no profiles for device `{device}`: kernels are weighted equally. Import one with `tl-evolve weights add --device {device} --workload NAME profile.txt`.");
    }
    let mut out = format!("device {device}, workloads: ");
    out.push_str(&p.workloads.iter().map(|w| format!("{} (weight {}, {:.2} ms busy, {:.0}% in Tree kernels)", w.workload, w.weight, w.total_busy_ms, 100.0 * w.kernels.iter().map(|k| k.busy_ms).sum::<f64>() / w.total_busy_ms.max(1e-9))).collect::<Vec<_>>().join("; "));
    out.push_str("\nshare   launches/step  us/launch  kernel\n");
    let mut rows: Vec<(&String, f64)> = p.weights.iter().map(|(k, v)| (k, *v)).collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let locked_share: f64 = p.weights.iter().filter(|(k, _)| locked(k)).map(|(_, v)| *v).sum();
    for (expr, share) in rows.into_iter().take(limit) {
        let (mut launches, mut ms) = (0u64, 0.0);
        let mut n = 0;
        for w in &p.workloads {
            if let Some(k) = w.kernels.iter().find(|k| &k.expr == expr) {
                launches += k.launches;
                ms += k.busy_ms;
                n += 1;
            }
        }
        let _ = n;
        out.push_str(&format!("{:>5.1}%  {:>12}  {:>9.1}  {expr}{}\n", share * 100.0, launches, if launches > 0 { ms * 1000.0 / launches as f64 } else { 0.0 }, if locked(expr) { "  [locked]" } else { "" }));
    }
    if locked_share > 0.05 {
        out.push_str(&format!("[locked]: {:.0}% of the profiled time is in kernels with cooperative matrices, which `optimize` skips: no pass changes them.\n", locked_share * 100.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_records_are_read_and_stored_per_device() {
        let text = "{\"label\":\"tree:(k:argmax)\",\"grid\":[4, 1, 1],\"push\":\"0a000000\",\"bind\":[64, 4],\"data\":[null,\"01000000\"],\"count\":3}\n{\"label\":\"copy_f32\",\"grid\":[1, 1, 1],\"push\":\"\",\"bind\":[],\"count\":1}\nnoise\n";
        let l = parse_launches(text);
        assert_eq!(l.len(), 1);
        assert_eq!((l[0].expr.as_str(), l[0].grid, l[0].push_bytes(), l[0].bind.clone()), ("(k:argmax)", [4, 1, 1], vec![10, 0, 0, 0], vec![64, 4]));
        assert_eq!((l[0].binding_data(0), l[0].binding_data(1)), (None, Some(vec![1, 0, 0, 0])));
        let kv = crate::kv::MemKv::default();
        store_launches(&kv, "d", "a", &l).unwrap();
        store_launches(&kv, "d", "b", &l).unwrap();
        let all = load_launches(&kv, "d").unwrap();
        assert_eq!((all.len(), all[0].count), (1, 6));
        assert!(load_launches(&kv, "other").unwrap().is_empty());
    }
    use crate::kv::MemKv;

    const GEMMA: &str = "vulkan profile (GPU time by kernel, NVIDIA GeForce RTX 3080):
kernel                                        launches     busy ms      gap ms  busy us/l
tree:(k:matvec 'q8_0 1)                           4200      30.000       2.000       7.1
tree:(k:rms-norm #t #f)                           1200       3.000       1.000       2.5
tree:(k:soft-max #t)                               300       1.000       0.500       3.3
copy_f32                                           500       1.000       0.100       2.0
total                                             6200      35.000       3.600
";

    #[test]
    fn parses_the_profile_table_with_spaces_in_labels() {
        let w = parse_profile(GEMMA, "gemma4-decode", "rtx3080", 1.0).unwrap();
        assert_eq!(w.kernels.len(), 3);
        assert_eq!(w.kernels[0].expr, "(k:matvec 'q8_0 1)");
        assert_eq!((w.kernels[0].launches, w.kernels[0].busy_ms), (4200, 30.0));
        assert_eq!(w.total_busy_ms, 35.0);
        assert!(parse_profile("nothing here", "w", "d", 1.0).is_err());
        // plain lines work too
        let w = parse_profile("(k:argmax) 10 2.5\n(k:argmax) 5 1.0\n", "w", "d", 1.0).unwrap();
        assert_eq!((w.kernels.len(), w.kernels[0].launches, w.kernels[0].busy_ms), (1, 15, 3.5));
    }

    /// Lines from a real `TL_METAL_PROFILE=1` run of tests/native.rs on an M1 (native-metal-tree).
    const METAL: &str = "running 1 test
metal profile: 0.30 ms over 13 kernels
      0.069 ms      3 x     23.0 us  tree:(k:matmul 'f32 64 16 16 4 1) grid=[1, 1, 8]
      0.033 ms      1 x     32.9 us  tree:(k:flash-attn 'f16 'f16 #t #f 2048) grid=[5, 4, 2]
      0.027 ms      2 x     13.5 us  tree:(k:unary 'copy 'same 'f32 'f32) grid=[10, 1, 1]
      0.021 ms      2 x     10.6 us  tree:(k:ew '((f32 full)) 'f16 '((t0 x0)) 't0 0) grid=[6, 1, 1]
      0.019 ms      1 x     18.9 us  tree:(k:unary 'copy 'same 'f32 'f32) grid=[6, 1, 1]
      0.010 ms      1 x     10.0 us  copy_f32 grid=[1, 1, 1]
test one ... ok
metal profile: 0.50 ms over 2 kernels
      0.040 ms      1 x     40.0 us  tree:(k:matmul 'f32 64 16 16 4 1) grid=[1, 1, 4]
";

    #[test]
    fn reads_the_cuda_and_metal_tables_summing_grids_and_tables() {
        let w = parse_profile(METAL, "native", "m1pro", 1.0).unwrap();
        let by = |e: &str| w.kernels.iter().find(|k| k.expr == e).unwrap().clone();
        // the same expression on two grids and in two tables adds up
        assert_eq!(by("(k:matmul 'f32 64 16 16 4 1)").launches, 4);
        assert!((by("(k:matmul 'f32 64 16 16 4 1)").busy_ms - 0.109).abs() < 1e-9);
        assert_eq!(by("(k:unary 'copy 'same 'f32 'f32)").launches, 3);
        // nested parentheses survive, the non-Tree row only counts toward the total
        assert_eq!(by("(k:ew '((f32 full)) 'f16 '((t0 x0)) 't0 0)").launches, 2);
        assert_eq!(w.kernels.len(), 4);
        assert!((w.total_busy_ms - 0.80).abs() < 1e-9, "{}", w.total_busy_ms);
        let cuda = "cuda profile: 3.00 ms over 2 kernels
      2.000 ms    100 x     20.0 us  tree:(k:matvec 'q8_0 1)
      1.000 ms     50 x     20.0 us  tree:(k:rms-norm #t #f)
";
        let w = parse_profile(cuda, "gemma", "rtx3080", 1.0).unwrap();
        assert_eq!((w.kernels.len(), w.total_busy_ms), (2, 3.0));
        assert_eq!(w.kernels[0].launches, 100);
        // a Metal-style label with the grid but without the prefix still resolves
        assert_eq!(tree_expr("(k:argmax) grid=[1, 1, 1]"), Some("(k:argmax)"));
        assert_eq!(tree_expr("copy_f32 grid=[1, 1, 1]"), None);
    }

    #[test]
    fn capture_runs_a_command_with_the_profiler_on() {
        let out = capture("metal", "echo \"metal profile: 1.0 ms over 1 kernels\"; echo \"1.000 ms 10 x 100.0 us tree:(k:argmax) grid=[1, 1, 1]\"; test \"$TL_METAL_PROFILE\" = 1").unwrap();
        let w = parse_profile(&out, "w", "d", 1.0).unwrap();
        assert_eq!(w.kernels[0].expr, "(k:argmax)");
        // the variable is per backend
        assert!(capture("cuda", "test -z \"$TL_METAL_PROFILE\" && test \"$TL_CUDA_PROFILE\" = 1").is_ok());
        assert!(capture("opencl", "true").is_err());
        let err = capture("vulkan", "echo boom >&2; exit 3").unwrap_err().to_string();
        assert!(err.contains("boom") && err.contains("failed"), "{err}");
    }

    #[test]
    fn weights_are_time_shares_averaged_over_workloads() {
        let kv = MemKv::new();
        assert!(load(&kv, "rtx3080").unwrap().weights.is_empty());
        store(&kv, &parse_profile(GEMMA, "gemma", "rtx3080", 1.0).unwrap()).unwrap();
        let p = load(&kv, "rtx3080").unwrap();
        assert!((p.weights["(k:matvec 'q8_0 1)"] - 30.0 / 34.0).abs() < 1e-9);
        assert!((p.weights.values().sum::<f64>() - 1.0).abs() < 1e-9);
        // a second workload that is all argmax, weighted 3x the first
        store(&kv, &parse_profile("(k:argmax) 1 1.0\n", "tiny", "rtx3080", 3.0).unwrap()).unwrap();
        let p = load(&kv, "rtx3080").unwrap();
        assert!((p.weights["(k:argmax)"] - 0.75).abs() < 1e-9);
        assert!((p.weights["(k:matvec 'q8_0 1)"] - 0.25 * 30.0 / 34.0).abs() < 1e-9);
        // other devices are separate
        assert!(load(&kv, "m1pro").unwrap().weights.is_empty());
        let table = render(&p, "rtx3080", 2, &|e| e == "(k:argmax)");
        assert!(table.contains("(k:argmax)  [locked]") && table.contains("75% of the profiled time"), "{table}");
        remove(&kv, "rtx3080", "tiny").unwrap();
        assert_eq!(load(&kv, "rtx3080").unwrap().workloads.len(), 1);
        assert!(store(&kv, &Workload { device: "../x".into(), ..parse_profile(GEMMA, "g", "d", 1.0).unwrap() }).is_err());
    }
}
