//! Backend conformance suite: the same scheduler behaviour (plans, barriers,
//! sessions, pools, asynchronous tickets) must hold on every
//! [`Backend`]. A backend's test file supplies an engine and its three tiny
//! kernels (`axpb`, `acc_add`, `matmul`) in its own language, then
//! instantiates the tests with `conformance!`.
#![allow(dead_code)]

use std::sync::Arc;

use tensorlisp_kernels::sched::{Backend, Engine, KernelBlob, LaunchDesc, Range, Space};

/// The three kernels every backend provides. Parameter layouts (all `u32`/`f32`
/// words, backend offsets handled by the backend):
///
/// * `axpb(a, b, out; n, s)`: `out[i] = a[i] * s + b[i]`, workgroup 64
/// * `acc_add(acc rw, x; n)`: `acc[i] += x[i]`, workgroup 64
/// * `matmul(a, b, c; m, n, k)`: `c[m, n] = a[m, k] * b[k, n]`, workgroup 8x8
pub struct Kernels {
    pub axpb: Arc<KernelBlob>,
    pub acc: Arc<KernelBlob>,
    pub matmul: Arc<KernelBlob>,
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

pub fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn step(off: u64, len: u64) -> Range {
    Range::new(Space::Step, off, len)
}

fn axpb(k: &Arc<KernelBlob>, a: Range, b: Range, out: Range, n: u32, s: f32) -> LaunchDesc {
    let mut push = n.to_ne_bytes().to_vec();
    push.extend(s.to_bits().to_ne_bytes());
    LaunchDesc { kernel: k.clone(), bindings: vec![a, b, out], push, grid: [n.div_ceil(64), 1, 1] }
}

fn acc_add(k: &Arc<KernelBlob>, acc: Range, x: Range, n: u32) -> LaunchDesc {
    LaunchDesc { kernel: k.clone(), bindings: vec![acc, x], push: n.to_ne_bytes().to_vec(), grid: [n.div_ceil(64), 1, 1] }
}

fn no_model<B: Backend>(e: &Engine<B>) -> Arc<tensorlisp_kernels::sched::ModelMem<B>> {
    e.upload_model(&[]).unwrap()
}

pub fn dependent_chain<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 1000u32;
    let bytes = 4 * n as u64;
    // x, y inputs; t1 = 2x+y; t2 = 3 t1 + y; out = t2 + x.
    let (x, y, t1, t2, out) = (step(0, bytes), step(4096, bytes), step(8192, bytes), step(12288, bytes), step(16384, bytes));
    let plan = e
        .compile(&[axpb(&k.axpb, x, y, t1, n, 2.0), axpb(&k.axpb, t1, y, t2, n, 3.0), axpb(&k.axpb, t2, x, out, n, 1.0)], 20480)
        .unwrap();
    assert_eq!(plan.group_sizes(), vec![1, 1, 1]);

    let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.25).collect();
    let ticket = e.submit(&plan, &no_model(e), None, &[(0, &f32_bytes(&xs)), (4096, &f32_bytes(&ys))]).unwrap();
    let got = to_f32(&ticket.read(16384, bytes as usize).unwrap());
    for i in 0..n as usize {
        let want = (xs[i] * 2.0 + ys[i]) * 3.0 + ys[i] + xs[i];
        assert!((got[i] - want).abs() <= 1e-3 * want.abs().max(1.0), "element {i}: {} vs {want}", got[i]);
    }
}

pub fn independent_group<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 256u32;
    let bytes = 4 * n as u64;
    let (x, y, p, q) = (step(0, bytes), step(2048, bytes), step(4096, bytes), step(6144, bytes));
    let plan = e.compile(&[axpb(&k.axpb, x, y, p, n, 2.0), axpb(&k.axpb, y, x, q, n, -1.0)], 8192).unwrap();
    assert_eq!(plan.group_sizes(), vec![2]);
    let xs: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i * i) as f32 * 1e-3).collect();
    let t = e.submit(&plan, &no_model(e), None, &[(0, &f32_bytes(&xs)), (2048, &f32_bytes(&ys))]).unwrap();
    let (gp, gq) = (to_f32(&t.read(4096, bytes as usize).unwrap()), to_f32(&t.read(6144, bytes as usize).unwrap()));
    for i in 0..n as usize {
        assert!((gp[i] - (xs[i] * 2.0 + ys[i])).abs() < 1e-3);
        assert!((gq[i] - (ys[i] * -1.0 + xs[i])).abs() < 1e-3);
    }
}

pub fn unaligned_offsets<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 100u32;
    let bytes = 4 * n as u64;
    // 20 bytes into their buffers: not a multiple of any usual descriptor alignment.
    let (x, y, out) = (step(20, bytes), step(4096 + 20, bytes), step(8192 + 20, bytes));
    let plan = e.compile(&[axpb(&k.axpb, x, y, out, n, 1.5)], 12288).unwrap();
    let xs: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ys: Vec<f32> = (0..n).map(|i| 100.0 - i as f32).collect();
    let t = e.submit(&plan, &no_model(e), None, &[(20, &f32_bytes(&xs)), (4096 + 20, &f32_bytes(&ys))]).unwrap();
    let got = to_f32(&t.read(8192 + 20, bytes as usize).unwrap());
    for i in 0..n as usize {
        assert!((got[i] - (xs[i] * 1.5 + ys[i])).abs() < 1e-4, "element {i}");
    }
}

pub fn weights_and_matmul<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let (m, n, kk) = (70u32, 50u32, 90u32);
    let a: Vec<f32> = (0..m * kk).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..kk * n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
    let model = e.upload_model(&[&f32_bytes(&b)]).unwrap();
    let (abytes, cbytes) = (4 * (m * kk) as u64, 4 * (m * n) as u64);
    let c_off = abytes.next_multiple_of(256);
    let mut push = Vec::new();
    for v in [m, n, kk] {
        push.extend(v.to_ne_bytes());
    }
    let launch = LaunchDesc {
        kernel: k.matmul.clone(),
        bindings: vec![step(0, abytes), Range::new(Space::Model(0), 0, 4 * (kk * n) as u64), step(c_off, cbytes)],
        push,
        grid: [n.div_ceil(8), m.div_ceil(8), 1],
    };
    let plan = e.compile(&[launch], c_off + cbytes).unwrap();
    // Two submissions through the same plan and weights.
    for round in 0..2 {
        let scale = 1.0 + round as f32;
        let a_in: Vec<f32> = a.iter().map(|v| v * scale).collect();
        let t = e.submit(&plan, &model, None, &[(0, &f32_bytes(&a_in))]).unwrap();
        let c = to_f32(&t.read(c_off, cbytes as usize).unwrap());
        for r in 0..m as usize {
            for col in 0..n as usize {
                let want: f32 = (0..kk as usize).map(|i| a_in[r * kk as usize + i] * b[i * n as usize + col]).sum();
                assert!((c[r * n as usize + col] - want).abs() < 1e-3, "c[{r},{col}] = {} vs {want}", c[r * n as usize + col]);
            }
        }
    }
}

pub fn sessions_and_pool_reuse<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 512u32;
    let bytes = 4 * n as u64;
    let plan = e.compile(&[acc_add(&k.acc, Range::new(Space::Session(0), 0, bytes), step(0, bytes), n)], bytes).unwrap();
    let model = no_model(e);
    let ones = f32_bytes(&vec![1.0; n as usize]);
    let twos = f32_bytes(&vec![2.0; n as usize]);

    let before = e.session_stats();
    let a = Arc::new(e.new_session(&[bytes]).unwrap());
    let b = Arc::new(e.new_session(&[bytes]).unwrap());
    for _ in 0..3 {
        e.submit(&plan, &model, Some(&a), &[(0, &ones)]).unwrap().wait().unwrap();
    }
    e.submit(&plan, &model, Some(&b), &[(0, &twos)]).unwrap().wait().unwrap();
    assert!(to_f32(&e.read_session(&a, 0, 0, bytes as usize).unwrap()).iter().all(|&v| v == 3.0));
    assert!(to_f32(&e.read_session(&b, 0, 0, bytes as usize).unwrap()).iter().all(|&v| v == 2.0));
    assert_eq!(e.session_stats().allocated - before.allocated, 2);

    // A finished session's memory serves the next session, zeroed.
    drop(a);
    let c: Arc<_> = Arc::new(e.new_session(&[bytes]).unwrap());
    assert_eq!(e.session_stats().reused - before.reused, 1);
    assert_eq!(e.session_stats().allocated - before.allocated, 2);
    assert!(to_f32(&e.read_session(&c, 0, 0, bytes as usize).unwrap()).iter().all(|&v| v == 0.0), "reused state starts zeroed");
    e.submit(&plan, &model, Some(&c), &[(0, &ones)]).unwrap().wait().unwrap();
    assert!(to_f32(&e.read_session(&c, 0, 0, bytes as usize).unwrap()).iter().all(|&v| v == 1.0));
}

pub fn async_tickets_and_arenas<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 4096u32;
    let bytes = 4 * n as u64;
    let (x, y, out) = (step(0, bytes), step(bytes, bytes), step(2 * bytes, bytes));
    let plan = e.compile(&[axpb(&k.axpb, x, y, out, n, 2.0)], 3 * bytes).unwrap();
    let model = no_model(e);
    let xs = f32_bytes(&vec![1.0; n as usize]);
    let ys1 = f32_bytes(&vec![10.0; n as usize]);
    let ys2 = f32_bytes(&vec![20.0; n as usize]);

    let before = e.step_stats();
    // Two tickets in flight at once need two arenas.
    let t1 = e.submit(&plan, &model, None, &[(0, &xs), (bytes, &ys1)]).unwrap();
    let t2 = e.submit(&plan, &model, None, &[(0, &xs), (bytes, &ys2)]).unwrap();
    assert_eq!(e.step_stats().allocated - before.allocated, 2);
    assert!(to_f32(&t1.read(2 * bytes, bytes as usize).unwrap()).iter().all(|&v| v == 12.0));
    assert!(to_f32(&t2.read(2 * bytes, bytes as usize).unwrap()).iter().all(|&v| v == 22.0));
    while !t1.poll().unwrap() {}
    drop((t1, t2));
    // Both arenas are back; the next submissions allocate nothing.
    for _ in 0..5 {
        let t = e.submit(&plan, &model, None, &[(0, &xs), (bytes, &ys1)]).unwrap();
        assert!(to_f32(&t.read(2 * bytes, 16).unwrap()).iter().all(|&v| v == 12.0));
    }
    assert_eq!(e.step_stats().allocated - before.allocated, 2);
    assert!(e.step_stats().reused - before.reused >= 5);
}

pub fn threads_with_sessions<B: Backend>(e: &Engine<B>, k: &Kernels) {
    // A second engine over the same backend, shareable across threads.
    let e = Arc::new(Engine::new(e.backend().clone()));
    let n = 1024u32;
    let bytes = 4 * n as u64;
    let plan = Arc::new(e.compile(&[acc_add(&k.acc, Range::new(Space::Session(0), 0, bytes), step(0, bytes), n)], bytes).unwrap());
    let model = no_model(&e);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let (e, plan, model) = (e.clone(), plan.clone(), model.clone());
            std::thread::spawn(move || {
                let session = Arc::new(e.new_session(&[bytes]).unwrap());
                let x = f32_bytes(&vec![(t + 1) as f32; n as usize]);
                for _ in 0..25 {
                    e.submit(&plan, &model, Some(&session), &[(0, &x)]).unwrap();
                }
                // Tickets were dropped (each waits); the session holds the sum.
                let got = to_f32(&e.read_session(&session, 0, 0, bytes as usize).unwrap());
                assert!(got.iter().all(|&v| v == 25.0 * (t + 1) as f32), "thread {t}: {}", got[0]);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// 300 dependent passes over 4 MiB ping-ponging between two buffers: with a
/// missing barrier a pass reads a half-written predecessor and the sum is off.
pub fn long_dependent_chain<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let n = 1u32 << 20;
    let bytes = 4 * n as u64;
    let ones = step(0, bytes);
    let (ping, pong) = (step(bytes, bytes), step(2 * bytes, bytes));
    let passes = 300;
    let mut launches = Vec::new();
    for i in 0..passes {
        let (src, dst) = if i % 2 == 0 { (ping, pong) } else { (pong, ping) };
        launches.push(axpb(&k.axpb, src, ones, dst, n, 1.0)); // dst = src + 1
    }
    let plan = e.compile(&launches, 3 * bytes).unwrap();
    assert_eq!(plan.group_sizes().len(), passes);
    let start: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let t0 = std::time::Instant::now();
    let t = e.submit(&plan, &no_model(e), None, &[(0, &f32_bytes(&vec![1.0; n as usize])), (bytes, &f32_bytes(&start))]).unwrap();
    let result = if passes % 2 == 0 { ping } else { pong };
    let got = to_f32(&t.read(result.off, bytes as usize).unwrap());
    println!("{passes} dependent passes over {n} floats on {}: {:.1} ms", e.backend().name(), t0.elapsed().as_secs_f64() * 1e3);
    for (i, (g, s)) in got.iter().zip(&start).enumerate() {
        assert_eq!(*g, s + passes as f32, "element {i}");
    }
}

pub fn bad_plans<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let r = step(0, 64);
    let ok = axpb(&k.axpb, r, r, step(64, 64), 16, 1.0);
    assert!(e.compile(&[ok.clone()], 128).is_ok());
    // Range past the arena.
    assert!(e.compile(&[ok.clone()], 100).is_err());
    // Wrong number of bindings / push bytes / empty grid.
    let mut bad = ok.clone();
    bad.bindings.pop();
    assert!(e.compile(&[bad], 128).is_err());
    let mut bad = ok.clone();
    bad.push.pop();
    assert!(e.compile(&[bad], 128).is_err());
    let mut bad = ok;
    bad.grid = [0, 1, 1];
    assert!(e.compile(&[bad], 128).is_err());

    // A plan that needs session state refuses to run without a session.
    let plan = e.compile(&[acc_add(&k.acc, Range::new(Space::Session(0), 0, 64), step(0, 64), 16)], 64).unwrap();
    assert!(e.submit(&plan, &no_model(e), None, &[]).is_err());
    // Small requests round up to a pool size class, so a tiny session still fits.
    let small = Arc::new(e.new_session(&[16]).unwrap());
    assert!(e.submit(&plan, &no_model(e), Some(&small), &[]).is_ok());
}

/// A kernel built for another backend is refused at compile time.
pub fn wrong_target_is_refused<B: Backend>(e: &Engine<B>, k: &Kernels) {
    let mut foreign = (*k.axpb).clone();
    foreign.target = match B::TARGET {
        tensorlisp_kernels::sched::Target::Vulkan => tensorlisp_kernels::sched::Target::Cuda,
        _ => tensorlisp_kernels::sched::Target::Vulkan,
    };
    let r = step(0, 64);
    let launch = LaunchDesc { kernel: Arc::new(foreign), bindings: vec![r, r, step(64, 64)], push: vec![0; 8], grid: [1, 1, 1] };
    assert!(e.compile(&[launch], 128).is_err());
}

/// Instantiates the suite for a backend: `$engine()` returns `Option<impl Deref<Target = Engine<B>>>`
/// (None to skip), `$kernels()` the backend's [`Kernels`].
#[macro_export]
macro_rules! conformance {
    ($engine:path, $kernels:path) => {
        mod conformance {
            use super::*;
            macro_rules! case {
                ($name:ident) => {
                    #[test]
                    fn $name() {
                        if let Some(e) = $engine() {
                            $crate::common::$name(&*e, &$kernels());
                        }
                    }
                };
            }
            case!(dependent_chain);
            case!(independent_group);
            case!(unaligned_offsets);
            case!(weights_and_matmul);
            case!(sessions_and_pool_reuse);
            case!(async_tickets_and_arenas);
            case!(threads_with_sessions);
            case!(long_dependent_chain);
            case!(bad_plans);
            case!(wrong_target_is_refused);
        }
    };
}
