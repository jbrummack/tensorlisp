//! ggml's own Vulkan shaders (vendor/ggml/src/ggml-vulkan/vulkan-shaders),
//! compiled with glslc/glslang, reflected into blobs and dispatched through
//! the scheduler. Parameter packing here mirrors ggml-vulkan.cpp.
//!
//! `cargo test -p tensorlisp-kernels --features vulkan --test ggml_vulkan -- --nocapture`
//! Skipped without a GLSL compiler (`TL_GLSLC`, Vulkan SDK, or vendor/glslang)
//! or, with `TL_ALLOW_NO_GPU=1`, without a Vulkan device.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::sync::Arc;

use tensorlisp_kernels::sched::spirv;
use tensorlisp_kernels::sched::{Backend, Engine, KernelBlob, KernelKey, LaunchDesc, OffsetMode, Range, Registry, Space, Target};
use tensorlisp_kernels::vulkan::ggml::SpvLibrary;
use tensorlisp_kernels::vulkan::glsl::{flatten, Compiler};
use tensorlisp_kernels::vulkan::{validation_errors, Options, Vulkan};
use tensorlisp_kernels::Error;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ggml/src/ggml-vulkan/vulkan-shaders")
}

struct Ctx {
    compiler: Compiler,
    engine: Engine<Vulkan>,
    align: u64,
}

fn ctx() -> Option<Ctx> {
    let Some(compiler) = Compiler::find() else {
        eprintln!("no GLSL compiler found (glslc or glslang), skipping");
        return None;
    };
    match Vulkan::open(Options { validate: true }) {
        Ok(v) => {
            let align = v.limits().storage_offset_align;
            println!("{} | compiler {} | offset align {align} | subgroup {} | validation {}", v.name(), compiler.describe(), v.subgroup_size(), v.validated());
            Some(Ctx { compiler, engine: Engine::new(Arc::new(v)), align })
        }
        Err(Error::NoDevice) if std::env::var_os("TL_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Vulkan device, skipping");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert_eq!(validation_errors(), 0, "the Vulkan validation layer reported errors");
        }
    }
}

/// Compiles `file` of the ggml shader directory with `defines` into a blob
/// that handles its own binding offsets, like ggml's host code does.
fn shader(c: &Compiler, file: &str, defines: &[(&str, &str)], spec: &[u32]) -> Arc<KernelBlob> {
    let dir = shader_dir();
    let source = flatten(&dir.join(file), &[dir]).unwrap();
    let defs: Vec<(String, String)> = defines.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let words = c.compile(file, &source, &defs).unwrap_or_else(|e| panic!("{e}"));
    Arc::new(spirv::blob(words, OffsetMode::Raw, spec.to_vec()).unwrap())
}

/// A contiguous-or-not f32 tensor placed in the step arena.
#[derive(Clone, Copy)]
struct T {
    ne: [u32; 4],
    /// Strides in elements (ggml's `nb / type_size`).
    nb: [u32; 4],
    /// Byte offset in the arena.
    off: u64,
}

impl T {
    fn contiguous(ne: [u32; 4], off: u64) -> T {
        T { ne, nb: [1, ne[0], ne[0] * ne[1], ne[0] * ne[1] * ne[2]], off }
    }

    fn nelements(&self) -> u32 {
        self.ne.iter().product()
    }

    fn bytes(&self) -> u64 {
        4 * self.nelements() as u64
    }

    fn range(&self) -> Range {
        Range::new(Space::Step, self.off, self.bytes())
    }

    /// Residual of the binding offset in elements, as ggml's `get_misalign_bytes / type_size`.
    fn misalign(&self, align: u64) -> u32 {
        ((self.off % align) / 4) as u32
    }
}

/// A binding the shader declares but the op does not use.
fn unused() -> Range {
    Range::new(Space::Step, 0, 0)
}

/// Workgroup counts for ggml's elementwise kernels (`wg_denoms = {512, 1, 1}`):
/// `elements = {512, ceil(ne / 512), 1}` past 512 elements, split again into z
/// past 262144, because the shaders index `z * 262144 + y * 512 + x`.
fn elementwise_grid(ne: u32) -> [u32; 3] {
    if ne > 262_144 {
        [1, 512, ne.div_ceil(262_144)]
    } else if ne > 512 {
        [1, ne.div_ceil(512), 1]
    } else {
        [1, 1, 1]
    }
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// `vk_op_binary_push_constants`.
fn binary_pc(a: &T, b: &T, d: &T, align: u64, p1: f32, p2: f32, p3: i32) -> Vec<u8> {
    let mut w = vec![a.nelements()];
    for t in [a, b, d] {
        w.extend(t.ne);
        w.extend(t.nb);
    }
    w.push((a.misalign(align) << 16) | (b.misalign(align) << 8) | d.misalign(align));
    w.extend([p1.to_bits(), p2.to_bits(), p3 as u32]);
    words(&w)
}

/// `init_fastdiv_values`.
fn fastdiv(d: u32) -> (u32, u32) {
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < d as u64 {
        l += 1;
    }
    let mp = ((1u64 << 32) * ((1u64 << l) - d as u64) / d as u64 + 1) as u32;
    (mp, l)
}

/// `vk_op_unary_push_constants` with `init_pushconst_fastdiv`.
fn unary_pc(a: &T, d: &T, align: u64, p1: f32, p2: f32) -> Vec<u8> {
    let mut w = vec![d.nelements()];
    for t in [a, d] {
        w.extend(t.ne);
        w.extend(t.nb);
    }
    w.push((a.misalign(align) << 16) | d.misalign(align));
    w.extend([p1.to_bits(), p2.to_bits(), 0, 0]);
    let fd = |t: &T| {
        let (m012, l012) = fastdiv(t.ne[2] * t.ne[1] * t.ne[0]);
        let (m01, l01) = fastdiv(t.ne[1] * t.ne[0]);
        let (m0, l0) = fastdiv(t.ne[0]);
        [m012, m01, m0, l012 | (l01 << 8) | (l0 << 16)]
    };
    w.extend(fd(a));
    w.extend(fd(d));
    words(&w)
}

/// One launch on a fresh arena holding `arena` (f32 words); returns the arena afterwards.
fn run_one(c: &Ctx, blob: &Arc<KernelBlob>, bindings: Vec<Range>, push: Vec<u8>, grid: [u32; 3], arena: &[f32]) -> Vec<f32> {
    let bytes: Vec<u8> = arena.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let plan = c
        .engine
        .compile(&[LaunchDesc { kernel: blob.clone(), bindings, push, grid }], bytes.len() as u64)
        .unwrap_or_else(|e| panic!("{e}"));
    let model = c.engine.upload_model(&[]).unwrap();
    let t = c.engine.submit(&plan, &model, None, &[(0, &bytes)]).unwrap();
    t.read(0, bytes.len()).unwrap().chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect()
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn put(arena: &mut [f32], t: &T, data: &[f32]) {
    let at = (t.off / 4) as usize;
    arena[at..at + data.len()].copy_from_slice(data);
}

fn get(arena: &[f32], t: &T) -> Vec<f32> {
    let at = (t.off / 4) as usize;
    arena[at..at + t.nelements() as usize].to_vec()
}

fn close(got: f32, want: f32, tol: f32) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

#[test]
fn reflection_of_ggml_blobs() {
    let Some(c) = ctx() else { return };
    let add = shader(&c.compiler, "add.comp", &[("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float"), ("ADD_RMS", "0")], &[0]);
    println!("add:      {:?}", add.sig);
    assert_eq!(add.sig.push_bytes, 116);
    assert_eq!(add.sig.workgroup, [256, 1, 1]);
    let rms = shader(&c.compiler, "rms_norm.comp", &[("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float"), ("FLOAT_TYPEV2", "vec2")], &[0, 0]);
    println!("rms_norm: {:?}", rms.sig);
    let scale = shader(&c.compiler, "scale.comp", &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")], &[]);
    println!("scale:    {:?}", scale.sig);
    assert_eq!(scale.sig.push_bytes, 120);
}

/// add/sub/mul/div with ggml's broadcasting rules, strided and misaligned operands.
#[test]
fn binary_ops_broadcast_with_misaligned_views() {
    let Some(c) = ctx() else { return };
    let mut seed = 5;
    for (op, f) in [("add", (|x, y| x + y) as fn(f32, f32) -> f32), ("sub", |x, y| x - y), ("mul", |x, y| x * y), ("div", |x, y| x / y)] {
        // src0 [37, 5, 3, 2]; src1 broadcasts over dims 1 and 3 ([37, 1, 3, 1]); every
        // operand starts a few elements past an alignment boundary.
        let a = T::contiguous([37, 5, 3, 2], 256 + 4 * 3);
        let b = T::contiguous([37, 1, 3, 1], 8192 + 4 * 5);
        let d = T::contiguous([37, 5, 3, 2], 16384 + 4 * 7);
        let mut arena = vec![0.0f32; 8192];
        let av: Vec<f32> = (0..a.nelements()).map(|_| lcg(&mut seed)).collect();
        // Keep the divisor away from zero.
        let bv: Vec<f32> = (0..b.nelements()).map(|_| 1.5 + lcg(&mut seed)).collect();
        put(&mut arena, &a, &av);
        put(&mut arena, &b, &bv);

        let blob = shader(
            &c.compiler,
            &format!("{op}.comp"),
            &[("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float"), ("ADD_RMS", "0")],
            &[0], // norepeat = false: src1 is broadcast
        );
        let mut bindings = vec![a.range(), b.range(), d.range()];
        bindings.resize(blob.sig.bindings.len(), unused());
        let pc = binary_pc(&a, &b, &d, c.align, 0.0, 0.0, 0);
        let grid = elementwise_grid(a.nelements());
        let out = run_one(&c, &blob, bindings, pc, grid, &arena);

        let got = get(&out, &d);
        let (n0, n1, n2, n3) = (37usize, 5usize, 3usize, 2usize);
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                for i1 in 0..n1 {
                    for i0 in 0..n0 {
                        let ia = ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
                        let ib = i2 * n0 + i0; // dims 1 and 3 of src1 are 1
                        let want = f(av[ia], bv[ib]);
                        assert!(close(got[ia], want, 1e-6), "{op}[{i0},{i1},{i2},{i3}] = {} vs {want}", got[ia]);
                    }
                }
            }
        }
        // Nothing outside dst was touched.
        assert_eq!(out[0], 0.0);
        println!("{op}: ok ({} bindings)", blob.sig.bindings.len());
    }
}

/// scale.comp: dst = src * s + b (contiguous only, like ggml routes it).
#[test]
fn scale_applies_scale_and_bias() {
    let Some(c) = ctx() else { return };
    let mut seed = 9;
    let a = T::contiguous([13, 7, 5, 1], 256 + 4);
    let d = T::contiguous([13, 7, 5, 1], 4096 + 4 * 2);
    let mut arena = vec![0.0f32; 3000];
    let src: Vec<f32> = (0..a.nelements()).map(|_| lcg(&mut seed)).collect();
    put(&mut arena, &a, &src);

    let blob = shader(&c.compiler, "scale.comp", &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")], &[]);
    let mut bindings = vec![a.range(), d.range()];
    bindings.resize(blob.sig.bindings.len(), unused());
    let pc = unary_pc(&a, &d, c.align, 2.5, -0.75);
    let out = run_one(&c, &blob, bindings, pc, elementwise_grid(d.nelements()), &arena);
    for (i, (g, s)) in get(&out, &d).iter().zip(&src).enumerate() {
        assert!(close(*g, s * 2.5 - 0.75, 1e-6), "scale[{i}] = {g}");
    }
}

/// copy.comp: a transposed, strided source into a contiguous destination,
/// through ggml's fastdiv index decomposition.
#[test]
fn copy_gathers_a_strided_source() {
    let Some(c) = ctx() else { return };
    let mut seed = 31;
    // Logical [13, 7, 5] read out of a [7, 13, 5] buffer with dims 0 and 1 swapped.
    let a = T { ne: [13, 7, 5, 1], nb: [7, 1, 91, 455], off: 256 + 4 };
    let d = T::contiguous([13, 7, 5, 1], 4096 + 4 * 2);
    let mut arena = vec![0.0f32; 3000];
    let src: Vec<f32> = (0..455).map(|_| lcg(&mut seed)).collect();
    put(&mut arena, &a, &src);

    let blob = shader(&c.compiler, "copy.comp", &[("A_TYPE", "float"), ("D_TYPE", "float")], &[]);
    let mut bindings = vec![Range::new(Space::Step, a.off, 4 * 455), d.range()];
    bindings.resize(blob.sig.bindings.len(), unused());
    let pc = unary_pc(&a, &d, c.align, 0.0, 0.0);
    let out = run_one(&c, &blob, bindings, pc, elementwise_grid(d.nelements()), &arena);

    let got = get(&out, &d);
    for i2 in 0..5usize {
        for i1 in 0..7usize {
            for i0 in 0..13usize {
                let want = src[i0 * 7 + i1 + i2 * 91];
                let g = got[(i2 * 7 + i1) * 13 + i0];
                assert_eq!(g, want, "copy[{i0},{i1},{i2}]");
            }
        }
    }
}

/// rms_norm.comp, plain and fused with the following multiply (the same
/// shader with `do_multiply` set through its specialization constant).
#[test]
fn rms_norm_plain_and_fused_with_mul() {
    let Some(c) = ctx() else { return };
    let mut seed = 21;
    let (ncols, nrows) = (100u32, 7u32);
    let x = T::contiguous([ncols, nrows, 1, 1], 256);
    let w = T::contiguous([ncols, 1, 1, 1], 4096 + 4 * 3);
    let y = T::contiguous([ncols, nrows, 1, 1], 8192);
    let eps = 1e-6f32;
    let mut arena = vec![0.0f32; 4096];
    let xv: Vec<f32> = (0..x.nelements()).map(|_| 3.0 * lcg(&mut seed)).collect();
    let wv: Vec<f32> = (0..ncols).map(|_| 1.0 + lcg(&mut seed)).collect();
    put(&mut arena, &x, &xv);
    put(&mut arena, &w, &wv);

    let defs = [("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float"), ("FLOAT_TYPEV2", "vec2")];
    for fused in [false, true] {
        let blob = shader(&c.compiler, "rms_norm.comp", &defs, &[0, fused as u32]);
        let b = if fused { w } else { x };
        let mut bindings = vec![x.range(), b.range(), y.range()];
        bindings.resize(blob.sig.bindings.len(), unused());
        let pc = binary_pc(&x, &b, &y, c.align, eps, 0.0, 0);
        let out = run_one(&c, &blob, bindings, pc, [nrows, 1, 1], &arena);
        let got = get(&out, &y);
        for r in 0..nrows as usize {
            let row = &xv[r * ncols as usize..(r + 1) * ncols as usize];
            let scale = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / ncols as f32 + eps).sqrt();
            for i in 0..ncols as usize {
                let want = row[i] * scale * if fused { wv[i] } else { 1.0 };
                assert!(close(got[r * ncols as usize + i], want, 1e-4), "fused={fused} [{i},{r}] = {} vs {want}", got[r * ncols as usize + i]);
            }
        }
        println!("rms_norm fused={fused}: ok ({} bindings, wg {:?})", blob.sig.bindings.len(), blob.sig.workgroup);
    }
}

/// Q4_0 as ggml stores it: per 32 values an f16 scale and 16 bytes of nibbles
/// (element j in the low nibble of byte j, element j + 16 in the high one).
/// Returns the bytes and the dequantized values the kernel should reproduce.
fn quantize_q4_0(x: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut deq = Vec::new();
    for block in x.chunks(32) {
        let max = block.iter().copied().fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d16 = half::f16::from_f32(d);
        bytes.extend(d16.to_le_bytes());
        let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for j in 0..16 {
            let (a, b) = (q(block[j]), q(block[j + 16]));
            bytes.push(a | (b << 4));
            lo[j] = a;
            hi[j] = b;
        }
        let d = d16.to_f32();
        deq.extend(lo.iter().map(|&v| (v as f32 - 8.0) * d));
        deq.extend(hi.iter().map(|&v| (v as f32 - 8.0) * d));
    }
    (bytes, deq)
}

/// `vk_mat_vec_push_constants` for a contiguous weight matrix `[k, n]`
/// times activations `[k, cols]` into `[n, cols]` (ggml_vk_mul_mat_vec_q_f16).
fn mat_vec_pc(k: u32, n: u32, cols: u32) -> Vec<u8> {
    let batch_n = cols > 1;
    let (bx, by, bd) = if batch_n { (0, k, n) } else { (k * n, k * cols, n * cols) };
    words(&[k, k, k, n, bx, by, bd, 0, 0, 1, 1, 1, 1])
}

fn mat_vec(c: &Ctx, tname: &str, spec: [u32; 3], weights: &[u8], k: u32, n: u32, cols: u32, y: &[f32]) -> Vec<f32> {
    let key = format!("DATA_A_{}", tname.to_uppercase());
    let defs = [
        (key.as_str(), "1"),
        ("FLOAT_TYPE", "float"),
        ("FLOAT_TYPEV2", "vec2"),
        ("B_TYPE", "float"),
        ("B_TYPEV2", "vec2"),
        ("B_TYPEV4", "vec4"),
        ("D_TYPE", "float"),
    ];
    let blob = shader(&c.compiler, "mul_mat_vec.comp", &defs, &spec);
    mat_vec_with(c, blob, spec, weights, k, n, cols, y)
}

fn mat_vec_with(c: &Ctx, blob: Arc<KernelBlob>, spec: [u32; 3], weights: &[u8], k: u32, n: u32, cols: u32, y: &[f32]) -> Vec<f32> {
    assert_eq!(blob.sig.bindings.len(), 5, "{:?}", blob.sig);

    // Weights live in model memory, activations and results in the step arena.
    let model = c.engine.upload_model(&[weights]).unwrap();
    let ybytes = 4 * (k * cols) as u64;
    let dbytes = 4 * (n * cols) as u64;
    let d_off = ybytes.next_multiple_of(256);
    let (yr, dr) = (Range::new(Space::Step, 0, ybytes), Range::new(Space::Step, d_off, dbytes));
    let launch = LaunchDesc {
        kernel: blob,
        // X, Y, D, then the two optional bias buffers (unused: bound to D, flags 0).
        bindings: vec![Range::new(Space::Model(0), 0, weights.len() as u64), yr, dr, dr, dr],
        push: mat_vec_pc(k, n, cols),
        grid: [n.div_ceil(spec[1]), 1, 1],
    };
    let plan = c.engine.compile(&[launch], d_off + dbytes).unwrap_or_else(|e| panic!("{e}"));
    let yb: Vec<u8> = y.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let t = c.engine.submit(&plan, &model, None, &[(0, &yb)]).unwrap();
    t.read(d_off, dbytes as usize).unwrap().chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect()
}

fn check_mat_vec(got: &[f32], w: &[f32], y: &[f32], k: usize, n: usize, cols: usize, tol: f32, what: &str) {
    for col in 0..cols {
        for row in 0..n {
            let want: f32 = (0..k).map(|i| w[row * k + i] * y[col * k + i]).sum();
            let g = got[col * n + row];
            assert!(close(g, want, tol), "{what}: d[{row},{col}] = {g} vs {want}");
        }
    }
}

#[test]
fn mat_vec_f32_weights() {
    let Some(c) = ctx() else { return };
    let mut seed = 77;
    let (k, n) = (300u32, 70u32);
    let w: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
    let y: Vec<f32> = (0..k).map(|_| lcg(&mut seed)).collect();
    let wb: Vec<u8> = w.iter().flat_map(|v| v.to_ne_bytes()).collect();
    // {workgroup size, rows per workgroup, columns}: ggml's shmem-reduction f32 pipeline.
    let got = mat_vec(&c, "f32", [128, 1, 1], &wb, k, n, 1, &y);
    check_mat_vec(&got, &w, &y, k as usize, n as usize, 1, 1e-4, "f32");
}

#[test]
fn mat_vec_q4_0_weights_with_batched_columns() {
    let Some(c) = ctx() else { return };
    let mut seed = 78;
    let (k, n, cols) = (256u32, 65u32, 4u32);
    let w: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
    let y: Vec<f32> = (0..k * cols).map(|_| lcg(&mut seed)).collect();
    let (wq, wdeq) = quantize_q4_0(&w);
    assert_eq!(wq.len(), (k * n / 32 * 18) as usize);
    let got = mat_vec(&c, "q4_0", [128, 2, cols], &wq, k, n, cols, &y);
    // The kernel reads the quantized weights: it must equal the dequantized product,
    // and stay close to the unquantized one.
    check_mat_vec(&got, &wdeq, &y, k as usize, n as usize, cols as usize, 1e-4, "q4_0");
    let exact: Vec<f32> = (0..cols as usize).flat_map(|col| (0..n as usize).map(move |row| (col, row))).map(|(col, row)| (0..k as usize).map(|i| w[row * k as usize + i] * y[col * k as usize + i]).sum()).collect();
    let err = got.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("q4_0 vs unquantized max abs error {err:.3} (values ~ sqrt(256)/3)");
    assert!(err < 2.0);
}

fn library() -> Option<SpvLibrary> {
    let lib = SpvLibrary::open_default().filter(|l| l.contains("add_f32_f32_f32"));
    if lib.is_none() {
        eprintln!("no precompiled ggml shaders (run scripts/build-ggml-vulkan-shaders.sh), skipping");
    }
    lib
}

/// Every module the generator produced (about 2000) goes through the reflector:
/// real-world SPIR-V with coopmat, bf16, int8 dot products and spec constants.
#[test]
fn precompiled_library_reflects_every_module() {
    let Some(lib) = library() else { return };
    let names = lib.names();
    let (mut max_bind, mut max_push, mut zero_push, mut failures) = (0usize, 0u32, 0usize, Vec::new());
    for name in &names {
        match lib.load(name, &[]) {
            Ok(b) => {
                max_bind = max_bind.max(b.sig.bindings.len());
                max_push = max_push.max(b.sig.push_bytes);
                zero_push += usize::from(b.sig.push_bytes == 0);
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    println!("{} modules: up to {max_bind} bindings, up to {max_push} push bytes, {zero_push} without push constants", names.len());
    assert!(failures.is_empty(), "{} modules failed to reflect, e.g. {:?}", failures.len(), &failures[..failures.len().min(3)]);
    assert!(names.len() > 1500, "expected the full variant set, found {}", names.len());
    assert!(max_push <= 256, "ggml keeps push constants within 256 bytes");
}

/// A blob loaded by name from the precompiled library, through a `Registry`,
/// computes the same thing as one compiled from source a moment ago.
#[test]
fn precompiled_mat_vec_matches_source_compiled() {
    let (Some(c), Some(lib)) = (ctx(), library()) else { return };
    let mut registry = Registry::new();
    registry.register(lib);
    let mut seed = 99;
    let (k, n, cols) = (256u32, 65u32, 4u32);
    let w: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed)).collect();
    let y: Vec<f32> = (0..k * cols).map(|_| lcg(&mut seed)).collect();
    let (wq, wdeq) = quantize_q4_0(&w);

    let spec = [128u32, 2, cols];
    let key = KernelKey { name: "mul_mat_vec_q4_0_f32_f32".into(), params: spec.iter().map(|&v| v as i64).collect() };
    let blob = Arc::new(registry.resolve(Target::Vulkan, &key).unwrap());
    let got = mat_vec_with(&c, blob, spec, &wq, k, n, cols, &y);
    check_mat_vec(&got, &wdeq, &y, k as usize, n as usize, cols as usize, 1e-4, "precompiled q4_0");
    let src = mat_vec(&c, "q4_0", spec, &wq, k, n, cols, &y);
    assert_eq!(got, src, "precompiled and source-compiled blobs must agree bit for bit");
}
