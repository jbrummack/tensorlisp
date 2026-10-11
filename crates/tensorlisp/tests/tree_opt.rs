//! The Tree optimization passes (stdlib/tree_opt.ss): each pass on small kernels, checked
//! structurally, and the whole pipeline on every generator (compiles, the SDK's spirv-val accepts it).
#![cfg(feature = "native-vulkan")]

use chez::{Scheme, Value};

const SPIRV_LIB: &str = include_str!("../src/scheme/stdlib/tree_spirv.ss");
const TREE_LIB: &str = include_str!("../src/scheme/stdlib/tree.ss");
const OPT_LIB: &str = include_str!("../src/scheme/stdlib/tree_opt.ss");
const KERNELS_LIB: &str = include_str!("../src/scheme/stdlib/tree_kernels.ss");

/// One Scheme at a time per process: tests take this lock for as long as they hold their `Scheme`.
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Guarded(Scheme, #[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
impl std::ops::Deref for Guarded {
    type Target = Scheme;
    fn deref(&self) -> &Scheme {
        &self.0
    }
}

fn scheme() -> Guarded {
    let guard = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let scheme = Scheme::new().unwrap();
    for lib in [SPIRV_LIB, TREE_LIB, OPT_LIB, KERNELS_LIB] {
        scheme.eval(lib).unwrap();
    }
    scheme
        .eval("(define-top-level-value '$env (environment '(rnrs) '(tl tree) '(tl tree opt) '(tl tree kernels)))")
        .unwrap();
    Guarded(scheme, guard)
}

/// Evaluates `src` in the environment above and expects a boolean.
fn check(scheme: &Scheme, what: &str, src: &str) {
    match scheme.eval(&format!("(eval '{src} $env)")).unwrap_or_else(|e| panic!("{what}: {e}")) {
        Value::Bool(true) => {}
        other => panic!("{what}: got {other:?}"),
    }
}

/// The body statements of a kernel.
const BODY: &str = "(lambda (k) (filter (lambda (f) (not (memq (car f) '(workgroup buffers push shared offsets)))) (cddr k)))";

/// How many times `x` occurs in the tree `t`.
const COUNT: &str = "(letrec ([count (lambda (x t) (cond [(equal? x t) 1] [(pair? t) (+ (count x (car t)) (count x (cdr t)))] [else 0]))]) count)";

#[test]
fn fold_evaluates_constants_and_drops_identities() {
    let s = scheme();
    let kernel = "'(kernel k (workgroup 1) (buffers (d wo f32)) (push (x u32)) (store d 0 (+ (* 2 3) (* x 1) 0)))";
    check(&s, "constants", &format!("(equal? ({BODY} (optimize {kernel} '(fold))) '((store d 0 (+ 6 x))))"));
    let dead = "'(kernel k (workgroup 1) (buffers (d wo f32)) (push (x u32)) (when #f (store d 0 1.0)) (when (< 2 3) (store d 1 2.0)) (store d 2 (select #t 1.0 2.0)))";
    check(&s, "dead arms", &format!("(equal? ({BODY} (optimize {dead} '(fold))) '((begin (store d 1 2.0)) (store d 2 1.0)))"));
    // an integer division that is not exact stays: a literal takes the type of its context
    let div = "'(kernel k (workgroup 1) (buffers (d wo f32)) (store d 0 (/ 1 2)))";
    check(&s, "division kept", &format!("(equal? ({BODY} (optimize {div} '(fold))) '((store d 0 (/ 1 2))))"));
}

#[test]
fn cse_names_a_value_computed_twice() {
    let s = scheme();
    let kernel = "'(kernel k (workgroup 1) (buffers (a ro f32) (d wo f32)) (push (i u32))
        (store d i (+ (* (load a i) 3.0) (* (load a i) 3.0)))
        (store d (+ i 1) (* (load a i) 3.0)))";
    // one load and one multiplication remain
    check(&s, "loads", &format!("(= ({COUNT} '(load a i) (optimize {kernel} '(cse))) 1)"));
    check(&s, "products", &format!("(= ({COUNT} 3.0 (optimize {kernel} '(cse))) 1)"));
    // a value read from a writable buffer is not shared: a store may change it in between
    let rw = "'(kernel k (workgroup 1) (buffers (a rw f32)) (push (i u32))
        (store a 0 (+ (* (load a i) 3.0) (* (load a i) 3.0))))";
    check(&s, "rw loads kept", &format!("(= ({COUNT} '(load a i) (optimize {rw} '(cse))) 2)"));
    // an assignment in between ends the sharing
    let assigned = "'(kernel k (workgroup 1) (buffers (d wo f32)) (push (n u32))
        (var v u32 1)
        (store d 0 (f32 (* (+ v n) (+ v n))))
        (set! v 2)
        (store d 1 (f32 (* (+ v n) (+ v n)))))";
    check(&s, "set! kills", &format!("(= ({COUNT} '(* (+ v n) (+ v n)) (optimize {assigned} '(cse))) 2)"));
}

#[test]
fn licm_hoists_invariant_arithmetic_only() {
    let s = scheme();
    let kernel = "'(kernel k (workgroup 1) (buffers (a ro f32) (d wo f32)) (push (n u32) (m u32))
        (var acc f32 0.0)
        (for (i 0 n)
          (set! acc (+ acc (load a (+ (* m m 17) i)))))
        (store d 0 acc))";
    // (* m m 17) is computed before the loop, i + that stays inside
    check(&s, "hoisted", &format!("(let ([o (optimize {kernel} '(licm))]) (and (= ({COUNT} '(* m m 17) o) 1) (let ([loop (car (filter (lambda (f) (eq? (car f) 'let)) ({BODY} o)))]) (eq? (car (caddr loop)) 'for))))"));
}

#[test]
fn dce_drops_unread_bindings() {
    let s = scheme();
    let kernel = "'(kernel k (workgroup 1) (buffers (d wo f32)) (push (x u32))
        (let ([unused (* x 5)] [used (+ x 1)])
          (var dead f32 0.0)
          (set! dead 2.0)
          (store d 0 (f32 used))))";
    check(&s, "unused let", &format!("(= ({COUNT} 'unused (optimize {kernel} '(dce))) 0)"));
    check(&s, "dead var", &format!("(= ({COUNT} 'dead (optimize {kernel} '(dce))) 0)"));
    check(&s, "used kept", &format!("(> ({COUNT} 'used (optimize {kernel} '(dce))) 0)"));
}

#[test]
fn user_passes_register_by_name() {
    let s = scheme();
    check(
        &s,
        "custom pass",
        "(begin (register-pass! 'rename (lambda (k) (cons 'kernel (cons 'renamed (cddr k)))))
                (eq? (cadr (optimize '(kernel k (workgroup 1) (buffers (d wo f32))) '(rename))) 'renamed))",
    );
}

/// Every generator, optimized and compiled: the optimizer must leave valid kernels.
#[test]
fn every_generator_survives_the_default_passes() {
    let s = scheme();
    let exprs = [
        "(k:binary 'add)",
        "(k:unary 'gelu 'same)",
        "(k:unary 'copy 'reshape)",
        "(k:concat 1)",
        "(k:get-rows 'q8_0)",
        "(k:norm)",
        "(k:rms-norm #t)",
        "(k:soft-max #t)",
        "(k:sum-rows)",
        "(k:matmul 'q8_0 64 64 32 4 4)",
        "(k:matvec 'q8_0 1 4)",
        "(k:matvec 'f16 3 1)",
        "(k:rope 'neox)",
        "(k:set-rows 'f16)",
        "(k:argmax)",
        "(k:quantize-q8)",
        "(k:matvec-q8 1 4)",
        "(k:flash-attn 'f16 'f16 #t #f 2048)",
        "(k:rms-norm #t #t)",
        "(k:ew '((f32 full) (f32 row) (f32 scalar)) 'f32 '((t0 (+ x0 x1)) (t1 (* t0 x2)) (t2 (unary gelu t1))) 't2 0)",
        "(k:ew '((i32 full)) 'f16 '((t0 x0)) 't0 0)",
        "(k:ew '() 'f32 '((t0 (+ c0 (* (f32 idx) c1))) (t1 (unary step t0))) 't1 2)",
        "(k:ew-q8 '((f32 full) (f32 full)) 'f32 '((t0 (unary gelu x0)) (t1 (* t0 x1))) 't1 0)",
        "(k:rowchain 1536 '((f32 full) (f32 full) (f32 full)) '((norm 1 0) (ew ((t0 (+ x2 x))) t0) (norm #f 1)) '((1 f32) (2 q8)) 2)",
        "(k:rowchain 256 '((f32 full) (f32 row)) '((norm 1 0)) '((0 f32)) 1)",
        "(k:flash-attn-part 'f16 'f16 #t #f 128 256 256)",
        "(k:flash-attn-part 'f32 'f32 #f #t 64 64 64)",
        "(k:flash-attn-combine)",
        "(k:flash-attn-combine #t)",
        "(k:norm-rope 256 #t 'neox 'f32 #f)",
        "(k:norm-rope 256 #t 'neox 'f16 #t)",
        "(k:norm-rope 512 #f #f 'f16 #t)",
        "(k:norm-rope 64 #t 'norm 'f32 #f)",
        "(k:gateup-q8 2 2)",
        "(k:gateup-q8 1 1)",
        "(k:matvec-q8 1 1 3)",
        "(k:matvec-q8 2 2 2)",
    ];
    for e in exprs {
        let plain = bytes(&s, &format!("(eval '(compile-kernel {e}) $env)"));
        let opt = bytes(&s, &format!("(eval '(compile-kernel (optimize {e})) $env)"));
        println!("{e}: {} bytes -> {} bytes", plain.len(), opt.len());
        assert!(opt.len() > 20 && opt.len() % 4 == 0);
    }
}

fn bytes(s: &Scheme, src: &str) -> Vec<u8> {
    match s.eval(src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Bytevector(b) => b,
        other => panic!("{src}: {other:?}"),
    }
}
