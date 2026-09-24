//! Chez is a process-global runtime, so everything runs in one test.
use chez::{Error, Scheme, Value};

extern "C" fn rust_mul_add(a: i64, b: i64, c: f64) -> f64 {
    (a * b) as f64 + c
}

#[test]
fn embed() {
    let scheme = Scheme::new().unwrap();
    assert!(matches!(Scheme::new(), Err(Error::AlreadyRunning)));

    // Round-tripping values.
    assert_eq!(scheme.eval("(+ 1 2)").unwrap(), Value::Int(3));
    assert_eq!(scheme.eval("(define x 10) (* x 1.5)").unwrap(), Value::Float(15.0));
    assert_eq!(scheme.eval("\"grüße\"").unwrap(), Value::String("grüße".into()));
    assert_eq!(
        scheme.eval("'(a 1 #t (2 . 3) #(4) #vu8(5))").unwrap(),
        Value::List(vec![
            Value::Symbol("a".into()),
            Value::Int(1),
            Value::Bool(true),
            Value::Pair(Box::new(Value::Int(2)), Box::new(Value::Int(3))),
            Value::Vector(vec![Value::Int(4)]),
            Value::Bytevector(vec![5]),
        ])
    );
    assert_eq!(scheme.eval("(expt 2 62)").unwrap(), Value::Int(1 << 62));
    assert_eq!(scheme.eval("(expt 2 64)").unwrap(), Value::Opaque("bignum"));
    assert_eq!(scheme.eval("car").unwrap(), Value::Opaque("procedure"));

    // Calling Scheme from Rust.
    scheme.eval("(define (greet name n) (list name (* n 2)))").unwrap();
    let args = [Value::String("hi".into()), Value::Int(21)];
    assert_eq!(
        scheme.call("greet", &args).unwrap(),
        Value::List(vec![Value::String("hi".into()), Value::Int(42)])
    );

    // Errors come back as values instead of aborting.
    let err = scheme.eval("(car 5)").unwrap_err().to_string();
    assert!(err.contains("car"), "{err}");
    assert!(scheme.call("no-such-procedure", &[]).is_err());
    assert!(scheme.eval("(raise 'boom)").unwrap_err().to_string().contains("boom"));
    // Still usable after an error, and the compiler (scheme.boot) is loaded.
    assert_eq!(scheme.eval("(compile '(+ 1 1))").unwrap(), Value::Int(2));

    // Calling Rust from Scheme through the foreign symbol table.
    unsafe { scheme.register_foreign("rust_mul_add", rust_mul_add as *const _) }.unwrap();
    scheme
        .eval("(define mul-add (foreign-procedure \"rust_mul_add\" (integer-64 integer-64 double) double))")
        .unwrap();
    assert_eq!(scheme.eval("(mul-add 6 7 0.5)").unwrap(), Value::Float(42.5));

    // The runtime can be booted again after being dropped.
    drop(scheme);
    let scheme = Scheme::new().unwrap();
    assert_eq!(scheme.eval("(+ 40 2)").unwrap(), Value::Int(42));
}
