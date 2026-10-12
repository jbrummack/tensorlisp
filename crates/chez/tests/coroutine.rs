//! A Scheme computation can yield to Rust and be resumed by a later, separate call: the basis for
//! pipelines that wait for input without occupying the Scheme thread.
use chez::{Scheme, Value};

#[test]
fn continuations_survive_returning_to_rust() {
    let scheme = Scheme::new().unwrap();
    scheme
        .eval(
            r#"
(define esc #f)
(define saved #f)
(define p (make-parameter 0))
;; Leaves the computation: the caller of start/resume gets (yielded tag).
(define (yield! tag)
  (call/cc (lambda (k) (set! saved k) (esc (list 'yielded tag)))))
(define (start thunk)
  (call/cc (lambda (r) (set! esc r) (let ([v (thunk)]) (esc (list 'done v))))))
(define (resume)
  (call/cc (lambda (r) (set! esc r) (saved #f))))
(define (body)
  (parameterize ([p 5])
    (let loop ([i 0] [acc '()])
      (if (= i 3)
          (list (p) (reverse acc))
          (begin (yield! i) (loop (+ i 1) (cons (* i i (p)) acc)))))))
"#,
        )
        .unwrap();
    let tagged = |tag: &str, v: Value| Value::List(vec![Value::Symbol(tag.into()), v]);
    assert_eq!(scheme.call("start", &[scheme.eval("body").unwrap_or(Value::Void)]).is_err(), true);
    // A procedure can't cross the boundary, so start through a named wrapper.
    scheme.eval("(define (start-body) (start body))").unwrap();
    assert_eq!(scheme.call("start-body", &[]).unwrap(), tagged("yielded", Value::Int(0)));
    scheme.eval("(collect)").unwrap();
    assert_eq!(scheme.call("resume", &[]).unwrap(), tagged("yielded", Value::Int(1)));
    // Something else runs on the Scheme thread in between.
    assert_eq!(scheme.eval("(+ 1 2)").unwrap(), Value::Int(3));
    assert_eq!(scheme.call("resume", &[]).unwrap(), tagged("yielded", Value::Int(2)));
    scheme.eval("(collect)").unwrap();
    assert_eq!(
        scheme.call("resume", &[]).unwrap(),
        tagged("done", Value::List(vec![Value::Int(5), Value::List(vec![Value::Int(0), Value::Int(5), Value::Int(20)])]))
    );
    // The parameter is back to its outer value afterwards.
    assert_eq!(scheme.eval("(p)").unwrap(), Value::Int(0));

    // An error after a resume is caught inside the computation and leaves through the current
    // escape, not through the (long gone) call that started it.
    scheme
        .eval(
            r#"
(define (safe-start thunk)
  (call/cc
    (lambda (r)
      (set! esc r)
      (esc (guard (e [#t (list 'failed (call-with-string-output-port (lambda (port) (display-condition e port))))])
             (list 'done (thunk)))))))
(define (failing) (yield! 'a) (yield! 'b) (error 'failing "boom" 42))
(define (start-failing) (safe-start failing))
"#,
        )
        .unwrap();
    assert_eq!(scheme.call("start-failing", &[]).unwrap(), tagged("yielded", Value::Symbol("a".into())));
    assert_eq!(scheme.call("resume", &[]).unwrap(), tagged("yielded", Value::Symbol("b".into())));
    let Value::List(items) = scheme.call("resume", &[]).unwrap() else { panic!() };
    assert_eq!(items[0], Value::Symbol("failed".into()));
    let Value::String(msg) = &items[1] else { panic!() };
    assert!(msg.contains("boom"), "{msg}");
    assert_eq!(scheme.eval("(+ 1 1)").unwrap(), Value::Int(2));
}
