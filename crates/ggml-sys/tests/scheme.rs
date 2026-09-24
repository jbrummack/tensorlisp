use ggml_sys::ffi::scheme;

#[test]
fn generated_scheme_bindings() {
    let symbols = scheme::symbols();
    assert!(symbols.iter().any(|(n, _)| *n == "ggml_add"));
    assert!(symbols.iter().all(|(_, p)| !p.is_null()));
    assert!(scheme::RAW.contains(r#"(define ggml_add (foreign-procedure "ggml_add" (uptr uptr uptr) uptr))"#));
    assert!(scheme::OPS.contains(
        "(define (ggml-add a b) (let* ([%c (%current-ctx 'ggml-add)] [a (%unwrap 'ggml-add 'a a)] [b (%unwrap 'ggml-add 'b b)]) (%wrap 'ggml-add (ggml_add %c a b))))"
    ));
    assert!(scheme::OPS.contains("(define (ggml-scale a s) (let* ([%c (%current-ctx 'ggml-scale)] [a (%unwrap 'ggml-scale 'a a)]) (%wrap 'ggml-scale (ggml_scale %c a s))))"));
    assert!(!scheme::OP_NAMES.split(' ').any(|n| n == "ggml-new-f32"));
    assert!(scheme::CONSTANTS.contains("(define GGML_TYPE_F32 0)"));
}
