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

#[test]
fn bool_parameters_are_int_sized() {
    // See the comment in ggml-codegen: stdbool stack arguments break Chez's arm64 macOS FFI.
    // ggml_im2col's trailing `ggml_type dst_type` enum is "unsigned-32" or
    // "integer-32" depending on the repr the C compiler (not ggml) picks for
    // a non-negative-valued enum — GCC/Clang choose unsigned, MSVC signed;
    // both are correct, so accept either.
    let prefix = r#"(define ggml_im2col (foreign-procedure "ggml_im2col" (uptr uptr uptr integer-32 integer-32 integer-32 integer-32 integer-32 integer-32 boolean "#;
    assert!(
        scheme::RAW.contains(&format!("{prefix}unsigned-32) uptr))")) || scheme::RAW.contains(&format!("{prefix}integer-32) uptr))")),
        "expected ggml_im2col's bool param to generate as `boolean` followed by a 32-bit integer enum type; got: {}",
        scheme::RAW.lines().find(|l| l.contains("ggml_im2col ")).unwrap_or("<not found>")
    );
    // Returns stay stdbool.
    assert!(scheme::RAW.contains(r#"(define ggml_is_contiguous (foreign-procedure "ggml_is_contiguous" (uptr) stdbool))"#));
}
