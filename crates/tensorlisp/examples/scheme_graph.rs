//! Builds and computes `(a + b) * 2` as a ggml graph from Scheme, with every
//! ggml function registered and declared by hand. Baseline for generating the
//! registration from the bindings.
use std::ffi::c_void;
use ggml_sys::ffi::*;

fn main() {
    let scheme = chez::Scheme::new().unwrap();
    let syms: &[(&str, *const c_void)] = &[
        ("ggml_init", ggml_init as *const _),
        ("ggml_free", ggml_free as *const _),
        ("ggml_new_tensor_1d", ggml_new_tensor_1d as *const _),
        ("ggml_set_f32", ggml_set_f32 as *const _),
        ("ggml_get_f32_1d", ggml_get_f32_1d as *const _),
        ("ggml_add", ggml_add as *const _),
        ("ggml_scale", ggml_scale as *const _),
        ("ggml_new_graph", ggml_new_graph as *const _),
        ("ggml_build_forward_expand", ggml_build_forward_expand as *const _),
        ("ggml_graph_compute_with_ctx", ggml_graph_compute_with_ctx as *const _),
    ];
    for (name, f) in syms {
        unsafe { scheme.register_foreign(name, *f) }.unwrap();
    }
    let r = scheme.eval(r#"
      (define-ftype ggml-init-params (struct [mem-size size_t] [mem-buffer void*] [no-alloc unsigned-8]))
      (define ggml-init (foreign-procedure "ggml_init" ((& ggml-init-params)) uptr))
      (define ggml-free (foreign-procedure "ggml_free" (uptr) void))
      (define ggml-new-tensor-1d (foreign-procedure "ggml_new_tensor_1d" (uptr int integer-64) uptr))
      (define ggml-set-f32 (foreign-procedure "ggml_set_f32" (uptr float) uptr))
      (define ggml-get-f32-1d (foreign-procedure "ggml_get_f32_1d" (uptr int) float))
      (define ggml-add (foreign-procedure "ggml_add" (uptr uptr uptr) uptr))
      (define ggml-scale (foreign-procedure "ggml_scale" (uptr uptr float) uptr))
      (define ggml-new-graph (foreign-procedure "ggml_new_graph" (uptr) uptr))
      (define ggml-build-forward-expand (foreign-procedure "ggml_build_forward_expand" (uptr uptr) void))
      (define ggml-graph-compute-with-ctx (foreign-procedure "ggml_graph_compute_with_ctx" (uptr uptr int) int))

      (define params (make-ftype-pointer ggml-init-params (foreign-alloc (ftype-sizeof ggml-init-params))))
      (ftype-set! ggml-init-params (mem-size) params (* 16 1024 1024))
      (ftype-set! ggml-init-params (mem-buffer) params 0)
      (ftype-set! ggml-init-params (no-alloc) params 0)
      (define ctx (ggml-init params))
      (foreign-free (ftype-pointer-address params))
      (define GGML_TYPE_F32 0)
      (define a (ggml-set-f32 (ggml-new-tensor-1d ctx GGML_TYPE_F32 4) 3.0))
      (define b (ggml-set-f32 (ggml-new-tensor-1d ctx GGML_TYPE_F32 4) 4.0))
      (define out (ggml-scale ctx (ggml-add ctx a b) 2.0))  ; (a + b) * 2
      (define gf (ggml-new-graph ctx))
      (ggml-build-forward-expand gf out)
      (ggml-graph-compute-with-ctx ctx gf 1)
      (let ([r (map (lambda (i) (ggml-get-f32-1d out i)) '(0 1 2 3))])
        (ggml-free ctx)
        r)
    "#);
    println!("{r:?}");
}
