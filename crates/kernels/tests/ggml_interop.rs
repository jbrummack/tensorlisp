//! A ggml graph and a tensorlisp-kernels launch operate on the same tensors,
//! in both orders, on one queue.
#![cfg(all(target_os = "macos", feature = "ggml"))]

use std::ptr::null_mut;

use ggml_sys::ffi::*;
use tensorlisp_kernels::metal::paged_attn::{copy_blocks, CopyBlocks};
use tensorlisp_kernels::metal::ggml::{device, tensor_buffer};
use tensorlisp_kernels::DType;

#[test]
fn kernels_run_on_ggml_tensors_between_ggml_graphs() {
    unsafe {
        let gpu = ggml_backend_dev_by_type(ggml_backend_dev_type::GGML_BACKEND_DEVICE_TYPE_GPU);
        assert!(!gpu.is_null());
        let backend = ggml_backend_dev_init(gpu, std::ptr::null());
        let ctx = ggml_init(ggml_init_params { mem_size: 1 << 20, mem_buffer: null_mut(), no_alloc: true });

        let (blocks, n) = (6usize, 40usize);
        let kc = ggml_new_tensor_1d(ctx, ggml_type::GGML_TYPE_F32, (blocks * n) as i64);
        let vc = ggml_new_tensor_1d(ctx, ggml_type::GGML_TYPE_F32, (blocks * n) as i64);
        let map = ggml_new_tensor_1d(ctx, ggml_type::GGML_TYPE_I64, 2);
        // ggml op that runs *before* our kernel: kc = kc_in * 2, so the kernel
        // consumes ggml output; and one that runs after.
        let kc_in = ggml_new_tensor_1d(ctx, ggml_type::GGML_TYPE_F32, (blocks * n) as i64);
        let pre = ggml_scale(ctx, kc_in, 2.0);
        let pre_cpy = ggml_cpy(ctx, pre, kc);
        let post = ggml_scale(ctx, kc, 10.0);
        let buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        assert!(!buf.is_null());

        let src: Vec<f32> = (0..blocks * n).map(|i| i as f32).collect();
        ggml_backend_tensor_set(kc_in, src.as_ptr().cast(), 0, src.len() * 4);
        let zero = vec![0f32; blocks * n];
        ggml_backend_tensor_set(vc, zero.as_ptr().cast(), 0, zero.len() * 4);
        ggml_backend_tensor_set(map, [1i64, 3].as_ptr().cast(), 0, 16);

        let g1 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g1, pre_cpy);
        assert_eq!(ggml_backend_graph_compute(backend, g1), ggml_status::GGML_STATUS_SUCCESS);

        let dev = device().expect("ggml's Metal device");
        let (kb, koff) = tensor_buffer(kc).expect("kc is a Metal tensor");
        let (vb, voff) = tensor_buffer(vc).unwrap();
        let (mb, moff) = tensor_buffer(map).unwrap();
        copy_blocks(
            &dev,
            &CopyBlocks {
                dtype: DType::F32,
                key_cache: kb.at(koff),
                value_cache: vb.at(voff),
                block_mapping: mb.at(moff),
                num_pairs: 1,
                numel_per_block_key: n,
                numel_per_block_value: n,
            },
        )
        .unwrap();
        dev.sync().unwrap();

        let g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, post);
        assert_eq!(ggml_backend_graph_compute(backend, g2), ggml_status::GGML_STATUS_SUCCESS);

        let mut out = vec![0f32; blocks * n];
        ggml_backend_tensor_get(post, out.as_mut_ptr().cast(), 0, out.len() * 4);
        // block 3 := block 1 (copied by our kernel), everything x2 (ggml, before) x10 (ggml, after)
        for i in 0..n {
            assert_eq!(out[3 * n + i], src[n + i] * 20.0, "copied block, element {i}");
            assert_eq!(out[n + i], src[n + i] * 20.0, "source block, element {i}");
            assert_eq!(out[5 * n + i], src[5 * n + i] * 20.0, "untouched block, element {i}");
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}
