// Kernels ggml-metal lacks or restricts, in the same style as ggml's (one
// argument struct at buffer 0, then tensors). Compiled as a separate library.
#include <metal_stdlib>
using namespace metal;

struct kargs_pad_ext {
    int64_t  ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int64_t  ne0,  ne1,  ne2,  ne3;
    uint64_t nb0,  nb1,  nb2,  nb3;
    int32_t  lp0, lp1, lp2, lp3;
};

// Zero padding with leading and trailing amounts (ggml_pad_ext). One
// threadgroup row handles up to 1024 outputs per x-group, like ggml's kernel_pad.
template <typename T>
kernel void kernel_pad_ext(
        constant kargs_pad_ext & args,
        device const char * src0,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;

    const int32_t k0 = tgpig.x / args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t s3 = i3 - args.lp3;
    const int32_t s2 = i2 - args.lp2;
    const int32_t s1 = i1 - args.lp1;
    const bool row_in = s3 >= 0 && s3 < args.ne03 && s2 >= 0 && s2 < args.ne02 && s1 >= 0 && s1 < args.ne01;

    device const char * src_row = src0 + s3*args.nb03 + s2*args.nb02 + s1*args.nb01;
    device       T    * dst_ptr = (device T *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }
        const int32_t s0 = i0 - args.lp0;
        if (row_in && s0 >= 0 && s0 < args.ne00) {
            dst_ptr[i0] = *((device const T *) (src_row + s0*args.nb00));
        } else {
            dst_ptr[i0] = T(0);
        }
    }
}

template [[host_name("kernel_pad_ext_f32")]] kernel void kernel_pad_ext<float>(constant kargs_pad_ext &, device const char *, device char *, uint3, uint3, uint3);
template [[host_name("kernel_pad_ext_f16")]] kernel void kernel_pad_ext<half>(constant kargs_pad_ext &, device const char *, device char *, uint3, uint3, uint3);
