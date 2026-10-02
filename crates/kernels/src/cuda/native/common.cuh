// Shared by every native-executor library: descriptor structs mirroring the
// Rust-side packer (cuda/lower.rs), element loads/stores by ggml type number,
// block reductions, and ggml's block-quant decoders.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef long long i64;
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ggml_type numbering: 0 f32, 1 f16, 2 q4_0, 3 q4_1, 6 q5_0, 7 q5_1, 8 q8_0,
// 10..14 q2_K..q6_K, 24 i8, 25 i16, 26 i32, 27 i64, 30 bf16.
struct T {
    i64 ne[4];
    i64 nb[4];
    int ty;
    int pad_;
};

// Sources 0..3 and the destination (slot 4), plus scalar slots (ints, or
// floats as their bit pattern, see xi/xf).
struct Op {
    T t[5];
    i64 x[12];
};

__device__ __forceinline__ int xi(i64 v) { return (int)v; }
__device__ __forceinline__ float xf(i64 v) { return __int_as_float((int)v); }

struct C4 {
    i64 i0, i1, i2, i3;
};

__device__ __forceinline__ C4 decomp(i64 i, const T& t) {
    C4 c;
    c.i0 = i % t.ne[0];
    i /= t.ne[0];
    c.i1 = i % t.ne[1];
    i /= t.ne[1];
    c.i2 = i % t.ne[2];
    c.i3 = i / t.ne[2];
    return c;
}

__device__ __forceinline__ i64 offs(const T& t, i64 i0, i64 i1, i64 i2, i64 i3) {
    return i0 * t.nb[0] + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

__device__ __forceinline__ i64 nelem(const T& t) { return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]; }

#define GRID_LOOP(i, n) for (i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (i64)gridDim.x * blockDim.x)

__device__ __forceinline__ int tysz(int ty) {
    switch (ty) {
        case 0: case 26: return 4;
        case 1: case 30: case 25: return 2;
        case 27: return 8;
        default: return 1;
    }
}

__device__ __forceinline__ float ld(const char* p, int ty) {
    switch (ty) {
        case 0: return *(const float*)p;
        case 1: return __half2float(*(const __half*)p);
        case 30: return __bfloat162float(*(const __nv_bfloat16*)p);
        case 26: return (float)*(const int*)p;
        case 27: return (float)*(const i64*)p;
        case 24: return (float)*(const signed char*)p;
        case 25: return (float)*(const short*)p;
        default: return 0.f;
    }
}

__device__ __forceinline__ void st(char* p, int ty, float v) {
    switch (ty) {
        case 0: *(float*)p = v; break;
        case 1: *(__half*)p = __float2half(v); break;
        case 30: *(__nv_bfloat16*)p = __float2bfloat16(v); break;
        case 26: *(int*)p = (int)v; break;
        case 27: *(i64*)p = (i64)v; break;
        case 24: *(signed char*)p = (signed char)v; break;
        case 25: *(short*)p = (short)v; break;
        default: break;
    }
}

// Integer element (indices): exact for i32/i64.
__device__ __forceinline__ i64 ldi(const char* p, int ty) {
    if (ty == 27) return *(const i64*)p;
    if (ty == 26) return (i64)*(const int*)p;
    return (i64)ld(p, ty);
}

__device__ __forceinline__ void cpy_el(char* d, const char* s, int sz) {
    switch (sz) {
        case 8: *(i64*)d = *(const i64*)s; break;
        case 4: *(int*)d = *(const int*)s; break;
        case 2: *(short*)d = *(const short*)s; break;
        default: *d = *s; break;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum / max; blockDim.x must be a multiple of 32 (<= 1024). Every
// thread gets the result, and the helpers can be called back to back.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.f;
    return warp_sum(v);
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : -INFINITY;
    return warp_max(v);
}

// ---------------------------------------------------------------- block quants

__device__ __forceinline__ float ldh(const unsigned char* p) { return __half2float(*(const __half*)p); }

__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ int q3k_scale(const unsigned char* sc, int is) {
    const int w = is >> 2, b = is & 3;
    int lo, hi;
    switch (w) {
        case 0: lo = sc[b] & 0xF; hi = sc[8 + b] & 3; break;
        case 1: lo = sc[4 + b] & 0xF; hi = (sc[8 + b] >> 2) & 3; break;
        case 2: lo = (sc[b] >> 4) & 0xF; hi = (sc[8 + b] >> 4) & 3; break;
        default: lo = (sc[4 + b] >> 4) & 0xF; hi = (sc[8 + b] >> 6) & 3; break;
    }
    return (lo | (hi << 4)) - 32;
}

// Element `k` of a row of type TY starting at `row`. `nb0` is the element
// stride, only meaningful for the unquantized types.
template <int TY>
__device__ __forceinline__ float dqf(const char* row, i64 k, i64 nb0) {
    const unsigned char* r = (const unsigned char*)row;
    if constexpr (TY == 0) {
        return *(const float*)(row + k * nb0);
    } else if constexpr (TY == 1) {
        return __half2float(*(const __half*)(row + k * nb0));
    } else if constexpr (TY == 30) {
        return __bfloat162float(*(const __nv_bfloat16*)(row + k * nb0));
    } else if constexpr (TY == 2) {  // q4_0: { half d; u8 qs[16] }
        const unsigned char* b = r + (k >> 5) * 18;
        const int j = (int)(k & 31);
        const int q = b[2 + (j & 15)];
        return ldh(b) * (float)((j < 16 ? (q & 0xF) : (q >> 4)) - 8);
    } else if constexpr (TY == 3) {  // q4_1: { half d, m; u8 qs[16] }
        const unsigned char* b = r + (k >> 5) * 20;
        const int j = (int)(k & 31);
        const int q = b[4 + (j & 15)];
        return ldh(b) * (float)(j < 16 ? (q & 0xF) : (q >> 4)) + ldh(b + 2);
    } else if constexpr (TY == 6) {  // q5_0: { half d; u8 qh[4]; u8 qs[16] }
        const unsigned char* b = r + (k >> 5) * 22;
        const int j = (int)(k & 31);
        const unsigned qh = b[2] | (b[3] << 8) | (b[4] << 16) | ((unsigned)b[5] << 24);
        const int q = b[6 + (j & 15)];
        const int v = (j < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> j) & 1) << 4);
        return ldh(b) * (float)(v - 16);
    } else if constexpr (TY == 7) {  // q5_1: { half d, m; u8 qh[4]; u8 qs[16] }
        const unsigned char* b = r + (k >> 5) * 24;
        const int j = (int)(k & 31);
        const unsigned qh = b[4] | (b[5] << 8) | (b[6] << 16) | ((unsigned)b[7] << 24);
        const int q = b[8 + (j & 15)];
        const int v = (j < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> j) & 1) << 4);
        return ldh(b) * (float)v + ldh(b + 2);
    } else if constexpr (TY == 8) {  // q8_0: { half d; i8 qs[32] }
        const unsigned char* b = r + (k >> 5) * 34;
        return ldh(b) * (float)((const signed char*)b)[2 + (k & 31)];
    } else if constexpr (TY == 10) {  // q2_K: { u8 scales[16]; u8 qs[64]; half d, dmin }
        const unsigned char* b = r + (k >> 8) * 84;
        const int e = (int)(k & 255), n = e >> 7, p = e & 127, j = p >> 5, h = (p >> 4) & 1, l = p & 15;
        const int sc = b[n * 8 + j * 2 + h];
        const int q = (b[16 + n * 32 + h * 16 + l] >> (2 * j)) & 3;
        return ldh(b + 80) * (float)(sc & 0xF) * (float)q - ldh(b + 82) * (float)(sc >> 4);
    } else if constexpr (TY == 11) {  // q3_K: { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d }
        const unsigned char* b = r + (k >> 8) * 110;
        const int e = (int)(k & 255), n = e >> 7, p = e & 127, j = p >> 5, h = (p >> 4) & 1, l = p & 15;
        const int sc = q3k_scale(b + 96, n * 8 + j * 2 + h);
        const int q = (b[32 + n * 32 + h * 16 + l] >> (2 * j)) & 3;
        const int hm = b[h * 16 + l] & (1 << (n * 4 + j));
        return ldh(b + 108) * (float)sc * (float)(q - (hm ? 0 : 4));
    } else if constexpr (TY == 12) {  // q4_K: { half d, dmin; u8 scales[12]; u8 qs[128] }
        const unsigned char* b = r + (k >> 8) * 144;
        const int e = (int)(k & 255), c = e >> 6, p = e & 63, hi = p >> 5, l = p & 31;
        int sc, m;
        scale_min_k4(c * 2 + hi, b + 4, sc, m);
        const int q = b[16 + c * 32 + l];
        return ldh(b) * (float)sc * (float)(hi ? (q >> 4) : (q & 0xF)) - ldh(b + 2) * (float)m;
    } else if constexpr (TY == 13) {  // q5_K: { half d, dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }
        const unsigned char* b = r + (k >> 8) * 176;
        const int e = (int)(k & 255), c = e >> 6, p = e & 63, hi = p >> 5, l = p & 31;
        int sc, m;
        scale_min_k4(c * 2 + hi, b + 4, sc, m);
        const int q = b[48 + c * 32 + l];
        const int v = (hi ? (q >> 4) : (q & 0xF)) + ((b[16 + l] & (1 << (c * 2 + hi))) ? 16 : 0);
        return ldh(b) * (float)sc * (float)v - ldh(b + 2) * (float)m;
    } else if constexpr (TY == 14) {  // q6_K: { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d }
        const unsigned char* b = r + (k >> 8) * 210;
        const int e = (int)(k & 255), n = e >> 7, p = e & 127, qi = p >> 5, l = p & 31;
        const int ql = b[n * 64 + l + (qi & 1) * 32];
        const int nib = qi < 2 ? (ql & 0xF) : (ql >> 4);
        const int qhv = (b[128 + n * 32 + l] >> (2 * qi)) & 3;
        const int sc = ((const signed char*)b)[192 + n * 8 + (l >> 4) + 2 * qi];
        return ldh(b + 208) * (float)sc * (float)((nib | (qhv << 4)) - 32);
    } else {
        return 0.f;
    }
}

__device__ __forceinline__ float dq(const char* row, int ty, i64 k, i64 nb0) {
    switch (ty) {
        case 0: return dqf<0>(row, k, nb0);
        case 1: return dqf<1>(row, k, nb0);
        case 30: return dqf<30>(row, k, nb0);
        case 2: return dqf<2>(row, k, nb0);
        case 3: return dqf<3>(row, k, nb0);
        case 6: return dqf<6>(row, k, nb0);
        case 7: return dqf<7>(row, k, nb0);
        case 8: return dqf<8>(row, k, nb0);
        case 10: return dqf<10>(row, k, nb0);
        case 11: return dqf<11>(row, k, nb0);
        case 12: return dqf<12>(row, k, nb0);
        case 13: return dqf<13>(row, k, nb0);
        case 14: return dqf<14>(row, k, nb0);
        default: return 0.f;
    }
}
