// Elementwise / strided / reduction kernels of the native CUDA executor.
// Every kernel takes the same arguments: a descriptor blob and up to four
// source pointers plus the destination (see common.cuh and cuda/lower.rs).
// Semantics follow ggml's CPU backend.

#define KARGS const Op op, const char* s0, const char* s1, const char* s2, const char* s3, char* dst
#define K extern "C" __global__ void

// --------------------------------------------------------------------- concat

K k_concat(KARGS) {
    const T& a = op.t[0];
    const T& b = op.t[1];
    const T& d = op.t[4];
    const int dim = xi(op.x[0]);
    const int sz = tysz(d.ty);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        i64 co[4] = {c.i0, c.i1, c.i2, c.i3};
        const char* p;
        if (co[dim] < a.ne[dim]) {
            p = s0 + offs(a, co[0], co[1], co[2], co[3]);
        } else {
            co[dim] -= a.ne[dim];
            p = s1 + offs(b, co[0], co[1], co[2], co[3]);
        }
        cpy_el(dst + offs(d, c.i0, c.i1, c.i2, c.i3), p, sz);
    }
}

// ------------------------------------------------------------------------ bin

K k_bin(KARGS) {
    const T& a = op.t[0];
    const T& b = op.t[1];
    const T& d = op.t[4];
    const int code = xi(op.x[0]);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const float x = ld(s0 + offs(a, c.i0, c.i1, c.i2, c.i3), a.ty);
        const float y = ld(s1 + offs(b, c.i0 % b.ne[0], c.i1 % b.ne[1], c.i2 % b.ne[2], c.i3 % b.ne[3]), b.ty);
        float r;
        switch (code) {
            case 0: r = x + y; break;
            case 1: r = x - y; break;
            case 2: r = x * y; break;
            default: r = x / y; break;
        }
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, r);
    }
}

// --------------------------------------------------------------------- repeat

K k_repeat(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const int sz = tysz(d.ty);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        cpy_el(dst + offs(d, c.i0, c.i1, c.i2, c.i3),
               s0 + offs(a, c.i0 % a.ne[0], c.i1 % a.ne[1], c.i2 % a.ne[2], c.i3 % a.ne[3]), sz);
    }
}

// ---------------------------------------------------------------------- unary

__device__ __forceinline__ float unary_f(int code, float x, const Op& op) {
    switch (code) {
        case 100: return x * xf(op.x[1]) + xf(op.x[2]);                              // scale
        case 101: return xf(op.x[1]);                                                // fill
        case 102: return fminf(fmaxf(x, xf(op.x[1])), xf(op.x[2]));                  // clamp
        case 103: return fmaxf(x, 0.f) + fminf(x, 0.f) * xf(op.x[1]);                // leaky relu
        case 104: return x * x;
        case 105: return sqrtf(x);
        case 106: return sinf(x);
        case 107: return cosf(x);
        case 108: return logf(x);
        case 0: return fabsf(x);
        case 1: return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f);
        case 2: return -x;
        case 3: return x > 0.f ? 1.f : 0.f;
        case 4: return tanhf(x);
        case 5: return x > 0.f ? x : expm1f(x);
        case 6: return fmaxf(x, 0.f);
        case 7: return 1.f / (1.f + expf(-x));
        case 8: return 0.5f * x * (1.f + tanhf(0.79788456080286535588f * x * (1.f + 0.044715f * x * x)));
        case 9: return x * (1.f / (1.f + expf(-1.702f * x)));
        case 10: return x / (1.f + expf(-x));
        case 11: return x * fminf(1.f, fmaxf(0.f, (x + 3.f) / 6.f));
        case 12: return fminf(1.f, fmaxf(0.f, (x + 3.f) / 6.f));
        case 13: return expf(x);
        case 14: return expm1f(x);
        case 15: return x > 20.f ? x : log1pf(expf(x));
        case 16: return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
        case 18: return floorf(x);
        case 19: return ceilf(x);
        case 20: return roundf(x);
        case 21: return truncf(x);
        default: return x;
    }
}

K k_unary(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const int code = xi(op.x[0]);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const float x = ld(s0 + offs(a, c.i0, c.i1, c.i2, c.i3), a.ty);
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, unary_f(code, x, op));
    }
}

// ------------------------------------------------------------------- sum_rows

K k_sum_rows(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const i64 row = blockIdx.x;
    const i64 i1 = row % a.ne[1], i2 = (row / a.ne[1]) % a.ne[2], i3 = row / (a.ne[1] * a.ne[2]);
    const char* base = s0 + offs(a, 0, i1, i2, i3);
    float s = 0.f;
    for (i64 i = threadIdx.x; i < a.ne[0]; i += blockDim.x) s += ld(base + i * a.nb[0], a.ty);
    s = block_sum(s);
    if (threadIdx.x == 0) {
        if (xi(op.x[0])) s /= (float)a.ne[0];
        st(dst + offs(d, 0, i1, i2, i3), d.ty, s);
    }
}

// ------------------------------------------------------------------- get_rows

K k_get_rows(KARGS) {
    const T& a = op.t[0];
    const T& idx = op.t[1];
    const T& d = op.t[4];
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const i64 r = ldi(s1 + c.i1 * idx.nb[0] + c.i2 * idx.nb[1] + c.i3 * idx.nb[2], idx.ty);
        const char* row = s0 + r * a.nb[1] + c.i2 * a.nb[2] + c.i3 * a.nb[3];
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, dq(row, a.ty, c.i0, a.nb[0]));
    }
}

// ------------------------------------------------------------------ soft_max

K k_soft_max(KARGS) {
    const T& a = op.t[0];
    const T& m = op.t[1];
    const T& d = op.t[4];
    const float scale = xf(op.x[0]), max_bias = xf(op.x[1]), m0 = xf(op.x[2]), m1 = xf(op.x[3]);
    const int nhl2 = xi(op.x[4]);
    const i64 row = blockIdx.x;
    const i64 i1 = row % a.ne[1], i2 = (row / a.ne[1]) % a.ne[2], i3 = row / (a.ne[1] * a.ne[2]);
    const char* sp = s0 + offs(a, 0, i1, i2, i3);
    const char* mp = s1 ? s1 + i1 * m.nb[1] + (i2 % m.ne[2]) * m.nb[2] + (i3 % m.ne[3]) * m.nb[3] : nullptr;
    float slope = 1.f;
    if (max_bias > 0.f) {
        const int h = (int)i2;
        slope = powf(h < nhl2 ? m0 : m1, (float)(h < nhl2 ? h + 1 : 2 * (h - nhl2) + 1));
    }
    const i64 n = a.ne[0];
    float mx = -INFINITY;
    for (i64 j = threadIdx.x; j < n; j += blockDim.x) {
        float v = ld(sp + j * a.nb[0], a.ty) * scale;
        if (mp) v += slope * ld(mp + j * m.nb[0], m.ty);
        mx = fmaxf(mx, v);
    }
    mx = block_max(mx);
    float sum = 0.f;
    for (i64 j = threadIdx.x; j < n; j += blockDim.x) {
        float v = ld(sp + j * a.nb[0], a.ty) * scale;
        if (mp) v += slope * ld(mp + j * m.nb[0], m.ty);
        sum += expf(v - mx);
    }
    sum = block_sum(sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    char* dp = dst + offs(d, 0, i1, i2, i3);
    for (i64 j = threadIdx.x; j < n; j += blockDim.x) {
        float v = ld(sp + j * a.nb[0], a.ty) * scale;
        if (mp) v += slope * ld(mp + j * m.nb[0], m.ty);
        st(dp + j * d.nb[0], d.ty, mx == -INFINITY ? 0.f : expf(v - mx) * inv);
    }
}

// ------------------------------------------------------------------------ cpy

K k_cpy(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const bool same = a.ty == d.ty;
    const int sz = tysz(d.ty);
    GRID_LOOP(i, nelem(a)) {
        const C4 c = decomp(i, a);
        const C4 e = decomp(i, d);
        char* dp = dst + offs(d, e.i0, e.i1, e.i2, e.i3);
        if (a.ty >= 2 && a.ty <= 14 && a.ty != 4 && a.ty != 5 && a.ty != 9) {
            st(dp, d.ty, dq(s0 + offs(a, 0, c.i1, c.i2, c.i3), a.ty, c.i0, a.nb[0]));
        } else if (same) {
            cpy_el(dp, s0 + offs(a, c.i0, c.i1, c.i2, c.i3), sz);
        } else {
            st(dp, d.ty, ld(s0 + offs(a, c.i0, c.i1, c.i2, c.i3), a.ty));
        }
    }
}

// ----------------------------------------------------------------------- norm

K k_norm(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const bool rms = xi(op.x[0]) != 0;
    const float eps = xf(op.x[1]);
    const i64 row = blockIdx.x;
    const i64 i1 = row % a.ne[1], i2 = (row / a.ne[1]) % a.ne[2], i3 = row / (a.ne[1] * a.ne[2]);
    const char* sp = s0 + offs(a, 0, i1, i2, i3);
    const i64 n = a.ne[0];
    float mean = 0.f;
    if (!rms) {
        float s = 0.f;
        for (i64 j = threadIdx.x; j < n; j += blockDim.x) s += ld(sp + j * a.nb[0], a.ty);
        mean = block_sum(s) / (float)n;
    }
    float sq = 0.f;
    for (i64 j = threadIdx.x; j < n; j += blockDim.x) {
        const float v = ld(sp + j * a.nb[0], a.ty) - mean;
        sq += v * v;
    }
    sq = block_sum(sq) / (float)n;
    const float scale = 1.0f / sqrtf(sq + eps);
    char* dp = dst + offs(d, 0, i1, i2, i3);
    for (i64 j = threadIdx.x; j < n; j += blockDim.x) {
        st(dp + j * d.nb[0], d.ty, (ld(sp + j * a.nb[0], a.ty) - mean) * scale);
    }
}

// --------------------------------------------------------------------- im2col

K k_im2col(KARGS) {
    const T& inp = op.t[1];
    const T& d = op.t[4];
    const int s0_ = xi(op.x[0]), s1_ = xi(op.x[1]), p0 = xi(op.x[2]), p1 = xi(op.x[3]), d0 = xi(op.x[4]), d1 = xi(op.x[5]);
    const bool is2d = op.x[6] != 0;
    const i64 KW = op.x[7], KH = op.x[8], IH = op.x[9];
    const i64 IW = inp.ne[0];
    const i64 KHW = KW * KH;
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const i64 ow = c.i1, oh = is2d ? c.i2 : 0, n = is2d ? c.i3 : c.i2;
        const i64 ic = c.i0 / KHW, r = c.i0 % KHW, ky = r / KW, kx = r % KW;
        const i64 iy = oh * s1_ + ky * d1 - p1, ix = ow * s0_ + kx * d0 - p0;
        float v = 0.f;
        if (iy >= 0 && iy < IH && ix >= 0 && ix < IW) {
            const i64 off = is2d ? n * inp.nb[3] + ic * inp.nb[2] + iy * inp.nb[1] + ix * inp.nb[0]
                                 : n * inp.nb[2] + ic * inp.nb[1] + ix * inp.nb[0];
            v = ld(s1 + off, inp.ty);
        }
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, v);
    }
}

// --------------------------------------------------------------------- arange

K k_arange(KARGS) {
    const T& d = op.t[4];
    const float start = xf(op.x[0]), step = xf(op.x[1]);
    GRID_LOOP(i, d.ne[0]) st(dst + i * d.nb[0], d.ty, start + (float)i * step);
}

// -------------------------------------------------------------------- upscale

__device__ __forceinline__ float bicubic_w1(float x) {
    const float a = -0.75f;
    return ((a + 2) * x - (a + 3)) * x * x + 1;
}
__device__ __forceinline__ float bicubic_w2(float x) {
    const float a = -0.75f;
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}
__device__ __forceinline__ float bicubic_1d(float p0, float p1, float p2, float p3, float x) {
    return p0 * bicubic_w2(x + 1) + p1 * bicubic_w1(x) + p2 * bicubic_w1(1 - x) + p3 * bicubic_w2(2 - x);
}

K k_upscale(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const float sf0 = xf(op.x[0]), sf1 = xf(op.x[1]), sf2 = xf(op.x[2]), sf3 = xf(op.x[3]);
    const float poff = xf(op.x[4]);
    const int mode = xi(op.x[5]);
    const bool aa = xi(op.x[6]) != 0;
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const i64 i02 = (i64)((float)c.i2 / sf2), i03 = (i64)((float)c.i3 / sf3);
        const char* plane = s0 + i02 * a.nb[2] + i03 * a.nb[3];
        float val;
        if (mode == 0) {
            const i64 i00 = (i64)((float)c.i0 / sf0), i01 = (i64)((float)c.i1 / sf1);
            val = ld(plane + i00 * a.nb[0] + i01 * a.nb[1], a.ty);
        } else if (mode == 1 && aa) {
            const float support1 = fmaxf(1.0f, 1.0f / sf1), invscale1 = 1.0f / support1;
            const float support0 = fmaxf(1.0f, 1.0f / sf0), invscale0 = 1.0f / support0;
            const float y = ((float)c.i1 + poff) / sf1;
            const float x = ((float)c.i0 + poff) / sf0;
            const i64 x_min = max((i64)(x - support0 + poff), (i64)0);
            const i64 x_max = min((i64)(x + support0 + poff), a.ne[0]);
            const i64 y_min = max((i64)(y - support1 + poff), (i64)0);
            const i64 y_max = min((i64)(y + support1 + poff), a.ne[1]);
            float sum = 0.f, total = 0.f;
            for (i64 sy = y_min; sy < y_max; sy++) {
                const float wy = fmaxf(1.0f - fabsf(((float)sy - y + poff) * invscale1), 0.0f);
                for (i64 sx = x_min; sx < x_max; sx++) {
                    const float wx = fmaxf(1.0f - fabsf(((float)sx - x + poff) * invscale0), 0.0f);
                    const float w = wx * wy;
                    if (w <= 0.0f) continue;
                    sum += ld(plane + sx * a.nb[0] + sy * a.nb[1], a.ty) * w;
                    total += w;
                }
            }
            val = total > 0.f ? sum / total : sum;
        } else if (mode == 1) {
            const float y = ((float)c.i1 + poff) / sf1 - poff;
            i64 y0 = (i64)floorf(y), y1 = y0 + 1;
            y0 = max((i64)0, min(y0, a.ne[1] - 1));
            y1 = max((i64)0, min(y1, a.ne[1] - 1));
            const float dy = fmaxf(0.0f, fminf(y - (float)y0, 1.0f));
            const float x = ((float)c.i0 + poff) / sf0 - poff;
            i64 x0 = (i64)floorf(x), x1 = x0 + 1;
            x0 = max((i64)0, min(x0, a.ne[0] - 1));
            x1 = max((i64)0, min(x1, a.ne[0] - 1));
            const float dx = fmaxf(0.0f, fminf(x - (float)x0, 1.0f));
            const float va = ld(plane + x0 * a.nb[0] + y0 * a.nb[1], a.ty);
            const float vb = ld(plane + x1 * a.nb[0] + y0 * a.nb[1], a.ty);
            const float vc = ld(plane + x0 * a.nb[0] + y1 * a.nb[1], a.ty);
            const float vd = ld(plane + x1 * a.nb[0] + y1 * a.nb[1], a.ty);
            val = va * (1 - dx) * (1 - dy) + vb * dx * (1 - dy) + vc * (1 - dx) * dy + vd * dx * dy;
        } else {
            const float y = ((float)c.i1 + poff) / sf1 - poff;
            const i64 y0 = (i64)floorf(y);
            const float dy = y - (float)y0;
            const float x = ((float)c.i0 + poff) / sf0 - poff;
            const i64 x0 = (i64)floorf(x);
            const float dx = x - (float)x0;
            float rows[4];
            for (int r = 0; r < 4; r++) {
                const i64 yy = max((i64)0, min(y0 + r - 1, a.ne[1] - 1));
                float p[4];
                for (int q = 0; q < 4; q++) {
                    const i64 xx = max((i64)0, min(x0 + q - 1, a.ne[0] - 1));
                    p[q] = ld(plane + xx * a.nb[0] + yy * a.nb[1], a.ty);
                }
                rows[r] = bicubic_1d(p[0], p[1], p[2], p[3], dx);
            }
            val = bicubic_1d(rows[0], rows[1], rows[2], rows[3], dy);
        }
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, val);
    }
}

// ----------------------------------------------------------------------- rope

K k_rope(KARGS) {
    const T& a = op.t[0];
    const T& pos = op.t[1];
    const T& d = op.t[4];
    const int n_dims = xi(op.x[0]), mode = xi(op.x[1]);
    const float freq_scale = xf(op.x[2]), ext_factor = xf(op.x[3]), attn_factor = xf(op.x[4]), theta_scale = xf(op.x[5]);
    const float corr0 = xf(op.x[6]), corr1 = xf(op.x[7]);
    const bool has_ff = op.x[8] != 0;
    const i64 half = d.ne[0] / 2;
    GRID_LOOP(i, half * d.ne[1] * d.ne[2] * d.ne[3]) {
        const i64 p = i % half;
        i64 r = i / half;
        const i64 i1 = r % d.ne[1];
        r /= d.ne[1];
        const i64 i2 = r % d.ne[2], i3 = r / d.ne[2];
        const char* sp = s0 + i1 * a.nb[1] + i2 * a.nb[2] + i3 * a.nb[3];
        char* dp = dst + i1 * d.nb[1] + i2 * d.nb[2] + i3 * d.nb[3];
        if (2 * p >= n_dims) {
            cpy_el(dp + 2 * p * d.nb[0], sp + 2 * p * a.nb[0], tysz(d.ty));
            cpy_el(dp + (2 * p + 1) * d.nb[0], sp + (2 * p + 1) * a.nb[0], tysz(d.ty));
            continue;
        }
        const float ff = has_ff ? *(const float*)(s2 + p * 4) : 1.0f;
        const float theta_extrap = (float)(*(const int*)(s1 + i2 * pos.nb[0])) * powf(theta_scale, (float)p) / ff;
        float theta = freq_scale * theta_extrap;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float y = ((float)p - corr0) / fmaxf(0.001f, corr1 - corr0);
            const float ramp_mix = (1.f - fminf(1.f, fmaxf(0.f, y))) * ext_factor;
            theta = freq_scale * theta_extrap * (1.f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float cs = cosf(theta) * mscale, sn = sinf(theta) * mscale;
        const i64 ia = mode == 0 ? 2 * p : p;
        const i64 ib = mode == 0 ? 2 * p + 1 : p + n_dims / 2;
        const float x0 = ld(sp + ia * a.nb[0], a.ty), x1 = ld(sp + ib * a.nb[0], a.ty);
        st(dp + ia * d.nb[0], d.ty, x0 * cs - x1 * sn);
        st(dp + ib * d.nb[0], d.ty, x0 * sn + x1 * cs);
    }
}

// -------------------------------------------------------------------- pool_2d

K k_pool_2d(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const int kind = xi(op.x[0]), k0 = xi(op.x[1]), k1 = xi(op.x[2]), s0_ = xi(op.x[3]), s1_ = xi(op.x[4]);
    const int p0 = xi(op.x[5]), p1 = xi(op.x[6]);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const char* plane = s0 + c.i2 * a.nb[2] + c.i3 * a.nb[3];
        float res = kind == 1 ? 0.f : -FLT_MAX;
        const i64 ix = -p0 + c.i0 * s0_, iy = -p1 + c.i1 * s1_;
        for (int ky = 0; ky < k1; ky++) {
            if (iy + ky < 0 || iy + ky >= a.ne[1]) continue;
            for (int kx = 0; kx < k0; kx++) {
                const i64 j = ix + kx;
                if (j < 0 || j >= a.ne[0]) continue;
                const float v = ld(plane + (iy + ky) * a.nb[1] + j * a.nb[0], a.ty);
                res = kind == 1 ? res + v : fmaxf(v, res);
            }
        }
        if (kind == 1) res /= (float)(k0 * k1);
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, res);
    }
}

// ------------------------------------------------------------------ set_rows

K k_set_rows(KARGS) {
    const T& a = op.t[0];
    const T& idx = op.t[1];
    const T& d = op.t[4];
    GRID_LOOP(i, nelem(a)) {
        const C4 c = decomp(i, a);
        const i64 i12 = c.i3 % idx.ne[2], i11 = c.i2 % idx.ne[1];
        const i64 row = ldi(s1 + c.i1 * idx.nb[0] + i11 * idx.nb[1] + i12 * idx.nb[2], idx.ty);
        st(dst + c.i0 * d.nb[0] + row * d.nb[1] + c.i2 * d.nb[2] + c.i3 * d.nb[3], d.ty,
           ld(s0 + offs(a, c.i0, c.i1, c.i2, c.i3), a.ty));
    }
}

// --------------------------------------------------------------------- argmax

K k_argmax(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const i64 row = blockIdx.x;
    const i64 i1 = row % a.ne[1], i2 = (row / a.ne[1]) % a.ne[2], i3 = row / (a.ne[1] * a.ne[2]);
    const char* sp = s0 + offs(a, 0, i1, i2, i3);
    float best = -INFINITY;
    int bi = 0;
    for (i64 j = threadIdx.x; j < a.ne[0]; j += blockDim.x) {
        const float v = ld(sp + j * a.nb[0], a.ty);
        // ggml's CPU backend keeps the last index among equal maxima.
        if (v >= best) {
            best = v;
            bi = (int)j;
        }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            const float vo = sv[threadIdx.x + s];
            const int io = si[threadIdx.x + s];
            if (vo > sv[threadIdx.x] || (vo == sv[threadIdx.x] && io > si[threadIdx.x])) {
                sv[threadIdx.x] = vo;
                si[threadIdx.x] = io;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *(int*)(dst + row * d.nb[0]) = si[0];
}

// ----------------------------------------------------------------- conv_2d_dw

K k_conv_2d_dw(KARGS) {
    const T& k = op.t[0];
    const T& x = op.t[1];
    const T& d = op.t[4];
    const int s0_ = xi(op.x[0]), s1_ = xi(op.x[1]), p0 = xi(op.x[2]), p1 = xi(op.x[3]), d0 = xi(op.x[4]), d1 = xi(op.x[5]);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const char* xp = s1 + c.i2 * x.nb[2] + c.i3 * x.nb[3];
        const char* kp = s0 + c.i2 * k.nb[3];
        float sum = 0.f;
        for (i64 ky = 0; ky < k.ne[1]; ky++) {
            const i64 iy = c.i1 * s1_ + ky * d1 - p1;
            if (iy < 0 || iy >= x.ne[1]) continue;
            for (i64 kx = 0; kx < k.ne[0]; kx++) {
                const i64 ix = c.i0 * s0_ + kx * d0 - p0;
                if (ix < 0 || ix >= x.ne[0]) continue;
                sum += ld(xp + iy * x.nb[1] + ix * x.nb[0], x.ty) * ld(kp + ky * k.nb[1] + kx * k.nb[0], k.ty);
            }
        }
        st(dst + offs(d, c.i0, c.i1, c.i2, c.i3), d.ty, sum);
    }
}

// ------------------------------------------------------------------------ pad

K k_pad(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const int sz = tysz(d.ty);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        const i64 j0 = c.i0 - op.x[0], j1 = c.i1 - op.x[1], j2 = c.i2 - op.x[2], j3 = c.i3 - op.x[3];
        char* dp = dst + offs(d, c.i0, c.i1, c.i2, c.i3);
        if (j0 >= 0 && j0 < a.ne[0] && j1 >= 0 && j1 < a.ne[1] && j2 >= 0 && j2 < a.ne[2] && j3 >= 0 && j3 < a.ne[3]) {
            cpy_el(dp, s0 + offs(a, j0, j1, j2, j3), sz);
        } else {
            st(dp, d.ty, 0.f);
        }
    }
}

K k_pad_reflect_1d(KARGS) {
    const T& a = op.t[0];
    const T& d = op.t[4];
    const i64 p0 = op.x[0];
    const int sz = tysz(d.ty);
    GRID_LOOP(i, nelem(d)) {
        const C4 c = decomp(i, d);
        i64 j = c.i0 - p0;
        if (j < 0) j = -j;
        if (j >= a.ne[0]) j = 2 * (a.ne[0] - 1) - j;
        cpy_el(dst + offs(d, c.i0, c.i1, c.i2, c.i3), s0 + offs(a, j, c.i1, c.i2, c.i3), sz);
    }
}
