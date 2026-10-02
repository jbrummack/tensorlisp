// Flash attention of the native CUDA executor, specialized per K/V type with
// `#define KTY` and `#define VTY` (ggml type numbers) ahead of this file.
//
// t[0] = q (f32 [D, nq, nh, nb]), t[1] = k [D, nkv, nhkv, nbk], t[2] = v
// [Dv, nkv, nhkv, nbk], t[3] = mask (f16, optional: s3 == null), t[4] = dst
// [Dv, nh, nq, nb]. x: 0 scale, 1 max_bias, 2 m0, 3 m1 (floats), 4 n_head_log2,
// 5 logit_softcap (float; scale was already divided by it).
//
// One block per (query, head, batch): keys are processed in chunks of FA_KC
// with an online softmax, so no scratch memory is needed. Each warp computes
// the scores of its keys (lanes stride over the head dimension), then all
// threads accumulate the weighted V rows (one output dimension per thread).

#define KARGS const Op op, const char* s0, const char* s1, const char* s2, const char* s3, char* dst
#define K extern "C" __global__ void

#define FA_THREADS 128
#define FA_KC 128
#define FA_ACC 4

K fa_main(KARGS) {
    const T& q = op.t[0];
    const T& k = op.t[1];
    const T& v = op.t[2];
    const T& m = op.t[3];
    const T& d = op.t[4];
    extern __shared__ float smem[];
    float* qs = smem;               // D floats
    float* sc = smem + q.ne[0];     // FA_KC scores / probabilities

    const i64 iq = blockIdx.x, h = blockIdx.y, ib = blockIdx.z;
    const i64 D = q.ne[0], Dv = v.ne[0], nkv = k.ne[1];
    const i64 hk = h / (q.ne[2] / k.ne[2]);
    const i64 bk = ib / (q.ne[3] / k.ne[3]);
    const float scale = xf(op.x[0]), max_bias = xf(op.x[1]), m0 = xf(op.x[2]), m1 = xf(op.x[3]);
    const int nhl2 = xi(op.x[4]);
    const float softcap = xf(op.x[5]);

    float slope = 1.f;
    if (max_bias > 0.f) {
        slope = powf(h < nhl2 ? m0 : m1, (float)(h < nhl2 ? h + 1 : 2 * (h - nhl2) + 1));
    }

    const char* qp = s0 + iq * q.nb[1] + h * q.nb[2] + ib * q.nb[3];
    for (i64 i = threadIdx.x; i < D; i += blockDim.x) qs[i] = *(const float*)(qp + i * q.nb[0]);
    const char* kb = s1 + hk * k.nb[2] + bk * k.nb[3];
    const char* vb = s2 + hk * v.nb[2] + bk * v.nb[3];
    const char* mp = s3 ? s3 + iq * m.nb[1] + (h % m.ne[2]) * m.nb[2] + (ib % m.ne[3]) * m.nb[3] : nullptr;
    __syncthreads();

    float acc[FA_ACC];
#pragma unroll
    for (int i = 0; i < FA_ACC; i++) acc[i] = 0.f;
    float mrun = -INFINITY, lrun = 0.f;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    for (i64 j0 = 0; j0 < nkv; j0 += FA_KC) {
        const i64 cn = min((i64)FA_KC, nkv - j0);
        for (i64 jj = warp; jj < cn; jj += nw) {
            const i64 j = j0 + jj;
            const char* kr = kb + j * k.nb[1];
            float s = 0.f;
            for (i64 e = lane; e < D; e += 32) s += qs[e] * dqf<KTY>(kr, e, k.nb[0]);
            s = warp_sum(s) * scale;
            if (softcap != 0.f) s = softcap * tanhf(s);
            if (mp) s += slope * __half2float(*(const __half*)(mp + j * m.nb[0]));
            if (lane == 0) sc[jj] = s;
        }
        __syncthreads();
        float cm = -INFINITY;
        for (i64 jj = threadIdx.x; jj < cn; jj += blockDim.x) cm = fmaxf(cm, sc[jj]);
        cm = block_max(cm);
        const float mnew = fmaxf(mrun, cm);
        const float corr = mnew == -INFINITY ? 1.f : expf(mrun - mnew);
        float ps = 0.f;
        for (i64 jj = threadIdx.x; jj < cn; jj += blockDim.x) {
            const float p = mnew == -INFINITY ? 0.f : expf(sc[jj] - mnew);
            sc[jj] = p;
            ps += p;
        }
        ps = block_sum(ps);
        lrun = lrun * corr + ps;
        mrun = mnew;
        // sc[] must be complete before anyone reads it below.
        __syncthreads();
#pragma unroll
        for (int i = 0; i < FA_ACC; i++) {
            const i64 e = threadIdx.x + (i64)i * FA_THREADS;
            if (e < Dv) {
                float a = acc[i] * corr;
                for (i64 jj = 0; jj < cn; jj++) a += sc[jj] * dqf<VTY>(vb + (j0 + jj) * v.nb[1], e, v.nb[0]);
                acc[i] = a;
            }
        }
        __syncthreads();
    }
    const float inv = lrun > 0.f ? 1.f / lrun : 0.f;
    char* dp = dst + h * d.nb[1] + iq * d.nb[2] + ib * d.nb[3];
#pragma unroll
    for (int i = 0; i < FA_ACC; i++) {
        const i64 e = threadIdx.x + (i64)i * FA_THREADS;
        if (e < Dv) *(float*)(dp + e * d.nb[0]) = acc[i] * inv;
    }
}

// Split-KV decoding (one query): the same online softmax over a slice of the
// keys per block, so that nh * nb blocks become nh * nb * P and every SM has
// work, then fa_reduce merges the slices (the structure of paged attention's
// v2 kernel, reading ggml's [D, rows] caches directly). Slices whose mask is
// all -inf (unused cache rows) exit at once.
//
// fa_split: as fa_main plus x6 = keys per slice, x7 = slices P; grid
// (P, nh, nb); dst = partials [nb][nh][P][Dv + 2] floats (max, sum, acc[Dv]).
K fa_split(KARGS) {
    const T& q = op.t[0];
    const T& k = op.t[1];
    const T& v = op.t[2];
    const T& m = op.t[3];
    extern __shared__ float smem[];
    float* qs = smem;
    float* sc = smem + q.ne[0];

    const i64 p = blockIdx.x, h = blockIdx.y, ib = blockIdx.z;
    const i64 D = q.ne[0], Dv = v.ne[0], nh = q.ne[2], nkv = k.ne[1];
    const i64 part_len = op.x[6], P = op.x[7];
    const i64 jb = p * part_len, je = min(nkv, jb + part_len);
    const i64 hk = h / (q.ne[2] / k.ne[2]);
    const i64 bk = ib / (q.ne[3] / k.ne[3]);
    const float scale = xf(op.x[0]), max_bias = xf(op.x[1]), m0 = xf(op.x[2]), m1 = xf(op.x[3]);
    const int nhl2 = xi(op.x[4]);
    const float softcap = xf(op.x[5]);
    float* out = (float*)dst + ((ib * nh + h) * P + p) * (Dv + 2);

    const char* mp = s3 ? s3 + (h % m.ne[2]) * m.nb[2] + (ib % m.ne[3]) * m.nb[3] : nullptr;
    if (mp) {
        int live = 0;
        for (i64 j = jb + threadIdx.x; j < je; j += blockDim.x) live |= *(const unsigned short*)(mp + j * m.nb[0]) != 0xFC00;
        if (!__syncthreads_or(live)) {
            if (threadIdx.x == 0) { out[0] = -INFINITY; out[1] = 0.f; }
            for (i64 e = threadIdx.x; e < Dv; e += blockDim.x) out[2 + e] = 0.f;
            return;
        }
    }

    float slope = 1.f;
    if (max_bias > 0.f) {
        slope = powf(h < nhl2 ? m0 : m1, (float)(h < nhl2 ? h + 1 : 2 * (h - nhl2) + 1));
    }
    const char* qp = s0 + h * q.nb[2] + ib * q.nb[3];
    for (i64 i = threadIdx.x; i < D; i += blockDim.x) qs[i] = *(const float*)(qp + i * q.nb[0]);
    const char* kb = s1 + hk * k.nb[2] + bk * k.nb[3];
    const char* vb = s2 + hk * v.nb[2] + bk * v.nb[3];
    __syncthreads();

    float acc[FA_ACC];
#pragma unroll
    for (int i = 0; i < FA_ACC; i++) acc[i] = 0.f;
    float mrun = -INFINITY, lrun = 0.f;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    for (i64 j0 = jb; j0 < je; j0 += FA_KC) {
        const i64 cn = min((i64)FA_KC, je - j0);
        for (i64 jj = warp; jj < cn; jj += nw) {
            const i64 j = j0 + jj;
            const char* kr = kb + j * k.nb[1];
            float s = 0.f;
            for (i64 e = lane; e < D; e += 32) s += qs[e] * dqf<KTY>(kr, e, k.nb[0]);
            s = warp_sum(s) * scale;
            if (softcap != 0.f) s = softcap * tanhf(s);
            if (mp) s += slope * __half2float(*(const __half*)(mp + j * m.nb[0]));
            if (lane == 0) sc[jj] = s;
        }
        __syncthreads();
        float cm = -INFINITY;
        for (i64 jj = threadIdx.x; jj < cn; jj += blockDim.x) cm = fmaxf(cm, sc[jj]);
        cm = block_max(cm);
        const float mnew = fmaxf(mrun, cm);
        const float corr = mnew == -INFINITY ? 1.f : expf(mrun - mnew);
        float ps = 0.f;
        for (i64 jj = threadIdx.x; jj < cn; jj += blockDim.x) {
            const float pr = mnew == -INFINITY ? 0.f : expf(sc[jj] - mnew);
            sc[jj] = pr;
            ps += pr;
        }
        ps = block_sum(ps);
        lrun = lrun * corr + ps;
        mrun = mnew;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < FA_ACC; i++) {
            const i64 e = threadIdx.x + (i64)i * FA_THREADS;
            if (e < Dv) {
                float a = acc[i] * corr;
                for (i64 jj = 0; jj < cn; jj++) a += sc[jj] * dqf<VTY>(vb + (j0 + jj) * v.nb[1], e, v.nb[0]);
                acc[i] = a;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[0] = mrun; out[1] = lrun; }
#pragma unroll
    for (int i = 0; i < FA_ACC; i++) {
        const i64 e = threadIdx.x + (i64)i * FA_THREADS;
        if (e < Dv) out[2 + e] = acc[i];
    }
}

// fa_reduce: s0 = partials, t[4] = dst [Dv, nh, 1, nb], x7 = P; grid (nh, 1, nb).
K fa_reduce(KARGS) {
    const T& d = op.t[4];
    const i64 h = blockIdx.x, ib = blockIdx.z;
    const i64 Dv = d.ne[0], nh = d.ne[1], P = op.x[7];
    const float* base = (const float*)s0 + (ib * nh + h) * P * (Dv + 2);
    float M = -INFINITY;
    for (i64 p = 0; p < P; p++) M = fmaxf(M, base[p * (Dv + 2)]);
    float L = 0.f;
    for (i64 p = 0; p < P; p++) {
        const float mp = base[p * (Dv + 2)];
        if (mp != -INFINITY) L += base[p * (Dv + 2) + 1] * expf(mp - M);
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    char* dp = dst + h * d.nb[1] + ib * d.nb[3];
    for (i64 e = threadIdx.x; e < Dv; e += blockDim.x) {
        float a = 0.f;
        for (i64 p = 0; p < P; p++) {
            const float mp = base[p * (Dv + 2)];
            if (mp != -INFINITY) a += base[p * (Dv + 2) + 2 + e] * expf(mp - M);
        }
        *(float*)(dp + e * d.nb[0]) = a * inv;
    }
}
