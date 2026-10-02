// Matrix multiplication of the native CUDA executor, specialized per source
// type with `#define MM_TY <ggml type number>` ahead of this file.
//
//   dst[m, n, i2, i3] = sum_k src0[k, m, i2 / r2, i3 / r3] * src1[k, n, i2, i3]
//
// op.t[0] = src0 (weights, any type), t[1] = src1 (f32 activations), t[4] = dst.
// x[0] = r2 (dst.ne[2] / src0.ne[2]), x[1] = r3.

#define KARGS const Op op, const char* s0, const char* s1, const char* s2, const char* s3, char* dst
#define K extern "C" __global__ void

// Few columns (decode): one warp per output row, lanes stride over K.
#define MV_ROWS 4
#define MV_COLS 8

K mm_vec(KARGS) {
    const T& a = op.t[0];
    const T& b = op.t[1];
    const T& d = op.t[4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const i64 m = (i64)blockIdx.x * MV_ROWS + warp;
    const i64 i2 = blockIdx.y, i3 = blockIdx.z;
    if (m >= d.ne[0]) return;
    const i64 K_ = a.ne[0], N = d.ne[1];
    const char* row = s0 + m * a.nb[1] + (i2 / op.x[0]) * a.nb[2] + (i3 / op.x[1]) * a.nb[3];
    const char* col = s1 + i2 * b.nb[2] + i3 * b.nb[3];
    float acc[MV_COLS];
#pragma unroll
    for (int j = 0; j < MV_COLS; j++) acc[j] = 0.f;
    for (i64 k = lane; k < K_; k += 32) {
        const float w = dqf<MM_TY>(row, k, a.nb[0]);
#pragma unroll
        for (int j = 0; j < MV_COLS; j++) {
            if (j < N) acc[j] += w * ld(col + k * b.nb[0] + j * b.nb[1], b.ty);
        }
    }
#pragma unroll
    for (int j = 0; j < MV_COLS; j++) {
        const float v = warp_sum(acc[j]);
        if (lane == 0 && j < N) st(dst + m * d.nb[0] + j * d.nb[1] + i2 * d.nb[2] + i3 * d.nb[3], d.ty, v);
    }
}

// Many columns (prefill, vision): 64x64 output tile per block, 16 deep in K,
// 256 threads each computing a 4x4 sub-tile.
#define BM 64
#define BN 64
#define BK 16

K mm_tile(KARGS) {
    const T& a = op.t[0];
    const T& b = op.t[1];
    const T& d = op.t[4];
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    const i64 i2 = blockIdx.z % d.ne[2], i3 = blockIdx.z / d.ne[2];
    const i64 m0 = (i64)blockIdx.x * BM, n0 = (i64)blockIdx.y * BN;
    const i64 K_ = a.ne[0], M = d.ne[0], N = d.ne[1];
    const char* abase = s0 + (i2 / op.x[0]) * a.nb[2] + (i3 / op.x[1]) * a.nb[3];
    const char* bbase = s1 + i2 * b.nb[2] + i3 * b.nb[3];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (i64 k0 = 0; k0 < K_; k0 += BK) {
#pragma unroll
        for (int r = 0; r < (BM * BK) / 256; r++) {
            const int idx = threadIdx.x + 256 * r;
            const int kk = idx % BK, mm = idx / BK;
            const i64 m = m0 + mm, k = k0 + kk;
            As[kk][mm] = (m < M && k < K_) ? dqf<MM_TY>(abase + m * a.nb[1], k, a.nb[0]) : 0.f;
            const i64 n = n0 + mm;
            Bs[kk][mm] = (n < N && k < K_) ? ld(bbase + k * b.nb[0] + n * b.nb[1], b.ty) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float av[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; i++) av[i] = As[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) bv[j] = Bs[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const i64 m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const i64 n = n0 + tx * 4 + j;
            if (n < N) st(dst + m * d.nb[0] + n * d.nb[1] + i2 * d.nb[2] + i3 * d.nb[3], d.ty, acc[i][j]);
        }
    }
}
