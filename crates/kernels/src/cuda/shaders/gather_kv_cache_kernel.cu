#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

namespace vllm {

/// Gather K and V from paged KV cache into contiguous output tensors.
///
/// One CUDA block per output token, 256 threads cooperatively copy
/// kv_heads * head_size elements for both K and V.
///
/// Uses binary search on cu_seq_lens to find batch_id, avoiding a
/// separate token_to_seq tensor.
///
/// K cache layout: [num_blocks, kv_heads, head_size/x, block_size, x]
/// V cache layout: [num_blocks, kv_heads, head_size, block_size]
/// K/V output:     [num_tokens, kv_heads, head_size]
template <typename cache_t, typename out_t, Fp8KVCacheDataType kv_dt>
__global__ void gather_kv_cache_kernel(
    const cache_t *__restrict__ key_cache,   // [num_blocks, kv_heads,
                                             //  head_size/x, block_size, x]
    const cache_t *__restrict__ value_cache, // [num_blocks, kv_heads,
                                             //  head_size, block_size]
    out_t *__restrict__ k_out,         // [num_tokens, kv_heads, head_size]
    out_t *__restrict__ v_out,         // [num_tokens, kv_heads, head_size]
    const float *__restrict__ k_scale, // scalar or nullptr
    const float *__restrict__ v_scale, // scalar or nullptr
    const int32_t *__restrict__ block_table, // [batch, max_blocks]
    const int32_t *__restrict__ cu_seq_lens, // [batch + 1]
    const int32_t num_tokens, const int32_t num_seqs, const int32_t block_size,
    const int32_t block_table_stride, const int32_t num_kv_heads,
    const int32_t head_size, const int32_t x) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  // Binary search cu_seq_lens to find batch_id.
  // cu_seq_lens is [batch+1] with cumulative token counts.
  // We want the largest i such that cu_seq_lens[i] <= token_id.
  int32_t lo = 0, hi = num_seqs;
  while (lo < hi) {
    int32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int32_t batch_id = lo;
  if (batch_id >= num_seqs) {
    return;
  }

  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int32_t n = num_kv_heads * head_size;
  const int64_t out_base =
      static_cast<int64_t>(token_id) * num_kv_heads * head_size;

  // Precompute strides
  const int64_t k_block_stride =
      static_cast<int64_t>(num_kv_heads) * (head_size / x) * block_size * x;
  const int64_t k_head_stride =
      static_cast<int64_t>(head_size / x) * block_size * x;
  const int64_t v_block_stride =
      static_cast<int64_t>(num_kv_heads) * head_size * block_size;
  const int64_t v_head_stride = static_cast<int64_t>(head_size) * block_size;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int d = i % head_size;

    // K: [block_id, head_idx, d/x, slot, d%x]
    const int x_idx = d / x;
    const int x_offset = d % x;
    const int64_t k_src_idx = static_cast<int64_t>(block_id) * k_block_stride +
                              head_idx * k_head_stride +
                              x_idx * block_size * x + slot * x + x_offset;

    // V: [block_id, head_idx, d, slot]
    const int64_t v_src_idx = static_cast<int64_t>(block_id) * v_block_stride +
                              head_idx * v_head_stride + d * block_size + slot;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      k_out[out_base + i] = key_cache[k_src_idx];
      v_out[out_base + i] = value_cache[v_src_idx];
    } else {
      k_out[out_base + i] = fp8::scaled_convert<out_t, cache_t, kv_dt>(
          key_cache[k_src_idx], *k_scale);
      v_out[out_base + i] = fp8::scaled_convert<out_t, cache_t, kv_dt>(
          value_cache[v_src_idx], *v_scale);
    }
  }
}

} // namespace vllm
