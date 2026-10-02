#include <stdint.h>

// tensorlisp: not vendored from vLLM/mistral.rs. Upstream's copy_blocks_kernel
// copies blocks across every layer's cache in one launch (`key_cache_ptrs` is
// an array of per-layer device pointers); the Metal port
// (../metal/shaders/copy_blocks.metal) already simplified this to one
// key_cache/value_cache pair per launch, so this mirrors that for a
// cross-backend `CopyBlocks` API instead of the multi-layer original.
template <typename T>
__global__ void copy_blocks_kernel(T *key_cache, T *value_cache, const int64_t *__restrict__ block_mapping,
                                    const int numel_per_block_key, const int numel_per_block_value) {
  const int pair_idx = blockIdx.x;
  const int64_t src_block_number = block_mapping[2 * pair_idx];
  const int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset_key = src_block_number * numel_per_block_key;
  const int64_t dst_block_offset_key = dst_block_number * numel_per_block_key;
  for (int i = threadIdx.x; i < numel_per_block_key; i += blockDim.x) {
    key_cache[dst_block_offset_key + i] = key_cache[src_block_offset_key + i];
  }

  const int64_t src_block_offset_value = src_block_number * numel_per_block_value;
  const int64_t dst_block_offset_value = dst_block_number * numel_per_block_value;
  for (int i = threadIdx.x; i < numel_per_block_value; i += blockDim.x) {
    value_cache[dst_block_offset_value + i] = value_cache[src_block_offset_value + i];
  }
}
