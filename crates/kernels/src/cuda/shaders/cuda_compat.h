#pragma once

#include <cstdint>

// tensorlisp: trimmed to the device-code macros. Upstream also defines
// VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize (a host function using
// cudaFuncGetAttributes/cudaFuncSetAttribute); that's the Rust host launcher's
// job here too (crate::cuda::Device::launch sets CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
// directly, like crate::metal::paged_attn's shared-memory sizing), and it's
// unavailable to NVRTC-compiled device code anyway (no <cuda_runtime.h>).

#define VLLM_LDG(arg) __ldg(arg)

#define VLLM_SHFL_XOR_SYNC(var, lane_mask)                                     \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)

#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
