#pragma once
// Raw Metal handles behind ggml's Metal backend, for launching other Metal
// kernels on the same device/queue against ggml tensor storage.
#include <stddef.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// id<MTLDevice> of ggml's first Metal device (not retained).
void * tl_ggml_metal_device(void);
// The command queue ggml submits to; serial, so command buffers committed by
// others on it run in commit order relative to ggml's.
void * tl_ggml_metal_queue(void);
// id<MTLBuffer> backing `t` (views resolve to their source) and the byte
// offset of its data inside it; NULL if the tensor is not in a Metal buffer.
void * tl_ggml_metal_tensor_buffer(const struct ggml_tensor * t, size_t * offset);

#ifdef __cplusplus
}
#endif
