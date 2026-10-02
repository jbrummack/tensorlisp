#include "metal_interop.h"

#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

static ggml_backend_dev_t metal_dev(void) {
    return ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
}

void * tl_ggml_metal_device(void) {
    return ggml_metal_device_get_obj((ggml_metal_device_t) metal_dev()->context);
}

void * tl_ggml_metal_queue(void) {
    return ggml_metal_device_get_queue((ggml_metal_device_t) metal_dev()->context);
}

void * tl_ggml_metal_tensor_buffer(const struct ggml_tensor * t, size_t * offset) {
    if (!t) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buffer || ggml_backend_buft_get_device(buffer->buft) != metal_dev()) {
        return nullptr;
    }
    ggml_metal_buffer_id id = ggml_metal_buffer_get_id((ggml_metal_buffer_t) buffer->context, t);
    if (offset) {
        *offset = id.offs;
    }
    return id.metal;
}
