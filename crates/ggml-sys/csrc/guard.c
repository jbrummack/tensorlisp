#include "guard.h"

#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include "ggml-impl.h"

static _Thread_local jmp_buf * tl_jump;
static _Thread_local char tl_message[2048];
static _Thread_local tl_abort_handler tl_handler;

/* ggml prefixes messages with __FILE__, an absolute build path; keep the file name. */
static const char * short_message(const char * message) {
    const char * colon = strchr(message, ':');
    const char * start = message;
    for (const char * p = message; colon && p < colon; p++) {
        if (*p == '/' || *p == '\\') {
            start = p + 1;
        }
    }
    return start;
}

static void on_abort(const char * message) {
    message = short_message(message);
    if (tl_jump) {
        snprintf(tl_message, sizeof tl_message, "%s", message);
        longjmp(*tl_jump, 1);
    }
    if (tl_handler) {
        tl_handler(message); /* does not return */
    }
    fprintf(stderr, "tensorlisp: unrecoverable ggml assertion (outside a guarded call, "
                    "e.g. on a worker thread): %s\n", message);
    ggml_print_backtrace();
    /* ggml_abort calls abort() when this returns. */
}

void tl_guard_install(void) {
    ggml_set_abort_callback(on_abort);
}

void tl_guard_set_thread_handler(tl_abort_handler handler) {
    tl_handler = handler;
}

/* Runs body with tl_jump pointing at a fresh jmp_buf; on an assertion inside
 * body, restores the previous jump target and reports the message. */
#define TL_TRY(body)                                          \
    jmp_buf buf;                                              \
    jmp_buf * prev = tl_jump;                                 \
    if (setjmp(buf)) {                                        \
        tl_jump = prev;                                       \
        snprintf(err, err_len, "%s", tl_message);             \
        return false;                                         \
    }                                                         \
    tl_jump = &buf;                                           \
    body;                                                     \
    tl_jump = prev;                                           \
    return true;

bool tl_try_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend,
                                      ggml_backend_buffer_t * buffer, char * err, size_t err_len) {
    TL_TRY(*buffer = ggml_backend_alloc_ctx_tensors(ctx, backend))
}

bool tl_try_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset,
                               size_t size, char * err, size_t err_len) {
    TL_TRY(ggml_backend_tensor_set(tensor, data, offset, size))
}

bool tl_try_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset,
                               size_t size, char * err, size_t err_len) {
    TL_TRY(ggml_backend_tensor_get(tensor, data, offset, size))
}

bool tl_try_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool * ok,
                              char * err, size_t err_len) {
    TL_TRY(*ok = ggml_backend_sched_alloc_graph(sched, graph))
}

bool tl_try_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph,
                                enum ggml_status * status, char * err, size_t err_len) {
    TL_TRY(*status = ggml_backend_sched_graph_compute(sched, graph))
}
