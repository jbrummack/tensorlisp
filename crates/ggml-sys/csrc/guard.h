/* Recovering from GGML_ASSERT / GGML_ABORT instead of aborting the process.
 *
 * tl_guard_install() registers an abort callback that, on the calling thread:
 *  - longjmps back into the innermost tl_try_* call, if there is one, or
 *  - calls the thread's handler (set with tl_guard_set_thread_handler), which
 *    must not return; the Scheme thread uses it to raise a Scheme error, and
 *    Chez unwinds its own and ggml's frames.
 * Otherwise (e.g. on a ggml worker thread) it prints the message and ggml
 * aborts as usual.
 *
 * The tl_try_* wrappers return false and write the assertion message to err
 * when an assertion fired. ggml frames skipped by longjmp are not cleaned up,
 * so state touched by the failed call must be considered broken. */
#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

typedef void (*tl_abort_handler)(const char * message);

void tl_guard_install(void);
void tl_guard_set_thread_handler(tl_abort_handler handler);

bool tl_try_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend,
                                      ggml_backend_buffer_t * buffer, char * err, size_t err_len);
bool tl_try_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset,
                               size_t size, char * err, size_t err_len);
bool tl_try_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset,
                               size_t size, char * err, size_t err_len);
bool tl_try_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool * ok,
                              char * err, size_t err_len);
bool tl_try_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph,
                                enum ggml_status * status, char * err, size_t err_len);
