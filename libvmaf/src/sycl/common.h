/**
 *
 *  Copyright 2016-2024 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef __VMAF_SRC_SYCL_COMMON_H__
#define __VMAF_SRC_SYCL_COMMON_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "config.h"
#include "picture.h"

#if HAVE_SYCL

#include <libvmaf/libvmaf_sycl.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * VmafSyclState internals — opaque to C callers, defined in common.cpp.
 *
 * The state owns:
 *   - A sycl::queue (in-order, optionally with profiling)
 *   - Shared frame buffers (USM device allocations for ref/dis Y planes)
 *   - Profiling accumulators
 */

/* ---- Device-memory helpers (USM wrappers) ---- */

/**
 * Allocate USM device memory on the SYCL device.
 *
 * @param state  The SYCL state.
 * @param size   Number of bytes to allocate.
 *
 * @return Non-NULL pointer to device memory, or NULL on failure.
 */
void *vmaf_sycl_malloc_device(VmafSyclState *state, size_t size);

/**
 * Allocate USM host memory (host-accessible, device-accessible on iGPU).
 *
 * @param state  The SYCL state.
 * @param size   Number of bytes to allocate.
 *
 * @return Non-NULL pointer to host memory, or NULL on failure.
 */
void *vmaf_sycl_malloc_host(VmafSyclState *state, size_t size);

/**
 * Free USM memory (device or host).
 *
 * @param state  The SYCL state.
 * @param ptr    Pointer returned by vmaf_sycl_malloc_device/host.
 */
void vmaf_sycl_free(VmafSyclState *state, void *ptr);

/**
 * Synchronous host→device copy.
 *
 * @param state  The SYCL state.
 * @param dst    Device pointer (destination).
 * @param src    Host pointer (source).
 * @param size   Number of bytes to copy.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_memcpy_h2d(VmafSyclState *state, void *dst, const void *src,
                          size_t size);

/**
 * Synchronous device→host copy.
 *
 * @param state  The SYCL state.
 * @param dst    Host pointer (destination).
 * @param src    Device pointer (source).
 * @param size   Number of bytes to copy.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_memcpy_d2h(VmafSyclState *state, void *dst, const void *src,
                          size_t size);

/**
 * Asynchronous host→device copy (returns immediately, work queued).
 *
 * @param state  The SYCL state.
 * @param dst    Device pointer (destination).
 * @param src    Host pointer (source).
 * @param size   Number of bytes to copy.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_memcpy_h2d_async(VmafSyclState *state, void *dst,
                                const void *src, size_t size);

/* ---- Queue synchronization ---- */

/**
 * Wait for all enqueued SYCL work to complete (primary queue).
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_queue_wait(VmafSyclState *state);

/**
 * Wait for all pending copy/upload operations on the copy queue.
 * Must be called after vmaf_sycl_shared_frame_upload() and before
 * extractor compute queues start work.
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_wait_copy_queue(VmafSyclState *state);

/**
 * Wait for just the last shared-frame upload event to complete.
 * Lighter than wait_copy_queue — only waits on the DMA event,
 * not the entire copy queue.
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_wait_last_upload(VmafSyclState *state);

/* ---- Per-extractor compute queue management ---- */

/**
 * Create a new in-order compute queue on the same device and context.
 * Each extractor should create its own queue to enable GPU-level
 * parallelism between extractors. Respects the profiling flag from
 * the original SYCL state.
 *
 * @param state  The SYCL state (provides device, context, profiling flag).
 *
 * @return Opaque pointer to sycl::queue (cast in C++ code), or NULL.
 *         Caller owns the queue and must free with vmaf_sycl_destroy_queue().
 */
void *vmaf_sycl_create_compute_queue(VmafSyclState *state);

/**
 * Destroy a queue created by vmaf_sycl_create_compute_queue().
 * Waits for any pending work before destruction.
 *
 * @param queue_ptr  Opaque pointer returned by vmaf_sycl_create_compute_queue().
 */
void vmaf_sycl_destroy_queue(void *queue_ptr);

/* ---- Shared frame buffer management ---- */

/**
 * Allocate shared ref+dis Y-plane device buffers.
 *
 * @param state  The SYCL state.
 * @param w      Frame width in pixels.
 * @param h      Frame height in pixels.
 * @param bpc    Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_shared_frame_init(VmafSyclState *state,
                                 unsigned w, unsigned h, unsigned bpc);

/**
 * Get pointers to the shared ref+dis device buffers.
 *
 * @param state    The SYCL state.
 * @param[out] ref Receives pointer to ref Y-plane device buffer.
 * @param[out] dis Receives pointer to dis Y-plane device buffer.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_shared_frame_get(VmafSyclState *state,
                                void **ref, void **dis);

/**
 * Upload host Y-plane data to the shared device buffers.
 * Used by the vmaf_read_pictures() path (non-zero-copy).
 *
 * @param state  The SYCL state.
 * @param ref    Reference picture (host buffer).
 * @param dis    Distorted picture (host buffer).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_shared_frame_upload(VmafSyclState *state,
                                   VmafPicture *ref, VmafPicture *dis);

/**
 * Upload a single Y-plane from a host buffer into a shared device buffer.
 * This is the platform-agnostic surface import path: the caller provides
 * a pointer to linear Y-plane pixels (e.g. from VPL MapFrame)
 * and this function performs an H2D copy into the appropriate shared buffer.
 *
 * @param state   The SYCL state (shared frame buffers must be initialised).
 * @param src     Host pointer to Y-plane pixel data (linear layout).
 * @param pitch   Row stride in bytes (may exceed w * bytes_per_pixel).
 * @param is_ref  If non-zero, upload to ref buffer; else dis buffer.
 * @param w       Frame width in pixels.
 * @param h       Frame height in pixels.
 * @param bpc     Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_upload_plane(VmafSyclState *state,
                            const void *src, unsigned pitch,
                            int is_ref,
                            unsigned w, unsigned h, unsigned bpc);

/**
 * Free the shared frame buffers.
 *
 * @param state  The SYCL state.
 */
void vmaf_sycl_shared_frame_close(VmafSyclState *state);

/* ---- Opaque queue handle for extractor kernels ---- */

/**
 * Get the underlying sycl::queue pointer for use in DPC++ kernel code.
 * The returned pointer is only valid while the state is alive.
 * Cast to sycl::queue* in C++ code.
 *
 * @param state  The SYCL state.
 *
 * @return Opaque pointer to sycl::queue (cast in C++ code), or NULL.
 */
void *vmaf_sycl_get_queue_ptr(VmafSyclState *state);

/**
 * Check if the SYCL device supports double precision (fp64).
 *
 * @param state  The SYCL state.
 *
 * @return true if fp64 is supported, false otherwise.
 */
bool vmaf_sycl_has_fp64(VmafSyclState *state);

/**
 * Get the shared ref device buffer pointer.
 *
 * @param state  The SYCL state.
 *
 * @return Pointer to ref Y-plane device buffer, or NULL if not initialised.
 */
void *vmaf_sycl_get_shared_ref(VmafSyclState *state);

/**
 * Get the shared dis device buffer pointer.
 *
 * @param state  The SYCL state.
 *
 * @return Pointer to dis Y-plane device buffer, or NULL if not initialised.
 */
void *vmaf_sycl_get_shared_dis(VmafSyclState *state);

/**
 * Get the shared ref device buffer pointer for a specific double-buffer slot.
 *
 * @param state  The SYCL state.
 * @param slot   Buffer slot index (0 or 1).
 *
 * @return Pointer to ref Y-plane device buffer, or NULL.
 */
void *vmaf_sycl_get_shared_ref_slot(VmafSyclState *state, int slot);

/**
 * Get the shared dis device buffer pointer for a specific double-buffer slot.
 *
 * @param state  The SYCL state.
 * @param slot   Buffer slot index (0 or 1).
 *
 * @return Pointer to dis Y-plane device buffer, or NULL.
 */
void *vmaf_sycl_get_shared_dis_slot(VmafSyclState *state, int slot);

/**
 * Get the current compute slot index (0 or 1).
 *
 * @param state  The SYCL state.
 *
 * @return Current compute slot index.
 */
int vmaf_sycl_get_compute_slot(VmafSyclState *state);

/**
 * Get a pointer to the sycl::event from the last shared-frame upload.
 * Extractors can depend on this event to avoid a CPU-side wait for DMA.
 *
 * @param state  The SYCL state.
 *
 * @return Opaque pointer to sycl::event, or NULL if not initialised.
 */
void *vmaf_sycl_get_last_upload_event(VmafSyclState *state);

/* ---- Combined command graph (merges all extractors into one replay) ---- */

/**
 * Callback types for combined command graph.
 *
 * Work for each extractor is split into three phases:
 *   pre_fn   → memset / clear operations (direct-enqueued OUTSIDE the graph)
 *   enqueue_fn → compute kernel launches (recorded INSIDE the command graph)
 *   post_fn  → D2H memcpy operations   (direct-enqueued OUTSIDE the graph)
 *
 * The Level Zero graph runtime does not reliably replay memcpy/memset
 * nodes, so only pure kernel launches go inside the recorded graph.
 *
 * enqueue_fn: Enqueue only compute kernels for this extractor.
 *   @param queue_ptr    Opaque pointer to sycl::queue.
 *   @param priv         Extractor private state.
 *   @param shared_ref   Device pointer to ref Y-plane.
 *   @param shared_dis   Device pointer to dis Y-plane.
 *
 * pre_fn / post_fn: Device-memory housekeeping (memset / memcpy).
 *   @param queue_ptr    Opaque pointer to sycl::queue.
 *   @param priv         Extractor private state.
 *   May be NULL if not needed.
 *
 * config_fn: Configure extractor state for a specific double-buffer slot.
 *   Called before enqueue_fn during graph recording so that the correct
 *   device pointers are captured (e.g. motion ping-pong buffers).
 *   @param priv  Extractor private state.
 *   @param slot  Double-buffer slot index (0 or 1).
 *   May be NULL if no per-slot configuration is needed.
 */
typedef void (*VmafSyclGraphEnqueueFn)(void *queue_ptr, void *priv,
                                        void *shared_ref, void *shared_dis);
typedef void (*VmafSyclGraphPreFn)(void *queue_ptr, void *priv);
typedef void (*VmafSyclGraphPostFn)(void *queue_ptr, void *priv);
typedef void (*VmafSyclGraphConfigFn)(void *priv, int slot);

/**
 * Register an extractor's GPU work with the combined command graph.
 * Must be called during init_fex_sycl for each SYCL extractor.
 *
 * @param state       The SYCL state.
 * @param enqueue_fn  Kernel launches (recorded inside the command graph).
 * @param pre_fn      Pre-graph work: memset ops (may be NULL).
 * @param post_fn     Post-graph work: D2H memcpy ops (may be NULL).
 * @param config_fn   Per-slot configuration for graph recording (may be NULL).
 * @param priv        Extractor private state (passed to callbacks).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_graph_register(VmafSyclState *state,
                              VmafSyclGraphEnqueueFn enqueue_fn,
                              VmafSyclGraphPreFn pre_fn,
                              VmafSyclGraphPostFn post_fn,
                              VmafSyclGraphConfigFn config_fn,
                              void *priv,
                              const char *name);

/**
 * Get the combined compute queue for direct submission.
 * All registered extractors share this single queue.
 *
 * @param state  The SYCL state.
 *
 * @return Opaque pointer to sycl::queue, or NULL.
 */
void *vmaf_sycl_get_combined_queue(VmafSyclState *state);

/**
 * Submit all registered extractors' GPU work for the current frame.
 * Idempotent per frame: enqueues work only when the last extractor submits;
 * earlier calls for the same frame are no-ops.
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_graph_submit(VmafSyclState *state);

/**
 * Wait for all GPU work to complete.
 * Idempotent per frame: only the first call per frame actually waits.
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_graph_wait(VmafSyclState *state);

/**
 * Check whether combined command graphs have been recorded.
 *
 * @param state  The SYCL state.
 *
 * @return 1 if graphs are recorded, 0 otherwise.
 */
int vmaf_sycl_graphs_recorded(VmafSyclState *state);

/**
 * Wait unconditionally for the combined compute queue to drain.
 * Unlike vmaf_sycl_graph_wait(), this is NOT idempotent — it always waits.
 * Used by vmaf_sycl_wait_compute() to ensure previous frame's GPU extractors
 * have finished before the VA import path overwrites shared buffers.
 *
 * @param state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_combined_queue_wait(VmafSyclState *state);

/**
 * Advance the internal frame counter.
 * Must be called once per frame in the zero-copy VA import path
 * (vmaf_read_pictures_sycl), because the host upload path
 * (vmaf_sycl_shared_frame_upload) increments it internally but
 * the VA import path does not.
 *
 * Without this, graph_submit/graph_wait synchronization breaks:
 *   - graph_wait idempotency returns stale results from frame 2+
 *   - graph_submit fires after every extractor instead of once per frame
 *
 * @param state  The SYCL state.
 */
void vmaf_sycl_advance_frame(VmafSyclState *state);

/* ---- VA import deferred DMA-BUF free ---- */

/**
 * Free all pending DMA-BUF import allocations.
 * Called after the primary queue is drained (de-tile kernels finished)
 * so the imported device pointers are no longer read.
 *
 * @param state  The SYCL state.
 */
void vmaf_sycl_flush_pending_imports(VmafSyclState *state);

/**
 * Print timing summary (cpu/gpu breakdown) to stderr.
 */
void vmaf_sycl_print_timing(VmafSyclState *state);

/**
 * Defer a DMA-BUF import pointer for freeing on the next queue wait.
 * The pointer will be freed by vmaf_sycl_flush_pending_imports().
 *
 * @param state  The SYCL state.
 * @param ptr    Device pointer from vmaf_sycl_dmabuf_import().
 */
void vmaf_sycl_defer_import_free(VmafSyclState *state, void *ptr);

/**
 * Record the last de-tile kernel event for cross-queue synchronization.
 * The compute queue will barrier on this event before running extractors.
 *
 * @param state      The SYCL state.
 * @param event_ptr  Pointer to sycl::event from the de-tile parallel_for.
 */
void vmaf_sycl_set_detile_event(VmafSyclState *state, void *event_ptr);

/* ---- Profiling ---- */

/**
 * Record a kernel timing entry (accumulated per kernel name).
 *
 * @param state        The SYCL state.
 * @param kernel_name  Human-readable kernel name.
 * @param delta_ns     Elapsed time in nanoseconds.
 */
void vmaf_sycl_profiling_record(VmafSyclState *state,
                                 const char *kernel_name, uint64_t delta_ns);

/**
 * Check whether profiling is enabled.
 *
 * @param state  The SYCL state.
 *
 * @return true if profiling is enabled.
 */
bool vmaf_sycl_profiling_is_enabled(VmafSyclState *state);

#ifdef __cplusplus
}
#endif

#endif /* HAVE_SYCL */

#endif /* __VMAF_SRC_SYCL_COMMON_H__ */
