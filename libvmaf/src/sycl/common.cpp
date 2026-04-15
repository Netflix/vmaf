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

#include "config.h"

#if HAVE_SYCL

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <level_zero/ze_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <time.h>

static double monotonic_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

extern "C" {
#include "common.h"
#include "log.h"
}

namespace syclex = sycl::ext::oneapi::experimental;
using exec_graph_t = syclex::command_graph<syclex::graph_state::executable>;

/* ------------------------------------------------------------------ */
/* VmafSyclState definition (opaque in the public header)             */
/* ------------------------------------------------------------------ */

struct VmafSyclState {
    sycl::queue queue;       // primary queue (legacy, misc ops)
    sycl::queue copy_queue;  // separate queue for H2D/D2H DMA transfers
    bool has_fp64;           // device supports double precision (fp64)

    // Double-buffered shared frame buffers: uploaded once per frame,
    // read by all extractors.  Two sets allow overlapping frame N+1
    // upload with frame N compute — no CPU-side wait between them.
    void *shared_ref_buf[2] = {};
    void *shared_dis_buf[2] = {};
    int   cur_upload = 0;       // index being uploaded (toggled each frame)
    int   cur_compute = 0;      // index being read by compute (previous upload)
    sycl::event last_upload_event;  // event from last copy_queue operation
    bool  has_uploaded = false;     // guard: true after first upload
    size_t shared_buf_size = 0;
    unsigned frame_w = 0;
    unsigned frame_h = 0;
    unsigned frame_bpc = 0;

    // VA import path: async de-tile + deferred DMA-BUF free
    // The de-tile kernel runs on the primary queue without q->wait().
    // The imported DMA-BUF pointers must stay alive until the de-tile
    // finishes reading from them, so we defer the free to the next frame
    // (after wait_compute confirms the primary queue is idle).
    sycl::event last_detile_event;    // event from last de-tile kernel
    bool  has_imported = false;       // guard: true after first VA import
    static constexpr int MAX_PENDING_IMPORTS = 4;
    void *pending_import_ptrs[MAX_PENDING_IMPORTS] = {};
    int   num_pending_imports = 0;

    // Profiling
    bool profiling_enabled = false;
    bool extractor_timing = false;  // per-extractor q.wait() timing
    std::mutex profiling_lock;
    struct ProfileEntry {
        uint64_t total_ns = 0;
        uint64_t count = 0;
    };
    std::map<std::string, ProfileEntry> profiling_data;

    // Frame-level timing
    double t_last_wait_done = 0;   // timestamp after graph_wait completes
    double t_submit_start = 0;     // timestamp at graph_submit entry
    double t_submit_done = 0;      // timestamp when graph_submit returns (all work enqueued)
    double sum_cpu_ms = 0;         // accumulated CPU time (between frames)
    double sum_gpu_ms = 0;         // accumulated GPU time (submit to wait)
    uint64_t timing_frames = 0;    // frames with valid timing

    // Combined command graph — merges all extractors into one replay
    sycl::queue *combined_queue = nullptr;
    exec_graph_t *combined_exec_graph[2] = {}; // one per double-buffer slot
    bool combined_graphs_recorded = false;
    uint64_t frame_counter = 0;        // incremented in shared_frame_upload
    uint64_t submit_frame = UINT64_MAX;  // frame for which submits are counted
    int submit_count = 0;                // submits received for submit_frame
    uint64_t graph_waited_frame = UINT64_MAX;     // last frame waited

    static constexpr int MAX_GRAPH_EXTRACTORS = 8;

    struct GraphExtractor {
        VmafSyclGraphEnqueueFn enqueue_fn = nullptr;
        VmafSyclGraphPreFn pre_fn = nullptr;
        VmafSyclGraphPostFn post_fn = nullptr;
        VmafSyclGraphConfigFn config_fn = nullptr;
        void *priv = nullptr;
        const char *name = "unknown";
    };
    GraphExtractor graph_extractors[MAX_GRAPH_EXTRACTORS];
    int num_graph_extractors = 0;

    VmafSyclState(sycl::queue q, sycl::queue cq)
        : queue(std::move(q)), copy_queue(std::move(cq)) {}
};

/* ------------------------------------------------------------------ */
/* State lifecycle                                                     */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_state_init(VmafSyclState **sycl_state,
                          VmafSyclConfiguration cfg)
{
    if (!sycl_state) return -EINVAL;

    try {
        // Enumerate Level Zero GPU devices.
        // Use gpu_selector_v which selects Intel GPU by default on oneAPI.
        sycl::device dev;

        if (cfg.device_index < 0) {
            dev = sycl::device(sycl::gpu_selector_v);
        } else {
            auto platforms = sycl::platform::get_platforms();
            std::vector<sycl::device> gpus;
            for (auto &p : platforms) {
                for (auto &d : p.get_devices(sycl::info::device_type::gpu))
                    gpus.push_back(d);
            }
            if (static_cast<unsigned>(cfg.device_index) >= gpus.size()) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR,
                         "SYCL: device_index %d out of range (%zu GPUs)\n",
                         cfg.device_index, gpus.size());
                return -ENODEV;
            }
            dev = gpus[cfg.device_index];
        }

        vmaf_log(VMAF_LOG_LEVEL_INFO, "SYCL: using device: %s\n",
                 dev.get_info<sycl::info::device::name>().c_str());

        bool has_fp64 = dev.has(sycl::aspect::fp64);
        if (!has_fp64) {
            vmaf_log(VMAF_LOG_LEVEL_WARNING,
                     "SYCL: device lacks fp64 support — "
                     "using int64 emulation for gain limiting\n");
        }

        sycl::property_list props;
        // Allow runtime profiling via environment variable
        bool profiling = cfg.enable_profiling;
        const char *env_prof = getenv("VMAF_SYCL_PROFILE");
        if (env_prof && env_prof[0] == '1') profiling = true;
        if (profiling) {
            props = sycl::property_list{
                sycl::property::queue::in_order{},
                sycl::property::queue::enable_profiling{}
            };
        } else {
            props = sycl::property_list{
                sycl::property::queue::in_order{}
            };
        }

        sycl::queue q(dev, props);

        // Create a separate copy queue for DMA transfers.
        // Uses the same context+device so USM pointers are interoperable.
        // Copy queue is always in-order (no profiling needed for memcpy).
        sycl::queue cq(q.get_context(), dev,
                       sycl::property_list{sycl::property::queue::in_order{}});

        auto *s = new VmafSyclState(std::move(q), std::move(cq));
        s->profiling_enabled = profiling;
        // Per-extractor timing via q.wait() — no enable_profiling needed
        const char *env_timing = getenv("VMAF_SYCL_TIMING");
        s->extractor_timing = (env_timing && env_timing[0] == '1');
        s->has_fp64 = has_fp64;
        *sycl_state = s;
        return 0;

    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL exception: %s\n", e.what());
        return -ENODEV;
    } catch (const std::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL init error: %s\n", e.what());
        return -EINVAL;
    }
}

extern "C"
void vmaf_sycl_state_free(VmafSyclState **sycl_state)
{
    if (!sycl_state || !*sycl_state) return;

    VmafSyclState *s = *sycl_state;

    // Wait for any outstanding work on both queues
    try { s->copy_queue.wait_and_throw(); } catch (...) {}
    try { s->queue.wait_and_throw(); } catch (...) {}

    // Free any deferred DMA-BUF imports
    vmaf_sycl_flush_pending_imports(s);

    // Free combined command graphs and queue
    for (int i = 0; i < 2; i++) {
        if (s->combined_exec_graph[i]) {
            delete s->combined_exec_graph[i];
            s->combined_exec_graph[i] = nullptr;
        }
    }
    if (s->combined_queue) {
        try { s->combined_queue->wait_and_throw(); } catch (...) {}
        delete s->combined_queue;
        s->combined_queue = nullptr;
    }

    vmaf_sycl_shared_frame_close(s);

    delete s;
    *sycl_state = nullptr;
}

/* ------------------------------------------------------------------ */
/* Per-extractor compute queue management                              */
/* ------------------------------------------------------------------ */

extern "C"
void *vmaf_sycl_create_compute_queue(VmafSyclState *state)
{
    if (!state) return nullptr;
    try {
        sycl::property_list props;
        if (state->profiling_enabled) {
            props = sycl::property_list{
                sycl::property::queue::in_order{},
                sycl::property::queue::enable_profiling{}
            };
        } else {
            props = sycl::property_list{
                sycl::property::queue::in_order{}
            };
        }
        // Same context+device so USM allocations are valid across queues
        auto *q = new sycl::queue(state->queue.get_context(),
                                  state->queue.get_device(), props);
        return q;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL create_compute_queue: %s\n", e.what());
        return nullptr;
    }
}

extern "C"
void vmaf_sycl_destroy_queue(void *queue_ptr)
{
    if (!queue_ptr) return;
    auto *q = static_cast<sycl::queue *>(queue_ptr);
    try { q->wait_and_throw(); } catch (...) {}
    delete q;
}

extern "C"
int vmaf_sycl_wait_copy_queue(VmafSyclState *state)
{
    if (!state) return -EINVAL;
    try {
        state->copy_queue.wait_and_throw();
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL copy queue wait: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
int vmaf_sycl_wait_last_upload(VmafSyclState *state)
{
    if (!state) return -EINVAL;
    if (!state->has_uploaded) return 0;
    try {
        state->last_upload_event.wait();
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL upload event wait: %s\n", e.what());
        return -EIO;
    }
}

/* ------------------------------------------------------------------ */
/* USM memory helpers                                                  */
/* ------------------------------------------------------------------ */

extern "C"
void *vmaf_sycl_malloc_device(VmafSyclState *state, size_t size)
{
    if (!state || !size) return nullptr;
    try {
        return sycl::malloc_device(size, state->queue);
    } catch (...) {
        return nullptr;
    }
}

extern "C"
void *vmaf_sycl_malloc_host(VmafSyclState *state, size_t size)
{
    if (!state || !size) return nullptr;
    try {
        return sycl::malloc_host(size, state->queue);
    } catch (...) {
        return nullptr;
    }
}

extern "C"
void vmaf_sycl_free(VmafSyclState *state, void *ptr)
{
    if (!state || !ptr) return;
    try {
        sycl::free(ptr, state->queue);
    } catch (...) {}
}

extern "C"
int vmaf_sycl_memcpy_h2d(VmafSyclState *state, void *dst, const void *src,
                          size_t size)
{
    if (!state || !dst || !src || !size) return -EINVAL;
    try {
        state->queue.memcpy(dst, src, size).wait();
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL memcpy H2D: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
int vmaf_sycl_memcpy_d2h(VmafSyclState *state, void *dst, const void *src,
                          size_t size)
{
    if (!state || !dst || !src || !size) return -EINVAL;
    try {
        state->queue.memcpy(dst, src, size).wait();
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL memcpy D2H: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
int vmaf_sycl_memcpy_h2d_async(VmafSyclState *state, void *dst,
                                const void *src, size_t size)
{
    if (!state || !dst || !src || !size) return -EINVAL;
    try {
        state->queue.memcpy(dst, src, size);
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL async memcpy: %s\n", e.what());
        return -EIO;
    }
}

/* ------------------------------------------------------------------ */
/* Queue synchronization                                               */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_queue_wait(VmafSyclState *state)
{
    if (!state) return -EINVAL;
    try {
        state->queue.wait_and_throw();

        // After the primary queue is idle, it's safe to free DMA-BUF
        // imports from the previous frame (the de-tile kernel that reads
        // from them has completed).
        vmaf_sycl_flush_pending_imports(state);

        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "SYCL queue wait: %s\n", e.what());
        return -EIO;
    }
}

/* ------------------------------------------------------------------ */
/* Shared frame buffers                                                */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_shared_frame_init(VmafSyclState *state,
                                 unsigned w, unsigned h, unsigned bpc)
{
    if (!state) return -EINVAL;
    if (!w || !h) return -EINVAL;

    // Already initialized (idempotent)
    if (state->shared_ref_buf[0]) return 0;

    unsigned bytes_per_pixel = (bpc + 7) / 8;
    size_t buf_size = (size_t)w * h * bytes_per_pixel;

    // Allocate two sets of ref+dis buffers for double-buffering.
    // Buffer [cur_upload] receives H2D data while compute reads [cur_compute].
    for (int i = 0; i < 2; i++) {
        state->shared_ref_buf[i] = vmaf_sycl_malloc_device(state, buf_size);
        if (!state->shared_ref_buf[i]) goto fail;
        state->shared_dis_buf[i] = vmaf_sycl_malloc_device(state, buf_size);
        if (!state->shared_dis_buf[i]) goto fail;
    }

    state->shared_buf_size = buf_size;
    state->frame_w = w;
    state->frame_h = h;
    state->frame_bpc = bpc;
    state->cur_upload = 0;
    state->cur_compute = 0;

    return 0;

fail:
    for (int i = 0; i < 2; i++) {
        if (state->shared_ref_buf[i]) vmaf_sycl_free(state, state->shared_ref_buf[i]);
        if (state->shared_dis_buf[i]) vmaf_sycl_free(state, state->shared_dis_buf[i]);
        state->shared_ref_buf[i] = nullptr;
        state->shared_dis_buf[i] = nullptr;
    }
    return -ENOMEM;
}

extern "C"
int vmaf_sycl_shared_frame_get(VmafSyclState *state, void **ref, void **dis)
{
    if (!state) return -EINVAL;
    int idx = state->cur_compute;
    if (!state->shared_ref_buf[idx] || !state->shared_dis_buf[idx]) return -EINVAL;

    if (ref) *ref = state->shared_ref_buf[idx];
    if (dis) *dis = state->shared_dis_buf[idx];

    return 0;
}

extern "C"
int vmaf_sycl_shared_frame_upload(VmafSyclState *state,
                                   VmafPicture *ref, VmafPicture *dis)
{
    if (!state || !ref || !dis) return -EINVAL;
    if (!state->shared_ref_buf[0]) return -EINVAL;

    double t0 = monotonic_ms();

    // Double-buffered upload via copy_queue (DMA engine).
    // In-order copy_queue ensures sequential uploads complete in order.
    int ui = state->cur_upload;
    unsigned bytes_per_pixel = (state->frame_bpc + 7) / 8;
    size_t row_bytes = (size_t)state->frame_w * bytes_per_pixel;

    try {
        // If stride matches width, single memcpy; otherwise row-by-row
        if ((unsigned)ref->stride[0] == row_bytes) {
            state->copy_queue.memcpy(state->shared_ref_buf[ui],
                           ref->data[0], state->shared_buf_size);
        } else {
            uint8_t *dst = (uint8_t *)state->shared_ref_buf[ui];
            const uint8_t *src = (const uint8_t *)ref->data[0];
            for (unsigned y = 0; y < state->frame_h; y++) {
                state->copy_queue.memcpy(dst, src, row_bytes);
                dst += row_bytes;
                src += ref->stride[0];
            }
        }

        double t1 = monotonic_ms();

        sycl::event last_ev;
        if ((unsigned)dis->stride[0] == row_bytes) {
            last_ev = state->copy_queue.memcpy(state->shared_dis_buf[ui],
                           dis->data[0], state->shared_buf_size);
        } else {
            uint8_t *dst = (uint8_t *)state->shared_dis_buf[ui];
            const uint8_t *src = (const uint8_t *)dis->data[0];
            for (unsigned y = 0; y < state->frame_h; y++) {
                last_ev = state->copy_queue.memcpy(dst, src, row_bytes);
                dst += row_bytes;
                src += dis->stride[0];
            }
        }

        double t2 = monotonic_ms();

        // Save the last DMA event so compute can depend on it
        // without a CPU-blocking wait.
        state->last_upload_event = last_ev;
        state->has_uploaded = true;

        // Swap: the buffer we just uploaded becomes compute, old compute
        // becomes the next upload target.
        state->cur_compute = ui;
        state->cur_upload  = 1 - ui;
        state->frame_counter++;

        if (state->extractor_timing && state->frame_counter <= 30) {
            fprintf(stderr, "UPLOAD frame %lu: ref=%.1fms dis=%.1fms total=%.1fms\n",
                    (unsigned long)state->frame_counter,
                    t1 - t0, t2 - t1, t2 - t0);
        }

    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL frame upload: %s\n", e.what());
        return -EIO;
    }

    return 0;
}

extern "C"
int vmaf_sycl_upload_plane(VmafSyclState *state,
                            const void *src, unsigned pitch,
                            int is_ref,
                            unsigned w, unsigned h, unsigned bpc)
{
    if (!state || !src) return -EINVAL;

    int ui = state->cur_upload;
    void *target_buf = is_ref
        ? state->shared_ref_buf[ui]
        : state->shared_dis_buf[ui];

    if (!target_buf) return -EINVAL;

    unsigned bytes_per_pixel = (bpc + 7) / 8;
    size_t row_bytes = (size_t)w * bytes_per_pixel;

    try {
        if (pitch == row_bytes) {
            /* Contiguous — single H2D copy via copy queue */
            state->copy_queue.memcpy(target_buf, src, row_bytes * h);
        } else {
            /* Pitched — row-by-row H2D copy via copy queue */
            uint8_t *dst = (uint8_t *)target_buf;
            const uint8_t *s = (const uint8_t *)src;
            for (unsigned y = 0; y < h; y++) {
                state->copy_queue.memcpy(dst, s, row_bytes);
                dst += row_bytes;
                s += pitch;
            }
        }
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL upload_plane: %s\n", e.what());
        return -EIO;
    }

    return 0;
}

extern "C"
void vmaf_sycl_shared_frame_close(VmafSyclState *state)
{
    if (!state) return;

    for (int i = 0; i < 2; i++) {
        if (state->shared_ref_buf[i]) {
            vmaf_sycl_free(state, state->shared_ref_buf[i]);
            state->shared_ref_buf[i] = nullptr;
        }
        if (state->shared_dis_buf[i]) {
            vmaf_sycl_free(state, state->shared_dis_buf[i]);
            state->shared_dis_buf[i] = nullptr;
        }
    }
    state->shared_buf_size = 0;
}

/* ------------------------------------------------------------------ */
/* Queue handle for extractors                                         */
/* ------------------------------------------------------------------ */

extern "C"
void *vmaf_sycl_get_queue_ptr(VmafSyclState *state)
{
    if (!state) return nullptr;
    return &state->queue;
}

extern "C"
bool vmaf_sycl_has_fp64(VmafSyclState *state)
{
    if (!state) return false;
    return state->has_fp64;
}

extern "C"
void *vmaf_sycl_get_shared_ref(VmafSyclState *state)
{
    if (!state) return nullptr;
    return state->shared_ref_buf[state->cur_compute];
}

extern "C"
void *vmaf_sycl_get_shared_dis(VmafSyclState *state)
{
    if (!state) return nullptr;
    return state->shared_dis_buf[state->cur_compute];
}

extern "C"
void *vmaf_sycl_get_shared_ref_slot(VmafSyclState *state, int slot)
{
    if (!state || slot < 0 || slot > 1) return nullptr;
    return state->shared_ref_buf[slot];
}

extern "C"
void *vmaf_sycl_get_shared_dis_slot(VmafSyclState *state, int slot)
{
    if (!state || slot < 0 || slot > 1) return nullptr;
    return state->shared_dis_buf[slot];
}

extern "C"
int vmaf_sycl_get_compute_slot(VmafSyclState *state)
{
    if (!state) return 0;
    return state->cur_compute;
}

extern "C"
void *vmaf_sycl_get_last_upload_event(VmafSyclState *state)
{
    if (!state || !state->has_uploaded) return nullptr;
    return &state->last_upload_event;
}

/* ------------------------------------------------------------------ */
/* Combined command graph                                              */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_graph_register(VmafSyclState *state,
                              VmafSyclGraphEnqueueFn enqueue_fn,
                              VmafSyclGraphPreFn pre_fn,
                              VmafSyclGraphPostFn post_fn,
                              VmafSyclGraphConfigFn config_fn,
                              void *priv,
                              const char *name)
{
    if (!state || !enqueue_fn) return -EINVAL;
    if (state->num_graph_extractors >= state->MAX_GRAPH_EXTRACTORS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL: too many graph extractors (max %d)\n",
                 state->MAX_GRAPH_EXTRACTORS);
        return -ENOSPC;
    }

    auto &ge = state->graph_extractors[state->num_graph_extractors++];
    ge.enqueue_fn = enqueue_fn;
    ge.pre_fn = pre_fn;
    ge.post_fn = post_fn;
    ge.config_fn = config_fn;
    ge.priv = priv;
    ge.name = name ? name : "unknown";

    // Create combined queue lazily (once, shared by all extractors)
    if (!state->combined_queue) {
        auto *q = static_cast<sycl::queue *>(
            vmaf_sycl_create_compute_queue(state));
        if (!q) return -ENOMEM;
        state->combined_queue = q;
    }

    return 0;
}

extern "C"
void *vmaf_sycl_get_combined_queue(VmafSyclState *state)
{
    if (!state) return nullptr;
    return state->combined_queue;
}

static void record_combined_graphs(VmafSyclState *state)
{
    sycl::queue &q = *state->combined_queue;
    auto ctx = q.get_context();
    auto dev = q.get_device();

    for (int slot = 0; slot < 2; slot++) {
        void *ref = state->shared_ref_buf[slot];
        void *dis = state->shared_dis_buf[slot];

        // Configure each extractor for this slot
        for (int i = 0; i < state->num_graph_extractors; i++) {
            auto &ge = state->graph_extractors[i];
            if (ge.config_fn)
                ge.config_fn(ge.priv, slot);
        }

        syclex::command_graph<syclex::graph_state::modifiable> graph(ctx, dev);
        graph.begin_recording(q);

        // Record ONLY compute kernel launches into the graph.
        for (int i = 0; i < state->num_graph_extractors; i++) {
            auto &ge = state->graph_extractors[i];
            ge.enqueue_fn(&q, ge.priv, ref, dis);
        }

        graph.end_recording(q);
        state->combined_exec_graph[slot] = new exec_graph_t(graph.finalize());
    }

    // Restore extractor state for the current slot
    int cur_slot = state->cur_compute;
    for (int i = 0; i < state->num_graph_extractors; i++) {
        auto &ge = state->graph_extractors[i];
        if (ge.config_fn)
            ge.config_fn(ge.priv, cur_slot);
    }

    state->combined_graphs_recorded = true;
    vmaf_log(VMAF_LOG_LEVEL_INFO,
            "[vmaf-sycl] combined graphs recorded (%d extractors)",
            state->num_graph_extractors);
}

extern "C"
int vmaf_sycl_graph_submit(VmafSyclState *state)
{
    if (!state || !state->combined_queue) return -EINVAL;

    uint64_t frame = state->frame_counter;

    // Count submits per frame — each extractor calls this once per frame.
    // Reset counter on a new frame.
    if (state->submit_frame != frame) {
        state->submit_frame = frame;
        state->submit_count = 0;
    }
    state->submit_count++;

    // Only enqueue when ALL extractors have submitted.
    if (state->submit_count < state->num_graph_extractors)
        return 0;

    sycl::queue &q = *state->combined_queue;

    // GPU-side barrier: compute queue waits for the last DMA upload event
    // without blocking the CPU.  This enables DMA/compute overlap.
    if (state->has_uploaded) {
        q.ext_oneapi_submit_barrier({state->last_upload_event});
    }

    // GPU-side barrier: compute queue waits for the last de-tile kernel
    // on the primary queue.  The de-tile writes shared ref/dis buffers;
    // extractors on the compute queue must see those writes.
    if (state->has_imported) {
        q.ext_oneapi_submit_barrier({state->last_detile_event});
    }

    // frame_counter is incremented in upload/advance, so frame 0 → 1, frame 1 → 2
    // Record combined graph on frame 2 for both host-upload and VA-import paths.
    // VA import always uses slot 0 (cur_compute stays 0), so only slot 0 graph
    // gets replayed — but we record both for generality.
    // Graph replay eliminates per-frame kernel launch overhead (~33% faster on
    // Arc A380 at 4K).  Set VMAF_SYCL_NO_GRAPH=1 to disable.
    if (frame == 2 && (state->has_uploaded || state->has_imported)) {
        const char *env_nograph = getenv("VMAF_SYCL_NO_GRAPH");
        if (!(env_nograph && env_nograph[0] == '1'))
            record_combined_graphs(state);
    }

    state->t_submit_start = monotonic_ms();

    try {
    // Phase 1: Pre-graph — memset operations (always direct, never in graph)
    for (int i = 0; i < state->num_graph_extractors; i++) {
        auto &ge = state->graph_extractors[i];
        if (ge.pre_fn)
            ge.pre_fn(&q, ge.priv);
    }

    // Phase 2: Compute kernels — graph replay when available, else direct
    if (!state->combined_graphs_recorded) {
        void *ref = state->shared_ref_buf[state->cur_compute];
        void *dis = state->shared_dis_buf[state->cur_compute];

        for (int i = 0; i < state->num_graph_extractors; i++) {
            auto &ge = state->graph_extractors[i];
            ge.enqueue_fn(&q, ge.priv, ref, dis);
        }
    } else {
        // Barrier to ensure pre_fn memsets complete before graph replay.
        // Required because graph replay doesn't honour in-order queue
        // dependencies with non-graph operations on Level Zero.
        q.ext_oneapi_submit_barrier();
        // Replay the pre-recorded combined graph
        int slot = state->cur_compute;
        q.ext_oneapi_graph(*state->combined_exec_graph[slot]);
        // Barrier after replay to ensure graph completes before post_fn
        q.ext_oneapi_submit_barrier();
    }

    // Phase 3: Post-graph — D2H memcpy operations (always direct)
    for (int i = 0; i < state->num_graph_extractors; i++) {
        auto &ge = state->graph_extractors[i];
        if (ge.post_fn)
            ge.post_fn(&q, ge.priv);
    }

    } catch (const sycl::exception &e) {
        fprintf(stderr, "libvmaf SYCL exception in graph_submit: %s\n", e.what());
        return -EIO;
    } catch (const std::exception &e) {
        fprintf(stderr, "libvmaf exception in graph_submit: %s\n", e.what());
        return -EIO;
    } catch (...) {
        fprintf(stderr, "libvmaf unknown exception in graph_submit\n");
        return -EIO;
    }

    // Invalidate wait guard — new work was enqueued, next graph_wait must wait
    state->graph_waited_frame = UINT64_MAX;
    state->t_submit_done = monotonic_ms();

    return 0;
}

extern "C"
int vmaf_sycl_graph_wait(VmafSyclState *state)
{
    if (!state || !state->combined_queue) return -EINVAL;

    uint64_t frame = state->frame_counter;

    // Idempotent: only wait once per frame
    if (state->graph_waited_frame == frame) return 0;
    state->graph_waited_frame = frame;

    try {
        double t_before_wait = monotonic_ms();
        state->combined_queue->wait_and_throw();
        double now = monotonic_ms();

        double wait_ms = now - t_before_wait;        // actual GPU wait
        double enqueue_ms = state->t_submit_done - state->t_submit_start; // enqueue time
        double between_ms = t_before_wait - state->t_submit_done;  // CPU between submit return and wait call

        if (state->t_submit_start > 0) {
            double gpu_ms = now - state->t_submit_start;
            state->sum_gpu_ms += gpu_ms;
            // Per-frame timing for first 30 frames
            if (state->extractor_timing && frame <= 30) {
                double cpu_ms_frame = (state->t_last_wait_done > 0)
                    ? state->t_submit_start - state->t_last_wait_done : 0;
                fprintf(stderr, "FRAME %3lu: total=%.1f  enqueue=%.1f  between=%.1f  wait=%.1f  cpu=%.1f  %s\n",
                        (unsigned long)frame, gpu_ms, enqueue_ms, between_ms, wait_ms, cpu_ms_frame,
                        state->combined_graphs_recorded ? "graph" : "direct");
            }
        }
        if (state->t_last_wait_done > 0 && state->t_submit_start > 0) {
            double cpu_ms = state->t_submit_start - state->t_last_wait_done;
            state->sum_cpu_ms += cpu_ms;
            state->timing_frames++;
        }
        state->t_last_wait_done = now;

        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL graph wait: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
int vmaf_sycl_graphs_recorded(VmafSyclState *state)
{
    if (!state) return 0;
    return state->combined_graphs_recorded ? 1 : 0;
}

extern "C"
int vmaf_sycl_combined_queue_wait(VmafSyclState *state)
{
    if (!state) return -EINVAL;
    if (!state->combined_queue) return 0;  // not yet initialised

    try {
        state->combined_queue->wait_and_throw();
        return 0;
    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL combined queue wait: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
void vmaf_sycl_advance_frame(VmafSyclState *state)
{
    if (state)
        state->frame_counter++;
}

/* ------------------------------------------------------------------ */
/* VA import deferred DMA-BUF free                                     */
/* ------------------------------------------------------------------ */

extern "C"
void vmaf_sycl_flush_pending_imports(VmafSyclState *state)
{
    if (!state) return;

    ze_context_handle_t ze_ctx = nullptr;
    for (int i = 0; i < state->num_pending_imports; i++) {
        if (!state->pending_import_ptrs[i]) continue;
        if (!ze_ctx) {
            try {
                sycl::queue *q = &state->queue;
                ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                    q->get_context());
            } catch (...) { break; }
        }
        zeMemFree(ze_ctx, state->pending_import_ptrs[i]);
        state->pending_import_ptrs[i] = nullptr;
    }
    state->num_pending_imports = 0;
}

extern "C"
void vmaf_sycl_print_timing(VmafSyclState *state)
{
    if (!state || state->timing_frames < 2) return;
    double avg_cpu = state->sum_cpu_ms / state->timing_frames;
    double avg_gpu = state->sum_gpu_ms / state->timing_frames;
    double avg_total = avg_cpu + avg_gpu;
    double fps = avg_total > 0 ? 1000.0 / avg_total : 0;
    fprintf(stderr,
        "[vmaf-sycl] timing: %lu frames, avg cpu=%.2fms gpu=%.2fms "
        "total=%.2fms (%.1f fps), gpu%%=%.0f%%\n",
        (unsigned long)state->timing_frames, avg_cpu, avg_gpu,
        avg_total, fps, 100.0 * avg_gpu / avg_total);
    fflush(stderr);

    // Print per-kernel profiling if enabled
    if (state->profiling_enabled) {
        vmaf_sycl_profiling_print(state);
    }
}

extern "C"
void vmaf_sycl_defer_import_free(VmafSyclState *state, void *ptr)
{
    if (!state || !ptr) return;
    if (state->num_pending_imports >= state->MAX_PENDING_IMPORTS) {
        /* Safety: if we're full, flush now (shouldn't happen with 2 per frame) */
        vmaf_sycl_flush_pending_imports(state);
    }
    state->pending_import_ptrs[state->num_pending_imports++] = ptr;
}

extern "C"
void vmaf_sycl_set_detile_event(VmafSyclState *state, void *event_ptr)
{
    if (!state || !event_ptr) return;
    state->last_detile_event = *static_cast<sycl::event *>(event_ptr);
    state->has_imported = true;
}

/* ------------------------------------------------------------------ */
/* Profiling                                                           */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_profiling_enable(VmafSyclState *state)
{
    if (!state) return -EINVAL;
    state->profiling_enabled = true;
    return 0;
}

extern "C"
void vmaf_sycl_profiling_disable(VmafSyclState *state)
{
    if (!state) return;
    state->profiling_enabled = false;
    std::lock_guard<std::mutex> lock(state->profiling_lock);
    state->profiling_data.clear();
}

extern "C"
void vmaf_sycl_profiling_record(VmafSyclState *state,
                                 const char *kernel_name, uint64_t delta_ns)
{
    if (!state || !kernel_name || !state->profiling_enabled) return;
    std::lock_guard<std::mutex> lock(state->profiling_lock);
    auto &entry = state->profiling_data[kernel_name];
    entry.total_ns += delta_ns;
    entry.count++;
}

extern "C"
bool vmaf_sycl_profiling_is_enabled(VmafSyclState *state)
{
    if (!state) return false;
    return state->profiling_enabled;
}

extern "C"
void vmaf_sycl_profiling_print(VmafSyclState *state)
{
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->profiling_lock);

    if (state->profiling_data.empty()) {
        printf("SYCL profiling: no data recorded\n");
        return;
    }

    printf("SYCL kernel profiling results:\n");
    printf("%-30s %10s %12s %12s\n",
           "Kernel", "Calls", "Total (ms)", "Avg (ms)");
    printf("%-30s %10s %12s %12s\n",
           "------", "-----", "----------", "--------");

    uint64_t grand_total_ns = 0;
    for (auto &[name, entry] : state->profiling_data) {
        double total_ms = entry.total_ns / 1e6;
        double avg_ms = entry.count > 0 ? total_ms / entry.count : 0.0;
        printf("%-30s %10lu %12.3f %12.3f\n",
               name.c_str(), (unsigned long)entry.count, total_ms, avg_ms);
        grand_total_ns += entry.total_ns;
    }
    printf("%-30s %10s %12.3f\n", "TOTAL", "", grand_total_ns / 1e6);
}

extern "C"
int vmaf_sycl_profiling_get_string(VmafSyclState *state, char **output)
{
    if (!state || !output) return -EINVAL;
    *output = nullptr;

    std::lock_guard<std::mutex> lock(state->profiling_lock);

    std::string result;
    result += "SYCL kernel profiling results:\n";

    char line[256];
    snprintf(line, sizeof(line), "%-30s %10s %12s %12s\n",
             "Kernel", "Calls", "Total (ms)", "Avg (ms)");
    result += line;

    uint64_t grand_total_ns = 0;
    for (auto &[name, entry] : state->profiling_data) {
        double total_ms = entry.total_ns / 1e6;
        double avg_ms = entry.count > 0 ? total_ms / entry.count : 0.0;
        snprintf(line, sizeof(line), "%-30s %10lu %12.3f %12.3f\n",
                 name.c_str(), (unsigned long)entry.count, total_ms, avg_ms);
        result += line;
        grand_total_ns += entry.total_ns;
    }
    snprintf(line, sizeof(line), "%-30s %10s %12.3f\n",
             "TOTAL", "", grand_total_ns / 1e6);
    result += line;

    *output = strdup(result.c_str());
    return *output ? 0 : -ENOMEM;
}

#endif /* HAVE_SYCL */
