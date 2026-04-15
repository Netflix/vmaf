/**
 *
 *  Copyright 2024-2026 Lusoris and Claude (Anthropic)
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

/**
 * SYCL/DPC++ Motion feature extractor.
 *
 * Implements separable 5-tap Gaussian blur + SAD (Sum of Absolute
 * Differences) between consecutive blurred reference frames.
 *
 * Algorithm:
 *   1. Blur the current reference frame with a separable 5-tap Gaussian
 *      kernel: {3571, 16004, 26386, 16004, 3571} (sum = 65536).
 *   2. Compute SAD between current blurred frame and previous blurred frame.
 *   3. motion_score = SAD / 256.0 / (width * height)
 *   4. motion2_score = min(prev_motion_score, cur_motion_score)
 *
 * Uses ping-pong buffers for blurred frames across temporal frames.
 *
 * Pattern: init -> submit (non-blocking) -> collect (wait + scores)
 * TEMPORAL flag: frames must be processed in sequential order.
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "sycl/common.h"
#include "log.h"
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

// 5-tap Gaussian filter: {3571, 16004, 26386, 16004, 3571}, sum = 65536
static constexpr int32_t blur_filter[5] = {
    3571, 16004, 26386, 16004, 3571
};

/* ------------------------------------------------------------------ */
/* Extractor private state                                             */
/* ------------------------------------------------------------------ */

struct MotionStateSycl {
    unsigned width, height;
    unsigned bpc;

    bool debug;
    bool motion_force_zero;

    VmafDictionary *feature_name_dict;

    // Frame tracking
    unsigned frame_index;
    double prev_motion_score;

    // Ping-pong blur buffers (device)
    int32_t *d_blur[2];   // alternating current / previous
    int cur_blur;         // index into d_blur for current frame

    // Vertical intermediate buffer
    int32_t *d_blur_tmp;  // vertical pass output

    // SAD accumulator (device + host)
    int64_t *d_sad_accum;
    int64_t *h_sad_accum;

    // Deferred state
    unsigned pending_index;
    bool has_pending;

    // Back-pointer for graph-mode checks
    VmafSyclState *sycl_state;
};

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateSycl, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = true},
    },
    {
        .name = "motion_force_zero",
        .help = "force motion score to zero",
        .offset = offsetof(MotionStateSycl, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/* ------------------------------------------------------------------ */
/* Device helpers                                                      */
/* ------------------------------------------------------------------ */

static inline int dev_mirror_motion(int idx, int sup) {
    if (idx < 0)   return -idx;
    if (idx >= sup) return 2 * sup - idx - 1;
    return idx;
}

/* ------------------------------------------------------------------ */
/* SYCL Kernel: Fused 2D Gaussian Blur + SAD (single dispatch)        */
/*                                                                     */
/* Eliminates the intermediate vertical output buffer by computing     */
/* the full separable filter from a 2D SLM tile in registers:          */
/*   Step A: vertical filter at 5 horizontal positions → vtmp[5]       */
/*   Step B: horizontal filter on vtmp[5] → final blurred pixel        */
/* This matches the V→H ordering for exact bit-identical results.      */
/* ------------------------------------------------------------------ */

static sycl::event launch_blur_sad_fused(sycl::queue &q,
                                          const void *input,
                                          int32_t *blur_out,
                                          const int32_t *prev_blur,
                                          int64_t *sad_accum,
                                          unsigned width, unsigned height,
                                          unsigned bpc, bool compute_sad)
{
    auto e_w = width;
    auto e_h = height;
    auto e_bpc = bpc;
    auto e_sad = compute_sad;
    auto p_in = input;

    constexpr int WG_X = 32, WG_Y = 8;
    constexpr int HALF_FW = 2;
    // 2D tile with 2-pixel halo on every edge for the 5-tap filter
    constexpr int TILE_H = WG_Y + 2 * HALF_FW; // 12
    constexpr int TILE_W = WG_X + 2 * HALF_FW; // 36
    constexpr int WG_SIZE = WG_X * WG_Y;        // 256

    sycl::range<2> global(((height + WG_Y - 1) / WG_Y) * WG_Y,
                          ((width  + WG_X - 1) / WG_X) * WG_X);
    sycl::range<2> local(WG_Y, WG_X);

    return q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int32_t, 2> s_tile(
            sycl::range<2>(TILE_H, TILE_W), cgh);
        constexpr int MAX_SUBGROUPS = 32;
        sycl::local_accessor<int64_t, 1> lmem(
            sycl::range<1>(MAX_SUBGROUPS), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item)
                [[intel::reqd_sub_group_size(32)]] {
                const int gx = item.get_global_id(1);
                const int gy = item.get_global_id(0);
                const unsigned lid = item.get_local_linear_id();
                const unsigned lx = item.get_local_id(1);
                const unsigned ly = item.get_local_id(0);
                const bool valid = (gx < (int)e_w && gy < (int)e_h);

                // --- Phase 1: Cooperative 2D tile load ---
                int tile_origin_y = (int)(item.get_group(0) * WG_Y) - HALF_FW;
                int tile_origin_x = (int)(item.get_group(1) * WG_X) - HALF_FW;

                constexpr unsigned tile_elems = TILE_H * TILE_W; // 432

                auto read_global = [&](int y, int x) -> int32_t {
                    if (e_bpc <= 8)
                        return static_cast<const uint8_t *>(p_in)[y * e_w + x];
                    else
                        return static_cast<const uint16_t *>(p_in)[y * e_w + x];
                };

                bool interior_wg =
                    (tile_origin_y >= 0) &&
                    (tile_origin_y + TILE_H <= (int)e_h) &&
                    (tile_origin_x >= 0) &&
                    (tile_origin_x + TILE_W <= (int)e_w);

                if (interior_wg) {
                    for (unsigned i = lid; i < tile_elems; i += WG_SIZE) {
                        unsigned tr = i / TILE_W;
                        unsigned tc = i % TILE_W;
                        int py = tile_origin_y + (int)tr;
                        int px = tile_origin_x + (int)tc;
                        s_tile[tr][tc] = read_global(py, px);
                    }
                } else {
                    for (unsigned i = lid; i < tile_elems; i += WG_SIZE) {
                        unsigned tr = i / TILE_W;
                        unsigned tc = i % TILE_W;
                        int py = dev_mirror_motion(
                            tile_origin_y + (int)tr, (int)e_h);
                        int px = dev_mirror_motion(
                            tile_origin_x + (int)tc, (int)e_w);
                        s_tile[tr][tc] = read_global(py, px);
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                // --- Phase 2: Separable 2D blur (V→H in registers) ---
                int64_t abs_diff = 0;

                if (valid) {
                    // Step A: vertical filter at 5 horizontal positions
                    //   vtmp[hx] = vert_filter(tile[ly..ly+4][lx+hx])
                    // int32 arithmetic safe for bpc <= 12:
                    //   max vsum = 65535 × 65536 = 4.2B fits in uint32
                    int32_t round1 = 1 << (e_bpc - 1);
                    int32_t vtmp[5];

                    #pragma unroll
                    for (int hx = 0; hx < 5; hx++) {
                        unsigned tcol = lx + (unsigned)hx;
                        int32_t vsum =
                            blur_filter[0] *
                                (s_tile[ly + 0][tcol] +
                                 s_tile[ly + 4][tcol]) +
                            blur_filter[1] *
                                (s_tile[ly + 1][tcol] +
                                 s_tile[ly + 3][tcol]) +
                            blur_filter[2] *
                                 s_tile[ly + 2][tcol];
                        vtmp[hx] = (vsum + round1) >> e_bpc;
                    }

                    // Step B: horizontal filter on vertically-filtered values
                    int64_t hsum =
                        (int64_t)blur_filter[0] * (vtmp[0] + vtmp[4]) +
                        (int64_t)blur_filter[1] * (vtmp[1] + vtmp[3]) +
                        (int64_t)blur_filter[2] *  vtmp[2];
                    int32_t blurred = (int32_t)((hsum + 32768) >> 16);

                    blur_out[gy * e_w + gx] = blurred;

                    // SAD with previous frame
                    if (e_sad) {
                        int32_t prev = prev_blur[gy * e_w + gx];
                        int32_t diff = blurred - prev;
                        abs_diff = (diff < 0) ? -(int64_t)diff : (int64_t)diff;
                    }
                }

                // --- Phase 3: Subgroup reduction for SAD ---
                if (e_sad) {
                    sycl::sub_group sg = item.get_sub_group();
                    int64_t sg_sum = sycl::reduce_over_group(
                        sg, abs_diff, sycl::plus<int64_t>{});

                    const uint32_t sg_id = sg.get_group_linear_id();
                    const uint32_t sg_lid = sg.get_local_linear_id();
                    const uint32_t n_subgroups = sg.get_group_linear_range();

                    if (sg_lid == 0)
                        lmem[sg_id] = sg_sum;

                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        int64_t total = 0;
                        for (uint32_t s = 0; s < n_subgroups; s++)
                            total += lmem[s];

                        sycl::atomic_ref<int64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            ref(*sad_accum);
                        ref.fetch_add(total);
                    }
                }
            });
    });
}

/* ------------------------------------------------------------------ */
/* Feature extractor callbacks                                         */
/* ------------------------------------------------------------------ */

// Forward declarations for combined graph callbacks (defined after init)
static void enqueue_motion_work(void *queue_ptr, void *priv,
                                 void *shared_ref, void *shared_dis);
static void motion_pre_graph(void *queue_ptr, void *priv);
static void motion_post_graph(void *queue_ptr, void *priv);
static void config_motion_slot(void *priv, int slot);

static int init_fex_sycl(VmafFeatureExtractor *fex,
                          enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<MotionStateSycl *>(fex->priv);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->prev_motion_score = 0.0;
    s->has_pending = false;
    s->cur_blur = 0;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_sycl: no SYCL state\n");
        return -EINVAL;
    }

    VmafSyclState *state = fex->sycl_state;

    // Initialize shared frame buffers (idempotent, first extractor wins)
    int err = vmaf_sycl_shared_frame_init(state, w, h, bpc);
    if (err) return err;

    size_t buf_size = (size_t)w * h * sizeof(int32_t);

    // Two blur buffers for ping-pong
    s->d_blur[0] = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, buf_size));
    s->d_blur[1] = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, buf_size));

    // Vertical intermediate
    s->d_blur_tmp = static_cast<int32_t *>(
        vmaf_sycl_malloc_device(state, buf_size));

    // SAD accumulator
    s->d_sad_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_device(state, sizeof(int64_t)));
    s->h_sad_accum = static_cast<int64_t *>(
        vmaf_sycl_malloc_host(state, sizeof(int64_t)));

    if (!s->d_blur[0] || !s->d_blur[1] || !s->d_blur_tmp ||
        !s->d_sad_accum || !s->h_sad_accum) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "motion_sycl: device memory allocation failed\n");
        return -ENOMEM;
    }

    s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
        fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    // Store back-pointer for graph-mode checks in post_fn
    s->sycl_state = state;

    // Register with combined command graph
    int err2 = vmaf_sycl_graph_register(state, enqueue_motion_work,
                                         motion_pre_graph, motion_post_graph,
                                         config_motion_slot, s, "MOTION");
    if (err2) return err2;

    return 0;
}

static int extract_force_zero(VmafFeatureExtractor *fex,
                               VmafPicture *ref, VmafPicture *ref_90,
                               VmafPicture *dist, VmafPicture *dist_90,
                               unsigned index,
                               VmafFeatureCollector *feature_collector)
{
    (void)ref; (void)ref_90; (void)dist; (void)dist_90;
    auto *s = static_cast<MotionStateSycl *>(fex->priv);

    int err = 0;
    if (s->frame_index > 0) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion_score",
            0.0, index);
    }
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
        s->feature_name_dict,
        "VMAF_integer_feature_motion2_score",
        0.0, index);

    s->frame_index++;
    return err;
}

/* ------------------------------------------------------------------ */
/* C-compatible callbacks for combined command graph                    */
/* ------------------------------------------------------------------ */

// Pre-graph: zero SAD accumulator (direct enqueue, outside graph)
static void motion_pre_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<MotionStateSycl *>(priv);
    if (s->frame_index > 0) {
        q.memset(s->d_sad_accum, 0, sizeof(int64_t));
    }
}

// Graph-recorded: compute kernels only
static void enqueue_motion_work(void *queue_ptr, void *priv,
                                 void *shared_ref, void *shared_dis)
{
    (void)shared_dis; // Motion only uses ref
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<MotionStateSycl *>(priv);

    bool compute_sad = (s->frame_index > 0);
    int cur = s->cur_blur;
    int prev = 1 - cur;

    launch_blur_sad_fused(q, shared_ref,
                          s->d_blur[cur], s->d_blur[prev],
                          s->d_sad_accum, s->width, s->height,
                          s->bpc, compute_sad);
}

// Post-graph: D2H SAD accumulator download (direct enqueue, outside graph)
// With two graph slots (config_fn sets cur_blur=slot), each slot writes
// to d_blur[slot] and reads from d_blur[1-slot].  The natural ping-pong
// alternation keeps the "previous" buffer correct — no copy needed.
static void motion_post_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<MotionStateSycl *>(priv);
    if (s->frame_index > 0) {
        q.memcpy(s->h_sad_accum, s->d_sad_accum, sizeof(int64_t));
    }
}

static void config_motion_slot(void *priv, int slot)
{
    auto *s = static_cast<MotionStateSycl *>(priv);
    s->cur_blur = slot;
}

/* ------------------------------------------------------------------ */
/* Submit / Collect                                                    */
/* ------------------------------------------------------------------ */

static int submit_fex_sycl(VmafFeatureExtractor *fex,
                            VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                            VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                            unsigned index)
{
    (void)ref_pic; (void)ref_pic_90;
    (void)dist_pic; (void)dist_pic_90;

    auto *s = static_cast<MotionStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph submit (idempotent per frame — first extractor wins)
    int err = vmaf_sycl_graph_submit(state);
    if (err) return err;

    s->pending_index = index;
    s->has_pending = true;

    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex,
                             unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<MotionStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    // Combined graph wait (idempotent per frame — first extractor wins)
    vmaf_sycl_graph_wait(state);

    double motion_score = 0.0;

    if (s->frame_index > 0) {
        int64_t sad = *s->h_sad_accum;
        motion_score = (double)sad / 256.0 /
                       ((double)s->width * s->height);
    }

    int err = 0;

    // Match CPU motion algorithm: motion2_score[i] is written at step i+1
    // using min(motion(i-1→i), motion(i→i+1)).
    //
    // frame_index 0 (first collect):
    //   Write motion2[index] = 0.0 (no prior frame)
    //
    // frame_index 1 (second collect):
    //   Only one motion score available, can't compute min yet.
    //   Just store it; don't write motion2 yet.
    //   CPU also skips: if (index == 1) return 0;
    //
    // frame_index >= 2:
    //   Write motion2[index-1] = min(prev_motion, current_motion)
    //   This is the delayed-by-one pattern from CPU.

    if (s->frame_index == 0) {
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "VMAF_integer_feature_motion2_score", 0.0, index);
        if (s->debug) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "VMAF_integer_feature_motion_score", 0.0, index);
        }
    } else if (s->frame_index == 1) {
        // Write motion_score if debug
        if (s->debug) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "VMAF_integer_feature_motion_score", motion_score, index);
        }
        // Don't write motion2 yet (CPU returns early at index 1)
    } else {
        // frame_index >= 2: write motion2 at previous index
        double motion2 = (motion_score < s->prev_motion_score) ?
                          motion_score : s->prev_motion_score;
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "VMAF_integer_feature_motion2_score", motion2, index - 1);
        if (s->debug) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict,
                "VMAF_integer_feature_motion_score", motion_score, index);
        }
    }

    // Advance state
    s->prev_motion_score = motion_score;
    s->cur_blur = 1 - s->cur_blur; // flip ping-pong (direct mode)
    s->frame_index++;
    s->has_pending = false;

    return err;
}

static int extract_fex_sycl(VmafFeatureExtractor *fex,
                              VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                              VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                              unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<MotionStateSycl *>(fex->priv);

    if (s->motion_force_zero)
        return extract_force_zero(fex, ref_pic, ref_pic_90,
                                   dist_pic, dist_pic_90,
                                   index, feature_collector);

    int err = submit_fex_sycl(fex, ref_pic, ref_pic_90,
                               dist_pic, dist_pic_90, index);
    if (err) return err;
    return collect_fex_sycl(fex, index, feature_collector);
}

static int flush_fex_sycl(VmafFeatureExtractor *fex,
                            VmafFeatureCollector *feature_collector)
{
    if (!fex) return -EINVAL;
    auto *s = static_cast<MotionStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;
    if (state) vmaf_sycl_queue_wait(state);

    int ret = 0;
    // Write the final motion2 score (delayed-by-one pattern).
    if (s->frame_index > 1) {
        ret = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "VMAF_integer_feature_motion2_score",
            s->prev_motion_score, s->frame_index - 1);
    }
    return (ret < 0) ? ret : !ret; // 1 = done, negative = error
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<MotionStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    if (state) {
        vmaf_sycl_queue_wait(state);

        if (s->d_blur[0]) vmaf_sycl_free(state, s->d_blur[0]);
        if (s->d_blur[1]) vmaf_sycl_free(state, s->d_blur[1]);
        if (s->d_blur_tmp) vmaf_sycl_free(state, s->d_blur_tmp);
        if (s->d_sad_accum) vmaf_sycl_free(state, s->d_sad_accum);
        if (s->h_sad_accum) vmaf_sycl_free(state, s->h_sad_accum);
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Feature extractor definition                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    NULL
};

extern "C"
VmafFeatureExtractor vmaf_fex_integer_motion_sycl = {
    .name = "motion_sycl",
    .init = init_fex_sycl,
    .extract = extract_fex_sycl,
    .flush = flush_fex_sycl,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options,
    .priv_size = sizeof(MotionStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features,
};
