/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature kernel on the SYCL backend (T7-23 / batch 3 part
 *  1c — ADR-0192 / ADR-0193). SYCL twin of motion_v2_vulkan (PR #146)
 *  and motion_v2_cuda (this PR's part 1b).
 *
 *  Stateless variant of `motion_sycl`: exploits convolution linearity
 *  (`SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)`) so each
 *  frame computes its score in one kernel launch over (prev_ref - cur_ref)
 *  without storing blurred frames across submits.
 *
 *  Self-contained submit / collect — does NOT register with
 *  vmaf_sycl_graph_register because motion_v2 needs the previous
 *  frame's raw ref pixels which the shared_frame buffer doesn't
 *  preserve across calls. Each submit copies the current ref Y plane
 *  into a private device-side ping-pong (`d_pix[2]`); the next
 *  frame's submit reads it as "prev". Same shape as ciede_sycl
 *  (PR #137 / ADR-0182) and the Vulkan / CUDA twins of this kernel.
 *
 *  motion2_v2_score = min(score[i], score[i+1]) emitted host-side in
 *  flush() — mirrors CPU integer_motion_v2.c::flush.
 *
 *  Mirror padding: edge-replicating reflective mirror
 *  (`2 * size - idx - 1` for idx >= size) — DIFFERS by one pixel
 *  from `motion_sycl`'s `dev_mirror_motion` (which uses skip-
 *  boundary `2 * size - idx - 2`). Same offset that produced
 *  max_abs_diff = 2.62e-3 on the Vulkan twin during bring-up
 *  before the divergence was caught.
 */

#include <sycl/sycl.hpp>

#include "sycl_compat.h"

#include <cerrno>
#include <cstdint>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "sycl/common.h"
}

namespace
{

struct MotionV2StateSycl {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    size_t plane_bytes;

    /* SYCL state back-pointer. */
    VmafSyclState *sycl_state;

    /* Pinned host staging — single buffer reused per submit for the
     * cur ref Y upload. */
    void *h_pix;

    /* Ping-pong of raw ref Y planes on device. d_pix[index%2] is the
     * current frame's slot; d_pix[(index+1)%2] is the previous. */
    void *d_pix[2];

    /* Single int64 SAD accumulator (device + host). */
    int64_t *d_sad;
    int64_t *h_sad;

    /* Submit/collect plumbing. */
    bool has_pending;
    unsigned pending_index;
    unsigned frame_index;
    /* fps-aware weight applied to the v2 SAD score in flush().
     * Default 1.0 is a no-op. Mirrors motion_sycl and motion_cuda
     * (ADR-0192 / PR #851). */
    double motion_fps_weight;

    VmafDictionary *feature_name_dict;
};

static constexpr int32_t MV2_FILTER[5] = {3571, 16004, 26386, 16004, 3571};
static constexpr int MV2_WG_X = 32;
static constexpr int MV2_WG_Y = 8;
static constexpr int MV2_HALF_FW = 2;
static constexpr int MV2_TILE_W = MV2_WG_X + 2 * MV2_HALF_FW; /* 36 */
static constexpr int MV2_TILE_H = MV2_WG_Y + 2 * MV2_HALF_FW; /* 12 */

static inline int dev_mirror_mv2(int idx, int sup)
{
    if (idx < 0)
        return -idx;
    if (idx >= sup)
        return 2 * sup - idx - 1;
    return idx;
}

static sycl::event launch_motion_v2(sycl::queue &q, const void *prev, const void *cur,
                                    int64_t *sad_accum, unsigned width, unsigned height,
                                    unsigned bpc)
{
    const unsigned e_w = width;
    const unsigned e_h = height;
    const unsigned e_bpc = bpc;
    const void *e_prev = prev;
    const void *e_cur = cur;

    sycl::range<2> global(((height + MV2_WG_Y - 1) / MV2_WG_Y) * MV2_WG_Y,
                          ((width + MV2_WG_X - 1) / MV2_WG_X) * MV2_WG_X);
    sycl::range<2> local(MV2_WG_Y, MV2_WG_X);

    return q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int32_t, 2> s_diff(sycl::range<2>(MV2_TILE_H, MV2_TILE_W), cgh);
        constexpr int MAX_SUBGROUPS = 32;
        sycl::local_accessor<int64_t, 1> lmem(sycl::range<1>(MAX_SUBGROUPS), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) VMAF_SYCL_REQD_SG_SIZE(32) {
                const int gx = (int)item.get_global_id(1);
                const int gy = (int)item.get_global_id(0);
                const unsigned lid = item.get_local_linear_id();
                const unsigned lx = item.get_local_id(1);
                const unsigned ly = item.get_local_id(0);
                const bool valid = (gx < (int)e_w && gy < (int)e_h);

                /* --- Phase 1: cooperative tile load of (prev - cur) --- */
                const int tile_oy = (int)(item.get_group(0) * MV2_WG_Y) - MV2_HALF_FW;
                const int tile_ox = (int)(item.get_group(1) * MV2_WG_X) - MV2_HALF_FW;
                constexpr unsigned tile_elems = MV2_TILE_H * MV2_TILE_W;
                constexpr unsigned WG_SIZE = MV2_WG_X * MV2_WG_Y;

                auto read_pixel = [&](const void *plane, int y, int x) -> int32_t {
                    if (e_bpc <= 8) {
                        return (int32_t)static_cast<const uint8_t *>(
                            plane)[(size_t)y * e_w + (size_t)x];
                    }
                    return (int32_t)static_cast<const uint16_t *>(
                        plane)[(size_t)y * e_w + (size_t)x];
                };

                const bool interior = (tile_oy >= 0) && (tile_oy + MV2_TILE_H <= (int)e_h) &&
                                      (tile_ox >= 0) && (tile_ox + MV2_TILE_W <= (int)e_w);

                for (unsigned i = lid; i < tile_elems; i += WG_SIZE) {
                    const unsigned tr = i / MV2_TILE_W;
                    const unsigned tc = i % MV2_TILE_W;
                    int py = tile_oy + (int)tr;
                    int px = tile_ox + (int)tc;
                    if (!interior) {
                        py = dev_mirror_mv2(py, (int)e_h);
                        px = dev_mirror_mv2(px, (int)e_w);
                    }
                    s_diff[tr][tc] = read_pixel(e_prev, py, px) - read_pixel(e_cur, py, px);
                }
                item.barrier(sycl::access::fence_space::local_space);

                /* --- Phase 2: separable V→H 5-tap filter on the diff --- */
                int64_t abs_h = 0;
                if (valid) {
                    const int64_t round_y = (int64_t)1 << ((int)e_bpc - 1);
                    const int shift_y = (int)e_bpc;
                    int32_t vtmp[5];

#pragma unroll
                    for (int hx = 0; hx < 5; hx++) {
                        const unsigned tcol = lx + (unsigned)hx;
                        /* int64 accumulator: at bpc=16 the diff range
                         * is ±65535 and filter[k] up to 26386 — the
                         * 5-tap sum overflows int32. */
                        int64_t vsum =
                            (int64_t)MV2_FILTER[0] * (s_diff[ly + 0][tcol] + s_diff[ly + 4][tcol]) +
                            (int64_t)MV2_FILTER[1] * (s_diff[ly + 1][tcol] + s_diff[ly + 3][tcol]) +
                            (int64_t)MV2_FILTER[2] * s_diff[ly + 2][tcol];
                        vtmp[hx] = (int32_t)((vsum + round_y) >> shift_y);
                    }

                    const int64_t hsum = (int64_t)MV2_FILTER[0] * (vtmp[0] + vtmp[4]) +
                                         (int64_t)MV2_FILTER[1] * (vtmp[1] + vtmp[3]) +
                                         (int64_t)MV2_FILTER[2] * vtmp[2];
                    const int64_t blurred = (hsum + 32768) >> 16;
                    abs_h = blurred < 0 ? -blurred : blurred;
                }

                /* --- Phase 3: subgroup + cross-subgroup SAD reduction --- */
                sycl::sub_group sg = item.get_sub_group();
                const int64_t sg_sum = sycl::reduce_over_group(sg, abs_h, sycl::plus<int64_t>{});
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
                    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        ref(*sad_accum);
                    ref.fetch_add(total);
                }
            });
    });
}

template <typename T>
static void copy_y_plane(const VmafPicture *pic, void *dst, unsigned w, unsigned h)
{
    const T *src = static_cast<const T *>(pic->data[0]);
    T *out = static_cast<T *>(dst);
    const ptrdiff_t src_stride_t = pic->stride[0] / static_cast<ptrdiff_t>(sizeof(T));
    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++)
            out[j] = src[j];
        src += src_stride_t;
        out += w;
    }
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_motion_v2_sycl[] = {
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = (int)offsetof(MotionV2StateSycl, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val = {.d = 1.0},
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<MotionV2StateSycl *>(fex->priv);

    /* The 5-tap SYCL motion_v2 kernel uses reflect-101 mirror padding;
     * dev_mirror_mv2() returns 2*sup - idx - 2, which is negative when sup < 3.
     * Refuse smaller frames up front to prevent out-of-bounds device reads.
     * Minimum: filter_width/2 + 1 = 3. */
    if (h < 3u || w < 3u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "motion_v2_sycl: frame %ux%u is below the 5-tap filter minimum 3x3; "
                 "refusing to avoid out-of-bounds mirror reads on device\n",
                 w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->has_pending = false;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_v2_sycl: no SYCL state\n");
        return -EINVAL;
    }
    VmafSyclState *state = fex->sycl_state;
    s->sycl_state = state;

    s->plane_bytes = (size_t)w * h * (bpc <= 8 ? 1u : 2u);

    s->h_pix = vmaf_sycl_malloc_host(state, s->plane_bytes);
    s->d_pix[0] = vmaf_sycl_malloc_device(state, s->plane_bytes);
    s->d_pix[1] = vmaf_sycl_malloc_device(state, s->plane_bytes);
    s->d_sad = static_cast<int64_t *>(vmaf_sycl_malloc_device(state, sizeof(int64_t)));
    s->h_sad = static_cast<int64_t *>(vmaf_sycl_malloc_host(state, sizeof(int64_t)));

    if (!s->h_pix || !s->d_pix[0] || !s->d_pix[1] || !s->d_sad || !s->h_sad) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_v2_sycl: USM allocation failed\n");
        return -ENOMEM;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    auto *s = static_cast<MotionV2StateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    /* Pack cur ref Y into pinned host staging (handles arbitrary
     * pic stride), then upload to d_pix[index%2]. */
    if (s->bpc <= 8)
        copy_y_plane<uint8_t>(ref_pic, s->h_pix, s->width, s->height);
    else
        copy_y_plane<uint16_t>(ref_pic, s->h_pix, s->width, s->height);

    const unsigned cur_idx = index % 2u;
    q.memcpy(s->d_pix[cur_idx], s->h_pix, s->plane_bytes);

    if (index > 0) {
        const unsigned prev_idx = (index + 1u) % 2u;
        q.memset(s->d_sad, 0, sizeof(int64_t));
        launch_motion_v2(q, s->d_pix[prev_idx], s->d_pix[cur_idx], s->d_sad, s->width, s->height,
                         s->bpc);
        q.memcpy(s->h_sad, s->d_sad, sizeof(int64_t));
    }

    s->pending_index = index;
    s->has_pending = true;
    s->frame_index = index + 1u;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<MotionV2StateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    qptr->wait();

    if (index == 0) {
        return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "VMAF_integer_feature_motion_v2_sad_score",
                                                       0.0, index);
    }

    const double sad_score = (double)*s->h_sad / 256.0 / ((double)s->width * (double)s->height);
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_motion_v2_sad_score",
                                                   sad_score, index);
}

static int flush_fex_sycl(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<MotionV2StateSycl *>(fex->priv);

    unsigned n_frames = 0;
    double dummy;
    while (!vmaf_feature_collector_get_score(
        feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &dummy, n_frames))
        n_frames++;

    if (n_frames < 2)
        return 1;

    for (unsigned i = 0; i < n_frames; i++) {
        double score_cur;
        double score_next;
        vmaf_feature_collector_get_score(feature_collector,
                                         "VMAF_integer_feature_motion_v2_sad_score", &score_cur, i);
        /* Apply fps weight — mirrors CPU integer_motion_v2.c flush logic.
         * Bit-exact when motion_fps_weight = 1.0 (default). */
        score_cur *= s->motion_fps_weight;

        double motion2;
        if (i + 1 < n_frames) {
            vmaf_feature_collector_get_score(
                feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &score_next, i + 1);
            score_next *= s->motion_fps_weight;
            motion2 = score_cur < score_next ? score_cur : score_next;
        } else {
            motion2 = score_cur;
        }

        const double motion2_weighted = motion2 * s->motion_fps_weight;
        vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion2_v2_score",
                                      motion2_weighted, i);
    }
    return 1;
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<MotionV2StateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->h_pix)
            vmaf_sycl_free(s->sycl_state, s->h_pix);
        if (s->d_pix[0])
            vmaf_sycl_free(s->sycl_state, s->d_pix[0]);
        if (s->d_pix[1])
            vmaf_sycl_free(s->sycl_state, s->d_pix[1]);
        if (s->d_sad)
            vmaf_sycl_free(s->sycl_state, s->d_sad);
        if (s->h_sad)
            vmaf_sycl_free(s->sycl_state, s->h_sad);
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_motion_v2_sycl[] = {
    "VMAF_integer_feature_motion_v2_sad_score", "VMAF_integer_feature_motion2_v2_score", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_integer_motion_v2_sycl = {
    .name = "motion_v2_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = flush_fex_sycl,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_motion_v2_sycl,
    .priv_size = sizeof(MotionV2StateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_motion_v2_sycl,
};

} /* extern "C" */
