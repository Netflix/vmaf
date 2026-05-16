/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the SYCL backend
 *  (T7-23 / ADR-0188 / ADR-0189, GPU long-tail batch 2 part 1c).
 *  SYCL twin of ssim_vulkan (PR #139) and ssim_cuda (this PR's
 *  batch 2 part 1b).
 *
 *  Self-contained submit / collect — does *not* register with
 *  vmaf_sycl_graph_register because shared_frame is luma-only
 *  packed at uint width and SSIM needs float [0, 255]
 *  intermediates with picture_copy normalisation. Same approach
 *  as ciede_sycl (PR #137).
 *
 *  Two-pass design mirrors ssim_vulkan / ssim_cuda:
 *    1. horizontal 11-tap separable Gaussian over ref / cmp /
 *       ref² / cmp² / ref·cmp into 5 device float buffers.
 *    2. nd_range vertical 11-tap + per-pixel SSIM combine +
 *       per-WG float partial sums via sycl::reduce_over_group.
 *
 *  Host accumulates partials in `double`, divides by
 *  (W-10)·(H-10) and emits `float_ssim`.
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan/cuda.
 *  fp64-free (Intel Arc A380 lacks native fp64).
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "../picture_copy.h"
#include "sycl/common.h"
}

namespace
{

static constexpr size_t SSIM_WG_X = 16;
static constexpr size_t SSIM_WG_Y = 8;
static constexpr int SSIM_K = 11;

/* Same 11-tap normalised Gaussian as the Vulkan + CUDA twins —
 * matches g_gaussian_window_h in iqa/ssim_tools.h byte-for-byte. */
static constexpr float G[SSIM_K] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f, 0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

struct SsimStateSycl {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    int scale_override;

    unsigned w_horiz;
    unsigned h_horiz;
    unsigned w_final;
    unsigned h_final;
    unsigned wg_count_x;
    unsigned wg_count_y;
    unsigned wg_count;

    float c1;
    float c2;

    /* SYCL state back-pointer. */
    VmafSyclState *sycl_state;

    /* Host-pinned float ref / cmp staging (post picture_copy). */
    float *h_ref;
    float *h_cmp;
    /* Device USM ref / cmp + 5 intermediates + WG partials. */
    float *d_ref;
    float *d_cmp;
    float *d_ref_mu;
    float *d_cmp_mu;
    float *d_ref_sq;
    float *d_cmp_sq;
    float *d_refcmp;
    float *d_partials;
    /* Host-pinned partials for D2H. */
    float *h_partials;

    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

static void launch_horiz(sycl::queue &q, const float *d_ref, const float *d_cmp, float *d_ref_mu,
                         float *d_cmp_mu, float *d_ref_sq, float *d_cmp_sq, float *d_refcmp,
                         unsigned width, unsigned w_horiz, unsigned h_horiz)
{
    sycl::range<2> global{(size_t)h_horiz, (size_t)w_horiz};
    const unsigned e_w = width;
    const unsigned e_w_horiz = w_horiz;
    const unsigned e_h_horiz = h_horiz;
    const float *e_ref = d_ref;
    const float *e_cmp = d_cmp;
    float *e_ref_mu = d_ref_mu;
    float *e_cmp_mu = d_cmp_mu;
    float *e_ref_sq = d_ref_sq;
    float *e_cmp_sq = d_cmp_sq;
    float *e_refcmp = d_refcmp;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(global, [=](sycl::id<2> id) {
            const size_t y = id[0];
            const size_t x = id[1];
            if (x >= (size_t)e_w_horiz || y >= (size_t)e_h_horiz)
                return;
            float ref_mu_h = 0.0f, cmp_mu_h = 0.0f;
            float ref_sq_h = 0.0f, cmp_sq_h = 0.0f, refcmp_h = 0.0f;
            for (int u = 0; u < SSIM_K; u++) {
                const size_t src_idx = y * (size_t)e_w + (x + (size_t)u);
                const float r = e_ref[src_idx];
                const float c = e_cmp[src_idx];
                const float w = G[u];
                ref_mu_h += w * r;
                cmp_mu_h += w * c;
                ref_sq_h += w * (r * r);
                cmp_sq_h += w * (c * c);
                refcmp_h += w * (r * c);
            }
            const size_t dst_idx = y * (size_t)e_w_horiz + x;
            e_ref_mu[dst_idx] = ref_mu_h;
            e_cmp_mu[dst_idx] = cmp_mu_h;
            e_ref_sq[dst_idx] = ref_sq_h;
            e_cmp_sq[dst_idx] = cmp_sq_h;
            e_refcmp[dst_idx] = refcmp_h;
        });
    });
}

static void launch_vert_combine(sycl::queue &q, const float *d_ref_mu, const float *d_cmp_mu,
                                const float *d_ref_sq, const float *d_cmp_sq, const float *d_refcmp,
                                float *d_partials, unsigned w_horiz, unsigned w_final,
                                unsigned h_final, float c1, float c2)
{
    /* Round work-item count up to WG-multiples. Per-WG sum
     * via sycl::reduce_over_group; one float per WG written
     * to partials. fp64-free (Arc A380). */
    const size_t global_x = ((w_final + SSIM_WG_X - 1) / SSIM_WG_X) * SSIM_WG_X;
    const size_t global_y = ((h_final + SSIM_WG_Y - 1) / SSIM_WG_Y) * SSIM_WG_Y;
    const size_t wg_count_x = global_x / SSIM_WG_X;
    sycl::nd_range<2> ndr{sycl::range<2>{global_y, global_x}, sycl::range<2>{SSIM_WG_Y, SSIM_WG_X}};
    const unsigned e_w_horiz = w_horiz;
    const unsigned e_w_final = w_final;
    const unsigned e_h_final = h_final;
    const float e_c1 = c1;
    const float e_c2 = c2;
    const size_t e_wg_count_x = wg_count_x;
    const float *e_ref_mu = d_ref_mu;
    const float *e_cmp_mu = d_cmp_mu;
    const float *e_ref_sq = d_ref_sq;
    const float *e_cmp_sq = d_cmp_sq;
    const float *e_refcmp = d_refcmp;
    float *e_partials = d_partials;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const size_t x = it.get_global_id(1);
            const size_t y = it.get_global_id(0);
            float my_ssim = 0.0f;
            if (x < (size_t)e_w_final && y < (size_t)e_h_final) {
                float ref_mu = 0.0f, cmp_mu = 0.0f;
                float ref_sq = 0.0f, cmp_sq = 0.0f, refcmp = 0.0f;
                for (int v = 0; v < SSIM_K; v++) {
                    const size_t src_idx = (y + (size_t)v) * (size_t)e_w_horiz + x;
                    const float w = G[v];
                    ref_mu += w * e_ref_mu[src_idx];
                    cmp_mu += w * e_cmp_mu[src_idx];
                    ref_sq += w * e_ref_sq[src_idx];
                    cmp_sq += w * e_cmp_sq[src_idx];
                    refcmp += w * e_refcmp[src_idx];
                }
                const float ref_var = ref_sq - ref_mu * ref_mu;
                const float cmp_var = cmp_sq - cmp_mu * cmp_mu;
                const float covar = refcmp - ref_mu * cmp_mu;
                const float mu_xy = ref_mu * cmp_mu;
                const float num = (2.0f * mu_xy + e_c1) * (2.0f * covar + e_c2);
                const float den =
                    (ref_mu * ref_mu + cmp_mu * cmp_mu + e_c1) * (ref_var + cmp_var + e_c2);
                my_ssim = num / den;
            }
            float wg_sum = sycl::reduce_over_group(it.get_group(), my_ssim, sycl::plus<float>{});
            if (it.get_local_id(0) == 0 && it.get_local_id(1) == 0) {
                const size_t wg_idx = it.get_group(0) * e_wg_count_x + it.get_group(1);
                e_partials[wg_idx] = wg_sum;
            }
        });
    });
}

} /* anonymous namespace */

extern "C" {

static int round_to_int(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}
static int min_int(int a, int b)
{
    return a < b ? a : b;
}
static int compute_scale(unsigned w, unsigned h, int override_)
{
    if (override_ > 0)
        return override_;
    int scaled = round_to_int((float)min_int((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static const VmafOption options_ssim_sycl[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimStateSycl, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<SsimStateSycl *>(fex->priv);

    int scale = compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_sycl: v1 supports scale=1 only (auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_sycl:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < (unsigned)SSIM_K || h < (unsigned)SSIM_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_sycl: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_K - 1);
    s->h_horiz = h;
    s->w_final = w - (SSIM_K - 1);
    s->h_final = h - (SSIM_K - 1);
    s->wg_count_x = (s->w_final + (unsigned)SSIM_WG_X - 1) / (unsigned)SSIM_WG_X;
    s->wg_count_y = (s->h_final + (unsigned)SSIM_WG_Y - 1) / (unsigned)SSIM_WG_Y;
    s->wg_count = s->wg_count_x * s->wg_count_y;
    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssim_sycl: no SYCL state\n");
        return -EINVAL;
    }
    s->sycl_state = fex->sycl_state;

    const size_t input_bytes = (size_t)w * h * sizeof(float);
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t partials_bytes = (size_t)s->wg_count * sizeof(float);

    s->h_ref = (float *)vmaf_sycl_malloc_host(s->sycl_state, input_bytes);
    s->h_cmp = (float *)vmaf_sycl_malloc_host(s->sycl_state, input_bytes);
    s->d_ref = (float *)vmaf_sycl_malloc_device(s->sycl_state, input_bytes);
    s->d_cmp = (float *)vmaf_sycl_malloc_device(s->sycl_state, input_bytes);
    s->d_ref_mu = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes);
    s->d_cmp_mu = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes);
    s->d_ref_sq = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes);
    s->d_cmp_sq = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes);
    s->d_refcmp = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes);
    s->d_partials = (float *)vmaf_sycl_malloc_device(s->sycl_state, partials_bytes);
    s->h_partials = (float *)vmaf_sycl_malloc_host(s->sycl_state, partials_bytes);
    if (!s->h_ref || !s->h_cmp || !s->d_ref || !s->d_cmp || !s->d_ref_mu || !s->d_cmp_mu ||
        !s->d_ref_sq || !s->d_cmp_sq || !s->d_refcmp || !s->d_partials || !s->h_partials) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssim_sycl: USM allocation failed\n");
        return -ENOMEM;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    s->has_pending = false;
    return 0;
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    auto *s = static_cast<SsimStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    /* Host-side picture_copy → uint sample → float [0, 255]
     * (matches CPU float_ssim.c::extract). The destination is
     * tightly packed at width*sizeof(float). */
    picture_copy(s->h_ref, (ptrdiff_t)((size_t)s->width * sizeof(float)), ref_pic, /*offset=*/0,
                 ref_pic->bpc, 0);
    picture_copy(s->h_cmp, (ptrdiff_t)((size_t)s->width * sizeof(float)), dist_pic, 0,
                 dist_pic->bpc, 0);

    const size_t input_bytes = (size_t)s->width * s->height * sizeof(float);
    q.memcpy(s->d_ref, s->h_ref, input_bytes);
    q.memcpy(s->d_cmp, s->h_cmp, input_bytes);

    launch_horiz(q, s->d_ref, s->d_cmp, s->d_ref_mu, s->d_cmp_mu, s->d_ref_sq, s->d_cmp_sq,
                 s->d_refcmp, s->width, s->w_horiz, s->h_horiz);
    launch_vert_combine(q, s->d_ref_mu, s->d_cmp_mu, s->d_ref_sq, s->d_cmp_sq, s->d_refcmp,
                        s->d_partials, s->w_horiz, s->w_final, s->h_final, s->c1, s->c2);

    q.memcpy(s->h_partials, s->d_partials, (size_t)s->wg_count * sizeof(float));

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<SsimStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    qptr->wait();

    /* Per-WG float partials → host double sum → mean SSIM
     * over (W-10)·(H-10) pixels. Same precision pattern as
     * ssim_vulkan / ssim_cuda. */
    double total = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += (double)s->h_partials[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_ssim", score, index);
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<SsimStateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->h_ref)
            vmaf_sycl_free(s->sycl_state, s->h_ref);
        if (s->h_cmp)
            vmaf_sycl_free(s->sycl_state, s->h_cmp);
        if (s->d_ref)
            vmaf_sycl_free(s->sycl_state, s->d_ref);
        if (s->d_cmp)
            vmaf_sycl_free(s->sycl_state, s->d_cmp);
        if (s->d_ref_mu)
            vmaf_sycl_free(s->sycl_state, s->d_ref_mu);
        if (s->d_cmp_mu)
            vmaf_sycl_free(s->sycl_state, s->d_cmp_mu);
        if (s->d_ref_sq)
            vmaf_sycl_free(s->sycl_state, s->d_ref_sq);
        if (s->d_cmp_sq)
            vmaf_sycl_free(s->sycl_state, s->d_cmp_sq);
        if (s->d_refcmp)
            vmaf_sycl_free(s->sycl_state, s->d_refcmp);
        if (s->d_partials)
            vmaf_sycl_free(s->sycl_state, s->d_partials);
        if (s->h_partials)
            vmaf_sycl_free(s->sycl_state, s->h_partials);
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_ssim_sycl[] = {"float_ssim", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_float_ssim_sycl = {
    .name = "float_ssim_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_ssim_sycl,
    .priv_size = sizeof(SsimStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_ssim_sycl,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};

} /* extern "C" */
