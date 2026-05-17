/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ms_ssim feature extractor on the SYCL backend
 *  (T7-23 / ADR-0188 / ADR-0190, GPU long-tail batch 2 part 2c).
 *  SYCL twin of ms_ssim_vulkan (PR #141) and ms_ssim_cuda (this
 *  PR's batch 2 part 2b).
 *
 *  Self-contained submit / collect — does *not* register with
 *  vmaf_sycl_graph_register because shared_frame is luma-only
 *  packed at uint width and MS-SSIM needs picture_copy-normalised
 *  float planes + a 5-level pyramid. Same pattern as ssim_sycl
 *  (PR #140).
 *
 *  5-level pyramid + 3-output SSIM per scale + host-side Wang
 *  product combine. Three SYCL kernels:
 *    - decimate (9-tap 9/7 biorthogonal LPF + 2× downsample)
 *    - horiz (11-tap separable Gaussian over 5 stats)
 *    - vert+lcs (vertical 11-tap + per-pixel l/c/s + per-WG
 *      partials × 3 via reduce_over_group)
 *
 *  fp64-free (Intel Arc A380 lacks native fp64 — same constraint
 *  as ssim_sycl).
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

static constexpr int MS_SSIM_SCALES = 5;
static constexpr int MS_SSIM_GAUSSIAN_LEN = 11;
static constexpr int MS_SSIM_K = 11;
static constexpr int LPF_LEN = 9;
static constexpr int LPF_HALF = 4;
static constexpr size_t WG_X = 16;
static constexpr size_t WG_Y = 8;

static constexpr float G[MS_SSIM_K] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f, 0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

static constexpr float LPF[LPF_LEN] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f,
    0.266846f, -0.078201f, -0.016828f, 0.026727f,
};

static constexpr float ALPHAS[MS_SSIM_SCALES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.1333f};
static constexpr float BETAS[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static constexpr float GAMMAS[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

struct MsSsimStateSycl {
    unsigned width;
    unsigned height;
    unsigned bpc;

    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    unsigned scale_w_horiz[MS_SSIM_SCALES];
    unsigned scale_h_horiz[MS_SSIM_SCALES];
    unsigned scale_w_final[MS_SSIM_SCALES];
    unsigned scale_h_final[MS_SSIM_SCALES];
    unsigned scale_wg_count_x[MS_SSIM_SCALES];
    unsigned scale_wg_count_y[MS_SSIM_SCALES];
    unsigned scale_wg_count[MS_SSIM_SCALES];

    float c1, c2, c3;

    VmafSyclState *sycl_state;

    /* Host pinned float planes for picture_copy upload. */
    float *h_ref;
    float *h_cmp;
    /* Pyramid: 5 ref + 5 cmp device USM buffers. */
    float *d_pyramid_ref[MS_SSIM_SCALES];
    float *d_pyramid_cmp[MS_SSIM_SCALES];
    /* SSIM intermediates sized for scale 0. */
    float *d_h_ref_mu;
    float *d_h_cmp_mu;
    float *d_h_ref_sq;
    float *d_h_cmp_sq;
    float *d_h_refcmp;
    /* 3 partials buffers + host pinned. */
    float *d_l_partials;
    float *d_c_partials;
    float *d_s_partials;
    float *h_l_partials;
    float *h_c_partials;
    float *h_s_partials;

    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

/* Period-2n mirror — matches ms_ssim_decimate.c::ms_ssim_decimate_mirror. */
static inline int mirror_idx(int idx, int n)
{
    int period = 2 * n;
    int r = idx % period;
    if (r < 0)
        r += period;
    if (r >= n)
        r = period - r - 1;
    return r;
}

static void launch_decimate(sycl::queue &q, const float *src, float *dst, unsigned w, unsigned h,
                            unsigned w_out, unsigned h_out)
{
    sycl::range<2> global{(size_t)h_out, (size_t)w_out};
    const unsigned e_w = w;
    const unsigned e_h = h;
    const unsigned e_w_out = w_out;
    const unsigned e_h_out = h_out;
    const float *e_src = src;
    float *e_dst = dst;

    q.submit([=](sycl::handler &h_) {
        h_.parallel_for(global, [=](sycl::id<2> id) {
            const size_t y_out = id[0];
            const size_t x_out = id[1];
            if (x_out >= (size_t)e_w_out || y_out >= (size_t)e_h_out)
                return;
            int x_src = (int)x_out * 2;
            int y_src = (int)y_out * 2;
            float acc = 0.0f;
            for (int kv = 0; kv < LPF_LEN; ++kv) {
                int yi = mirror_idx(y_src + kv - LPF_HALF, (int)e_h);
                float row_acc = 0.0f;
                for (int ku = 0; ku < LPF_LEN; ++ku) {
                    int xi = mirror_idx(x_src + ku - LPF_HALF, (int)e_w);
                    row_acc += e_src[yi * (int)e_w + xi] * LPF[ku];
                }
                acc += row_acc * LPF[kv];
            }
            e_dst[y_out * (size_t)e_w_out + x_out] = acc;
        });
    });
}

static void launch_horiz(sycl::queue &q, const float *ref, const float *cmp, float *h_ref_mu,
                         float *h_cmp_mu, float *h_ref_sq, float *h_cmp_sq, float *h_refcmp,
                         unsigned width, unsigned w_horiz, unsigned h_horiz)
{
    sycl::range<2> global{(size_t)h_horiz, (size_t)w_horiz};
    const unsigned e_w = width;
    const unsigned e_w_horiz = w_horiz;
    const unsigned e_h_horiz = h_horiz;
    const float *e_ref = ref;
    const float *e_cmp = cmp;
    float *e_h_ref_mu = h_ref_mu;
    float *e_h_cmp_mu = h_cmp_mu;
    float *e_h_ref_sq = h_ref_sq;
    float *e_h_cmp_sq = h_cmp_sq;
    float *e_h_refcmp = h_refcmp;

    q.submit([=](sycl::handler &h_) {
        h_.parallel_for(global, [=](sycl::id<2> id) {
            const size_t y = id[0];
            const size_t x = id[1];
            if (x >= (size_t)e_w_horiz || y >= (size_t)e_h_horiz)
                return;
            float ref_mu = 0.0f, cmp_mu = 0.0f, ref_sq = 0.0f, cmp_sq = 0.0f, refcmp = 0.0f;
            for (int u = 0; u < MS_SSIM_K; ++u) {
                const size_t src_idx = y * (size_t)e_w + (x + (size_t)u);
                const float r = e_ref[src_idx];
                const float c = e_cmp[src_idx];
                const float w = G[u];
                ref_mu += w * r;
                cmp_mu += w * c;
                ref_sq += w * (r * r);
                cmp_sq += w * (c * c);
                refcmp += w * (r * c);
            }
            const size_t dst_idx = y * (size_t)e_w_horiz + x;
            e_h_ref_mu[dst_idx] = ref_mu;
            e_h_cmp_mu[dst_idx] = cmp_mu;
            e_h_ref_sq[dst_idx] = ref_sq;
            e_h_cmp_sq[dst_idx] = cmp_sq;
            e_h_refcmp[dst_idx] = refcmp;
        });
    });
}

static void launch_vert_lcs(sycl::queue &q, const float *h_ref_mu, const float *h_cmp_mu,
                            const float *h_ref_sq, const float *h_cmp_sq, const float *h_refcmp,
                            float *l_partials, float *c_partials, float *s_partials,
                            unsigned w_horiz, unsigned w_final, unsigned h_final, float c1,
                            float c2, float c3)
{
    const size_t global_x = ((w_final + WG_X - 1) / WG_X) * WG_X;
    const size_t global_y = ((h_final + WG_Y - 1) / WG_Y) * WG_Y;
    const size_t wg_count_x = global_x / WG_X;
    sycl::nd_range<2> ndr{sycl::range<2>{global_y, global_x}, sycl::range<2>{WG_Y, WG_X}};
    const unsigned e_w_horiz = w_horiz;
    const unsigned e_w_final = w_final;
    const unsigned e_h_final = h_final;
    const float e_c1 = c1, e_c2 = c2, e_c3 = c3;
    const size_t e_wg_count_x = wg_count_x;
    const float *e_h_ref_mu = h_ref_mu;
    const float *e_h_cmp_mu = h_cmp_mu;
    const float *e_h_ref_sq = h_ref_sq;
    const float *e_h_cmp_sq = h_cmp_sq;
    const float *e_h_refcmp = h_refcmp;
    float *e_l = l_partials;
    float *e_c = c_partials;
    float *e_s = s_partials;

    q.submit([=](sycl::handler &h_) {
        h_.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const size_t x = it.get_global_id(1);
            const size_t y = it.get_global_id(0);
            float my_l = 0.0f, my_c = 0.0f, my_s = 0.0f;
            if (x < (size_t)e_w_final && y < (size_t)e_h_final) {
                float ref_mu = 0.0f, cmp_mu = 0.0f, ref_sq = 0.0f, cmp_sq = 0.0f, refcmp = 0.0f;
                for (int v = 0; v < MS_SSIM_K; ++v) {
                    const size_t src_idx = (y + (size_t)v) * (size_t)e_w_horiz + x;
                    const float w = G[v];
                    ref_mu += w * e_h_ref_mu[src_idx];
                    cmp_mu += w * e_h_cmp_mu[src_idx];
                    ref_sq += w * e_h_ref_sq[src_idx];
                    cmp_sq += w * e_h_cmp_sq[src_idx];
                    refcmp += w * e_h_refcmp[src_idx];
                }
                /* Clamp σ² ≥ 0 before sqrt — matches MAX(0, ...)
                 * in iqa/ssim_tools.c::ssim_variance_scalar. */
                const float ref_var = sycl::fmax(ref_sq - ref_mu * ref_mu, 0.0f);
                const float cmp_var = sycl::fmax(cmp_sq - cmp_mu * cmp_mu, 0.0f);
                const float covar = refcmp - ref_mu * cmp_mu;
                const float sigma_xy_geom = sycl::sqrt(ref_var * cmp_var);
                const float clamped_covar = (covar < 0.0f && sigma_xy_geom <= 0.0f) ? 0.0f : covar;

                my_l = (2.0f * ref_mu * cmp_mu + e_c1) / (ref_mu * ref_mu + cmp_mu * cmp_mu + e_c1);
                my_c = (2.0f * sigma_xy_geom + e_c2) / (ref_var + cmp_var + e_c2);
                my_s = (clamped_covar + e_c3) / (sigma_xy_geom + e_c3);
            }

            float wg_l = sycl::reduce_over_group(it.get_group(), my_l, sycl::plus<float>{});
            float wg_c = sycl::reduce_over_group(it.get_group(), my_c, sycl::plus<float>{});
            float wg_s = sycl::reduce_over_group(it.get_group(), my_s, sycl::plus<float>{});
            if (it.get_local_id(0) == 0 && it.get_local_id(1) == 0) {
                const size_t wg_idx = it.get_group(0) * e_wg_count_x + it.get_group(1);
                e_l[wg_idx] = wg_l;
                e_c[wg_idx] = wg_c;
                e_s[wg_idx] = wg_s;
            }
        });
    });
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_ms_ssim_sycl[] = {{0}};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<MsSsimStateSycl *>(fex->priv);

    const unsigned min_dim = (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1);
    if (w < min_dim || h < min_dim) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ms_ssim_sycl: input %ux%u too small; %d-level %d-tap MS-SSIM pyramid"
                 " needs >= %ux%u (Netflix#1414 / ADR-0153)\n",
                 w, h, MS_SSIM_SCALES, MS_SSIM_GAUSSIAN_LEN, min_dim, min_dim);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2) + (s->scale_w[i - 1] & 1);
        s->scale_h[i] = (s->scale_h[i - 1] / 2) + (s->scale_h[i - 1] & 1);
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        s->scale_w_horiz[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_horiz[i] = s->scale_h[i];
        s->scale_w_final[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_final[i] = s->scale_h[i] - (MS_SSIM_K - 1);
        s->scale_wg_count_x[i] = (s->scale_w_final[i] + (unsigned)WG_X - 1) / (unsigned)WG_X;
        s->scale_wg_count_y[i] = (s->scale_h_final[i] + (unsigned)WG_Y - 1) / (unsigned)WG_Y;
        s->scale_wg_count[i] = s->scale_wg_count_x[i] * s->scale_wg_count_y[i];
    }

    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);
    s->c3 = s->c2 * 0.5f;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ms_ssim_sycl: no SYCL state\n");
        return -EINVAL;
    }
    s->sycl_state = fex->sycl_state;

    const size_t input_bytes = (size_t)w * h * sizeof(float);
    s->h_ref = (float *)vmaf_sycl_malloc_host(s->sycl_state, input_bytes);
    s->h_cmp = (float *)vmaf_sycl_malloc_host(s->sycl_state, input_bytes);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const size_t plane_bytes = (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
        s->d_pyramid_ref[i] = (float *)vmaf_sycl_malloc_device(s->sycl_state, plane_bytes);
        s->d_pyramid_cmp[i] = (float *)vmaf_sycl_malloc_device(s->sycl_state, plane_bytes);
    }
    const size_t horiz_bytes_max =
        (size_t)s->scale_w_horiz[0] * s->scale_h_horiz[0] * sizeof(float);
    s->d_h_ref_mu = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes_max);
    s->d_h_cmp_mu = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes_max);
    s->d_h_ref_sq = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes_max);
    s->d_h_cmp_sq = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes_max);
    s->d_h_refcmp = (float *)vmaf_sycl_malloc_device(s->sycl_state, horiz_bytes_max);
    const size_t partials_bytes_max = (size_t)s->scale_wg_count[0] * sizeof(float);
    s->d_l_partials = (float *)vmaf_sycl_malloc_device(s->sycl_state, partials_bytes_max);
    s->d_c_partials = (float *)vmaf_sycl_malloc_device(s->sycl_state, partials_bytes_max);
    s->d_s_partials = (float *)vmaf_sycl_malloc_device(s->sycl_state, partials_bytes_max);
    s->h_l_partials = (float *)vmaf_sycl_malloc_host(s->sycl_state, partials_bytes_max);
    s->h_c_partials = (float *)vmaf_sycl_malloc_host(s->sycl_state, partials_bytes_max);
    s->h_s_partials = (float *)vmaf_sycl_malloc_host(s->sycl_state, partials_bytes_max);

    if (!s->h_ref || !s->h_cmp || !s->d_h_ref_mu || !s->d_h_cmp_mu || !s->d_h_ref_sq ||
        !s->d_h_cmp_sq || !s->d_h_refcmp || !s->d_l_partials || !s->d_c_partials ||
        !s->d_s_partials || !s->h_l_partials || !s->h_c_partials || !s->h_s_partials) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ms_ssim_sycl: USM allocation failed\n");
        return -ENOMEM;
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (!s->d_pyramid_ref[i] || !s->d_pyramid_cmp[i])
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
    auto *s = static_cast<MsSsimStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    /* picture_copy host-side → upload to pyramid level 0. */
    picture_copy(s->h_ref, (ptrdiff_t)((size_t)s->width * sizeof(float)), ref_pic, 0, ref_pic->bpc,
                 0);
    picture_copy(s->h_cmp, (ptrdiff_t)((size_t)s->width * sizeof(float)), dist_pic, 0,
                 dist_pic->bpc, 0);

    const size_t input_bytes = (size_t)s->width * s->height * sizeof(float);
    q.memcpy(s->d_pyramid_ref[0], s->h_ref, input_bytes);
    q.memcpy(s->d_pyramid_cmp[0], s->h_cmp, input_bytes);

    /* Build pyramid scales 1..4. */
    for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
        launch_decimate(q, s->d_pyramid_ref[i], s->d_pyramid_ref[i + 1], s->scale_w[i],
                        s->scale_h[i], s->scale_w[i + 1], s->scale_h[i + 1]);
        launch_decimate(q, s->d_pyramid_cmp[i], s->d_pyramid_cmp[i + 1], s->scale_w[i],
                        s->scale_h[i], s->scale_w[i + 1], s->scale_h[i + 1]);
    }

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<MsSsimStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr)
        return -EINVAL;
    sycl::queue &q = *qptr;

    /* Per-scale SSIM compute + readback. The intermediates are
     * shared so scales must run sequentially with q.wait()
     * between them so the host readback gets fresh partials. */
    double l_means[MS_SSIM_SCALES] = {0}, c_means[MS_SSIM_SCALES] = {0},
           s_means[MS_SSIM_SCALES] = {0};
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        launch_horiz(q, s->d_pyramid_ref[i], s->d_pyramid_cmp[i], s->d_h_ref_mu, s->d_h_cmp_mu,
                     s->d_h_ref_sq, s->d_h_cmp_sq, s->d_h_refcmp, s->scale_w[i],
                     s->scale_w_horiz[i], s->scale_h_horiz[i]);
        launch_vert_lcs(q, s->d_h_ref_mu, s->d_h_cmp_mu, s->d_h_ref_sq, s->d_h_cmp_sq,
                        s->d_h_refcmp, s->d_l_partials, s->d_c_partials, s->d_s_partials,
                        s->scale_w_horiz[i], s->scale_w_final[i], s->scale_h_final[i], s->c1, s->c2,
                        s->c3);
        const size_t partials_bytes = (size_t)s->scale_wg_count[i] * sizeof(float);
        q.memcpy(s->h_l_partials, s->d_l_partials, partials_bytes);
        q.memcpy(s->h_c_partials, s->d_c_partials, partials_bytes);
        q.memcpy(s->h_s_partials, s->d_s_partials, partials_bytes);
        q.wait();

        double total_l = 0.0, total_c = 0.0, total_s = 0.0;
        for (unsigned j = 0; j < s->scale_wg_count[i]; j++) {
            total_l += (double)s->h_l_partials[j];
            total_c += (double)s->h_c_partials[j];
            total_s += (double)s->h_s_partials[j];
        }
        const double n_pixels = (double)s->scale_w_final[i] * (double)s->scale_h_final[i];
        l_means[i] = total_l / n_pixels;
        c_means[i] = total_c / n_pixels;
        s_means[i] = total_s / n_pixels;
    }

    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        msssim *= std::pow(l_means[i], (double)ALPHAS[i]) * std::pow(c_means[i], (double)BETAS[i]) *
                  std::pow(std::fabs(s_means[i]), (double)GAMMAS[i]);
    }

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_ms_ssim", msssim, index);
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<MsSsimStateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->h_ref)
            vmaf_sycl_free(s->sycl_state, s->h_ref);
        if (s->h_cmp)
            vmaf_sycl_free(s->sycl_state, s->h_cmp);
        for (int i = 0; i < MS_SSIM_SCALES; i++) {
            if (s->d_pyramid_ref[i])
                vmaf_sycl_free(s->sycl_state, s->d_pyramid_ref[i]);
            if (s->d_pyramid_cmp[i])
                vmaf_sycl_free(s->sycl_state, s->d_pyramid_cmp[i]);
        }
        if (s->d_h_ref_mu)
            vmaf_sycl_free(s->sycl_state, s->d_h_ref_mu);
        if (s->d_h_cmp_mu)
            vmaf_sycl_free(s->sycl_state, s->d_h_cmp_mu);
        if (s->d_h_ref_sq)
            vmaf_sycl_free(s->sycl_state, s->d_h_ref_sq);
        if (s->d_h_cmp_sq)
            vmaf_sycl_free(s->sycl_state, s->d_h_cmp_sq);
        if (s->d_h_refcmp)
            vmaf_sycl_free(s->sycl_state, s->d_h_refcmp);
        if (s->d_l_partials)
            vmaf_sycl_free(s->sycl_state, s->d_l_partials);
        if (s->d_c_partials)
            vmaf_sycl_free(s->sycl_state, s->d_c_partials);
        if (s->d_s_partials)
            vmaf_sycl_free(s->sycl_state, s->d_s_partials);
        if (s->h_l_partials)
            vmaf_sycl_free(s->sycl_state, s->h_l_partials);
        if (s->h_c_partials)
            vmaf_sycl_free(s->sycl_state, s->h_c_partials);
        if (s->h_s_partials)
            vmaf_sycl_free(s->sycl_state, s->h_s_partials);
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_ms_ssim_sycl[] = {"float_ms_ssim", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_float_ms_ssim_sycl = {
    .name = "float_ms_ssim_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_ms_ssim_sycl,
    .priv_size = sizeof(MsSsimStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_ms_ssim_sycl,
    .chars =
        {
            .n_dispatches_per_frame = 18,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};

} /* extern "C" */
