/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#include <errno.h>
#include <math.h>
#include <stddef.h>

#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "log.h"
#include "opt.h"

#include "mem.h"
#include "ms_ssim.h"
#include "picture_copy.h"
#include "iqa/ssim_simd.h"
#include "iqa/ssim_tools.h"

#if ARCH_X86
#include "x86/ssim_avx2.h"
#include "x86/convolve_avx2.h"
#if HAVE_AVX512
#include "x86/ssim_avx512.h"
#include "x86/convolve_avx512.h"
#endif
#endif

#if ARCH_AARCH64
#include "arm64/convolve_neon.h"
#include "arm64/ssim_neon.h"
#endif

typedef struct MsSsimState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    bool enable_chroma;
    double max_db;
} MsSsimState;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(MsSsimState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write MS-SSIM values as dB",
        .offset = offsetof(MsSsimState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(MsSsimState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_chroma",
        .help = "enable calculation for chroma channels",
        .offset = offsetof(MsSsimState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0}};

/* Wire the SSIM + convolve SIMD dispatch tables for the host ISA. */
static void ms_ssim_init_simd_dispatch(void)
{
#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        iqa_ssim_set_dispatch(ssim_precompute_avx2, ssim_variance_avx2, ssim_accumulate_avx2);
        iqa_convolve_set_dispatch(iqa_convolve_avx2);
    }
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        iqa_ssim_set_dispatch(ssim_precompute_avx512, ssim_variance_avx512, ssim_accumulate_avx512);
        iqa_convolve_set_dispatch(iqa_convolve_avx512);
    }
#endif
#elif ARCH_AARCH64
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_ARM_CPU_FLAG_NEON) {
        iqa_ssim_set_dispatch(ssim_precompute_neon, ssim_variance_neon, ssim_accumulate_neon);
        iqa_convolve_set_dispatch(iqa_convolve_neon);
    }
#endif
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    MsSsimState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;

    /* The 5-level MS-SSIM pyramid with an 11-tap Gaussian needs
     * every scale to satisfy min(w, h) >= 11. Starting from w, h
     * and dividing by 2 per scale, the minimum admissible input
     * dimension is GAUSSIAN_LEN << (SCALES - 1) = 11 << 4 = 176.
     * Below that the pyramid walks off the kernel footprint at a
     * mid-level scale and `ms_ssim_check_scale_ok` fails mid-run
     * with a confusing "scale below 1x1!" print. Reject cleanly
     * at init with a helpful message. Netflix#1414 / ADR-0153. */
    const unsigned min_dim = GAUSSIAN_LEN << (SCALES - 1);
    if (w < min_dim || h < min_dim) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "%s: input resolution %ux%u is too small; the %d-level "
                 "%d-tap MS-SSIM pyramid requires at least %ux%u (Netflix#1414)\n",
                 fex->name, w, h, SCALES, GAUSSIAN_LEN, min_dim, min_dim);
        return -EINVAL;
    }

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    ms_ssim_init_simd_dispatch();

    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref)
        goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist)
        goto free_ref;

    return 0;

free_ref:
    free(s->ref);
fail:
    return -ENOMEM;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static const char *const ms_ssim_feature_names[3] = {
    "float_ms_ssim",
    "float_ms_ssim_cb",
    "float_ms_ssim_cr",
};

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    MsSsimState *s = fex->priv;
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    const unsigned n_planes = s->enable_chroma ? 3 : 1;

    for (unsigned p = 0; p < n_planes; p++) {
        const size_t plane_float_stride = ALIGN_CEIL(ref_pic->w[p] * sizeof(float));
        picture_copy(s->ref, plane_float_stride, ref_pic, 0, ref_pic->bpc, p);
        picture_copy(s->dist, plane_float_stride, dist_pic, 0, dist_pic->bpc, p);

        double score;
        double l_scores[5];
        double c_scores[5];
        double s_scores[5];
        err = compute_ms_ssim(s->ref, s->dist, ref_pic->w[p], ref_pic->h[p], plane_float_stride,
                              plane_float_stride, &score, l_scores, c_scores, s_scores);
        if (err)
            return err;

        if (s->enable_db)
            score = convert_to_db(score, s->max_db);

        err = vmaf_feature_collector_append(feature_collector, ms_ssim_feature_names[p], score,
                                            index);
        if (p == 0 && s->enable_lcs) {
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_l_scale0",
                                                 l_scores[0], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_l_scale1",
                                                 l_scores[1], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_l_scale2",
                                                 l_scores[2], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_l_scale3",
                                                 l_scores[3], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_l_scale4",
                                                 l_scores[4], index);

            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_c_scale0",
                                                 c_scores[0], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_c_scale1",
                                                 c_scores[1], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_c_scale2",
                                                 c_scores[2], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_c_scale3",
                                                 c_scores[3], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_c_scale4",
                                                 c_scores[4], index);

            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_s_scale0",
                                                 s_scores[0], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_s_scale1",
                                                 s_scores[1], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_s_scale2",
                                                 s_scores[2], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_s_scale3",
                                                 s_scores[3], index);
            err |= vmaf_feature_collector_append(feature_collector, "float_ms_ssim_s_scale4",
                                                 s_scores[4], index);
        }
        if (err)
            return err;
    }

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    MsSsimState *s = fex->priv;
    if (s->ref)
        aligned_free(s->ref);
    if (s->dist)
        aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "float_ms_ssim",
    "float_ms_ssim_cb",
    "float_ms_ssim_cr",
    NULL,
};

// extern-registered in feature_extractor.c registry
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ms_ssim = {
    .name = "float_ms_ssim",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(MsSsimState),
    .provided_features = provided_features,
};
