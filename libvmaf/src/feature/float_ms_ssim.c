/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "ms_ssim.h"
#include "picture_copy.h"

typedef struct MsSsimState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
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
    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) pix_fmt;

    MsSsimState *s = fex->priv;

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto free_ref;

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

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MsSsimState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, 0, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, 0, dist_pic->bpc);

    double score, l_scores[5], c_scores[5], s_scores[5];
    err = compute_ms_ssim(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                          s->float_stride, s->float_stride,
                          &score, l_scores, c_scores, s_scores);
    if (err) return err;

    if (s->enable_db)
        score = convert_to_db(score, s->max_db);

    err = vmaf_feature_collector_append(feature_collector, "float_ms_ssim",
                                        score, index);
    if (s->enable_lcs) {
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_l_scale0", l_scores[0], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_l_scale1", l_scores[1], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_l_scale2", l_scores[2], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_l_scale3", l_scores[3], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_l_scale4", l_scores[4], index);

        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_c_scale0", c_scores[0], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_c_scale1", c_scores[1], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_c_scale2", c_scores[2], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_c_scale3", c_scores[3], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_c_scale4", c_scores[4], index);

        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_s_scale0", s_scores[0], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_s_scale1", s_scores[1], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_s_scale2", s_scores[2], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_s_scale3", s_scores[3], index);
        err |= vmaf_feature_collector_append(feature_collector,
                "float_ms_ssim_s_scale4", s_scores[4], index);
    }

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    MsSsimState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "float_ms_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ms_ssim = {
    .name = "float_ms_ssim",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(MsSsimState),
    .provided_features = provided_features,
};
