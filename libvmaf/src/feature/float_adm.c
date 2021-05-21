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
#include <string.h>
#include <stddef.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "adm.h"
#include "adm_options.h"
#include "mem.h"
#include "picture_copy.h"

typedef struct AdmState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
} AdmState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_enhn_gain_limit",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
    },
    {
        .name = "adm_norm_view_dist",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
    },
    {
        .name = "adm_ref_display_height",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
    },
    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmState *s = fex->priv;
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

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    AdmState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc);

    double score, score_num, score_den;
    double scores[8];
    err = compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR,
                      s->adm_enhn_gain_limit,
                      s->adm_norm_view_dist, s->adm_ref_display_height);
    if (err) return err;

    const char *key =
        s->adm_enhn_gain_limit != DEFAULT_ADM_ENHN_GAIN_LIMIT ?
        "adm_enhn_gain_limit" : NULL;
    const double val = s->adm_enhn_gain_limit;

    err = vmaf_feature_collector_append_templated(feature_collector,
                                                  "VMAF_feature_adm2_score",
                                                  key, val,
                                                  score, index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                "VMAF_feature_adm_scale0_score",
                                                key, val,
                                                scores[0] / scores[1], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                "VMAF_feature_adm_scale1_score",
                                                key, val,
                                                scores[2] / scores[3], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                "VMAF_feature_adm_scale2_score",
                                                key, val,
                                                scores[4] / scores[5], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                "VMAF_feature_adm_scale3_score",
                                                key, val,
                                                scores[6] / scores[7], index);

    if (!s->debug) return err;

    err |= vmaf_feature_collector_append_templated(feature_collector, "adm",
                                                   key, val, score, index);

    err |= vmaf_feature_collector_append_templated(feature_collector, "adm_num",
                                                   key, val, score_num, index);

    err |= vmaf_feature_collector_append_templated(feature_collector, "adm_den",
                                                   key, val, score_den, index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_num_scale0",
                                                    key, val, scores[0], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_den_scale0",
                                                   key, val, scores[1], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_num_scale1",
                                                   key, val, scores[2], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_den_scale1",
                                                   key, val, scores[3], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_num_scale2",
                                                   key, val, scores[4], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_den_scale2",
                                                   key, val, scores[5], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_num_scale3",
                                                   key, val, scores[6], index);

    err |= vmaf_feature_collector_append_templated(feature_collector,
                                                   "adm_den_scale3",
                                                   key, val, scores[7], index);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_feature_adm2_score",
    "VMAF_feature_adm2_score",
    "VMAF_feature_adm_scale0_score", "VMAF_feature_adm_scale1_score",
    "VMAF_feature_adm_scale2_score", "VMAF_feature_adm_scale3_score",
    "adm_scale0", "adm_scale1", "adm_scale2", "adm_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_adm = {
    .name = "float_adm",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
