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
#include "mem.h"

#include "vif.h"
#include "vif_options.h"
#include "picture_copy.h"

typedef struct VifState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    double vif_enhn_gain_limit;
} VifState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
    },
    { NULL }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    VifState *s = fex->priv;
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
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifState *s = fex->priv;
    int err = 0;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc);

    double score, score_num, score_den;
    double scores[8];
    err = compute_vif(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride,
                      &score, &score_num, &score_den, scores,
                      s->vif_enhn_gain_limit);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_vif_scale0_score'",
                                        scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale1_score'",
                                         scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale2_score'",
                                         scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append(feature_collector,
                                         "'VMAF_feature_vif_scale3_score'",
                                         scores[6] / scores[7], index);

    if (s->debug) {
        err |= vmaf_feature_collector_append(feature_collector,
                                             "vif", score, index);
        err |= vmaf_feature_collector_append(feature_collector,
                                             "vif_num", score_num, index);
        err |= vmaf_feature_collector_append(feature_collector,
                                             "vif_den", score_den, index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_num_scale0", scores[0], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_den_scale0", scores[1], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_num_scale1", scores[2], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_den_scale1", scores[3], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_num_scale2", scores[4], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_den_scale2", scores[5], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_num_scale3", scores[6], index);
        err |= vmaf_feature_collector_append(feature_collector,
                                            "vif_den_scale3", scores[7], index);
    }

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    VifState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_vif_scale0_score'", "'VMAF_feature_vif_scale1_score'",
    "'VMAF_feature_vif_scale2_score'", "'VMAF_feature_vif_scale3_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_vif = {
    .name = "float_vif",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(VifState),
    .provided_features = provided_features,
};
