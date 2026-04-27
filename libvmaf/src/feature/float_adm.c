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

#include "dict.h"
#include "feature_collector.h"
#include "framesync.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "adm.h"
#include "adm_options.h"
#include "mem.h"
#include "picture_copy.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

typedef struct AdmState {
    size_t float_stride;
    float *ref;
    float *dist;
    bool debug;
    bool adm_skip_scale0;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    double adm_noise_weight;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_min_val;
    int adm_ref_display_height;
    int adm_csf_mode;
    int adm_adm3_apply_hm;
    int adm_bypass_cm;
    double adm_p_norm;
    double adm_dlm_weight;
    double adm_f1s0;
    double adm_f1s1;
    double adm_f1s2;
    double adm_f1s3;
    double adm_f2s0;
    double adm_f2s1;
    double adm_f2s2;
    double adm_f2s3;
    int adm_skip_aim_scale;
    VmafDictionary *feature_name_dict;
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
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_noise_weight",
        .alias = "nw",
        .help = "noise weight",
        .offset = offsetof(AdmState, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_scale",
        .alias = "scf",
        .help = "scale factor for the CSF",
        .offset = offsetof(AdmState, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_diag_scale",
        .alias = "scfd",
        .help = "scale factor for the CSF diag",
        .offset = offsetof(AdmState, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdf",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(AdmState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_bypass_cm",
        .alias = "bcm",
        .help = "bypass contrast masking (CM)",
        .offset = offsetof(AdmState, adm_bypass_cm),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 1,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_adm3_apply_hm",
        .alias = "aah",
        .help = "apply harmonic mean to combine DLM and AIM",
        .offset = offsetof(AdmState, adm_adm3_apply_hm),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_p_norm",
        .alias = "apn",
        .help = "p-norm for energy vector",
        .offset = offsetof(AdmState, adm_p_norm),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 3.0,
        .min = 1.0,
        .max = 20.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_dlm_weight",
        .alias = "dlmw",
        .help = "linear weighting between DLM and AIM; 1 corresponds to DLM-only",
        .offset = offsetof(AdmState, adm_dlm_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.5,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
            .name = "adm_min_val",
            .alias = "min",
            .help = "minimum value allowed; lower values will be clipped to this value",
            .offset = offsetof(AdmState, adm_min_val),
            .type = VMAF_OPT_TYPE_DOUBLE,
            .default_val.d = DEFAULT_ADM_MIN_VAL,
            .min = 0.0,
            .max = 1.0,
            .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s0",
        .alias = "f1s0",
        .help = "factor1 scale0",
        .offset = offsetof(AdmState, adm_f1s0),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s1",
        .alias = "f1s1",
        .help = "factor1 scale1",
        .offset = offsetof(AdmState, adm_f1s1),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s2",
        .alias = "f1s2",
        .help = "factor1 scale2",
        .offset = offsetof(AdmState, adm_f1s2),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f1s3",
        .alias = "f1s3",
        .help = "factor1 scale3",
        .offset = offsetof(AdmState, adm_f1s3),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s0",
        .alias = "f2s0",
        .help = "factor2 scale0",
        .offset = offsetof(AdmState, adm_f2s0),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s1",
        .alias = "f2s1",
        .help = "factor2 scale1",
        .offset = offsetof(AdmState, adm_f2s1),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s2",
        .alias = "f2s2",
        .help = "factor2 scale2",
        .offset = offsetof(AdmState, adm_f2s2),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_f2s3",
        .alias = "f2s3",
        .help = "factor2 scale3",
        .offset = offsetof(AdmState, adm_f2s3),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = -1.0,
        .min = -1.0,
        .max = 10.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_aim_scale",
        .alias = "sasc",
        .help = "when set, skip AIM calculations for that scale",
        .offset = offsetof(AdmState, adm_skip_aim_scale),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = -1,
        .min = 0,
        .max = 3,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_scale0",
        .alias = "ssz",
        .help = "skip the calculation of scale 0",
        .offset = offsetof(AdmState, adm_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    AdmState *s = fex->priv;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto fail;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    vmaf_dictionary_free(&s->feature_name_dict);
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

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc, 0);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc, 0);

	double luminance_level = DEFAULT_ADM_CSF_LUMINANCE_LEVEL;

    double score, score_num, score_den, score_aim;
    double scores[8];
    err = compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR,
                      s->adm_enhn_gain_limit,
                      s->adm_norm_view_dist, s->adm_ref_display_height,
                      s->adm_csf_mode, luminance_level, s->adm_csf_scale, s->adm_csf_diag_scale,
                      s->adm_noise_weight, s->adm_bypass_cm, s->adm_p_norm, &score_aim,
                      s->adm_f1s0, s->adm_f1s1, s->adm_f1s2, s->adm_f1s3,
                      s->adm_f2s0, s->adm_f2s1, s->adm_f2s2, s->adm_f2s3,
                      s->adm_skip_aim_scale, s->adm_skip_scale0);
    if (err) return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm2_score", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
        s->feature_name_dict, "VMAF_feature_aim_score", score_aim, index);

    if (s->adm_adm3_apply_hm) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm3_score", MAX(2 * score * score_aim / (score + score_aim), s->adm_min_val), index);
    } else {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm3_score", MAX(score * s->adm_dlm_weight + (1 - score_aim) * (1 - s->adm_dlm_weight), s->adm_min_val), index);
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm_scale0_score",
            scores[0] / scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm_scale1_score",
            scores[2] / scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm_scale2_score",
            scores[4] / scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_adm_scale3_score",
            scores[6] / scores[7], index);

    if (!s->debug) return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_num_scale0", scores[0], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_den_scale0", scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_num_scale1", scores[2], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_den_scale1", scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_num_scale2", scores[4], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_den_scale2", scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_num_scale3", scores[6], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "adm_den_scale3", scores[7], index);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_feature_adm2_score", "VMAF_feature_aim_score", "VMAF_feature_adm3_score", "VMAF_feature_adm_scale0_score",
    "VMAF_feature_adm_scale1_score", "VMAF_feature_adm_scale2_score",
    "VMAF_feature_adm_scale3_score", "adm_num", "adm_den", "adm_scale0",
    "adm_num_scale0", "adm_den_scale0", "adm_num_scale1", "adm_den_scale1",
    "adm_num_scale2", "adm_den_scale2", "adm_num_scale3", "adm_den_scale3",
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
