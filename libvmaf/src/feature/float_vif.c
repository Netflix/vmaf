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
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"

#include "vif.h"
#include "vif_options.h"
#include "vif_tools.h"
#include "picture_copy.h"

/* Default minimum value allowed for the feature */
#define DEFAULT_VIF_MIN_VAL (0.0)
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct VifState {
    size_t float_stride;
    size_t scaled_float_stride;
    size_t scaled_w;
    size_t scaled_h;
    float *ref;
    float *ref_scaled;
    float *dist;
    float *dist_scaled;
    bool debug;
    double vif_sigma_nsq;
    bool vif_skip_scale0;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_prescale;
    double vif_scale1_min_val;
    double vif_scale2_min_val;
    double vif_scale3_min_val;
    char *vif_prescale_method;
    VmafDictionary *feature_name_dict;
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
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_kernelscale",
        .alias = "ks",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(VifState, vif_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_prescale",
        .alias = "ps",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(VifState, vif_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_scale1_min_val",
        .help = "minimum value allowed; smaller values will be set to this value",
        .offset = offsetof(VifState, vif_scale1_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "s1miv",
    },
        {
        .name = "vif_scale2_min_val",
        .help = "minimum value allowed; smaller values will be set to this value",
        .offset = offsetof(VifState, vif_scale2_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "s2miv",
    },
        {
        .name = "vif_scale3_min_val",
        .help = "minimum value allowed; smaller values will be set to this value",
        .offset = offsetof(VifState, vif_scale3_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "s3miv",
    },
    {
        .name = "vif_prescale_method",
        .alias = "pm",
        .help = "scaling method for the frame, supported options: [nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(VifState, vif_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_VIF_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_sigma_nsq",
        .help = "neural noise variance",
        .offset = offsetof(VifState, vif_sigma_nsq),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 2.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snsq",
    },
    {
        .name = "vif_skip_scale0",
        .alias = "ssclz",
        .help = "when set, skip scale 0 calculations",
        .offset = offsetof(VifState, vif_skip_scale0),
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

    VifState *s = fex->priv;

    enum vif_scaling_method scaling_method;
    if (vif_get_scaling_method(s->vif_prescale_method, &scaling_method)) {
        return -EINVAL;
    }

    s->scaled_w = (int)(w * s->vif_prescale + 0.5);
    s->scaled_h = (int)(h * s->vif_prescale + 0.5);
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->scaled_float_stride = ALIGN_CEIL(s->scaled_w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto fail;
    s->ref_scaled = aligned_malloc(s->scaled_float_stride * s->scaled_h, 32);
    if (!s->ref_scaled) goto fail;
    s->dist_scaled = aligned_malloc(s->scaled_float_stride * s->scaled_h, 32);
    if (!s->dist_scaled) goto fail;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    if (s->ref_scaled) aligned_free(s->ref_scaled);
    if (s->dist_scaled) aligned_free(s->dist_scaled);
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc, 0);
    picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc, 0);

    // The scaling method has been checked for validity in the init callback
    enum vif_scaling_method scaling_method;
    vif_get_scaling_method(s->vif_prescale_method, &scaling_method);

    vif_scale_frame_s(
        scaling_method, s->ref, s->ref_scaled,
        ref_pic->w[0], ref_pic->h[0], s->float_stride / sizeof(float),
        s->scaled_w, s->scaled_h, s->scaled_float_stride / sizeof(float)
    );

    vif_scale_frame_s(
        scaling_method, s->dist, s->dist_scaled,
        dist_pic->w[0], dist_pic->h[0], s->float_stride / sizeof(float),
        s->scaled_w, s->scaled_h, s->scaled_float_stride / sizeof(float)
    );

    double score, score_num, score_den;
    double scores[8];
    err = compute_vif(s->ref_scaled, s->dist_scaled, s->scaled_w, s->scaled_h,
                      s->scaled_float_stride, s->scaled_float_stride,
                      &score, &score_num, &score_den, scores,
                      s->vif_enhn_gain_limit,
                      s->vif_kernelscale, s->vif_skip_scale0, s->vif_sigma_nsq);
    if (err) return err;

    if (s->vif_skip_scale0) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_vif_scale0_score",
            0.0f, index);
    } else {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_vif_scale0_score",
            scores[0] / scores[1], index);
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_vif_scale1_score",
            MAX(scores[2] / scores[3], s->vif_scale1_min_val), index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_vif_scale2_score",
            MAX(scores[4] / scores[5], s->vif_scale2_min_val), index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_feature_vif_scale3_score",
            MAX(scores[6] / scores[7], s->vif_scale3_min_val), index);

    if (!s->debug) return err;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den", score_den, index);

    if (s->vif_skip_scale0) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale0", 0.0f, index);

        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale0", -1.0f, index);
    } else {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale0", scores[0], index);

        err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale0", scores[1], index);
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale1", scores[2], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale1", scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale2", scores[4], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale2", scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_num_scale3", scores[6], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "vif_den_scale3", scores[7], index);

    return err;
}

static int close(VmafFeatureExtractor *fex)
{
    VifState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    if (s->ref_scaled) aligned_free(s->ref_scaled);
    if (s->dist_scaled) aligned_free(s->dist_scaled);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_feature_vif_scale0_score", "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score", "VMAF_feature_vif_scale3_score",
    "vif", "vif_num", "vif_den", "vif_num_scale0", "vif_den_scale0",
    "vif_num_scale1", "vif_den_scale1", "vif_num_scale2", "vif_den_scale2",
    "vif_num_scale3", "vif_den_scale3",
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
