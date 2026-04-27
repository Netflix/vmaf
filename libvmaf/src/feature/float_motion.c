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

#include "common/convolution.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "mem.h"
#include "motion.h"
#include "motion_blend_tools.h"
#include "motion_tools.h"
#include "vif_tools.h"

#include "picture_copy.h"

/* Default maximum value allowed for motion */
#define DEFAULT_MOTION_MAX_VAL (10000.0)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct MotionState {
    size_t float_stride;
    float *ref, *ref_u, *ref_v;
    float *tmp, *tmp_u, *tmp_v;
    float *blur[3], *blur_u[3], *blur_v[3];
    unsigned index;
    double score;
    bool debug;
    bool motion_add_scale1;
    bool motion_add_uv;
    bool motion_force_zero;
    double motion_fps_weight;
    double motion_blend_factor;
    double motion_blend_offset;
    int motion_filter_size;
    double motion_max_val;

    VmafDictionary *feature_name_dict;
} MotionState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionState, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionState, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_add_scale1",
        .alias = "mdc",
        .help = "add motion score from scale1",
        .offset = offsetof(MotionState, motion_add_scale1),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_filter_size",
        .alias = "mfs",
        .help = "filtering size",
        .offset = offsetof(MotionState, motion_filter_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_MOTION_FILTER_SIZE,
        .min = 0,
        .max = 9,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_add_uv",
        .alias = "mau",
        .help = "include U and V terms",
        .offset = offsetof(MotionState, motion_add_uv),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionState, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mmxv",
    },
    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    MotionState *s = fex->priv;

    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    s->tmp = aligned_malloc(s->float_stride * h, 32);
    s->blur[0] = aligned_malloc(s->float_stride * h, 32);
    s->blur[1] = aligned_malloc(s->float_stride * h, 32);
    s->blur[2] = aligned_malloc(s->float_stride * h, 32);
    if (s->motion_add_uv) {
        unsigned h_u, h_v;
        switch (pix_fmt) {
            case VMAF_PIX_FMT_UNKNOWN:
            case VMAF_PIX_FMT_YUV400P:
                return -EINVAL;
            case VMAF_PIX_FMT_YUV420P:
                h_u = h / 2;
                h_v = h / 2;
                break;
            case VMAF_PIX_FMT_YUV422P:
            case VMAF_PIX_FMT_YUV444P:
                h_u = h;
                h_v = h;
                break;
        }
        s->ref_u = aligned_malloc(s->float_stride * h_u, 32);
        s->ref_v = aligned_malloc(s->float_stride * h_v, 32);
        s->tmp_u = aligned_malloc(s->float_stride * h_u, 32);
        s->tmp_v = aligned_malloc(s->float_stride * h_v, 32);
        s->blur_u[0] = aligned_malloc(s->float_stride * h_u, 32);
        s->blur_u[1] = aligned_malloc(s->float_stride * h_u, 32);
        s->blur_u[2] = aligned_malloc(s->float_stride * h_u, 32);
        s->blur_v[0] = aligned_malloc(s->float_stride * h_v, 32);
        s->blur_v[1] = aligned_malloc(s->float_stride * h_v, 32);
        s->blur_v[2] = aligned_malloc(s->float_stride * h_v, 32);
    }
    if (!s->ref || !s->tmp || !s->blur[0] || !s->blur[1] || !s->blur[2])
        goto fail;
    if (s->motion_add_uv) {
        if (!s->ref_u || !s->ref_v || !s->tmp_u || !s->tmp_v || !s->blur_u[0] || !s->blur_u[1] || !s->blur_u[2] || !s->blur_v[0] || !s->blur_v[1] || !s->blur_v[2])
            goto fail;
    }
    if (s->motion_force_zero)
        fex->flush = NULL;
    s->score = 0;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    if (s->motion_add_uv) {
        if (s->ref_u) aligned_free(s->ref_u);
        if (s->ref_v) aligned_free(s->ref_v);
        if (s->tmp_u) aligned_free(s->tmp_u);
        if (s->tmp_v) aligned_free(s->tmp_v);
        if (s->blur_u[0]) aligned_free(s->blur_u[0]);
        if (s->blur_u[1]) aligned_free(s->blur_u[1]);
        if (s->blur_u[2]) aligned_free(s->blur_u[2]);
        if (s->blur_v[0]) aligned_free(s->blur_v[0]);
        if (s->blur_v[1]) aligned_free(s->blur_v[1]);
        if (s->blur_v[2]) aligned_free(s->blur_v[2]);
    }

    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;

}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int ret = 0;

    if (s->index > 0) {
        ret = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                            "VMAF_feature_motion2_score",
                                            MIN(s->score * s->motion_fps_weight, s->motion_max_val), s->index);
        ret |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                             "VMAF_feature_motion3_score",
                                             MIN(motion_blend(s->score * s->motion_fps_weight, s->motion_blend_factor, s->motion_blend_offset), s->motion_max_val),
                                             s->index);
    } else if (s->index == 0) {
        ret |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                             "VMAF_feature_motion3_score",
                                             0, s->index);
    }

    return (ret < 0) ? ret : !ret;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int err = 0;

    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    if (s->motion_force_zero) {
        int err =
            vmaf_feature_collector_append_with_dict(feature_collector,
                    s->feature_name_dict, "VMAF_feature_motion2_score",
                    0., index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                    s->feature_name_dict, "VMAF_feature_motion3_score",
                    0., index);
        if (s->debug) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector,
                    s->feature_name_dict, "VMAF_feature_motion_score", 0.,
                    index);
        }
        return err;
    }

    s->index = index;
    unsigned blur_idx_0 = (index + 0) % 3;
    unsigned blur_idx_1 = (index + 1) % 3;
    unsigned blur_idx_2 = (index + 2) % 3;

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc, 0);
    if (s->motion_add_uv) {
        picture_copy(s->ref_u, s->float_stride, ref_pic, -128, ref_pic->bpc, 1);
        picture_copy(s->ref_v, s->float_stride, ref_pic, -128, ref_pic->bpc, 2);
    }

    if (s->motion_filter_size == 1) {
        convolution_f32_c_s(FILTER_5_NO_OP_s, 5, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->float_stride / sizeof(float),
                        s->float_stride / sizeof(float));
        if (s->motion_add_uv) {
            convolution_f32_c_s(FILTER_5_NO_OP_s, 5, s->ref_u, s->blur_u[blur_idx_0], s->tmp_u,
                            ref_pic->w[1], ref_pic->h[1],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
            convolution_f32_c_s(FILTER_5_NO_OP_s, 5, s->ref_v, s->blur_v[blur_idx_0], s->tmp_v,
                            ref_pic->w[2], ref_pic->h[2],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
        }
    } else if (s->motion_filter_size == 3) {
        convolution_f32_c_s(FILTER_3_s, 3, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->float_stride / sizeof(float),
                        s->float_stride / sizeof(float));
        if (s->motion_add_uv) {
            convolution_f32_c_s(FILTER_3_s, 3, s->ref_u, s->blur_u[blur_idx_0], s->tmp_u,
                            ref_pic->w[1], ref_pic->h[1],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
            convolution_f32_c_s(FILTER_3_s, 3, s->ref_v, s->blur_v[blur_idx_0], s->tmp_v,
                            ref_pic->w[2], ref_pic->h[2],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
        }
    } else {
        convolution_f32_c_s(FILTER_5_s, 5, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->float_stride / sizeof(float),
                        s->float_stride / sizeof(float));
        if (s->motion_add_uv) {
            convolution_f32_c_s(FILTER_5_s, 5, s->ref_u, s->blur_u[blur_idx_0], s->tmp_u,
                            ref_pic->w[1], ref_pic->h[1],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
            convolution_f32_c_s(FILTER_5_s, 5, s->ref_v, s->blur_v[blur_idx_0], s->tmp_v,
                            ref_pic->w[2], ref_pic->h[2],
                            s->float_stride / sizeof(float),
                            s->float_stride / sizeof(float));
        }
    }

    if (index == 0) {
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                             "VMAF_feature_motion2_score",
                                             0., index);
        if (s->debug) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                 "VMAF_feature_motion_score",
                                                 0., index);
        }
        return err;
    }

    double score;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_0],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score, s->motion_add_scale1);
    if (s->motion_add_uv) {
        double score_u, score_v;
        err = compute_motion(s->blur_u[blur_idx_2], s->blur_u[blur_idx_0],
                         ref_pic->w[1], ref_pic->h[1],
                         s->float_stride, s->float_stride, &score_u, s->motion_add_scale1);
        err = compute_motion(s->blur_v[blur_idx_2], s->blur_v[blur_idx_0],
                         ref_pic->w[2], ref_pic->h[2],
                         s->float_stride, s->float_stride, &score_v, s->motion_add_scale1);
        score += score_u;
        score += score_v;
    }

    if (err) return err;

    if (s->debug) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                            "VMAF_feature_motion_score",
                                            MIN(score * s->motion_fps_weight, s->motion_max_val), index);
    }
    if (err) return err;
    s->score = score;

    if (index == 1) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                             "VMAF_feature_motion3_score",
                                             MIN(motion_blend(score * s->motion_fps_weight, s->motion_blend_factor, s->motion_blend_offset), s->motion_max_val), 0);
        return err;
    }

    double score2;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_1],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score2, s->motion_add_scale1);
    if (s->motion_add_uv) {
        double score2_u, score2_v;
        err = compute_motion(s->blur_u[blur_idx_2], s->blur_u[blur_idx_1],
                         ref_pic->w[1], ref_pic->h[1],
                         s->float_stride, s->float_stride, &score2_u, s->motion_add_scale1);
        err = compute_motion(s->blur_v[blur_idx_2], s->blur_v[blur_idx_1],
                         ref_pic->w[2], ref_pic->h[2],
                         s->float_stride, s->float_stride, &score2_v, s->motion_add_scale1);
        score2 += score2_u;
        score2 += score2_v;
    }

    if (err) return err;
    score2 = score2 < score ? score2 : score;
    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                        "VMAF_feature_motion2_score",
                                        MIN(score2 * s->motion_fps_weight, s->motion_max_val), index - 1);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                         "VMAF_feature_motion3_score",
                                         MIN(motion_blend(score2 * s->motion_fps_weight, s->motion_blend_factor, s->motion_blend_offset), s->motion_max_val),
                                         index - 1);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    MotionState *s = fex->priv;

    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    if (s->motion_add_uv) {
        if (s->ref_u) aligned_free(s->ref_u);
        if (s->blur_u[0]) aligned_free(s->blur_u[0]);
        if (s->blur_u[1]) aligned_free(s->blur_u[1]);
        if (s->blur_u[2]) aligned_free(s->blur_u[2]);
        if (s->tmp_u) aligned_free(s->tmp_u);
        if (s->ref_v) aligned_free(s->ref_v);
        if (s->blur_v[2]) aligned_free(s->blur_v[2]);
        if (s->blur_v[0]) aligned_free(s->blur_v[0]);
        if (s->blur_v[1]) aligned_free(s->blur_v[1]);
        if (s->tmp_v) aligned_free(s->tmp_v);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_feature_motion_score", "VMAF_feature_motion2_score",
    "VMAF_feature_motion3_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_motion = {
    .name = "float_motion",
    .init = init,
    .extract = extract,
    .options = options,
    .flush = flush,
    .close = close,
    .priv_size = sizeof(MotionState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
