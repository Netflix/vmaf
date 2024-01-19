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
#include "motion_tools.h"

#include "picture_copy.h"

typedef struct MotionState {
    size_t float_stride;
    float *ref;
    float *tmp;
    float *blur[3];
    unsigned index;
    double score;
    bool debug;
    bool motion_force_zero;
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
    if (!s->ref || !s->tmp || !s->blur[0] || !s->blur[1] || !s->blur[2])
        goto fail;
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
    vmaf_dictionary_free(&s->feature_name_dict);
    return -ENOMEM;

}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int ret = 0;

    if (s->index > 0) {
        ret = vmaf_feature_collector_append(feature_collector,
                                            "VMAF_feature_motion2_score",
                                            s->score, s->index);
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

    picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc);
    convolution_f32_c_s(FILTER_5_s, 5, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->float_stride / sizeof(float),
                        s->float_stride / sizeof(float));

    if (index == 0) {
        err = vmaf_feature_collector_append(feature_collector,
                                             "VMAF_feature_motion2_score",
                                             0., index);
        if (s->debug) {
            err |= vmaf_feature_collector_append(feature_collector,
                                                 "VMAF_feature_motion_score",
                                                 0., index);
        }
        return err;
    }

    double score;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_0],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score);
    if (err) return err;

    if (s->debug) {
        err |= vmaf_feature_collector_append(feature_collector,
                                            "VMAF_feature_motion_score",
                                            score, index);
    }
    if (err) return err;
    s->score = score;

    if (index == 1)
        return 0;
    
    double score2;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_1],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score2);
    if (err) return err;
    score2 = score2 < score ? score2 : score;
    err = vmaf_feature_collector_append(feature_collector,
                                        "VMAF_feature_motion2_score",
                                        score2, index - 1);
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
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_feature_motion_score", "VMAF_feature_motion2_score",
    "VMAF_feature_motion2_score",
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
