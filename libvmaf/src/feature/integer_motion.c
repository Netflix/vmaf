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
#include <string.h>

#include "common/convolution.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"
#include "integer_motion_function.h"
#include "motion_tools.h"

#include "picture_copy.h"

typedef struct Integer_MotionState {
    size_t integer_stride;
    int16_t *ref;
    int16_t *tmp;
    int16_t *blur[3];
    unsigned index;
    double score;
} Integer_MotionState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    Integer_MotionState *s = fex->priv;

    s->integer_stride = sizeof(int16_t) * w;
    s->ref = aligned_malloc(s->integer_stride * h, 32);
    s->tmp = aligned_malloc(s->integer_stride * h, 32);
    s->blur[0] = aligned_malloc(s->integer_stride * h, 32);
    s->blur[1] = aligned_malloc(s->integer_stride * h, 32);
    s->blur[2] = aligned_malloc(s->integer_stride * h, 32);
    if (!s->ref || !s->tmp || !s->blur[0] || !s->blur[1] || !s->blur[2])
        goto fail;

    s->score = 0;
    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    return -ENOMEM;

}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    Integer_MotionState *s = fex->priv;
    int ret = vmaf_feature_collector_append(feature_collector,
                                            "'VMAF_feature_motion2_integer_score'",
                                            s->score, s->index);
    return (ret < 0) ? ret : !ret;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    Integer_MotionState *s = fex->priv;
    int err = 0;

    s->index = index;
    unsigned blur_idx_0 = (index + 0) % 3;
    unsigned blur_idx_1 = (index + 1) % 3;
    unsigned blur_idx_2 = (index + 2) % 3;

    integer_picture_copy(s->ref, ref_pic, 0 - (1<<(ref_pic->bpc - 1)), ref_pic->bpc);
    integer_convolution_f32_c_s(INTEGER_FILTER_5_s, 5, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->integer_stride / sizeof(int16_t),
                        s->integer_stride / sizeof(int16_t), ref_pic->bpc);

    if (index == 0)
        return vmaf_feature_collector_append(feature_collector,
                                             "'VMAF_feature_motion2_integer_score'",
                                             0., index);

    double score;
    //the stride pass to integer_compute_motion is in multiple of sizeof(float)
    err = integer_compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_0],
                         ref_pic->w[0], ref_pic->h[0],
                         (s->integer_stride) * 2, (s->integer_stride) * 2, &score);
    if (err) return err;
    s->score = score;

    if (index == 1)
        return 0;
    
    double score2;
    //the stride pass to integer_compute_motion is in multiple of sizeof(float)
    err = integer_compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_1],
                         ref_pic->w[0], ref_pic->h[0],
                         (s->integer_stride) * 2, (s->integer_stride) * 2, &score2);
    if (err) return err;
    score2 = score2 < score ? score2 : score;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_motion2_integer_score'",
                                        score2, index - 1);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    Integer_MotionState *s = fex->priv;

    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_motion2_integer_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion = {
    .name = "motion",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close,
    .priv_size = sizeof(Integer_MotionState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
