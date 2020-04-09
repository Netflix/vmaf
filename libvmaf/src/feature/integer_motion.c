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

#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"
#include "integer_motion_function.h"
#include "picture.h"

typedef struct Integer_MotionState {
    VmafPicture tmp;
    VmafPicture blur[3];
    unsigned index;
    double score;
} Integer_MotionState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    Integer_MotionState *s = fex->priv;

    // VmafPicture buffers are in uint16_t format to handle all bitdepth 8,10,12...
    // VmafPicture buffers are allocated in uint16 to preserve the precision after convolution as coefficient used in convolution have high precision
    unsigned bit16 = 16;
    int ret1 = vmaf_picture_alloc(&s->tmp, pix_fmt, bit16, w, h);
    int ret2 = vmaf_picture_alloc(&s->blur[0], pix_fmt, bit16, w, h);
    int ret3 = vmaf_picture_alloc(&s->blur[1], pix_fmt, bit16, w, h);
    int ret4 = vmaf_picture_alloc(&s->blur[2], pix_fmt, bit16, w, h);
    
    if (ret1 < 0 || ret2 < 0 || ret3 < 0 || ret4 < 0)
        goto fail;

    s->score = 0;
    return 0;

fail:
    if (&s->blur[0].data[0]) vmaf_picture_unref(&s->blur[0]);
    if (&s->blur[1].data[0]) vmaf_picture_unref(&s->blur[1]);
    if (&s->blur[2].data[0]) vmaf_picture_unref(&s->blur[2]);
    if (&s->tmp.data[0]) vmaf_picture_unref(&s->tmp);
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

    if (ref_pic->bpc == 8)
    {
        // 8 bit input and 16 bit output for integer_convolution_8
        // ref_pic->stride[0] is pass as src_stride is in multiple of sizeof(uint8_t)
        // s->blur[blur_idx_0].stride[0] >> 1 is pass as dst_stride is in multiple of sizeof(uint16_t)
        integer_convolution_8(INTEGER_FILTER_5_s, 5, ref_pic->data[0], s->blur[blur_idx_0].data[0], s->tmp.data[0],
                        ref_pic->w[0], ref_pic->h[0],
                        ref_pic->stride[0] ,
                        s->blur[blur_idx_0].stride[0] >> 1, ref_pic->bpc);
    }
    else
    {
        // 16 bit input and 16 bit output for integer_convolution_16
        // ref_pic->stride[0] >> 1 is pass as src_stride is in multiple of sizeof(uint8_t)
        // s->blur[blur_idx_0].stride[0] >> 1 is pass as dst_stride is in multiple of sizeof(uint16_t)
        integer_convolution_16(INTEGER_FILTER_5_s, 5, ref_pic->data[0], s->blur[blur_idx_0].data[0], s->tmp.data[0],
                        ref_pic->w[0], ref_pic->h[0],
                        ref_pic->stride[0] >> 1,
                        s->blur[blur_idx_0].stride[0] >> 1 , ref_pic->bpc);
    }

    if (index == 0)
        return vmaf_feature_collector_append(feature_collector,
                                             "'VMAF_feature_motion2_integer_score'",
                                             0., index);

    double score;
    //the stride pass to integer_compute_motion is in multiple of sizeof(uint16_t)
    err = integer_compute_motion(s->blur[blur_idx_2].data[0], s->blur[blur_idx_0].data[0],
                         ref_pic->w[0], ref_pic->h[0],
                         s->blur[blur_idx_2].stride[0], s->blur[blur_idx_0].stride[0], &score);
    if (err) return err;
    s->score = score;

    if (index == 1)
        return 0;
    
    double score2;
    //the stride pass to integer_compute_motion is in multiple of sizeof(uint16_t)
    err = integer_compute_motion(s->blur[blur_idx_2].data[0], s->blur[blur_idx_1].data[0],
                         ref_pic->w[0], ref_pic->h[0],
                         s->blur[blur_idx_2].stride[0], s->blur[blur_idx_1].stride[0], &score2);
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

    if (&s->blur[0].data[0]) vmaf_picture_unref(&s->blur[0]);
    if (&s->blur[1].data[0]) vmaf_picture_unref(&s->blur[1]);
    if (&s->blur[2].data[0]) vmaf_picture_unref(&s->blur[2]);
    if (&s->tmp.data[0]) vmaf_picture_unref(&s->tmp);
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
