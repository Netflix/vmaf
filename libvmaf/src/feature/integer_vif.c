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

#include "integer_vif_function.h"
#include "picture.h"

static const float log2_poly_s[9] = { -0.012671635276421, 0.064841182402670, -0.157048836463065, 0.257167726303123, -0.353800560300520, 0.480131410397451, -0.721314327952201, 1.442694803896991, 0 };

static float horner_s(const float *poly, float x, int n)
{
    float var = 0;
    int i;

    for (i = 0; i < n; ++i) {
        var = var * x + poly[i];
    }
    return var;
}

static float log2f_approx(float x)
{
    const uint32_t exp_zero_const = 0x3F800000UL;

    const uint32_t exp_expo_mask = 0x7F800000UL;
    const uint32_t exp_mant_mask = 0x007FFFFFUL;

    float remain;
    float log_base, log_remain;
    uint32_t u32, u32remain;
    uint32_t exponent, mant;

    if (x == 0)
        return -INFINITY;
    if (x < 0)
        return NAN;

    memcpy(&u32, &x, sizeof(float));

    exponent = (u32 & exp_expo_mask) >> 23;
    mant = (u32 & exp_mant_mask) >> 0;
    u32remain = mant | exp_zero_const;

    memcpy(&remain, &u32remain, sizeof(float));

    log_base = (int32_t)exponent - 127;
    log_remain = horner_s(log2_poly_s, (remain - 1.0f), sizeof(log2_poly_s) / sizeof(float));

    return log_base + log_remain;
}

static void log_generate(uint16_t *log_values)
{
    int i;
    for (i = 32767; i < 65536; i++)

    {
        log_values[i] = (uint16_t)round(log2f_approx((float)i) * 2048);
    }
}

typedef struct Integer_VifState {
    size_t buf_stride;          //stride size for intermidate buffers
    size_t buf_frame_size;      //size of frame buffer
    pixel *buf;                 //buffer for vif intermidiate data calculations
    uint16_t log_values[65537]; //log value table for integer implementation
} Integer_VifState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
    unsigned bpc, unsigned w, unsigned h)
{
    Integer_VifState *s = fex->priv;

    //one time log table generation for integer vif
    log_generate(&s->log_values);

    s->buf_stride = ALIGN_CEIL(w * sizeof(uint32_t));       //stride in uint32_t format
    s->buf_frame_size = (size_t)s->buf_stride * h ;

    //7 frame size buffer + 7 stride size buffer for intermidate calculations in integer vif
    s->buf = aligned_malloc((s->buf_frame_size * 7) + (s->buf_stride * 7), MAX_ALIGN);
    if (!s->buf) goto free_ref;

    return 0;

free_ref:
    free(s->buf);
fail:
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    Integer_VifState *s = fex->priv;
    int err = 0;

    double score, score_num, score_den;
    double scores[8];
    ptrdiff_t integer_stride = ref_pic->bpc == 8 ? ref_pic->stride[0] << 2 : ref_pic->stride[0] << 1;
    
    //the stride pass to integer_compute_vif is in multiple of sizeof(uint32_t)
    err = integer_compute_vif(ref_pic->data[0], dist_pic->data[0], ref_pic->w[0], ref_pic->h[0],
                      integer_stride, integer_stride,
                      &score, &score_num, &score_den, scores, ref_pic->bpc, s->buf, s->buf_stride, s->buf_frame_size, &s->log_values);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_vif_scale0_integer_score'",
                                        scores[0] / scores[1], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_vif_scale1_integer_score'",
                                        scores[2] / scores[3], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_vif_scale2_integer_score'",
                                        scores[4] / scores[5], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_vif_scale3_integer_score'",
                                        scores[6] / scores[7], index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    Integer_VifState *s = fex->priv;
    if (s->buf) aligned_free(s->buf);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_vif_scale0_integer_score'", "'VMAF_feature_vif_scale1_integer_score'",
    "'VMAF_feature_vif_scale2_integer_score'", "'VMAF_feature_vif_scale3_integer_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif = {
    .name = "vif",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(Integer_VifState),
    .provided_features = provided_features,
};
