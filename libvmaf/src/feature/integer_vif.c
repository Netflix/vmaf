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
#include "picture_copy.h"

typedef struct Integer_VifState {
    size_t integer_stride;
    int16_t *ref;
    int16_t *dist;
} Integer_VifState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    Integer_VifState *s = fex->priv;
    s->integer_stride = sizeof(int16_t) * w;            
    s->ref = aligned_malloc(s->integer_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->integer_stride * h, 32);
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
    Integer_VifState *s = fex->priv;
    int err = 0;

    integer_picture_copy(s->ref, ref_pic, 0 - (1 << (ref_pic->bpc - 1)), ref_pic->bpc);
    integer_picture_copy(s->dist, dist_pic, 0 - (1 << (dist_pic->bpc - 1)), dist_pic->bpc);

    double score, score_num, score_den;
    double scores[8];
    //the stride pass to integer_compute_motion is in multiple of sizeof(float)
    err = integer_compute_vif(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      (s->integer_stride)*2, (s->integer_stride)*2,
                      &score, &score_num, &score_den, scores, ref_pic->bpc);
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
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
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
