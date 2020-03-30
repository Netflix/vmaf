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

#include "integer_adm_function.h"
#include "adm_options.h"
#include "mem.h"
#include "picture_copy.h"

typedef struct Integer_AdmState {
    size_t integer_stride;
    int16_t *ref;
    int16_t *dist;
} Integer_AdmState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    Integer_AdmState *s = fex->priv;
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
    Integer_AdmState *s = fex->priv;
    int err = 0;

    integer_picture_copy(s->ref, ref_pic, 0 - (1 << (ref_pic->bpc - 1)), ref_pic->bpc);
    integer_picture_copy(s->dist, dist_pic, 0 - (1 << (ref_pic->bpc - 1)), dist_pic->bpc);

    double score, score_num, score_den;
    double scores[8];
    //the stride pass to integer_compute_motion is in multiple of sizeof(float)
    err = integer_compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      (s->integer_stride)*2, (s->integer_stride)*2, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR, ref_pic->bpc);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_adm2_integer_score'",
                                        score, index);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale0",
                                        scores[0] / scores[1], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale1",
                                        scores[2] / scores[3], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale2",
                                        scores[4] / scores[5], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale3",
                                        scores[6] / scores[7], index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    Integer_AdmState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_adm2_integer_score'",
    "integer_adm_scale0", "integer_adm_scale1",
    "integer_adm_scale2", "integer_adm_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_adm = {
    .name = "adm",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(Integer_AdmState),
    .provided_features = provided_features,
};
