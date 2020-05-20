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

#include "adm.h"
#include "adm_options.h"
#include "mem.h"
#include "picture_copy.h"

#define PROFILE_FLOAT_ADM 1 && (defined(__MINGW32__) || (!defined(_WIN32)))

#if PROFILE_FLOAT_ADM
#include <sys/time.h>
#endif

typedef struct AdmState {
    size_t float_stride;
    float *ref;
    float *dist;
} AdmState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmState *s = fex->priv;
    s->float_stride = sizeof(float) * w;
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
    AdmState *s = fex->priv;
    int err = 0;

    picture_copy(s->ref, ref_pic, -128, ref_pic->bpc);
    picture_copy(s->dist, dist_pic, -128, dist_pic->bpc);

    double score, score_num, score_den;
    double scores[8];

#if PROFILE_FLOAT_ADM
    double time_useconds;
    struct timeval s_tv;
    struct timeval e_tv;
    gettimeofday(&s_tv, NULL);
#endif // PROFILE_FLOAT_ADM

    err = compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR);
#if PROFILE_FLOAT_ADM
    gettimeofday(&e_tv, NULL);
    time_useconds = ((e_tv.tv_sec - s_tv.tv_sec) * 1000000) +
        (e_tv.tv_usec - s_tv.tv_usec);
    printf("Frame_No %d, time(ms) %lf \n", index, time_useconds);
#endif // PROFILE_FLOAT_ADM

    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_adm2_score'",
                                        score, index);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "adm_scale0",
                                        scores[0] / scores[1], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "adm_scale1",
                                        scores[2] / scores[3], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "adm_scale2",
                                        scores[4] / scores[5], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "adm_scale3",
                                        scores[6] / scores[7], index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_adm2_score'",
    "adm_scale0", "adm_scale1",
    "adm_scale2", "adm_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_adm = {
    .name = "float_adm",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
