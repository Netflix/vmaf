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
#include "moment.h"
#include "picture_copy.h"

typedef struct MomentState {
    size_t float_stride;
    float *ref;
    float *dist;
} MomentState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    MomentState *s = fex->priv;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
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
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MomentState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, 0, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, 0, dist_pic->bpc);

    double score[4];
    err = compute_1st_moment(s->ref, ref_pic->w[0], ref_pic->h[0],
                             s->float_stride, &score[0]);
    if (err) return err;
    err = compute_1st_moment(s->dist, dist_pic->w[0], dist_pic->h[0],
                             s->float_stride, &score[1]);
    if (err) return err;
    err = compute_2nd_moment(s->ref, ref_pic->w[0], ref_pic->h[0],
                             s->float_stride, &score[2]);
    if (err) return err;
    err = compute_2nd_moment(s->dist, dist_pic->w[0], dist_pic->h[0],
                             s->float_stride, &score[3]);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "float_moment_ref1st",
                                        score[0], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "float_moment_dis1st",
                                        score[1], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "float_moment_ref2nd",
                                        score[2], index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector,
                                        "float_moment_dis2nd",
                                        score[3], index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    MomentState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "float_moment",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_moment = {
    .name = "float_moment",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(MomentState),
    .provided_features = provided_features,
};
