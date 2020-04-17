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
#include "mem.h"
#include "adm_options.h"

#ifndef NUM_BUFS_ADM
#define NUM_BUFS_ADM 30
#endif

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    Integer_AdmState *s = fex->priv;

    s->integer_stride = ALIGN_CEIL(w * sizeof(int32_t));
    s->ind_size_x   = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->ind_size_y   = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    s->buf_sz_one   = s->ind_size_x * ((h + 1) / 2);

    s->data_buf     = aligned_malloc(s->buf_sz_one * NUM_BUFS_ADM, MAX_ALIGN);
    if (!s->data_buf) goto free_ref;
    s->tmp_ref      = aligned_malloc(s->integer_stride * 4, MAX_ALIGN);
    if (!s->tmp_ref) goto free_ref;
    s->buf_x_orig   = aligned_malloc(s->ind_size_x * 4, MAX_ALIGN);
    if (!s->buf_x_orig) goto free_ref;
    s->buf_y_orig   = aligned_malloc(s->ind_size_y * 4, MAX_ALIGN);
    if (!s->buf_y_orig) goto free_ref;

    return 0;

free_ref:
    if (s->data_buf)    aligned_free(s->data_buf);
    if (s->tmp_ref)     aligned_free(s->tmp_ref);
    if (s->buf_x_orig)  aligned_free(s->buf_x_orig);
    if (s->buf_y_orig)  aligned_free(s->buf_y_orig);

    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    Integer_AdmState *s = fex->priv;
    int err = 0;

    double score, score_num, score_den;
    double scores[8];

    ptrdiff_t integer_stride = ref_pic->bpc == 8 ? ref_pic->stride[0] << 2 : ref_pic->stride[0] << 1;

    //the stride pass to integer_compute_adm is in multiple of sizeof(uint32_t)
    err = integer_compute_adm(ref_pic->data[0], dist_pic->data[0], ref_pic->w[0], ref_pic->h[0],
                      integer_stride, integer_stride, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR, ref_pic->bpc, s);
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

    if (s->data_buf)    aligned_free(s->data_buf);
    if (s->tmp_ref)     aligned_free(s->tmp_ref);
    if (s->buf_x_orig)  aligned_free(s->buf_x_orig);
    if (s->buf_y_orig)  aligned_free(s->buf_y_orig);

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
