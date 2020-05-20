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

#if defined(__MINGW32__) || !defined(_WIN32)
#define PROFILE_FLOAT_ADM 1
#endif

#if PROFILE_INTEGER_ADM
#include <sys/time.h>
#endif

static inline void *init_dwt_band(adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = (int16_t *)data_top; data_top += stride;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *init_index(int32_t **index, char *data_top, size_t stride)
{
    index[0] = (int32_t *)data_top; data_top += stride;
    index[1] = (int32_t *)data_top; data_top += stride;
    index[2] = (int32_t *)data_top; data_top += stride;
    index[3] = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *i4_init_dwt_band(i4_adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = (int32_t *)data_top; data_top += stride;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *init_dwt_band_hvd(adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static inline void *i4_init_dwt_band_hvd(i4_adm_dwt_band_t *band, char *data_top, size_t stride)
{
    band->band_a = NULL;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmState *s = fex->priv;

    s->integer_stride   = ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x   = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y   = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    size_t buf_sz_one   = s->buf.ind_size_x * ((h + 1) / 2);

    s->buf.data_buf     = aligned_malloc(buf_sz_one * NUM_BUFS_ADM, MAX_ALIGN);
    if (!s->buf.data_buf) goto free_ref;
    s->buf.tmp_ref      = aligned_malloc(s->integer_stride * 4, MAX_ALIGN);
    if (!s->buf.tmp_ref) goto free_ref;
    s->buf.buf_x_orig   = aligned_malloc(s->buf.ind_size_x * 4, MAX_ALIGN);
    if (!s->buf.buf_x_orig) goto free_ref;
    s->buf.buf_y_orig   = aligned_malloc(s->buf.ind_size_y * 4, MAX_ALIGN);
    if (!s->buf.buf_y_orig) goto free_ref;

    void *data_top = s->buf.data_buf;
    data_top = init_dwt_band(&s->buf.ref_dwt2, data_top, buf_sz_one / 2);
    data_top = init_dwt_band(&s->buf.dis_dwt2, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.decouple_r, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.decouple_a, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.csf_a, data_top, buf_sz_one / 2);
    data_top = init_dwt_band_hvd(&s->buf.csf_f, data_top, buf_sz_one / 2);

    data_top = i4_init_dwt_band(&s->buf.i4_ref_dwt2, data_top, buf_sz_one);
    data_top = i4_init_dwt_band(&s->buf.i4_dis_dwt2, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_decouple_r, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_decouple_a, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_csf_a, data_top, buf_sz_one);
    data_top = i4_init_dwt_band_hvd(&s->buf.i4_csf_f, data_top, buf_sz_one);

    void *ind_buf_y = s->buf.buf_y_orig;
    init_index(s->buf.ind_y, ind_buf_y, s->buf.ind_size_y);
    void *ind_buf_x = s->buf.buf_x_orig;
    init_index(s->buf.ind_x, ind_buf_x, s->buf.ind_size_x);

    return 0;

free_ref:
    if (s->buf.data_buf)    aligned_free(s->buf.data_buf);
    if (s->buf.tmp_ref)     aligned_free(s->buf.tmp_ref);
    if (s->buf.buf_x_orig)  aligned_free(s->buf.buf_x_orig);
    if (s->buf.buf_y_orig)  aligned_free(s->buf.buf_y_orig);

    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    AdmState *s = fex->priv;
    int err = 0;

    double score;
    double scores[8];

#if PROFILE_INTEGER_ADM
    double time_useconds;
    struct timeval s_tv;
    struct timeval e_tv;
    gettimeofday(&s_tv, NULL);
#endif // PROFILE_INTEGER_ADM

    integer_compute_adm(ref_pic, dist_pic, &score, scores, ADM_BORDER_FACTOR, s->buf);

#if PROFILE_INTEGER_ADM
    gettimeofday(&e_tv, NULL);
    time_useconds = ((e_tv.tv_sec - s_tv.tv_sec) * 1000000) +
        (e_tv.tv_usec - s_tv.tv_usec);
    printf("Frame_No %d, time(ms) %lf \n", index, time_useconds);
#endif // PROFILE_INTEGER_ADM

    err |= vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_adm2_integer_score'",
                                        score, index);

    err |= vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale0",
                                        scores[0] / scores[1], index);

    err |= vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale1",
                                        scores[2] / scores[3], index);

    err |= vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale2",
                                        scores[4] / scores[5], index);

    err |= vmaf_feature_collector_append(feature_collector,
                                        "integer_adm_scale3",
                                        scores[6] / scores[7], index);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;

    if (s->buf.data_buf)    aligned_free(s->buf.data_buf);
    if (s->buf.tmp_ref)     aligned_free(s->buf.tmp_ref);
    if (s->buf.buf_x_orig)  aligned_free(s->buf.buf_x_orig);
    if (s->buf.buf_y_orig)  aligned_free(s->buf.buf_y_orig);

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
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
