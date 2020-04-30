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
#include "picture.h"

typedef struct MotionState {
    VmafPicture tmp;
    VmafPicture blur[3];
    unsigned index;
    double score;
    void (*convolution)(VmafPicture *src, VmafPicture *dst, VmafPicture *tmp);
    void (*sad)(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad);
} MotionState;

static const uint16_t filter[5] = { 3571, 16004, 26386, 16004, 3571 };
static const int filter_width = sizeof(filter) / sizeof(filter[0]);

static inline uint32_t
edge_16(bool horizontal, const uint16_t *src, int width,
        int height, int stride, int i, int j)
{
    const int radius = filter_width / 2;
    uint32_t accum = 0;

    // MIRROR | ЯOЯЯIM
    for (unsigned k = 0; k < filter_width; ++k) {
        int i_tap = horizontal ? i : i - radius + k;
        int j_tap = horizontal ? j - radius + k : j;

        if (horizontal) {
            if (j_tap < 0)
                j_tap = -j_tap;
            else if (j_tap >= width)
                j_tap = width - (j_tap - width + 1);
        } else {
            if (i_tap < 0)
                i_tap = -i_tap;
            else if (i_tap >= height)
                i_tap = height - (i_tap - height + 1);
        }
        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

static inline void
x_convolution_16(const uint16_t *src, uint16_t *dst, unsigned width,
                 unsigned height, ptrdiff_t src_stride,
                 ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;

    uint16_t *src_p = (uint16_t*) src + (left_edge - radius);
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        uint16_t *src_p1 = src_p;
        for (unsigned j = left_edge; j < right_edge; j++) {
            uint32_t accum = 0;
            uint16_t *src_p2 = src_p1;
            for (unsigned k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1++;
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
        }

        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }

        src_p += src_stride;
    }
}

static inline void
y_convolution_16(const uint16_t *src, uint16_t *dst, unsigned width,
                 unsigned height, ptrdiff_t src_stride,
                 ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned add_before_shift = (int) pow(2, (inp_size_bits - 1));
    const unsigned shift_var = inp_size_bits;

    uint16_t *src_p = (uint16_t*) src + (top_edge - radius) * src_stride;
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    for (unsigned i = top_edge; i < bottom_edge; i++) {
        uint16_t *src_p1 = src_p;
        for (unsigned j = 0; j < width; ++j) {
            uint16_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (unsigned k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }
        src_p += src_stride;
    }

    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

static void convolution_16(VmafPicture *src, VmafPicture *dst, VmafPicture *tmp)
{
    y_convolution_16(src->data[0], tmp->data[0], src->w[0], src->h[0],
                     src->stride[0] / 2, tmp->stride[0] / 2, src->bpc);
    x_convolution_16(tmp->data[0], dst->data[0], tmp->w[0], tmp->h[0],
                     tmp->stride[0] / 2, dst->stride[0] / 2);
}

static inline uint32_t
edge_8(const uint8_t *src, int height, int stride, int i, int j)
{
    int radius = filter_width / 2;
    uint32_t accum = 0;

    // MIRROR | ЯOЯЯIM
    for (unsigned k = 0; k < filter_width; ++k) {
        int i_tap = i - radius + k;
        int j_tap = j;

        if (i_tap < 0)
            i_tap = -i_tap;
        else if (i_tap >= height)
            i_tap = height - (i_tap - height + 1);

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

static inline void
y_convolution_8(const uint8_t *src, uint16_t *dst, unsigned width,
                unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned shift_var = 8;
    const unsigned add_before_shift = (int) pow(2, (shift_var - 1));

    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    uint8_t *src_p = (uint8_t*) src + (top_edge - radius) * src_stride;
    for (unsigned i = top_edge; i < bottom_edge; i++) {
        uint8_t *src_p1 = src_p;
        for (unsigned j = 0; j < width; ++j) {
            uint8_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (unsigned k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }
        src_p += src_stride;
    }

    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

static void convolution_8(VmafPicture *src, VmafPicture *dst, VmafPicture *tmp)
{
    y_convolution_8(src->data[0], tmp->data[0], src->w[0], src->h[0],
                    src->stride[0], dst->stride[0] / 2);
    x_convolution_16(tmp->data[0], dst->data[0], tmp->w[0], tmp->h[0],
                     tmp->stride[0] / 2, dst->stride[0] / 2);
}

static void sad_c(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad)
{
    *sad = 0;

    uint16_t *a = pic_a->data[0];
    uint16_t *b = pic_b->data[0];
    for (unsigned i = 0; i < pic_a->h[0]; i++) {
        uint32_t inner_sad = 0;
        for (unsigned j = 0; j < pic_a->w[0]; j++) {
            inner_sad += abs(a[j] - b[j]);
        }
        *sad += inner_sad;
        a += (pic_a->stride[0] / 2);
        b += (pic_b->stride[0] / 2);
    }
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    MotionState *s = fex->priv;

    int err = 0;
    err |= vmaf_picture_alloc(&s->tmp, pix_fmt, 16, w, h);
    err |= vmaf_picture_alloc(&s->blur[0], pix_fmt, 16, w, h);
    err |= vmaf_picture_alloc(&s->blur[1], pix_fmt, 16, w, h);
    err |= vmaf_picture_alloc(&s->blur[2], pix_fmt, 16, w, h);
    if (err) goto fail;

    s->convolution = bpc == 8 ? convolution_8 : convolution_16;
    s->sad = sad_c;
    s->score = 0.;
    return 0;

fail:
    err |= vmaf_picture_unref(&s->blur[0]);
    err |= vmaf_picture_unref(&s->blur[1]);
    err |= vmaf_picture_unref(&s->blur[2]);
    err |= vmaf_picture_unref(&s->tmp);
    return err;
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int ret =
        vmaf_feature_collector_append(feature_collector,
                                      "'VMAF_feature_motion2_integer_score'",
                                      s->score, s->index);
    return (ret < 0) ? ret : !ret;
}

static inline double normalize_and_scale_sad(uint64_t sad,
                                             unsigned w, unsigned h)
{
    return (float) (sad / 256.) / (w * h);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    (void) dist_pic;
    int err = 0;

    s->index = index;
    const unsigned blur_idx_0 = (index + 0) % 3;
    const unsigned blur_idx_1 = (index + 1) % 3;
    const unsigned blur_idx_2 = (index + 2) % 3;

    s->convolution(ref_pic, &s->blur[blur_idx_0], &s->tmp);

    if (index == 0)
        return vmaf_feature_collector_append(feature_collector,
                                             "'VMAF_feature_motion2_integer_score'",
                                             0., index);

    uint64_t sad;
    s->sad(&s->blur[blur_idx_2], &s->blur[blur_idx_0], &sad);
    double score = s->score =
        normalize_and_scale_sad(sad, ref_pic->h[0], ref_pic->w[0]);

    if (index == 1) return 0;

    uint64_t sad2;
    s->sad(&s->blur[blur_idx_2], &s->blur[blur_idx_1], &sad2);
    double score2 = normalize_and_scale_sad(sad2, ref_pic->w[0], ref_pic->h[0]);

    score2 = score2 < score ? score2 : score;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_motion2_integer_score'",
                                        score2, index - 1);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    MotionState *s = fex->priv;

    int err = 0;
    err |= vmaf_picture_unref(&s->blur[0]);
    err |= vmaf_picture_unref(&s->blur[1]);
    err |= vmaf_picture_unref(&s->blur[2]);
    err |= vmaf_picture_unref(&s->tmp);
    return err;
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
    .priv_size = sizeof(MotionState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
