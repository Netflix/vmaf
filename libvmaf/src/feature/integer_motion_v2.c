/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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

// Pipelined motion feature extractor.
//
// Computes the same motion score as integer_motion.c but without storing
// blurred frames across extract calls. Instead, it exploits the linearity
// of convolution: SAD(blur[N-1], blur[N]) == sum(|blur(f[N-1] - f[N])|).
//
// The frame difference, blur, and absolute-sum are fused into a single
// row-at-a-time pipeline, requiring only one row of scratch memory.
//
// The framework provides the previous reference frame via fex->prev_ref,
// making each extract call stateless with respect to pixel data.

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "integer_motion.h"

#if ARCH_X86_64
#include "x86/motion_v2_avx2.h"
#if HAVE_AVX512
#include "x86/motion_v2_avx512.h"
#endif
#endif

typedef uint64_t (*motion_pipeline_fn)(const uint8_t *, ptrdiff_t,
                                       const uint8_t *, ptrdiff_t,
                                       int32_t *, unsigned, unsigned,
                                       unsigned bpc);

typedef struct MotionV2State {
    int32_t *y_row;
    unsigned w, h, bpc;
    motion_pipeline_fn pipeline;
} MotionV2State;

static inline int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 1;
    return idx;
}

static uint64_t
motion_score_pipeline_8(const uint8_t *prev, ptrdiff_t prev_stride,
                        const uint8_t *cur, ptrdiff_t cur_stride,
                        int32_t *y_row, unsigned w, unsigned h,
                        unsigned bpc)
{
    (void)bpc;
    const int radius = filter_width / 2;
    const int32_t y_round = 1 << 7;
    const int32_t x_round = 1 << 15;

    uint64_t sad = 0;

    for (unsigned i = 0; i < h; i++) {
        // Fused diff + y_conv for row i (shift by 8, matching v1 precision)
        int32_t any_nonzero = 0;
        for (unsigned j = 0; j < w; j++) {
            int32_t accum = 0;
            for (int k = 0; k < filter_width; k++) {
                const int row = mirror((int)i - radius + k, (int)h);
                int32_t diff = prev[row * prev_stride + j]
                             - cur[row * cur_stride + j];
                accum += (int32_t)filter[k] * diff;
            }
            y_row[j] = (accum + y_round) >> 8;
            any_nonzero |= y_row[j];
        }

        if (!any_nonzero) continue;

        // x_conv + abs + accumulate for row i
        uint32_t row_sad = 0;
        for (unsigned j = 0; j < w; j++) {
            int64_t accum = 0;
            for (int k = 0; k < filter_width; k++) {
                const int col = mirror((int)j - radius + k, (int)w);
                accum += (int64_t)filter[k] * y_row[col];
            }
            int32_t val = (int32_t)((accum + x_round) >> 16);
            row_sad += abs(val);
        }
        sad += row_sad;
    }

    return sad;
}

static inline uint64_t
motion_score_pipeline_16(const uint8_t *prev_u8, ptrdiff_t prev_stride,
                         const uint8_t *cur_u8, ptrdiff_t cur_stride,
                         int32_t *y_row, unsigned w, unsigned h,
                         unsigned bpc)
{
    const uint16_t *prev = (const uint16_t *)prev_u8;
    const uint16_t *cur = (const uint16_t *)cur_u8;
    const ptrdiff_t p_stride = prev_stride / 2;
    const ptrdiff_t c_stride = cur_stride / 2;

    const int radius = filter_width / 2;
    const int32_t y_round = 1 << (bpc - 1);
    const int32_t x_round = 1 << 15;

    uint64_t sad = 0;

    for (unsigned i = 0; i < h; i++) {
        // Fused diff + y_conv for row i
        int32_t any_nonzero = 0;
        for (unsigned j = 0; j < w; j++) {
            int64_t accum = 0;
            for (int k = 0; k < filter_width; k++) {
                const int row = mirror((int)i - radius + k, (int)h);
                int32_t diff = prev[row * p_stride + j]
                             - cur[row * c_stride + j];
                accum += (int64_t)filter[k] * diff;
            }
            y_row[j] = (int32_t)((accum + y_round) >> bpc);
            any_nonzero |= y_row[j];
        }

        if (!any_nonzero) continue;

        // x_conv + abs + accumulate for row i
        uint32_t row_sad = 0;
        for (unsigned j = 0; j < w; j++) {
            int64_t accum = 0;
            for (int k = 0; k < filter_width; k++) {
                const int col = mirror((int)j - radius + k, (int)w);
                accum += (int64_t)filter[k] * y_row[col];
            }
            int32_t val = (int32_t)((accum + x_round) >> 16);
            row_sad += abs(val);
        }
        sad += row_sad;
    }

    return sad;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) pix_fmt;
    MotionV2State *s = fex->priv;

    s->w = w;
    s->h = h;
    s->bpc = bpc;

    s->y_row = malloc(sizeof(*s->y_row) * w);
    if (!s->y_row) return -ENOMEM;

    if (bpc == 8)
        s->pipeline = motion_score_pipeline_8;
    else
        s->pipeline = motion_score_pipeline_16;

#if ARCH_X86_64
    if (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX2) {
        if (bpc == 8)
            s->pipeline = motion_score_pipeline_8_avx2;
        else
            s->pipeline = motion_score_pipeline_16_avx2;
    }
#if HAVE_AVX512
    if (vmaf_get_cpu_flags() & VMAF_X86_CPU_FLAG_AVX512) {
        if (bpc == 8)
            s->pipeline = motion_score_pipeline_8_avx512;
        else
            s->pipeline = motion_score_pipeline_16_avx512;
    }
#endif
#endif

    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionV2State *s = fex->priv;

    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    if (index == 0) {
        return vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion_v2_sad_score", 0., index);
    }

    if (!fex->prev_ref.ref)
        return -EINVAL;

    const unsigned w = s->w;
    const unsigned h = s->h;
    const uint8_t *prev_data = (const uint8_t *)fex->prev_ref.data[0];
    const uint8_t *cur_data = (const uint8_t *)ref_pic->data[0];

    uint64_t sad = s->pipeline(prev_data, fex->prev_ref.stride[0],
                               cur_data, ref_pic->stride[0],
                               s->y_row, w, h, s->bpc);

    double score = (double)sad / 256. / (w * h);

    return vmaf_feature_collector_append(feature_collector,
            "VMAF_integer_feature_motion_v2_sad_score", score, index);
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionV2State *s = fex->priv;
    free(s->y_row);
    return 0;
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    (void) fex;

    unsigned n_frames = 0;
    double dummy;
    while (!vmaf_feature_collector_get_score(feature_collector,
            "VMAF_integer_feature_motion_v2_sad_score", &dummy, n_frames))
        n_frames++;

    if (n_frames < 2) return 1;

    for (unsigned i = 0; i < n_frames; i++) {
        double score_cur, score_next;
        vmaf_feature_collector_get_score(feature_collector,
            "VMAF_integer_feature_motion_v2_sad_score", &score_cur, i);

        double motion2;
        if (i + 1 < n_frames) {
            vmaf_feature_collector_get_score(feature_collector,
                "VMAF_integer_feature_motion_v2_sad_score", &score_next, i + 1);
            motion2 = score_cur < score_next ? score_cur : score_next;
        } else {
            motion2 = score_cur;
        }

        vmaf_feature_collector_append(feature_collector,
            "VMAF_integer_feature_motion2_v2_score", motion2, i);
    }

    return 1;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_v2_sad_score",
    "VMAF_integer_feature_motion2_v2_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion_v2 = {
    .name = "motion_v2",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(MotionV2State),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_PREV_REF,
};
