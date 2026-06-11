/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_motion.h"
#include "motion_blend_tools.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Default maximum value allowed for motion */
#define DEFAULT_MOTION_MAX_VAL (10000.0)

#if ARCH_X86
#include "x86/motion_avx2.h"
#if HAVE_AVX512
#include "x86/motion_avx512.h"
#endif
#endif

typedef uint64_t (*motion_pipeline_fn)(const uint8_t *, ptrdiff_t,
                                       const uint8_t *, ptrdiff_t,
                                       int32_t *, unsigned, unsigned,
                                       unsigned bpc);

typedef struct MotionState {
    int32_t *y_row;
    unsigned w, h, bpc;
    motion_pipeline_fn pipeline;
    double motion_max_val;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    bool motion_five_frame_window;
    bool motion_moving_average;
    bool motion_force_zero;
    bool debug;
    VmafDictionary *feature_name_dict;
} MotionState;

static const VmafOption options[] = {
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionState, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionState, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionState, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window",
        .offset = offsetof(MotionState, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion scores after first frame",
        .offset = offsetof(MotionState, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

static inline int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
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
    MotionState *s = fex->priv;

    s->w = w;
    s->h = h;
    s->bpc = bpc;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    s->y_row = malloc(sizeof(*s->y_row) * w);
    if (!s->y_row) return -ENOMEM;

    if (bpc == 8)
        s->pipeline = motion_score_pipeline_8;
    else
        s->pipeline = motion_score_pipeline_16;

#if ARCH_X86
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
    MotionState *s = fex->priv;

    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    double score = 0.;
    int err = 0;

    if (s->motion_force_zero) goto write_score;

    const unsigned min_idx = s->motion_five_frame_window ? 2 : 1;
    if (index >= min_idx) {
        const VmafPicture *prev = s->motion_five_frame_window
    	    ? &fex->prev_prev_ref
    	    : &fex->prev_ref;
        if (!prev->ref)
    	return -EINVAL;
    
        const unsigned w = s->w;
        const unsigned h = s->h;
        const uint8_t *prev_data = (const uint8_t *)prev->data[0];
        const uint8_t *cur_data = (const uint8_t *)ref_pic->data[0];
    
        uint64_t sad = s->pipeline(prev_data, prev->stride[0],
    			       cur_data, ref_pic->stride[0],
    			       s->y_row, w, h, s->bpc);
    
        score = MIN((double)sad / 256. / (w * h) * s->motion_fps_weight,
    		s->motion_max_val);
    }

write_score:
    err = vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion_sad_score", score, index);
    if (err) return err;

    if (s->debug) {
        return vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict,
                "VMAF_integer_feature_motion_score", score, index);
    }

    return 0;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionState *s = fex->priv;
    free(s->y_row);
    return vmaf_dictionary_free(&s->feature_name_dict);
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;

    if (!s->feature_name_dict) {
        s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
                fex->provided_features, fex->options, s);
        if (!s->feature_name_dict) return -ENOMEM;
    }

    const VmafDictionaryEntry *sad_entry = vmaf_dictionary_get(
            &s->feature_name_dict, "VMAF_integer_feature_motion_sad_score", 0);
    if (!sad_entry) return -EINVAL;
    const char *sad_name = sad_entry->val;

    unsigned n = 0;
    double score;
    while (!vmaf_feature_collector_get_score(feature_collector, sad_name, &score, n))
        n++;
    const unsigned stride = s->motion_five_frame_window ? 2 : 1;
    const unsigned min_idx = s->motion_five_frame_window ? 2 : 1;
    if (!n) {
        vmaf_dictionary_free(&s->feature_name_dict);
        return 1;
    }

    double stamp_value = 0.;
    if (n > min_idx) {
        double sad_at_min_idx;
        if (!vmaf_feature_collector_get_score(feature_collector, sad_name,
                                              &sad_at_min_idx, min_idx)) {
            stamp_value = MIN(motion_blend(sad_at_min_idx,
                                           s->motion_blend_factor,
                                           s->motion_blend_offset),
                              s->motion_max_val);
        }
    }

    double prev_processed = 0.;
    for (unsigned i = 0; i < n; i++) {
        double sad_i;
        vmaf_feature_collector_get_score(feature_collector, sad_name, &sad_i, i);

        double motion2;

        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const int hi_idx = (int)i + 1;
            double hi;
            const bool has_hi = !vmaf_feature_collector_get_score(
                    feature_collector, sad_name, &hi, hi_idx);
            if (!has_hi) {
                motion2 = sad_i;
            } else if (lo_idx >= (int)min_idx) {
                double lo;
                vmaf_feature_collector_get_score(
                        feature_collector, sad_name, &lo, lo_idx);
                motion2 = lo < hi ? lo : hi;
            } else {
                motion2 = hi;
            }
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion2_score", motion2, i);

        double motion3;
        if (i < min_idx) {
            motion3 = stamp_value;
            prev_processed = stamp_value;
        } else {
            double processed = MIN(motion_blend(motion2,
                                                s->motion_blend_factor,
                                                s->motion_blend_offset),
                                   s->motion_max_val);
            motion3 = s->motion_moving_average
                    ? (processed + prev_processed) / 2.0
                    : processed;
            prev_processed = processed;
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion3_score", motion3, i);
    }

    vmaf_dictionary_free(&s->feature_name_dict);
    return 1;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_sad_score",
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_motion3_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion = {
    .name = "motion",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(MotionState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_PREV_REF,
};
