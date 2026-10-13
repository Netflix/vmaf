/**
 * Copyright 2026 Dan Trapp.
 *
 * Licensed under the BSD+Patent License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://opensource.org/licenses/BSDplusPatent
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "feature/vif_tools.h"
#include "test.h"

static char *test_filter_dec16(void)
{
    static const struct { int w, h; } sizes[] = {
        { 16, 16 }, { 17, 31 }, { 31, 17 }, { 32, 32 }, { 33, 33 },
        { 63, 65 }, { 65, 63 }, { 80, 80 }, { 81, 95 }, { 95, 81 },
        { 96, 97 }, { 97, 96 }, { 128, 129 }, { 160, 161 }, { 320, 180 },
    };
    static const float scales[] = { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f };
    uint32_t state = 123456789;
    vmaf_init_cpu();
    vmaf_set_cpu_flags_mask(0);

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(*sizes); s++) {
        const int w = sizes[s].w, h = sizes[s].h;
        for (unsigned layout = 0; layout < 2; layout++) {
            const int src_stride = w + (layout ? 3 : 0);
            const int full_stride = w + (layout ? 7 : 0);
            const int dst_stride = w / 16 + (layout ? 5 : 0);
            const size_t src_count = (size_t)(h - 1) * src_stride + w;
            const size_t dst_count = (size_t)(h / 16) * dst_stride;
            float *allocation = malloc((src_count + layout) * sizeof(float));
            float *src = allocation ? allocation + layout : NULL;
            float *full = malloc((size_t)h * full_stride * sizeof(float));
            float *tmp = malloc((size_t)w * sizeof(float));
            float *expected = malloc(dst_count * sizeof(float));
            float *actual = malloc(dst_count * sizeof(float));
            float *simd = malloc(dst_count * sizeof(float));
            mu_assert("filter buffer allocation failed", src && full && tmp && expected && actual && simd);

            for (unsigned pattern = 0; pattern < 5; pattern++) {
                for (size_t i = 0; i < src_count; i++) src[i] = NAN;
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        float value;
                        state = state * 1664525u + 1013904223u;
                        switch (pattern) {
                        case 0: value = ((int)(state & 65535) - 32768) / 128.f; break;
                        case 1: value = 127.5f; break;
                        case 2: value = ((i + j) & 1) ? 255.f : 0.f; break;
                        case 3: value = (i == 0 || i == h - 1) &&
                                        (j == 0 || j == w - 1) ? 255.f : 0.f; break;
                        default: value = (i * w + j) / 257.f; break;
                        }
                        src[i * src_stride + j] = value;
                    }
                }
                for (unsigned k = 0; k < sizeof(scales) / sizeof(*scales); k++) {
                    const int fwidth = vif_get_filter_size(1, scales[k]);
                    if (fwidth / 2 >= w || fwidth / 2 >= h) continue;
                    float filter[128];
                    speed_get_antialias_filter(filter, 4, scales[k]);
                    for (size_t i = 0; i < dst_count; i++)
                        expected[i] = actual[i] = simd[i] = -12345.f;
                    vif_filter1d_s(filter, src, full, tmp, w, h,
                                   src_stride * sizeof(float),
                                   full_stride * sizeof(float), fwidth);
                    vif_dec16_s(full, expected, w, h,
                                full_stride * sizeof(float),
                                dst_stride * sizeof(float));
                    vif_filter1d_dec16_s(filter, src, actual, tmp, w, h,
                                         src_stride * sizeof(float),
                                         dst_stride * sizeof(float), fwidth);
                    if (memcmp(expected, actual, dst_count * sizeof(float))) {
                        fprintf(stderr, "%dx%d, layout %u, pattern %u, filter %d\n",
                                w, h, layout, pattern, fwidth);
                        return "decimated filter differs from full filter";
                    }

                    vmaf_set_cpu_flags_mask(UINT_MAX);
                    vif_filter1d_dec16_s(filter, src, simd, tmp, w, h,
                                         src_stride * sizeof(float),
                                         dst_stride * sizeof(float), fwidth);
                    vmaf_set_cpu_flags_mask(0);
                    if (memcmp(expected, simd, dst_count * sizeof(float))) {
                        fprintf(stderr, "%dx%d, layout %u, pattern %u, filter %d\n",
                                w, h, layout, pattern, fwidth);
                        return "SIMD decimated filter differs from full filter";
                    }
                }
            }
            free(allocation);
            free(full);
            free(tmp);
            free(expected);
            free(actual);
            free(simd);
        }
    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_filter_dec16);
    return NULL;
}
