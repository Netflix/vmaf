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

#include <stdlib.h>

#include "test.h"
#include "feature/motion.h"

/* float_motion keeps the chroma planes in buffers sized for the luma width,
 * so a chroma plane is read with a stride that is larger than its own
 * width needs. vmaf_image_sad_c() must score the same pixels whatever the
 * stride, including with motion_add_scale1. */

#define W 50
#define H 21
/* ALIGN_CEIL(W * sizeof(float)) / sizeof(float) */
#define COMPACT_STRIDE 56
#define WIDE_STRIDE 128

static void fill(float *img, int stride, unsigned seed)
{
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < stride; j++) {
            seed = seed * 1664525u + 1013904223u;
            img[i * stride + j] = (float)((seed >> 16) & 0xff);
        }
    }
}

static void repack(const float *src, int src_stride, float *dst, int dst_stride)
{
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            dst[i * dst_stride + j] = src[i * src_stride + j];
}

static char *test_sad_stride(int add_scale1)
{
    float *wide1 = malloc(sizeof(float) * WIDE_STRIDE * H);
    float *wide2 = malloc(sizeof(float) * WIDE_STRIDE * H);
    float *compact1 = malloc(sizeof(float) * COMPACT_STRIDE * H);
    float *compact2 = malloc(sizeof(float) * COMPACT_STRIDE * H);
    mu_assert("allocation failed", wide1 && wide2 && compact1 && compact2);

    fill(wide1, WIDE_STRIDE, 1);
    fill(wide2, WIDE_STRIDE, 2);
    repack(wide1, WIDE_STRIDE, compact1, COMPACT_STRIDE);
    repack(wide2, WIDE_STRIDE, compact2, COMPACT_STRIDE);

    const float expected =
        vmaf_image_sad_c(compact1, compact2, W, H, COMPACT_STRIDE, COMPACT_STRIDE, add_scale1);
    const float actual =
        vmaf_image_sad_c(wide1, wide2, W, H, WIDE_STRIDE, WIDE_STRIDE, add_scale1);

    free(wide1);
    free(wide2);
    free(compact1);
    free(compact2);

    mu_assert("the sad must not depend on the stride", actual == expected);
    return NULL;
}

static char *test_sad_stride_scale0()
{
    return test_sad_stride(0);
}

static char *test_sad_stride_scale1()
{
    return test_sad_stride(1);
}

char *run_tests()
{
    mu_run_test(test_sad_stride_scale0);
    mu_run_test(test_sad_stride_scale1);
    return NULL;
}
