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

#include "test.h"
#include "feature/ciede.c"

static int close_enough(float a, float b)
{
    const float epsilon = 1e-9f;
    return fabs(a - b) < epsilon;
}

static const KSubArgs default_ksub = { .l = 0.65, .c = 1.0, .h = 4.0 };

static char *test_ciede()
{
    const LABColor color_1 = { .l = 0.052488625, .a = -0.587470829, .b = -8.98771572 };
    const LABColor color_2 = { .l = 0.465437293, .a = 0.386364758, .b = -12.7648535 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 2.54780269",
              close_enough(de00, 2.54780269));

    return NULL;
}

static char *test_ciede2()
{
    const LABColor color_1 = { .l = 87.156334, .a = -12.049645, .b = -1.205325 };
    const LABColor color_2 = { .l = 83.455727, .a = -9.040445, .b = -8.894289 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.22714281",
              close_enough(de00, 4.22714281));

    return NULL;
}

static char *test_ciede3()
{
    const LABColor color_1 = { .l = 79.718491, .a = 9.109915, .b = 13.727915 };
    const LABColor color_2 = { .l = 78.717224, .a = 7.526546, .b = 5.597448 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 4.26012468",
              close_enough(de00, 4.26012468));

    return NULL;
}

static char *test_ciede4()
{
    const LABColor color_1 = { .l = 99.205299, .a = -3.339410, .b = 1.205873 };
    const LABColor color_2 = { .l = 97.991730, .a = -2.497345, .b = 2.473533 };

    const float de00 = ciede2000(color_1, color_2, default_ksub);
    mu_assert("de00 for this input should be 1.26915979",
              close_enough(de00, 1.26915979));

    return NULL;
}

static char *test_chroma_422()
{
    /* Four distinct rows must be preserved; each chroma column is repeated
     * horizontally. Padding is initialized so an incorrect column lookup
     * cannot happen to read the expected value. */
    static const unsigned expected[4][8] = {
        { 1, 1, 2, 2, 3, 3, 4, 4 },
        { 6, 6, 7, 7, 8, 8, 9, 9 },
        { 11, 11, 12, 12, 13, 13, 14, 14 },
        { 16, 16, 17, 17, 18, 18, 19, 19 },
    };
    const unsigned depths[] = { 8, 10, 12, 16 };
    for (unsigned k = 0; k < sizeof(depths) / sizeof(depths[0]); k++) {
        const unsigned bpc = depths[k], scale = 1U << (bpc - 8);
        VmafPicture in, out;
        int err = vmaf_picture_alloc(&in, VMAF_PIX_FMT_YUV422P, bpc, 8, 4);
        mu_assert("could not allocate 4:2:2 picture", !err);
        err = vmaf_picture_alloc(&out, VMAF_PIX_FMT_YUV444P, bpc, 8, 4);
        if (err) vmaf_picture_unref(&in);
        mu_assert("could not allocate 4:4:4 picture", !err);
        for (unsigned p = 0; p < 3; p++) {
            memset(in.data[p], 0, in.stride[p] * in.h[p]);
            for (unsigned y = 0; y < in.h[p]; y++) {
                uint8_t *row = (uint8_t *)in.data[p] + y * in.stride[p];
                for (unsigned x = 0; x < in.w[p]; x++) {
                    const unsigned value = (20 * p + 5 * y + x + 1) * scale;
                    if (bpc == 8) row[x] = value;
                    else ((uint16_t *)row)[x] = value;
                }
            }
        }
        if (bpc == 8) scale_chroma_planes(&in, &out);
        else scale_chroma_planes_hbd(&in, &out);
        int matches = 1;
        for (unsigned p = 0; p < 3; p++) {
            for (unsigned y = 0; y < out.h[p]; y++) {
                const uint8_t *row = (uint8_t *)out.data[p] + y * out.stride[p];
                for (unsigned x = 0; x < out.w[p]; x++) {
                    const unsigned value = bpc == 8 ? row[x] : ((const uint16_t *)row)[x];
                    const unsigned want = p ? (20 * p + expected[y][x]) * scale
                                            : (5 * y + x + 1) * scale;
                    if (value != want) matches = 0;
                }
            }
        }
        vmaf_picture_unref(&in);
        vmaf_picture_unref(&out);
        mu_assert("4:2:2 must duplicate columns and preserve rows", matches);
    }
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_ciede);
    mu_run_test(test_ciede2);
    mu_run_test(test_ciede3);
    mu_run_test(test_ciede4);
    mu_run_test(test_chroma_422);
    return NULL;
}
