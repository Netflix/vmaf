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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "common/macros.h"
#include "common/alignment.h"
#include "mem.h"

 /**
  * Works similar to floating-point function convolution_edge_s
 */
FORCE_INLINE inline uint32_t integer_convolution_edge_16_s(bool horizontal, const uint16_t *filter, int filter_width, const uint16_t *src, int width, int height, int stride, int i, int j)
{
    int radius = filter_width / 2;

    uint32_t accum = 0;
    for (int k = 0; k < filter_width; ++k) {
        int i_tap = horizontal ? i : i - radius + k;
        int j_tap = horizontal ? j - radius + k : j;

        // Handle edges by mirroring.
        if (horizontal) {
            if (j_tap < 0)
                j_tap = -j_tap;
            else if (j_tap >= width)
                j_tap = width - (j_tap - width + 1);
        }
        else {
            if (i_tap < 0)
                i_tap = -i_tap;
            else if (i_tap >= height)
                i_tap = height - (i_tap - height + 1);
        }

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

/**
 * The input src here is of type Q16, Filter coefficients is Q16
 * Hence accum is shifted by 16 bits to store in dst as Q16
*/
FORCE_INLINE inline void integer_convolution_x_16(const uint16_t *filter, int filter_width, const uint16_t *src, uint16_t *dst, int width, int height, int src_stride, int dst_stride, int step)
{
    int radius = filter_width / 2;
    int borders_left = vmaf_ceiln(radius, step);
    int borders_right = vmaf_floorn(width - (filter_width - radius), step);
    int shift_add_round = 32768;

    //pointers for optimize data manapulation
    uint16_t *src_p, *src_p1, *src_p2;
    src_p = src + (borders_left - radius);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < borders_left; j += step) {
            dst[i * dst_stride + j / step] = (integer_convolution_edge_16_s(true, filter, filter_width, src, width, height, src_stride, i, j) + shift_add_round) >> 16;
        }
        src_p1 = src_p;
        for (int j = borders_left; j < borders_right; j += step) {
            uint32_t accum = 0;
            src_p2 = src_p1;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1 += step;
            dst[i * dst_stride + j / step] = (accum + shift_add_round) >> 16;
        }
        src_p += src_stride;
        for (int j = borders_right; j < width; j += step) {
            dst[i * dst_stride + j / step] = (integer_convolution_edge_16_s(true, filter, filter_width, src, width, height, src_stride, i, j) + shift_add_round) >> 16;
        }
    }
}

/**
 * The input src here is of type Q8, Filter coefficients is Q16
 * Hence accum is shifted by 8 bits to store in dst as Q16
*/
FORCE_INLINE inline void integer_convolution_y_16(const uint16_t *filter, int filter_width, const uint16_t *src, uint16_t *dst, int width, int height, int src_stride, int dst_stride, int step, int inp_size_bits)
{
    int radius = filter_width / 2;
    int borders_top = vmaf_ceiln(radius, step);
    int borders_bottom = vmaf_floorn(height - (filter_width - radius), step);
    int add_before_shift = (int)pow(2, (inp_size_bits - 1));
    int shift_var = inp_size_bits;

    //pointers for optimize data manapulation
    uint16_t *src_p, *src_p1, *src_p2;
    src_p = src + (borders_top - radius)*src_stride;

    uint16_t step_stride = step * src_stride;

    for (int i = 0; i < borders_top; i += step) {
        for (int j = 0; j < width; ++j) {
            dst[(i / step) * dst_stride + j] = (integer_convolution_edge_16_s(false, filter, filter_width, src, width, height, src_stride, i, j) + add_before_shift) >> shift_var;
        }
    }
    for (int i = borders_top; i < borders_bottom; i += step) {
        src_p1 = src_p;
        for (int j = 0; j < width; ++j) {
            src_p2 = src_p1;
            uint32_t accum = 0;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += step_stride;
            }
            dst[(i / step) * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }
        src_p += step_stride;
    }
    for (int i = borders_bottom; i < height; i += step) {
        for (int j = 0; j < width; ++j) {
            dst[(i / step) * dst_stride + j] = (integer_convolution_edge_16_s(false, filter, filter_width, src, width, height, src_stride, i, j) + add_before_shift) >> shift_var;
        }
    }
}

/**
 * Works similar to floating-point function vmaf_image_sad_c
 */
FORCE_INLINE inline float integer_vmaf_image_sad_c(const uint16_t *img1, const uint16_t *img2, int width, int height, int img1_stride, int img2_stride)
{
    uint64_t accum = 0;


    for (int i = 0; i < height; ++i) {
    	uint32_t accum_inner = 0;
        for (int j = 0; j < width; ++j) {
            uint16_t img1px = img1[i * img1_stride + j];
            uint16_t img2px = img2[i * img2_stride + j];

            accum_inner += (uint32_t) abs(img1px - img2px);
            //assuming it is 4k video, max accum_inner is 2^16*3840
        }
        accum += (uint64_t) accum_inner;
        //assuming it is 4k video, max accum is 2^16*3840*1920 which uses upto 39bits
    }
    float f_accum = (float) accum/256.0;
    return (float) (f_accum / (width * height));
}

/**
 * Works similar to floating-point function convolution_f32_c_s for 16 bit input
 */
void integer_convolution_16(const uint16_t *filter, int filter_width, const uint16_t *src, uint16_t *dst, uint16_t *tmp, int width, int height, int src_stride, int dst_stride, int inp_size_bits)
{
    // convolve along y first then x
    integer_convolution_y_16(filter, filter_width, src, tmp, width, height, src_stride, dst_stride, 1, inp_size_bits);
    integer_convolution_x_16(filter, filter_width, tmp, dst, width, height, dst_stride, dst_stride, 1);
}

/**
* Works similar to floating-point function convolution_edge_s
*/
FORCE_INLINE inline uint32_t integer_convolution_edge_8_s(bool horizontal, const uint16_t *filter, int filter_width, const uint8_t *src, int width, int height, int stride, int i, int j)
{
    int radius = filter_width / 2;

    uint32_t accum = 0;
    for (int k = 0; k < filter_width; ++k) {
        int i_tap = horizontal ? i : i - radius + k;
        int j_tap = horizontal ? j - radius + k : j;

        // Handle edges by mirroring.
        if (horizontal) {
            if (j_tap < 0)
                j_tap = -j_tap;
            else if (j_tap >= width)
                j_tap = width - (j_tap - width + 1);
        }
        else {
            if (i_tap < 0)
                i_tap = -i_tap;
            else if (i_tap >= height)
                i_tap = height - (i_tap - height + 1);
        }

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

/**
 * The input src here is of type Q8, Filter coefficients is Q16
 * Hence accum is shifted by 8 bits to store in dst as Q16
*/
FORCE_INLINE inline void integer_convolution_y_8(const uint16_t *filter, int filter_width, const uint8_t *src, uint16_t *dst, int width, int height, int src_stride, int dst_stride, int step, int inp_size_bits)
{
    int radius = filter_width / 2;
    int borders_top = vmaf_ceiln(radius, step);
    int borders_bottom = vmaf_floorn(height - (filter_width - radius), step);
    int add_before_shift = (int)pow(2, (inp_size_bits - 1));
    int shift_var = inp_size_bits;

    //pointers for optimize data manapulation
    uint8_t *src_p, *src_p1, *src_p2;
    src_p = src + (borders_top - radius)*src_stride;

    int step_stride = step * src_stride;

    for (int i = 0; i < borders_top; i += step) {
        for (int j = 0; j < width; ++j) {
            dst[(i / step) * dst_stride + j] = (integer_convolution_edge_8_s(false, filter, filter_width, src, width, height, src_stride, i, j) + add_before_shift) >> shift_var;
        }
    }
    for (int i = borders_top; i < borders_bottom; i += step) {
        src_p1 = src_p;
        for (int j = 0; j < width; ++j) {
            src_p2 = src_p1;
            uint32_t accum = 0;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += step_stride;
            }
            dst[(i / step) * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }
        src_p += step_stride;
    }
    for (int i = borders_bottom; i < height; i += step) {
        for (int j = 0; j < width; ++j) {
            dst[(i / step) * dst_stride + j] = (integer_convolution_edge_8_s(false, filter, filter_width, src, width, height, src_stride, i, j) + add_before_shift) >> shift_var;
        }
    }
}

/**
 * Works similar to floating-point function convolution_f32_c_s for 8 bit input
 */
void integer_convolution_8(const uint16_t *filter, int filter_width, const uint8_t *src, uint16_t *dst, uint16_t *tmp, int width, int height, int src_stride, int dst_stride, int inp_size_bits)
{
    // convolve along y first then x
    integer_convolution_y_8(filter, filter_width, src, tmp, width, height, src_stride, dst_stride, 1, inp_size_bits);
    integer_convolution_x_16(filter, filter_width, tmp, dst, width, height, dst_stride, dst_stride, 1);
}

/**
 * Works similar to floating-point function compute_motion
 */
int integer_compute_motion(const uint16_t *ref, const uint16_t *dis, int w, int h, int ref_stride, int dis_stride, double *score)
{

    if (ref_stride % sizeof(uint16_t) != 0)
    {
        printf("error: ref_stride %% sizeof(uint16_t) != 0, ref_stride = %d, sizeof(uint16_t) = %zu.\n", ref_stride, sizeof(uint16_t));
        fflush(stdout);
        goto fail;
    }
    if (dis_stride % sizeof(uint16_t) != 0)
    {
        printf("error: dis_stride %% sizeof(uint16_t) != 0, dis_stride = %d, sizeof(uint16_t) = %zu.\n", dis_stride, sizeof(uint16_t));
        fflush(stdout);
        goto fail;
    }
    // stride for integer_vmaf_image_sad_c is in terms of (sizeof(uint16_t) bytes)
    *score = integer_vmaf_image_sad_c(ref, dis, w, h, ref_stride >> 1, dis_stride >> 1);

    return 0;

fail:
    return 1;
}