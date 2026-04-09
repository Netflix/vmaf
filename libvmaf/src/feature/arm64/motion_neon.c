/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>

#include "feature/integer_motion.h"
#include "feature/common/alignment.h"

void x_convolution_16_neon(const uint16_t *src, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;

    /* Edge pixels: left */
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }

    /* Interior pixels: NEON vectorized, 8 uint16 per iteration */
    const uint16x4_t k0 = vdup_n_u16(3571);
    const uint16x4_t k1 = vdup_n_u16(16004);
    const uint16x4_t k2 = vdup_n_u16(26386);
    const uint32x4_t addnum = vdupq_n_u32(32768);

    for (unsigned i = 0; i < height; ++i) {
        const uint16_t *src_row = src + i * src_stride + (left_edge - radius);
        unsigned j = left_edge;

        for (; j + 8 <= right_edge; j += 8) {
            const uint16_t *sp = src_row;

            /* Load 5 overlapping vectors of 8 uint16 for the 5-tap filter */
            uint16x8_t s0 = vld1q_u16(sp);
            uint16x8_t s1 = vld1q_u16(sp + 1);
            uint16x8_t s2 = vld1q_u16(sp + 2);
            uint16x8_t s3 = vld1q_u16(sp + 3);
            uint16x8_t s4 = vld1q_u16(sp + 4);

            /* Widening multiply-accumulate: filter[0]*s0 + filter[1]*s1 + ... */
            /* Process low 4 elements */
            uint32x4_t acc_lo = vmull_u16(vget_low_u16(s0), k0);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(s1), k1);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(s2), k2);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(s3), k1);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(s4), k0);
            acc_lo = vaddq_u32(acc_lo, addnum);
            acc_lo = vshrq_n_u32(acc_lo, 16);

            /* Process high 4 elements */
            uint32x4_t acc_hi = vmull_u16(vget_high_u16(s0), k0);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(s1), k1);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(s2), k2);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(s3), k1);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(s4), k0);
            acc_hi = vaddq_u32(acc_hi, addnum);
            acc_hi = vshrq_n_u32(acc_hi, 16);

            /* Narrow back to uint16 */
            uint16x4_t res_lo = vmovn_u32(acc_lo);
            uint16x4_t res_hi = vmovn_u32(acc_hi);
            uint16x8_t result = vcombine_u16(res_lo, res_hi);

            vst1q_u16(dst + i * dst_stride + j, result);

            src_row += 8;
        }

        /* Scalar tail for remaining interior pixels */
        for (; j < right_edge; j++) {
            uint32_t accum = 0;
            const uint16_t *sp = src_row;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * sp[k];
            }
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
            src_row++;
        }
    }

    /* Edge pixels: right */
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }
}
