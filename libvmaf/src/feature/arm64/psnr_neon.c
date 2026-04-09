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
#include <stdint.h>

#include "psnr_neon.h"

uint32_t psnr_sse_line_8_neon(const uint8_t *ref, const uint8_t *dis,
                               unsigned w)
{
    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);
    unsigned j = 0;

    for (; j + 32 <= w; j += 32) {
        /* Load 16 bytes each, two iterations */
        uint8x16_t r0 = vld1q_u8(ref + j);
        uint8x16_t d0 = vld1q_u8(dis + j);
        uint8x16_t r1 = vld1q_u8(ref + j + 16);
        uint8x16_t d1 = vld1q_u8(dis + j + 16);

        /* Absolute difference */
        uint8x16_t abs0 = vabdq_u8(r0, d0);
        uint8x16_t abs1 = vabdq_u8(r1, d1);

        /* Widen to u16 and accumulate squared diffs */
        uint16x8_t a0_lo = vmull_u8(vget_low_u8(abs0), vget_low_u8(abs0));
        uint16x8_t a0_hi = vmull_u8(vget_high_u8(abs0), vget_high_u8(abs0));
        uint16x8_t a1_lo = vmull_u8(vget_low_u8(abs1), vget_low_u8(abs1));
        uint16x8_t a1_hi = vmull_u8(vget_high_u8(abs1), vget_high_u8(abs1));

        /* Widen u16 → u32 and accumulate */
        sum0 = vaddq_u32(sum0, vaddl_u16(vget_low_u16(a0_lo),
                                          vget_high_u16(a0_lo)));
        sum1 = vaddq_u32(sum1, vaddl_u16(vget_low_u16(a0_hi),
                                          vget_high_u16(a0_hi)));
        sum0 = vaddq_u32(sum0, vaddl_u16(vget_low_u16(a1_lo),
                                          vget_high_u16(a1_lo)));
        sum1 = vaddq_u32(sum1, vaddl_u16(vget_low_u16(a1_hi),
                                          vget_high_u16(a1_hi)));
    }

    /* Horizontal sum */
    uint32x4_t total = vaddq_u32(sum0, sum1);
    uint32_t result = vaddvq_u32(total);

    /* Scalar tail */
    for (; j < w; j++) {
        const int16_t e = ref[j] - dis[j];
        result += (uint32_t)(e * e);
    }

    return result;
}

uint64_t psnr_sse_line_16_neon(const uint16_t *ref, const uint16_t *dis,
                                unsigned w)
{
    uint64x2_t sum0 = vdupq_n_u64(0);
    uint64x2_t sum1 = vdupq_n_u64(0);
    unsigned j = 0;

    for (; j + 8 <= w; j += 8) {
        /* Load 8 uint16 */
        uint16x8_t r = vld1q_u16(ref + j);
        uint16x8_t d = vld1q_u16(dis + j);

        /* Absolute difference → uint16 (safe for all bit depths) */
        uint16x8_t absdiff = vabdq_u16(r, d);

        /* Widen to uint32 and square */
        uint32x4_t lo = vmull_u16(vget_low_u16(absdiff), vget_low_u16(absdiff));
        uint32x4_t hi = vmull_u16(vget_high_u16(absdiff), vget_high_u16(absdiff));

        /* Widen to uint64 and accumulate */
        sum0 = vaddq_u64(sum0, vaddl_u32(vget_low_u32(lo), vget_high_u32(lo)));
        sum1 = vaddq_u64(sum1, vaddl_u32(vget_low_u32(hi), vget_high_u32(hi)));
    }

    /* Horizontal sum */
    uint64x2_t total = vaddq_u64(sum0, sum1);
    uint64_t result = vaddvq_u64(total);

    /* Scalar tail */
    for (; j < w; j++) {
        const int32_t e = (int32_t)ref[j] - (int32_t)dis[j];
        result += (uint64_t)((uint32_t)(e * e));
    }

    return result;
}
