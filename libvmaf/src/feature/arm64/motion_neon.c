#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "feature/integer_motion.h"

static inline int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

// Phase 2: x_conv + abs + SAD for one row of int32 y_row.
// Processes 4 int32 columns at a time via widening s32->s64 multiply-accumulate.
static inline uint32_t
x_conv_row_sad_neon(const int32_t *y_row, unsigned w)
{
    uint32_t row_sad = 0;

    // Scalar left edge (columns 0, 1) — mirror boundary
    unsigned j;
    for (j = 0; j < 2 && j < w; j++) {
        int64_t accum = 0;
        for (int k = 0; k < 5; k++) {
            int col = mirror((int)j - 2 + k, (int)w);
            accum += (int64_t)filter[k] * y_row[col];
        }
        int32_t val = (int32_t)((accum + (1 << 15)) >> 16);
        row_sad += (uint32_t)abs(val);
    }

    // SIMD middle: need y_row[j-2]..y_row[j+5], so j+6 <= w
    int32x4_t sad_acc = vdupq_n_s32(0);
    const int64x2_t round64 = vdupq_n_s64(1 << 15);
    for (; j + 6 <= w; j += 4) {
        int32x4_t y0 = vld1q_s32(y_row + j - 2);
        int32x4_t y1 = vld1q_s32(y_row + j - 1);
        int32x4_t y2 = vld1q_s32(y_row + j);
        int32x4_t y3 = vld1q_s32(y_row + j + 1);
        int32x4_t y4 = vld1q_s32(y_row + j + 2);

        int64x2_t acc_lo = vmull_n_s32(vget_low_s32(y0), (int32_t)filter[0]);
        acc_lo = vmlal_n_s32(acc_lo, vget_low_s32(y1), (int32_t)filter[1]);
        acc_lo = vmlal_n_s32(acc_lo, vget_low_s32(y2), (int32_t)filter[2]);
        acc_lo = vmlal_n_s32(acc_lo, vget_low_s32(y3), (int32_t)filter[3]);
        acc_lo = vmlal_n_s32(acc_lo, vget_low_s32(y4), (int32_t)filter[4]);

        int64x2_t acc_hi = vmull_n_s32(vget_high_s32(y0), (int32_t)filter[0]);
        acc_hi = vmlal_n_s32(acc_hi, vget_high_s32(y1), (int32_t)filter[1]);
        acc_hi = vmlal_n_s32(acc_hi, vget_high_s32(y2), (int32_t)filter[2]);
        acc_hi = vmlal_n_s32(acc_hi, vget_high_s32(y3), (int32_t)filter[3]);
        acc_hi = vmlal_n_s32(acc_hi, vget_high_s32(y4), (int32_t)filter[4]);

        acc_lo = vshrq_n_s64(vaddq_s64(acc_lo, round64), 16);
        acc_hi = vshrq_n_s64(vaddq_s64(acc_hi, round64), 16);

        int32x2_t res_lo = vmovn_s64(acc_lo);
        int32x2_t res_hi = vmovn_s64(acc_hi);
        int32x4_t result = vcombine_s32(res_lo, res_hi);

        sad_acc = vaddq_s32(sad_acc, vabsq_s32(result));
    }
    row_sad += vaddvq_s32(sad_acc);

    // Scalar right edge + tail
    for (; j < w; j++) {
        int64_t accum = 0;
        for (int k = 0; k < 5; k++) {
            int col = mirror((int)j - 2 + k, (int)w);
            accum += (int64_t)filter[k] * y_row[col];
        }
        int32_t val = (int32_t)((accum + (1 << 15)) >> 16);
        row_sad += (uint32_t)abs(val);
    }

    return row_sad;
}

uint64_t motion_score_pipeline_8_neon(const uint8_t *prev, ptrdiff_t prev_stride,
                                      const uint8_t *cur, ptrdiff_t cur_stride,
                                      int32_t *y_row, unsigned w, unsigned h,
                                      unsigned bpc)
{
    (void)bpc;
    uint64_t sad = 0;

    for (unsigned i = 0; i < h; i++) {
        const uint8_t *p[5], *c[5];
        for (int k = 0; k < 5; k++) {
            int r = mirror((int)i - 2 + k, (int)h);
            p[k] = prev + r * prev_stride;
            c[k] = cur + r * cur_stride;
        }

        // Phase 1: diff + y_conv -> y_row (16 columns at a time, shift >>8)
        unsigned j;
        int32x4_t nz_acc = vdupq_n_s32(0);
        const int32x4_t round8 = vdupq_n_s32(1 << 7);
        for (j = 0; j + 16 <= w; j += 16) {
            uint8x16_t p0 = vld1q_u8(p[0] + j), c0 = vld1q_u8(c[0] + j);
            uint8x16_t p1 = vld1q_u8(p[1] + j), c1 = vld1q_u8(c[1] + j);
            uint8x16_t p2 = vld1q_u8(p[2] + j), c2 = vld1q_u8(c[2] + j);
            uint8x16_t p3 = vld1q_u8(p[3] + j), c3 = vld1q_u8(c[3] + j);
            uint8x16_t p4 = vld1q_u8(p[4] + j), c4 = vld1q_u8(c[4] + j);

            int16x8_t d0_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p0), vget_low_u8(c0)));
            int16x8_t d0_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(p0), vget_high_u8(c0)));
            int16x8_t d1_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p1), vget_low_u8(c1)));
            int16x8_t d1_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(p1), vget_high_u8(c1)));
            int16x8_t d2_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p2), vget_low_u8(c2)));
            int16x8_t d2_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(p2), vget_high_u8(c2)));
            int16x8_t d3_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p3), vget_low_u8(c3)));
            int16x8_t d3_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(p3), vget_high_u8(c3)));
            int16x8_t d4_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(p4), vget_low_u8(c4)));
            int16x8_t d4_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(p4), vget_high_u8(c4)));

            int32x4_t acc0 = vdupq_n_s32(0); // columns j+0..j+3
            int32x4_t acc1 = vdupq_n_s32(0); // columns j+4..j+7
            int32x4_t acc2 = vdupq_n_s32(0); // columns j+8..j+11
            int32x4_t acc3 = vdupq_n_s32(0); // columns j+12..j+15

            acc0 = vmlal_n_s16(acc0, vget_low_s16(d0_lo), (int16_t)filter[0]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(d0_lo), (int16_t)filter[0]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(d0_hi), (int16_t)filter[0]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(d0_hi), (int16_t)filter[0]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(d1_lo), (int16_t)filter[1]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(d1_lo), (int16_t)filter[1]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(d1_hi), (int16_t)filter[1]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(d1_hi), (int16_t)filter[1]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(d2_lo), (int16_t)filter[2]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(d2_lo), (int16_t)filter[2]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(d2_hi), (int16_t)filter[2]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(d2_hi), (int16_t)filter[2]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(d3_lo), (int16_t)filter[3]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(d3_lo), (int16_t)filter[3]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(d3_hi), (int16_t)filter[3]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(d3_hi), (int16_t)filter[3]);

            acc0 = vmlal_n_s16(acc0, vget_low_s16(d4_lo), (int16_t)filter[4]);
            acc1 = vmlal_n_s16(acc1, vget_high_s16(d4_lo), (int16_t)filter[4]);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(d4_hi), (int16_t)filter[4]);
            acc3 = vmlal_n_s16(acc3, vget_high_s16(d4_hi), (int16_t)filter[4]);

            acc0 = vshrq_n_s32(vaddq_s32(acc0, round8), 8);
            acc1 = vshrq_n_s32(vaddq_s32(acc1, round8), 8);
            acc2 = vshrq_n_s32(vaddq_s32(acc2, round8), 8);
            acc3 = vshrq_n_s32(vaddq_s32(acc3, round8), 8);

            vst1q_s32(y_row + j, acc0);
            vst1q_s32(y_row + j + 4, acc1);
            vst1q_s32(y_row + j + 8, acc2);
            vst1q_s32(y_row + j + 12, acc3);

            nz_acc = vorrq_s32(nz_acc, vorrq_s32(vorrq_s32(acc0, acc1), vorrq_s32(acc2, acc3)));
        }

        // Scalar tail for phase 1
        int32_t nz_tail = 0;
        for (; j < w; j++) {
            int32_t accum = 0;
            for (int k = 0; k < 5; k++) {
                int32_t diff = p[k][j] - c[k][j];
                accum += (int32_t)filter[k] * diff;
            }
            y_row[j] = (accum + (1 << 7)) >> 8;
            nz_tail |= y_row[j];
        }

        uint32_t nz_scalar = vmaxvq_u32(vreinterpretq_u32_s32(nz_acc));
        if (!nz_scalar && !nz_tail) continue;

        // Phase 2: SIMD x_conv + abs + accumulate
        sad += x_conv_row_sad_neon(y_row, w);
    }

    return sad;
}
