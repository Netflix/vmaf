#include "feature/integer_adm.h"

#include <arm_neon.h>

// Signed 32 Bits //
// The macro instance int32x4_t accumulators and accumlates the multiplication of 4 int16x8_t vectors with a 4 elements filter.
#define NEON_ADM_INSTANCE_ACCUM_AND_MACC_VEC_4_ELEMS_ARR_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_name, init_vec, vec_name, filter_vec) \
    int32x4_t accum_name##_l = vmlal_lane_s16(init_vec, vget_low_s16(vec_name[0]), filter_vec, 0);                                 \
    int32x4_t accum_name##_h = vmlal_high_lane_s16(init_vec, vec_name[0], filter_vec, 0);                                          \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_name[1]), filter_vec, 1);                                     \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_name[1], filter_vec, 1);                                              \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_name[2]), filter_vec, 2);                                     \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_name[2], filter_vec, 2);                                              \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_name[3]), filter_vec, 3);                                     \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_name[3], filter_vec, 3);

// The macro instance int32x4_t accumulators and accumlates the multiplication of 2 int16x8x2_t vectors with a 4 elements filter.
#define NEON_ADM_INSTANCE_ACCUM_AND_MACC_PAIR_VEC_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_name, vec_pair_1, vec_pair_2, init_vec, filter_vec) \
    int32x4_t accum_name##_l = vmlal_lane_s16(init_vec, vget_low_s16(vec_pair_1.val[0]), filter_vec, 0);                                  \
    int32x4_t accum_name##_h = vmlal_high_lane_s16(init_vec, vec_pair_1.val[0], filter_vec, 0);                                           \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_pair_1.val[1]), filter_vec, 1);                                      \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_pair_1.val[1], filter_vec, 1);                                               \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_pair_2.val[0]), filter_vec, 2);                                      \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_pair_2.val[0], filter_vec, 2);                                               \
    accum_name##_l = vmlal_lane_s16(accum_name##_l, vget_low_s16(vec_pair_2.val[1]), filter_vec, 3);                                      \
    accum_name##_h = vmlal_high_lane_s16(accum_name##_h, vec_pair_2.val[1], filter_vec, 3);

// The macro takes low and high accumulators, shift them, unzip them into single int16x8_t vector, and stores it
#define NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(accum_name, shift_vec, store_pointer)  \
    {                                                                                                   \
        int16x8_t accum_name = vuzp1q_s16(vreinterpretq_s16_s32(vshlq_s32(accum_name##_l, shift_vec)),  \
                                          vreinterpretq_s16_s32(vshlq_s32(accum_name##_h, shift_vec))); \
        vst1q_s16(store_pointer, accum_name);                                                           \
    }

void adm_dwt2_8_neon(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride)
{
    const int16_t shift_VP = 8;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 128;
    const int32_t add_shift_HP = 32768;

    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;

    const int16x4_t filter_lo_vec = vld1_s16(dwt2_db2_coeffs_lo);
    const int16x4_t filter_hi_vec = vld1_s16(dwt2_db2_coeffs_hi);
    const int32x4_t normalize_vec_vp_lo = vdupq_n_s32((-1 * (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP) + add_shift_VP);
    const int32x4_t normalize_vec_vp_hi = vdupq_n_s32((-1 * (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP) + add_shift_VP);
    const int32x4_t shift_vp_vec = vdupq_n_s32(-shift_VP);
    const int32x4_t add_shift_hp_vec = vdupq_n_s32(add_shift_HP);
    const int32x4_t shift_hp_vec = vdupq_n_s32(-shift_HP);

    for (int i = 0; i < (h + 1) / 2; ++i)
    {
        /* Vertical pass. */
        const uint8_t *p_src_0 = src + ind_y[0][i] * src_stride;
        const uint8_t *p_src_1 = src + ind_y[1][i] * src_stride;
        const uint8_t *p_src_2 = src + ind_y[2][i] * src_stride;
        const uint8_t *p_src_3 = src + ind_y[3][i] * src_stride;

        for (int j = 0; j < w - 15; j += 16, p_src_0 += 16, p_src_1 += 16, p_src_2 += 16, p_src_3 += 16)
        {
            uint8x16_t u_8[4];
            int16x8_t s_16_l[4], s_16_h[4];

            u_8[0] = vld1q_u8(p_src_0);
            u_8[1] = vld1q_u8(p_src_1);
            u_8[2] = vld1q_u8(p_src_2);
            u_8[3] = vld1q_u8(p_src_3);

            s_16_l[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_8[0])));
            s_16_h[0] = vreinterpretq_s16_u16(vmovl_high_u8(u_8[0]));
            s_16_l[1] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_8[1])));
            s_16_h[1] = vreinterpretq_s16_u16(vmovl_high_u8(u_8[1]));
            s_16_l[2] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_8[2])));
            s_16_h[2] = vreinterpretq_s16_u16(vmovl_high_u8(u_8[2]));
            s_16_l[3] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_8[3])));
            s_16_h[3] = vreinterpretq_s16_u16(vmovl_high_u8(u_8[3]));

            NEON_ADM_INSTANCE_ACCUM_AND_MACC_VEC_4_ELEMS_ARR_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_lo_l, normalize_vec_vp_lo, s_16_l, filter_lo_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_VEC_4_ELEMS_ARR_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_lo_h, normalize_vec_vp_lo, s_16_h, filter_lo_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_VEC_4_ELEMS_ARR_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_hi_l, normalize_vec_vp_hi, s_16_l, filter_hi_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_VEC_4_ELEMS_ARR_BY_4_ELEMENTS_FILTER_S32X4_LH(accum_hi_h, normalize_vec_vp_hi, s_16_h, filter_hi_vec);

            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(accum_lo_l, shift_vp_vec, tmplo + j);
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(accum_lo_h, shift_vp_vec, tmplo + j + 8);
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(accum_hi_l, shift_vp_vec, tmphi + j);
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(accum_hi_h, shift_vp_vec, tmphi + j + 8);
        }

        /* Horizontal pass (lo and hi). */
        // j = 0 is a special case (entry src_ind_x[0][0] is mirrored 101 instead of -1).
        // Note that j = ((w + 1) / 2) has same mirroring yet that value is ignored/overriden so no need to implement it seperatly.
        /* from: dwt2_src_indices_filt()
            src_ind_x[0][0] = 1;
            src_ind_x[1][0] = 0;
            src_ind_x[2][0] = 1;
            src_ind_x[3][0] = 2;
        */
        int32_t accum_a = add_shift_HP;
        int32_t accum_v = add_shift_HP;
        int32_t accum_h = add_shift_HP;
        int32_t accum_d = add_shift_HP;

        for (int idx = 0; idx < 3; idx++)
        {
            int j_idx = ind_x[idx][0];
            int16_t s_lo = tmplo[j_idx];
            int16_t s_hi = tmphi[j_idx];
            accum_a += (int32_t)dwt2_db2_coeffs_lo[idx] * s_lo;
            accum_v += (int32_t)dwt2_db2_coeffs_hi[idx] * s_lo;
            accum_h += (int32_t)dwt2_db2_coeffs_lo[idx] * s_hi;
            accum_d += (int32_t)dwt2_db2_coeffs_hi[idx] * s_hi;
        }

        dst->band_a[i * dst_stride] = accum_a >> shift_HP;
        dst->band_v[i * dst_stride] = accum_v >> shift_HP;
        dst->band_h[i * dst_stride] = accum_h >> shift_HP;
        dst->band_d[i * dst_stride] = accum_d >> shift_HP;

        /* Vectorize code assumes w is even (assumption is valid as we call the whole function only in case !(w%8) )
            As so the whole ind_x can be ignored as:
                ind1 = 2 * j;
                ind0 = ind1 - 1;
                ind2 = ind1 + 1;
                ind3 = ind1 + 2;
                src_ind_x[0][j] = ind0; \\ 2*j-1
                src_ind_x[1][j] = ind1; \\ 2*j
                src_ind_x[2][j] = ind2; \\ 2*j+1
                src_ind_x[3][j] = ind3; \\ 2*j+2
         */

        int16_t *p_low = tmplo + 2;  // 2*j (j=1) - 1 --> 2 -1 = 1
        int16_t *p_high = tmphi + 2; // 2*j (j=1) - 1 --> 2 -1 = 1
        int stride_h = i * dst_stride + 1;
        for (int j = 1; j < ((w + 1) / 2); j += 8, p_low += 16, p_high += 16, stride_h += 8)
        {
            int16x8x2_t low_s0s1_vec_s16, low_s2s3_vec_s16;
            int16x8x2_t high_s0s1_vec_s16, high_s2s3_vec_s16;

            low_s0s1_vec_s16 = vld2q_s16(p_low - 1);
            low_s2s3_vec_s16 = vld2q_s16(p_low + 1);
            high_s0s1_vec_s16 = vld2q_s16(p_high - 1);
            high_s2s3_vec_s16 = vld2q_s16(p_high + 1);

            NEON_ADM_INSTANCE_ACCUM_AND_MACC_PAIR_VEC_BY_4_ELEMENTS_FILTER_S32X4_LH(low_accum_vec_lo, low_s0s1_vec_s16, low_s2s3_vec_s16, add_shift_hp_vec, filter_lo_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_PAIR_VEC_BY_4_ELEMENTS_FILTER_S32X4_LH(low_accum_vec_hi, low_s0s1_vec_s16, low_s2s3_vec_s16, add_shift_hp_vec, filter_hi_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_PAIR_VEC_BY_4_ELEMENTS_FILTER_S32X4_LH(high_accum_vec_lo, high_s0s1_vec_s16, high_s2s3_vec_s16, add_shift_hp_vec, filter_lo_vec);
            NEON_ADM_INSTANCE_ACCUM_AND_MACC_PAIR_VEC_BY_4_ELEMENTS_FILTER_S32X4_LH(high_accum_vec_hi, high_s0s1_vec_s16, high_s2s3_vec_s16, add_shift_hp_vec, filter_hi_vec);

            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(low_accum_vec_lo, shift_hp_vec, (dst->band_a + stride_h));
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(low_accum_vec_hi, shift_hp_vec, (dst->band_v + stride_h));
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(high_accum_vec_lo, shift_hp_vec, (dst->band_h + stride_h));
            NEON_ADM_STORE_ZIPPED_ACCUM_LO_HI_WITH_RIGHT_SHIFT_S16x8(high_accum_vec_hi, shift_hp_vec, (dst->band_d + stride_h));
        }
    }
}
