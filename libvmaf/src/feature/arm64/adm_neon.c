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

    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;
    const int w_half = (w + 1) / 2;
    const int half_w_mod8 = (w_half - 2) - ((w_half - 3) % 8);

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

        for (int idx = 0; idx < 4; idx++)
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
        for (int j = 1; j < half_w_mod8; j += 8, p_low += 16, p_high += 16, stride_h += 8)
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

        for (int j = half_w_mod8; j < w_half; ++j)
        {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            int32_t accum = add_shift_HP;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (int16_t)(accum >> shift_HP);

            accum = add_shift_HP;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (int16_t)(accum >> shift_HP);

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = add_shift_HP;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (int16_t)(accum >> shift_HP);

            accum = add_shift_HP;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (int16_t)(accum >> shift_HP);
        }
    }
}

static inline float32x4_t adm_dot_s16(int16x4_t ah, int16x4_t av,
                                     int16x4_t bh, int16x4_t bv)
{
    const int32x4_t h = vmull_s16(ah, bh);
    const int32x4_t v = vmull_s16(av, bv);
    const int64x2_t lo = vaddl_s32(vget_low_s32(h), vget_low_s32(v));
    const int64x2_t hi = vaddl_s32(vget_high_s32(h), vget_high_s32(v));
    return vcombine_f32(vcvt_f32_f64(vcvtq_f64_s64(lo)),
                        vcvt_f32_f64(vcvtq_f64_s64(hi)));
}

static inline uint32x2_t adm_angle_f64(float32x2_t dot, float32x2_t omag,
                                      float32x2_t tmag, double cos_sq)
{
    const float64x2_t d = vmulq_n_f64(vcvt_f64_f32(dot), 1.0 / 4096.0);
    const float64x2_t o = vmulq_n_f64(vcvt_f64_f32(omag), 1.0 / 4096.0);
    const float64x2_t t = vmulq_n_f64(vcvt_f64_f32(tmag), 1.0 / 4096.0);
    return vmovn_u64(vandq_u64(vcgeq_f64(d, vdupq_n_f64(0)),
        vcgeq_f64(vmulq_f64(d, d), vmulq_f64(vmulq_n_f64(o, cos_sq), t))));
}

static inline int16x4_t adm_decouple_band(const int16_t *ref, int16x4_t o,
                                         int16x4_t t, uint32x4_t angle,
                                         int gain, const int32_t *lookup)
{
    const int32_t div[4] = { lookup[ref[0] + 32768], lookup[ref[1] + 32768],
                             lookup[ref[2] + 32768], lookup[ref[3] + 32768] };
    const int32x4_t recip = vld1q_s32(div);
    const int32x4_t dis = vmovl_s16(t), orig = vmovl_s16(o);
    const int32x4_t ratio = vcombine_s32(
        vrshrn_n_s64(vmull_s32(vget_low_s32(recip), vget_low_s32(dis)), 15),
        vrshrn_n_s64(vmull_s32(vget_high_s32(recip), vget_high_s32(dis)), 15));
    const int32x4_t k = vbslq_s32(vceqq_s32(orig, vdupq_n_s32(0)),
        vdupq_n_s32(32768), vmaxq_s32(vdupq_n_s32(0),
                                      vminq_s32(ratio, vdupq_n_s32(32768))));
    const int32x4_t rst = vrshrq_n_s32(vmulq_s32(k, orig), 15);
    if (!vmaxvq_u32(angle)) return vmovn_s32(rst);
    const int32x4_t scaled = vmulq_n_s32(rst, gain);
    const uint32x4_t active = vandq_u32(angle, vcgtq_s32(k, vdupq_n_s32(0)));
    return vmovn_s32(vbslq_s32(
        vandq_u32(active, vcgtq_s32(orig, vdupq_n_s32(0))), vminq_s32(scaled, dis),
        vbslq_s32(vandq_u32(active, vcltq_s32(orig, vdupq_n_s32(0))),
                   vmaxq_s32(scaled, dis), rst)));
}

void adm_decouple_neon(AdmBuffer *buf, int w, int h, int stride,
                       double adm_enhn_gain_limit, int32_t *adm_div_lookup)
{
    const float cos_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2, bottom = h - top + 2;
    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > w ? w : right;
    bottom = bottom > h ? h : bottom;
    if (right - left < 4 || adm_enhn_gain_limit != (int) adm_enhn_gain_limit) {
        adm_decouple(buf, w, h, stride, adm_enhn_gain_limit, adm_div_lookup);
        return;
    }
    for (int i = top; i < bottom; i++) {
        for (int j = left; ; j += 4) {
            if (j > right - 4) j = right - 4;
            const int off = i * stride + j;
            const int16x4_t oh = vld1_s16(buf->ref_dwt2.band_h + off);
            const int16x4_t ov = vld1_s16(buf->ref_dwt2.band_v + off);
            const int16x4_t od = vld1_s16(buf->ref_dwt2.band_d + off);
            const int16x4_t th = vld1_s16(buf->dis_dwt2.band_h + off);
            const int16x4_t tv = vld1_s16(buf->dis_dwt2.band_v + off);
            const int16x4_t td = vld1_s16(buf->dis_dwt2.band_d + off);
            uint32x4_t angle = vdupq_n_u32(0);
            // With gain 1, the Q15 reconstruction already lies between zero and dis.
            if (adm_enhn_gain_limit != 1.0) {
                const float32x4_t dot = adm_dot_s16(oh, ov, th, tv);
                const float32x4_t omag = adm_dot_s16(oh, ov, oh, ov);
                const float32x4_t tmag = adm_dot_s16(th, tv, th, tv);
                angle = vcombine_u32(
                    adm_angle_f64(vget_low_f32(dot), vget_low_f32(omag),
                                   vget_low_f32(tmag), cos_sq),
                    adm_angle_f64(vget_high_f32(dot), vget_high_f32(omag),
                                   vget_high_f32(tmag), cos_sq));
            }
            const int16x4_t rh = adm_decouple_band(buf->ref_dwt2.band_h + off,
                oh, th, angle, adm_enhn_gain_limit, adm_div_lookup);
            const int16x4_t rv = adm_decouple_band(buf->ref_dwt2.band_v + off,
                ov, tv, angle, adm_enhn_gain_limit, adm_div_lookup);
            const int16x4_t rd = adm_decouple_band(buf->ref_dwt2.band_d + off,
                od, td, angle, adm_enhn_gain_limit, adm_div_lookup);
            vst1_s16(buf->decouple_r.band_h + off, rh);
            vst1_s16(buf->decouple_r.band_v + off, rv);
            vst1_s16(buf->decouple_r.band_d + off, rd);
            vst1_s16(buf->decouple_a.band_h + off, vsub_s16(th, rh));
            vst1_s16(buf->decouple_a.band_v + off, vsub_s16(tv, rv));
            vst1_s16(buf->decouple_a.band_d + off, vsub_s16(td, rd));
            if (j == right - 4) break;
        }
    }
}
