#include "feature/integer_adm.h"
#include "feature/barten_csf_tools.h"

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

static inline float
dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
        double adm_norm_view_dist, int adm_ref_display_height)
{
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    float temp = log10(pow(2.0, lambda + 1)*params->f0*params->g[theta] / r);
    float Q = 2.0*params->a*pow(10.0, params->k*temp*temp) /
        dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

static void adm_csf_factors(int scale, double adm_norm_view_dist,
                            int adm_ref_display_height, int adm_csf_mode,
                            double adm_csf_scale, double adm_csf_diag_scale,
                            float factor[2])
{
    if (adm_csf_mode == ADM_CSF_MODE_BARTEN) {
        factor[0] = barten_csf(scale, adm_norm_view_dist, adm_ref_display_height, DEFAULT_ADM_CSF_LUM, adm_csf_scale);
        factor[1] = barten_csf(scale, adm_norm_view_dist, adm_ref_display_height, DEFAULT_ADM_CSF_LUM, adm_csf_diag_scale);
    } else if (adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND) {
        factor[0] = barten_watson_blend_csf(scale, 0, adm_norm_view_dist, adm_ref_display_height);
        factor[1] = barten_watson_blend_csf(scale, 1, adm_norm_view_dist, adm_ref_display_height);
    } else if (adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND_MAE) {
        factor[0] = barten_watson_blend_csf_mae(scale, 0, adm_norm_view_dist, adm_ref_display_height);
        factor[1] = barten_watson_blend_csf_mae(scale, 1, adm_norm_view_dist, adm_ref_display_height);
    } else {
        factor[0] = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
        factor[1] = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    }
}

static void adm_cm_sum_row(const adm_dwt_band_t *f, int row, int stride,
                           int left, int right, int32_t *sum)
{
    const int16_t *h = f->band_h + row * stride;
    const int16_t *v = f->band_v + row * stride;
    const int16_t *d = f->band_d + row * stride;
    int j = left;
    for (; j + 4 <= right; j += 4) {
        const int32x4_t hv = vaddl_s16(vld1_s16(h + j), vld1_s16(v + j));
        vst1q_s32(sum + j, vaddw_s16(hv, vld1_s16(d + j)));
    }
    for (; j < right; j++)
        sum[j] = h[j] + v[j] + d[j];
}

static inline int32x4_t adm_cm_threshold_neon(const adm_dwt_band_t *a,
                                             const int32_t *center,
                                             const int32_t *vertical,
                                             int offset, int col)
{
    int32x4_t threshold = vaddq_s32(vld1q_s32(vertical + col - 1),
                                   vld1q_s32(vertical + col));
    threshold = vaddq_s32(threshold, vld1q_s32(vertical + col + 1));
    threshold = vsubq_s32(threshold, vld1q_s32(center + col));
    const int16_t *angles[3] = { a->band_h, a->band_v, a->band_d };
    for (int band = 0; band < 3; band++) {
        int32x4_t x = vabsq_s32(vmovl_s16(vld1_s16(angles[band] + offset)));
        x = vmulq_n_s32(x, ONE_BY_15);
        x = vshrq_n_s32(vaddq_s32(x, vdupq_n_s32(2048)), 12);
        threshold = vaddw_s16(threshold, vmovn_s32(x));
    }
    return threshold;
}

static inline int64x2_t adm_cm_cube_neon(int32x4_t x, bool shift_30, int shift_cub)
{
    const int32x2_t lo = vget_low_s32(x), hi = vget_high_s32(x);
    const int64x2_t round_sq = vdupq_n_s64(shift_30 ? 536870912 : 268435456);
    const int64x2_t sq_lo = vaddq_s64(vmull_s32(lo, lo), round_sq);
    const int64x2_t sq_hi = vaddq_s64(vmull_s32(hi, hi), round_sq);
    const int32x2_t low = shift_30 ? vshrn_n_s64(sq_lo, 30) : vshrn_n_s64(sq_lo, 29);
    const int32x2_t high = shift_30 ? vshrn_n_s64(sq_hi, 30) : vshrn_n_s64(sq_hi, 29);
    const int64x2_t shift = vdupq_n_s64(-shift_cub);
    return vaddq_s64(vrshlq_s64(vmull_s32(low, lo), shift),
                     vrshlq_s64(vmull_s32(high, hi), shift));
}

static inline int64x2_t adm_cm_accum_neon(const int16_t *src, int32x4_t threshold,
                                        uint32x4_t mask, uint16_t factor,
                                        bool diagonal, int shift_cub)
{
    int32x4_t x = vmulq_n_s32(vmovl_s16(vld1_s16(src)), factor);
    threshold = diagonal ? vshlq_n_s32(threshold, 12) : vshlq_n_s32(threshold, 10);
    x = vmaxq_s32(vsubq_s32(vabsq_s32(x), threshold), vdupq_n_s32(0));
    x = vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(x), mask));
    return adm_cm_cube_neon(x, diagonal, shift_cub);
}

float adm_cm_neon(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                  double adm_norm_view_dist, int adm_ref_display_height,
                  int adm_csf_mode, double adm_csf_scale, double adm_csf_diag_scale,
                  double adm_noise_weight, bool measure_aim)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left, bottom = h - top;
    if (w < 32 || left < 1 || top < 1 || right >= w || bottom >= h || right - left < 4)
        goto scalar;

    const adm_dwt_band_t *src = measure_aim ? &buf->decouple_a : &buf->decouple_r;
    const adm_dwt_band_t *a = measure_aim ? &buf->csf_f : &buf->csf_a;
    const adm_dwt_band_t *f = measure_aim ? &buf->csf_a : &buf->csf_f;
    float factor[2];
    adm_csf_factors(0, adm_norm_view_dist, adm_ref_display_height, adm_csf_mode,
                    adm_csf_scale, adm_csf_diag_scale, factor);
    const float rfactor1[3] = { factor[0], factor[0], factor[1] };

    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8 &&
        adm_csf_mode == ADM_CSF_MODE_WATSON97) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    }
    else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        if (!(rfactor1[0] * pow2_21 >= 0 && rfactor1[0] * pow2_21 < 65536 &&
              rfactor1[2] * pow2_23 >= 0 && rfactor1[2] * pow2_23 < 65536))
            goto scalar;
        i_rfactor[0] = (uint16_t) (rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t) (rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t) (rfactor1[2] * pow2_23);
    }

    const int shift_hv = (int)ceil(log2(w) - 4);
    const int shift_d = (int)ceil(log2(w) - 3);
    const int shift_inner = (int)ceil(log2(h));
    const int64_t round_inner = (uint32_t)pow(2, shift_inner - 1);
    const int32x4_t lanes = { 0, 1, 2, 3 };
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;

    /* DWT scratch is free until the next scale. */
    int32_t *tmp = buf->tmp_ref;
    int32_t *rows[3] = { tmp, tmp + w, tmp + 2 * w };
    int32_t *vertical = tmp + 3 * w;
    adm_cm_sum_row(f, top - 1, csf_a_stride, left - 1, right + 1, rows[(top - 1) % 3]);
    adm_cm_sum_row(f, top, csf_a_stride, left - 1, right + 1, rows[top % 3]);
    for (int i = top; i < bottom; i++) {
        const int32_t *prev = rows[(i - 1) % 3];
        const int32_t *center = rows[i % 3];
        int32_t *next = rows[(i + 1) % 3];
        adm_cm_sum_row(f, i + 1, csf_a_stride, left - 1, right + 1, next);
        int j = left - 1;
        for (; j + 4 <= right + 1; j += 4) {
            int32x4_t sum = vaddq_s32(vld1q_s32(prev + j), vld1q_s32(center + j));
            vst1q_s32(vertical + j, vaddq_s32(sum, vld1q_s32(next + j)));
        }
        for (; j <= right; j++)
            vertical[j] = prev[j] + center[j] + next[j];
        int64x2_t inner_h = vdupq_n_s64(0);
        int64x2_t inner_v = vdupq_n_s64(0);
        int64x2_t inner_d = vdupq_n_s64(0);
        for (int j = left; j < right; j += 4) {
            /* Mask samples already counted in the overlapping final vector. */
            const int col = j < right - 4 ? j : right - 4;
            const uint32x4_t mask = vcgeq_s32(lanes, vdupq_n_s32(j - col));
            const int32x4_t threshold = adm_cm_threshold_neon(a, center,
                                                        vertical, i * csf_a_stride + col, col);
            const int off = i * src_stride + col;
            inner_h = vaddq_s64(inner_h, adm_cm_accum_neon(src->band_h + off,
                                  threshold, mask, i_rfactor[0], false, shift_hv));
            inner_v = vaddq_s64(inner_v, adm_cm_accum_neon(src->band_v + off,
                                  threshold, mask, i_rfactor[1], false, shift_hv));
            inner_d = vaddq_s64(inner_d, adm_cm_accum_neon(src->band_d + off,
                                  threshold, mask, i_rfactor[2], true, shift_d));
        }
        /* The scalar path rounds once per row. */
        accum_h += (vaddvq_s64(inner_h) + round_inner) >> shift_inner;
        accum_v += (vaddvq_s64(inner_v) + round_inner) >> shift_inner;
        accum_d += (vaddvq_s64(inner_d) + round_inner) >> shift_inner;
    }
    const float fh = (float)(accum_h / pow(2, 52 - shift_hv - shift_inner));
    const float fv = (float)(accum_v / pow(2, 52 - shift_hv - shift_inner));
    const float fd = (float)(accum_d / pow(2, 57 - shift_d - shift_inner));
    const float noise = powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    const float nh = powf(fh, 1.0f / 3.0f) + noise;
    const float nv = powf(fv, 1.0f / 3.0f) + noise;
    const float nd = powf(fd, 1.0f / 3.0f) + noise;
    return nh + nv + nd;

scalar:
    return adm_cm(buf, w, h, src_stride, csf_a_stride, adm_norm_view_dist,
                  adm_ref_display_height, adm_csf_mode, adm_csf_scale,
                  adm_csf_diag_scale, adm_noise_weight, measure_aim);
}

static inline int32x4_t adm_dwt_filter_neon(int32x4_t s0, int32x4_t s1,
                                          int32x4_t s2, int32x4_t s3,
                                          const int16_t *filter, int shift)
{
    if (!shift) {
        int32x4_t sum = vmulq_n_s32(s0, filter[0]);
        sum = vmlaq_n_s32(sum, s1, filter[1]);
        sum = vmlaq_n_s32(sum, s2, filter[2]);
        return vmlaq_n_s32(sum, s3, filter[3]);
    }
    int64x2_t lo = vmull_n_s32(vget_low_s32(s0), filter[0]);
    int64x2_t hi = vmull_n_s32(vget_high_s32(s0), filter[0]);
    lo = vmlal_n_s32(lo, vget_low_s32(s1), filter[1]);
    hi = vmlal_n_s32(hi, vget_high_s32(s1), filter[1]);
    lo = vmlal_n_s32(lo, vget_low_s32(s2), filter[2]);
    hi = vmlal_n_s32(hi, vget_high_s32(s2), filter[2]);
    lo = vmlal_n_s32(lo, vget_low_s32(s3), filter[3]);
    hi = vmlal_n_s32(hi, vget_high_s32(s3), filter[3]);
    const int64x2_t shift_v = vdupq_n_s64(-shift);
    return vcombine_s32(vmovn_s64(vrshlq_s64(lo, shift_v)),
                        vmovn_s64(vrshlq_s64(hi, shift_v)));
}

static inline int32_t adm_dwt_filter_sample(const int32_t *src,
                                           int i0, int i1, int i2, int i3,
                                           const int16_t *filter, int shift)
{
    int64_t sum = (int64_t)src[i0] * filter[0] +
                  (int64_t)src[i1] * filter[1] +
                  (int64_t)src[i2] * filter[2] +
                  (int64_t)src[i3] * filter[3];
    return (sum + (shift ? 1 << (shift - 1) : 0)) >> shift;
}

static void adm_dwt2_s123_neon(const int32_t *src, const i4_adm_dwt_band_t *dst,
                              AdmBuffer *buf, int w, int h, int src_stride,
                              int dst_stride, int scale)
{
    const int shift_v = scale == 1 ? 0 : 16;
    const int shift_h = scale == 2 ? 16 : 15;
    int32_t *lo = buf->tmp_ref, *hi = lo + w;
    const int w_half = (w + 1) / 2;

    for (int i = 0; i < (h + 1) / 2; i++) {
        const int y0 = buf->ind_y[0][i] * src_stride;
        const int y1 = buf->ind_y[1][i] * src_stride;
        const int y2 = buf->ind_y[2][i] * src_stride;
        const int y3 = buf->ind_y[3][i] * src_stride;
        int j = 0;
        for (; j + 4 <= w; j += 4) {
            const int32x4_t s0 = vld1q_s32(src + y0 + j);
            const int32x4_t s1 = vld1q_s32(src + y1 + j);
            const int32x4_t s2 = vld1q_s32(src + y2 + j);
            const int32x4_t s3 = vld1q_s32(src + y3 + j);
            vst1q_s32(lo + j, adm_dwt_filter_neon(s0, s1, s2, s3,
                                                dwt2_db2_coeffs_lo, shift_v));
            vst1q_s32(hi + j, adm_dwt_filter_neon(s0, s1, s2, s3,
                                                dwt2_db2_coeffs_hi, shift_v));
        }
        for (; j < w; j++) {
            lo[j] = adm_dwt_filter_sample(src + j, y0, y1, y2, y3,
                                          dwt2_db2_coeffs_lo, shift_v);
            hi[j] = adm_dwt_filter_sample(src + j, y0, y1, y2, y3,
                                          dwt2_db2_coeffs_hi, shift_v);
        }

        j = 0;
        while (j < w_half) {
            const int off = i * dst_stride + j;
            if (j > 0 && 2 * j + 8 < w) {
                const int32x4x2_t l01 = vld2q_s32(lo + 2 * j - 1);
                const int32x4x2_t l23 = vld2q_s32(lo + 2 * j + 1);
                const int32x4x2_t h01 = vld2q_s32(hi + 2 * j - 1);
                const int32x4x2_t h23 = vld2q_s32(hi + 2 * j + 1);
                vst1q_s32(dst->band_a + off, adm_dwt_filter_neon(
                    l01.val[0], l01.val[1], l23.val[0], l23.val[1],
                    dwt2_db2_coeffs_lo, shift_h));
                vst1q_s32(dst->band_v + off, adm_dwt_filter_neon(
                    l01.val[0], l01.val[1], l23.val[0], l23.val[1],
                    dwt2_db2_coeffs_hi, shift_h));
                vst1q_s32(dst->band_h + off, adm_dwt_filter_neon(
                    h01.val[0], h01.val[1], h23.val[0], h23.val[1],
                    dwt2_db2_coeffs_lo, shift_h));
                vst1q_s32(dst->band_d + off, adm_dwt_filter_neon(
                    h01.val[0], h01.val[1], h23.val[0], h23.val[1],
                    dwt2_db2_coeffs_hi, shift_h));
                j += 4;
            } else {
                const int x0 = buf->ind_x[0][j], x1 = buf->ind_x[1][j];
                const int x2 = buf->ind_x[2][j], x3 = buf->ind_x[3][j];
                dst->band_a[off] = adm_dwt_filter_sample(lo, x0, x1, x2, x3,
                                                         dwt2_db2_coeffs_lo, shift_h);
                dst->band_v[off] = adm_dwt_filter_sample(lo, x0, x1, x2, x3,
                                                         dwt2_db2_coeffs_hi, shift_h);
                dst->band_h[off] = adm_dwt_filter_sample(hi, x0, x1, x2, x3,
                                                         dwt2_db2_coeffs_lo, shift_h);
                dst->band_d[off] = adm_dwt_filter_sample(hi, x0, x1, x2, x3,
                                                         dwt2_db2_coeffs_hi, shift_h);
                j++;
            }
        }
    }
}

void adm_dwt2_s123_combined_neon(const int32_t *ref, const int32_t *dis,
                                AdmBuffer *buf, int w, int h, int ref_stride,
                                int dis_stride, int dst_stride, int scale)
{
    adm_dwt2_s123_neon(ref, &buf->i4_ref_dwt2, buf, w, h,
                       ref_stride, dst_stride, scale);
    adm_dwt2_s123_neon(dis, &buf->i4_dis_dwt2, buf, w, h,
                       dis_stride, dst_stride, scale);
}

static inline int32x2_t adm_decouple_ratio_s123(int32x2_t recip, int32x2_t dis,
                                               int32x2_t shift)
{
    int64x2_t ratio = vrshlq_s64(vmull_s32(recip, dis),
                                vnegq_s64(vmovl_s32(vadd_s32(shift, vdup_n_s32(15)))));
    ratio = vbslq_s64(vcltq_s64(ratio, vdupq_n_s64(0)), vdupq_n_s64(0), ratio);
    ratio = vbslq_s64(vcgtq_s64(ratio, vdupq_n_s64(32768)), vdupq_n_s64(32768), ratio);
    return vmovn_s64(ratio);
}

static inline int32x4_t adm_decouple_band_s123(int32x4_t orig, int32x4_t dis,
                                              const int32_t *lookup,
                                              uint32x4_t *clamp)
{
    const uint32x4_t mag = vreinterpretq_u32_s32(vabsq_s32(orig));
    const int32x4_t shift = vmaxq_s32(vsubq_s32(vdupq_n_s32(17),
                                      vreinterpretq_s32_u32(vclzq_u32(mag))), vdupq_n_s32(0));
    uint32_t index[4];
    vst1q_u32(index, vrshlq_u32(mag, vnegq_s32(shift)));
    const int32_t div[4] = { lookup[index[0] + 32768], lookup[index[1] + 32768],
                            lookup[index[2] + 32768], lookup[index[3] + 32768] };
    int32x4_t recip = vld1q_s32(div);
    recip = vbslq_s32(vcltq_s32(orig, vdupq_n_s32(0)), vnegq_s32(recip), recip);
    int32x4_t k = vcombine_s32(
        adm_decouple_ratio_s123(vget_low_s32(recip), vget_low_s32(dis), vget_low_s32(shift)),
        adm_decouple_ratio_s123(vget_high_s32(recip), vget_high_s32(dis), vget_high_s32(shift)));
    k = vbslq_s32(vceqq_s32(orig, vdupq_n_s32(0)), vdupq_n_s32(32768), k);
    const int32x4_t rst = vcombine_s32(
        vrshrn_n_s64(vmull_s32(vget_low_s32(k), vget_low_s32(orig)), 15),
        vrshrn_n_s64(vmull_s32(vget_high_s32(k), vget_high_s32(orig)), 15));
    *clamp = vandq_u32(vcgtq_s32(k, vdupq_n_s32(0)), vorrq_u32(
        vandq_u32(vcgtq_s32(orig, vdupq_n_s32(0)), vcgtq_s32(rst, dis)),
        vandq_u32(vcltq_s32(orig, vdupq_n_s32(0)), vcltq_s32(rst, dis))));
    return rst;
}

static inline float32x4_t adm_dot_s32(int32x4_t ah, int32x4_t av,
                                     int32x4_t bh, int32x4_t bv, bool exact_double)
{
    const int64x2_t lo = vmlal_s32(vmull_s32(vget_low_s32(ah), vget_low_s32(bh)),
                                  vget_low_s32(av), vget_low_s32(bv));
    const int64x2_t hi = vmlal_s32(vmull_s32(vget_high_s32(ah), vget_high_s32(bh)),
                                  vget_high_s32(av), vget_high_s32(bv));
    if (exact_double)
        return vcombine_f32(vcvt_f32_f64(vcvtq_f64_s64(lo)),
                            vcvt_f32_f64(vcvtq_f64_s64(hi)));
    /* Convert directly above 2^53 to avoid double rounding. */
    const float32x4_t result = {
        (float)vgetq_lane_s64(lo, 0), (float)vgetq_lane_s64(lo, 1),
        (float)vgetq_lane_s64(hi, 0), (float)vgetq_lane_s64(hi, 1)
    };
    return result;
}

void adm_decouple_s123_neon(AdmBuffer *buf, int w, int h, int stride,
                            double adm_enhn_gain_limit, int32_t *adm_div_lookup)
{
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2, bottom = h - top + 2;
    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > w ? w : right;
    bottom = bottom > h ? h : bottom;
    if (right - left < 4 || adm_enhn_gain_limit != 1.0) {
        adm_decouple_s123(buf, w, h, stride, adm_enhn_gain_limit, adm_div_lookup);
        return;
    }
    const float cos_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    for (int i = top; i < bottom; i++) {
        for (int j = left; ; j += 4) {
            if (j > right - 4) j = right - 4;
            const int off = i * stride + j;
            const int32x4_t oh = vld1q_s32(buf->i4_ref_dwt2.band_h + off);
            const int32x4_t ov = vld1q_s32(buf->i4_ref_dwt2.band_v + off);
            const int32x4_t od = vld1q_s32(buf->i4_ref_dwt2.band_d + off);
            const int32x4_t th = vld1q_s32(buf->i4_dis_dwt2.band_h + off);
            const int32x4_t tv = vld1q_s32(buf->i4_dis_dwt2.band_v + off);
            const int32x4_t td = vld1q_s32(buf->i4_dis_dwt2.band_d + off);
            uint32x4_t ch, cv, cd;
            int32x4_t rh = adm_decouple_band_s123(oh, th, adm_div_lookup, &ch);
            int32x4_t rv = adm_decouple_band_s123(ov, tv, adm_div_lookup, &cv);
            int32x4_t rd = adm_decouple_band_s123(od, td, adm_div_lookup, &cd);
            /* At gain 1, only overshooting reconstructions need the angle test. */
            const uint32x4_t active = vorrq_u32(vorrq_u32(ch, cv), cd);
            if (vmaxvq_u32(active)) {
                const int32x4_t magnitude = vmaxq_s32(
                    vmaxq_s32(vabsq_s32(oh), vabsq_s32(ov)),
                    vmaxq_s32(vabsq_s32(th), vabsq_s32(tv)));
                const bool exact_double = vmaxvq_s32(magnitude) < (1 << 26);
                const float32x4_t dot = adm_dot_s32(oh, ov, th, tv, exact_double);
                const float32x4_t omag = adm_dot_s32(oh, ov, oh, ov, exact_double);
                const float32x4_t tmag = adm_dot_s32(th, tv, th, tv, exact_double);
                const uint32x4_t a = vcombine_u32(
                    adm_angle_f64(vget_low_f32(dot), vget_low_f32(omag),
                                   vget_low_f32(tmag), cos_sq),
                    adm_angle_f64(vget_high_f32(dot), vget_high_f32(omag),
                                   vget_high_f32(tmag), cos_sq));
                rh = vbslq_s32(vandq_u32(a, ch), th, rh);
                rv = vbslq_s32(vandq_u32(a, cv), tv, rv);
                rd = vbslq_s32(vandq_u32(a, cd), td, rd);
            }
            vst1q_s32(buf->i4_decouple_r.band_h + off, rh);
            vst1q_s32(buf->i4_decouple_r.band_v + off, rv);
            vst1q_s32(buf->i4_decouple_r.band_d + off, rd);
            vst1q_s32(buf->i4_decouple_a.band_h + off, vsubq_s32(th, rh));
            vst1q_s32(buf->i4_decouple_a.band_v + off, vsubq_s32(tv, rv));
            vst1q_s32(buf->i4_decouple_a.band_d + off, vsubq_s32(td, rd));
            if (j == right - 4) break;
        }
    }
}

static void i4_adm_cm_sum_row(const i4_adm_dwt_band_t *f, int row, int stride,
                             int left, int right, int32_t *sum)
{
    const int32_t *h = f->band_h + row * stride;
    const int32_t *v = f->band_v + row * stride;
    const int32_t *d = f->band_d + row * stride;
    int j = left;
    for (; j + 4 <= right; j += 4) {
        const int32x4_t hv = vaddq_s32(vld1q_s32(h + j), vld1q_s32(v + j));
        vst1q_s32(sum + j, vaddq_s32(hv, vld1q_s32(d + j)));
    }
    for (; j < right; j++)
        sum[j] = h[j] + v[j] + d[j];
}

static inline int32x4_t i4_adm_cm_threshold_neon(const i4_adm_dwt_band_t *a,
                                                const int32_t *center,
                                                const int32_t *vertical,
                                                int offset, int col)
{
    int32x4_t threshold = vaddq_s32(vld1q_s32(vertical + col - 1),
                                   vld1q_s32(vertical + col));
    threshold = vaddq_s32(threshold, vld1q_s32(vertical + col + 1));
    threshold = vsubq_s32(threshold, vld1q_s32(center + col));
    const int32_t *angles[3] = { a->band_h, a->band_v, a->band_d };
    for (int band = 0; band < 3; band++) {
        const int32x4_t x = vabsq_s32(vld1q_s32(angles[band] + offset));
        /* Match the scalar tap's signed int32_t rounding constant. */
        const int64x2_t rounding = vdupq_n_s64(INT32_MIN);
        const int32x4_t c = vcombine_s32(
            vshrn_n_s64(vaddq_s64(vmull_n_s32(vget_low_s32(x), I4_ONE_BY_15), rounding), 32),
            vshrn_n_s64(vaddq_s64(vmull_n_s32(vget_high_s32(x), I4_ONE_BY_15), rounding), 32));
        threshold = vaddq_s32(threshold, c);
    }
    return threshold;
}

static inline int64x2_t i4_adm_cm_accum_neon(const int32_t *src, int32x4_t threshold,
                                           uint32x4_t mask, int32_t factor, int shift_cub)
{
    const int32x4_t s = vld1q_s32(src);
    int32x4_t x = vcombine_s32(
        vrshrn_n_s64(vmull_n_s32(vget_low_s32(s), factor), 28),
        vrshrn_n_s64(vmull_n_s32(vget_high_s32(s), factor), 28));
    x = vmaxq_s32(vsubq_s32(vabsq_s32(x), threshold), vdupq_n_s32(0));
    x = vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(x), mask));
    return adm_cm_cube_neon(x, true, shift_cub);
}

float i4_adm_cm_neon(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                     int scale, double adm_norm_view_dist, int adm_ref_display_height,
                     int adm_csf_mode, double adm_csf_scale, double adm_csf_diag_scale,
                     double adm_noise_weight, bool measure_aim)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left, bottom = h - top;
    if (left < 1 || top < 1 || right >= w || bottom >= h || right - left < 4)
        goto scalar;

    float factor[2];
    adm_csf_factors(scale, adm_norm_view_dist, adm_ref_display_height, adm_csf_mode,
                    adm_csf_scale, adm_csf_diag_scale, factor);
    if (!(factor[0] >= 0 && factor[0] < 0.5f && factor[1] >= 0 && factor[1] < 0.5f))
        goto scalar;
    const int32_t rf[2] = { factor[0] * pow(2, 32), factor[1] * pow(2, 32) };
    const i4_adm_dwt_band_t *src = measure_aim ? &buf->i4_decouple_a : &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *a = measure_aim ? &buf->i4_csf_f : &buf->i4_csf_a;
    const i4_adm_dwt_band_t *f = measure_aim ? &buf->i4_csf_a : &buf->i4_csf_f;
    const int shift_cub = (int)ceil(log2(w));
    const int shift_inner = (int)ceil(log2(h));
    const int64_t round_inner = (uint32_t)pow(2, shift_inner - 1);
    const int32x4_t lanes = { 0, 1, 2, 3 };
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int32_t *tmp = buf->tmp_ref;
    int32_t *rows[3] = { tmp, tmp + w, tmp + 2 * w };
    int32_t *vertical = tmp + 3 * w;
    i4_adm_cm_sum_row(f, top - 1, csf_a_stride, left - 1, right + 1, rows[(top - 1) % 3]);
    i4_adm_cm_sum_row(f, top, csf_a_stride, left - 1, right + 1, rows[top % 3]);
    for (int i = top; i < bottom; i++) {
        const int32_t *prev = rows[(i - 1) % 3];
        const int32_t *center = rows[i % 3];
        int32_t *next = rows[(i + 1) % 3];
        i4_adm_cm_sum_row(f, i + 1, csf_a_stride, left - 1, right + 1, next);
        int j = left - 1;
        for (; j + 4 <= right + 1; j += 4) {
            int32x4_t sum = vaddq_s32(vld1q_s32(prev + j), vld1q_s32(center + j));
            vst1q_s32(vertical + j, vaddq_s32(sum, vld1q_s32(next + j)));
        }
        for (; j <= right; j++)
            vertical[j] = prev[j] + center[j] + next[j];
        int64x2_t inner_h = vdupq_n_s64(0);
        int64x2_t inner_v = vdupq_n_s64(0);
        int64x2_t inner_d = vdupq_n_s64(0);
        for (j = left; j < right; j += 4) {
            const int col = j < right - 4 ? j : right - 4;
            const uint32x4_t mask = vcgeq_s32(lanes, vdupq_n_s32(j - col));
            const int32x4_t threshold = i4_adm_cm_threshold_neon(a, center,
                                                        vertical, i * csf_a_stride + col, col);
            const int off = i * src_stride + col;
            inner_h = vaddq_s64(inner_h, i4_adm_cm_accum_neon(src->band_h + off,
                                  threshold, mask, rf[0], shift_cub));
            inner_v = vaddq_s64(inner_v, i4_adm_cm_accum_neon(src->band_v + off,
                                  threshold, mask, rf[0], shift_cub));
            inner_d = vaddq_s64(inner_d, i4_adm_cm_accum_neon(src->band_d + off,
                                  threshold, mask, rf[1], shift_cub));
        }
        accum_h += (vaddvq_s64(inner_h) + round_inner) >> shift_inner;
        accum_v += (vaddvq_s64(inner_v) + round_inner) >> shift_inner;
        accum_d += (vaddvq_s64(inner_d) + round_inner) >> shift_inner;
    }
    const int shifts[3] = { 45, 39, 36 };
    const float final_shift = pow(2, shifts[scale - 1] - shift_cub - shift_inner);
    const float fh = (float)accum_h / final_shift;
    const float fv = (float)accum_v / final_shift;
    const float fd = (float)accum_d / final_shift;
    const float noise = powf((bottom - top) * (right - left) * adm_noise_weight, 1.0f / 3.0f);
    const float nh = powf(fh, 1.0f / 3.0f) + noise;
    const float nv = powf(fv, 1.0f / 3.0f) + noise;
    const float nd = powf(fd, 1.0f / 3.0f) + noise;
    return nh + nv + nd;

scalar:
    return i4_adm_cm(buf, w, h, src_stride, csf_a_stride, scale, adm_norm_view_dist,
                     adm_ref_display_height, adm_csf_mode, adm_csf_scale,
                     adm_csf_diag_scale, adm_noise_weight, measure_aim);
}
