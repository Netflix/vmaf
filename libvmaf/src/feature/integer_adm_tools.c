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

#include <math.h>
#include <stddef.h>
#include <string.h>
#include "mem.h"
#include "adm_options.h"
#include "integer_adm_tools.h"

#ifndef M_PI
  #define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

#include <emmintrin.h>

static const int16_t dwt2_db2_coeffs_lo_fix_s[4] = { 15826, 27411, 7345, -4240 };
static const int16_t dwt2_db2_coeffs_hi_fix_s[4] = {-4240, -7345, 27411, -15826};

#ifndef INTEGER_ONE_BY_15
#define INTEGER_ONE_BY_15 8738
#endif

#ifndef INTEGER_I4_ONE_BY_15
#define INTEGER_I4_ONE_BY_15 286331153
#endif

/**
 * Works similar to floating-point function adm_decouple_s
 */
void integer_adm_decouple_s(const integer_adm_dwt_band_t_s *ref, const integer_adm_dwt_band_t_s *dis, const integer_adm_dwt_band_t_s *r, const integer_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor)
{
#ifdef ADM_OPT_AVOID_ATAN
	const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif

	int ref_px_stride = ref_stride / sizeof(float);
	int dis_px_stride = dis_stride / sizeof(float);
	int r_px_stride = r_stride / sizeof(float);
	int a_px_stride = a_stride / sizeof(float);

	/* The computation of the score is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5 - 1; // -1 for filter tap
	int top = h * border_factor - 0.5 - 1;
	int right = w - left + 2; // +2 for filter tap
	int bottom = h - top + 2;

	if (left < 0) {
		left = 0;
	}
	if (right > w) {
		right = w;
	}
	if (top < 0) {
		top = 0;
	}
	if (bottom > h) {
		bottom = h;
	}

	int16_t oh, ov, od, th, tv, td;
	int32_t kh, kv, kd;
	int16_t tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
	int64_t ot_dp, o_mag_sq, t_mag_sq;
#else
	float oa, ta, diff;
#endif
	int angle_flag;
	int i, j;

	for (i = top; i < bottom; ++i) {
		for (j = left; j < right; ++j) {
			oh = ref->band_h[i * ref_px_stride + j];
			ov = ref->band_v[i * ref_px_stride + j];
			od = ref->band_d[i * ref_px_stride + j];
			th = dis->band_h[i * dis_px_stride + j];
			tv = dis->band_v[i * dis_px_stride + j];
			td = dis->band_d[i * dis_px_stride + j];

			/**
			 * This shift is done so that precision is maintained after division
			 * division is Q31/Q16
			 */
			int32_t tmp_th = ((int32_t) th)<<15;
			int32_t tmp_tv = ((int32_t) tv)<<15;
			int32_t tmp_td = ((int32_t) td)<<15;

			int32_t tmp_kh = (oh == 0) ? 32768 : (tmp_th/oh);
			int32_t tmp_kv = (ov == 0) ? 32768 : (tmp_tv/ov);
			int32_t tmp_kd = (od == 0) ? 32768 : (tmp_td/od); 

			kh = tmp_kh < 0 ? 0 : (tmp_kh > (int32_t) 32768 ? (int32_t) 32768 : (int32_t) tmp_kh);
			kv = tmp_kv < 0 ? 0 : (tmp_kv > (int32_t) 32768 ? (int32_t) 32768 : (int32_t) tmp_kv);
			kd = tmp_kd < 0 ? 0 : (tmp_kd > (int32_t) 32768 ? (int32_t) 32768 : (int32_t) tmp_kd);

			/**
			 * kh,kv,kd are in Q15 type and oh,ov,od are in Q16 type hence shifted by 15 to make result Q16
			 */
			tmph = ((kh * oh)+ 16384) >>15;
			tmpv = ((kv * ov)+ 16384) >>15;
			tmpd = ((kd * od)+ 16384) >>15;
#ifdef ADM_OPT_AVOID_ATAN
			/* Determine if angle between (oh,ov) and (th,tv) is less than 1 degree.
			 * Given that u is the angle (oh,ov) and v is the angle (th,tv), this can
			 * be done by testing the inequvality.
			 *
			 * { (u.v.) >= 0 } AND { (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2 }
			 *
			 * Proof:
			 *
			 * cos(theta) = (u.v) / (||u|| * ||v||)
			 *
			 * IF u.v >= 0 THEN
			 *   cos(theta)^2 = (u.v)^2 / (||u||^2 * ||v||^2)
			 *   (u.v)^2 = cos(theta)^2 * ||u||^2 * ||v||^2
			 *
			 *   IF |theta| < 1deg THEN
			 *     (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2
			 *   END
			 * ELSE
			 *   |theta| > 90deg
			 * END
			 */
			ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
			o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
			t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

			/**
			 * angle_flag is calculated in floating-point by converting fixed-point variables back to floating-point
			 */
			angle_flag = (((float)ot_dp/4096.0) >= 0.0f) && (((float)ot_dp/4096.0) * ((float)ot_dp/4096.0) >= cos_1deg_sq * ((float)o_mag_sq/4096.0) * ((float)t_mag_sq/4096.0));
#else
			oa = atanf(DIVS(ov, oh + eps));
			ta = atanf(DIVS(tv, th + eps));

			if (oh < 0.0f)
				oa += (float)M_PI;
			if (th < 0.0f)
				ta += (float)M_PI;

			diff = fabsf(oa - ta) * 180.0f / M_PI;
			angle_flag = diff < 1.0f;
#endif
			if (angle_flag) {
				tmph = th;
				tmpv = tv;
				tmpd = td;
			}

			r->band_h[i * r_px_stride + j] = tmph;
			r->band_v[i * r_px_stride + j] = tmpv;
			r->band_d[i * r_px_stride + j] = tmpd;

			a->band_h[i * a_px_stride + j] = th - tmph;
			a->band_v[i * a_px_stride + j] = tv - tmpv;
			a->band_d[i * a_px_stride + j] = td - tmpd;
		}
	}
}

/**
 * Works similar to integer_adm_decouple_s function for scale 1,2,3
 */
void integer_adm_decouple_scale123_s(const integer_i4_adm_dwt_band_t_s *ref, const integer_i4_adm_dwt_band_t_s *dis, const integer_i4_adm_dwt_band_t_s *r, const integer_i4_adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride, double border_factor, int scale)
{
#ifdef ADM_OPT_AVOID_ATAN
	const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif

	int ref_px_stride = ref_stride / sizeof(float);
	int dis_px_stride = dis_stride / sizeof(float);
	int r_px_stride = r_stride / sizeof(float);
	int a_px_stride = a_stride / sizeof(float);
	float f_shift[3] = {pow(2, 21), pow(2, 19), pow(2, 18)};
	/* The computation of the score is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5 - 1; // -1 for filter tap
	int top = h * border_factor - 0.5 - 1;
	int right = w - left + 2; // +2 for filter tap
	int bottom = h - top + 2;

	if (left < 0)
	{
		left = 0;
	}
	if (right > w)
	{
		right = w;
	}
	if (top < 0)
	{
		top = 0;
	}
	if (bottom > h)
	{
		bottom = h;
	}

	int32_t oh, ov, od, th, tv, td;
	int64_t kh, kv, kd;
	int32_t tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
	int64_t ot_dp, o_mag_sq, t_mag_sq;
#else
	float oa, ta, diff;
#endif
	int angle_flag;
	int i, j;

	for (i = top; i < bottom; ++i)
	{
		for (j = left; j < right; ++j)
		{
			oh = ref->band_h[i * ref_px_stride + j];
			ov = ref->band_v[i * ref_px_stride + j];
			od = ref->band_d[i * ref_px_stride + j];
			th = dis->band_h[i * dis_px_stride + j];
			tv = dis->band_v[i * dis_px_stride + j];
			td = dis->band_d[i * dis_px_stride + j];

			int64_t tmp_th = ((int64_t)th) << 15;
			int64_t tmp_tv = ((int64_t)tv) << 15;
			int64_t tmp_td = ((int64_t)td) << 15;

			int64_t tmp_kh = (oh == 0) ? 32768 : (tmp_th / oh);
			int64_t tmp_kv = (ov == 0) ? 32768 : (tmp_tv / ov);
			int64_t tmp_kd = (od == 0) ? 32768 : (tmp_td / od);

			kh = tmp_kh < 0 ? 0 : (tmp_kh > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kh);
			kv = tmp_kv < 0 ? 0 : (tmp_kv > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kv);
			kd = tmp_kd < 0 ? 0 : (tmp_kd > (int32_t)32768 ? (int32_t)32768 : (int32_t)tmp_kd);

			tmph = ((kh * oh) + 16384) >> 15;
			tmpv = ((kv * ov) + 16384) >> 15;
			tmpd = ((kd * od) + 16384) >> 15;
#ifdef ADM_OPT_AVOID_ATAN
			/* Determine if angle between (oh,ov) and (th,tv) is less than 1 degree.
			 * Given that u is the angle (oh,ov) and v is the angle (th,tv), this can
			 * be done by testing the inequvality.
			 *
			 * { (u.v.) >= 0 } AND { (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2 }
			 *
			 * Proof:
			 *
			 * cos(theta) = (u.v) / (||u|| * ||v||)
			 *
			 * IF u.v >= 0 THEN
			 *   cos(theta)^2 = (u.v)^2 / (||u||^2 * ||v||^2)
			 *   (u.v)^2 = cos(theta)^2 * ||u||^2 * ||v||^2
			 *
			 *   IF |theta| < 1deg THEN
			 *     (u.v)^2 >= cos(1deg)^2 * ||u||^2 * ||v||^2
			 *   END
			 * ELSE
			 *   |theta| > 90deg
			 * END
			 */
			ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
			o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
			t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;

			angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) && (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >= cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
#else
			oa = atanf(DIVS(ov, oh + eps));
			ta = atanf(DIVS(tv, th + eps));

			if (oh < 0.0f)
				oa += (float)M_PI;
			if (th < 0.0f)
				ta += (float)M_PI;

			diff = fabsf(oa - ta) * 180.0f / M_PI;
			angle_flag = diff < 1.0f;
#endif
			if (angle_flag)
			{
				tmph = th;
				tmpv = tv;
				tmpd = td;
			}

			r->band_h[i * r_px_stride + j] = tmph;
			r->band_v[i * r_px_stride + j] = tmpv;
			r->band_d[i * r_px_stride + j] = tmpd;

			a->band_h[i * a_px_stride + j] = th - tmph;
			a->band_v[i * a_px_stride + j] = tv - tmpv;
			a->band_d[i * a_px_stride + j] = td - tmpd;
		}
	}
}

void integer_adm_csf_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *dst, const integer_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor)
{
	const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
	int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
	int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

	const int16_t *src_ptr;
	int16_t *dst_ptr;
	int16_t *flt_ptr;

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
	float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
	float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

	/**
	 * rfactor is converted to fixed-point for scale0 and stored in i_rfactor multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2]
	 */
	uint16_t i_rfactor[3] = {36453,36453,49417};

	/**
	 * Shifts pending from previous stage is 6
	 * hence variables multiplied by i_rfactor[0,1] has to be shifted by 21+6=27 to convert into floating-point. But shifted by 15 to make it Q16
	 * and variables multiplied by i_factor[2] has to be shifted by 23+6=29 to convert into floating-point. But shifted by 17 to make it Q16
	 * Hence remaining shifts after shifting by i_shifts is 12 to make it equivalent to floating-point
	 */
	uint8_t i_shifts[3] = {15,15,17};
	uint16_t i_shiftsadd[3] = {16384, 16384, 65535};
	uint16_t FIX_ONE_BY_30 = 4369; //(1/30)*2^17
	/* The computation of the csf values is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5 - 1; // -1 for filter tap
	int top = h * border_factor - 0.5 - 1;
	int right = w - left + 2; // +2 for filter tap
	int bottom = h - top + 2;

	if (left < 0) {
		left = 0;
	}
	if (right > w) {
		right = w;
	}
	if (top < 0) {
		top = 0;
	}
	if (bottom > h) {
		bottom = h;
	}

	int i, j, theta, src_offset, dst_offset;
	int32_t dst_val;

	for (theta = 0; theta < 3; ++theta) {
		src_ptr = src_angles[theta];
		dst_ptr = dst_angles[theta];
		flt_ptr = flt_angles[theta];

		for (i = top; i < bottom; ++i) {
			src_offset = i * src_px_stride;
			dst_offset = i * dst_px_stride;

			for (j = left; j < right; ++j) {
				dst_val = i_rfactor[theta] * (int32_t)src_ptr[src_offset + j];
				int16_t i16_dst_val = ((int16_t)((dst_val + i_shiftsadd[theta])>>i_shifts[theta]));
				dst_ptr[dst_offset + j] = i16_dst_val;
				flt_ptr[dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t) i16_dst_val))+ 2048)>>12));//shifted by 12 to make it Q16. Remaining shifts to make it equivalent to floating point is 12+17-12=12
			}
		}
	}
}

/**
 * Works similar to fixed-point function integer_adm_csf_s
 */
void integer_i4_adm_csf_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *dst, const integer_i4_adm_dwt_band_t_s *flt, int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor)
{
	const int32_t *src_angles[3] = {src->band_h, src->band_v, src->band_d};
	int32_t *dst_angles[3] = {dst->band_h, dst->band_v, dst->band_d};
	int32_t *flt_angles[3] = {flt->band_h, flt->band_v, flt->band_d};

	const int32_t *src_ptr;
	int32_t *dst_ptr;
	int32_t *flt_ptr;

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
	float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
	float rfactor1[3] = {1.0f / factor1, 1.0f / factor1, 1.0f / factor2};

	//rfactor in fixed-point
	uint32_t i_rfactor[3] = {(uint32_t)(rfactor1[0] * pow(2, 32)), (uint32_t)(rfactor1[1] * pow(2, 32)), (uint32_t)(rfactor1[2] * pow(2, 32))};

	uint32_t FIX_ONE_BY_30 = 143165577;

	/**
	 * The shifts are done such that overflow doesn't happen i.e results are less than 2^31 i.e 1Q31
	 */
	int32_t shift_dst[3] = {28, 28, 28};
	int32_t shift_flt[3] = {32, 32, 32};

	int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

	int idx;
	for (idx = 0; idx < 3; idx++)
	{
		add_bef_shift_dst[idx] = (int32_t)pow(2, (shift_dst[idx] - 1));
		add_bef_shift_flt[idx] = (int32_t)pow(2, (shift_flt[idx] - 1));
	}

	/* The computation of the csf values is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5 - 1; // -1 for filter tap
	int top = h * border_factor - 0.5 - 1;
	int right = w - left + 2; // +2 for filter tap
	int bottom = h - top + 2;

	if (left < 0)
	{
		left = 0;
	}
	if (right > w)
	{
		right = w;
	}
	if (top < 0)
	{
		top = 0;
	}
	if (bottom > h)
	{
		bottom = h;
	}

	int i, j, theta, src_offset, dst_offset;
	int32_t dst_val;

	for (theta = 0; theta < 3; ++theta)
	{
		src_ptr = src_angles[theta];
		dst_ptr = dst_angles[theta];
		flt_ptr = flt_angles[theta];

		for (i = top; i < bottom; ++i)
		{
			src_offset = i * src_px_stride;
			dst_offset = i * dst_px_stride;

			for (j = left; j < right; ++j)
			{
				dst_val = (int32_t)(((i_rfactor[theta] * (int64_t)src_ptr[src_offset + j]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				dst_ptr[dst_offset + j] = dst_val;
				flt_ptr[dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) + add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);
			}
		}
	}
}

/* Combination of adm_csf_s and adm_sum_cube_s for csf_o based den_scale */
float integer_adm_csf_den_scale_s(const integer_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor)
{
	int16_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;
	
	int src_px_stride = src_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
	float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
	float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

	uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
	uint64_t accum_inner_h, accum_inner_v, accum_inner_d;
	float den_scale_h, den_scale_v, den_scale_d;

	uint64_t val;

	/* The computation of the denominator scales is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5;
	int top = h * border_factor - 0.5;
	int right = w - left;
	int bottom = h - top;

	int32_t shift_inner_accum = (int32_t)ceil(log2((bottom - top)*(right-left))-20);
	shift_inner_accum = shift_inner_accum>0? shift_inner_accum : 0;
	int32_t add_before_shift_inner_accum = (int32_t)pow(2, (shift_inner_accum - 1));
	add_before_shift_inner_accum = add_before_shift_inner_accum>0? add_before_shift_inner_accum : 0;
	uint8_t shift_accum = 18 - shift_inner_accum;

	int i, j;
	/**
	 * The rfactor is multiplied at the end after cubing
	 * Because d+ = (a[i]^3)*(r^3)
	 * is equivalent to d+=a[i]^3 and d=d*(r^3)
	 */
	for (i = top; i < bottom; ++i) {
		accum_inner_h = 0;
		accum_inner_v = 0;
		accum_inner_d = 0;
		src_h = src->band_h + i * src_px_stride;
		src_v = src->band_v + i * src_px_stride;
		src_d = src->band_d + i * src_px_stride;
		for (j = left; j < right; ++j) {

			uint16_t h = (uint16_t) abs(src_h[j]);
			uint16_t v = (uint16_t) abs(src_v[j]);
			uint16_t d = (uint16_t) abs(src_d[j]);

			val = ((uint64_t)h * h) * h;

			accum_inner_h += val;
			val = ((uint64_t)v * v) * v;
			accum_inner_v += val;
			val = ((uint64_t)d * d) * d;
			accum_inner_d += val;
		}
		/**
		 * max_value of h^3, v^3, d^3 is 1.205624776×10^13
		 * accum_h can hold till 1.844674407×10^19
		 * accum_h's maximum is reached when it is 2^20 * max(h^3)
		 * Therefore the accum_h,v,d is shifted based on width and height subtracted by 20
		 */
		accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
	}
	double csf_h, csf_v, csf_d;

	/**
	 * rfactor is multiplied after cubing
	 * accum_h,v,d is converted to floating-point for score calculation
	 * 6bits are yet to be shifted from previous stage that is after dwt hence after cubing 18bits are to shifted
	 * Hence final shift is 18-shift_inner_accum
	 */
	csf_h = (double) (accum_h / pow(2,shift_accum)) * pow(rfactor[0],3);
	csf_v = (double) (accum_v / pow(2,shift_accum)) * pow(rfactor[1],3);
	csf_d = (double) (accum_d / pow(2,shift_accum)) * pow(rfactor[2],3);

	den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return(den_scale_h + den_scale_v + den_scale_d);

}

/* Combination of adm_csf_s and adm_sum_cube_s for csf_o based den_scale */
float integer_adm_csf_den_scale123_s(const integer_i4_adm_dwt_band_t_s *src, int orig_h, int scale, int w, int h, int src_stride, double border_factor)
{
	int32_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

	int src_px_stride = src_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
	float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
	float rfactor[3] = {1.0f / factor1, 1.0f / factor1, 1.0f / factor2};

	uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
	uint64_t accum_inner_h, accum_inner_v, accum_inner_d;
	float den_scale_h, den_scale_v, den_scale_d;

	uint32_t shift_sq[3] = {31, 30, 31};
	uint32_t add_before_shift_sq[3];
	int idx;
	for (idx = 0; idx < 3; idx++)
	{
		add_before_shift_sq[idx] = (uint32_t)pow(2, shift_sq[idx]);
	}

	uint64_t val;

	/* The computation of the denominator scales is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5;
	int top = h * border_factor - 0.5;
	int right = w - left;
	int bottom = h - top;

	uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
	uint32_t add_before_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

	uint32_t shift_inner_accum = (uint32_t)ceil(log2(bottom - top));
	uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

	int accum_convert_float[3] = {32, 27, 23};
	int i, j;
	/**
	 * The shifts are done such that overflow doesn't happen i.e results are less than 2^64
	 */
	for (i = top; i < bottom; ++i)
	{
		accum_inner_h = 0;
		accum_inner_v = 0;
		accum_inner_d = 0;
		src_h = src->band_h + i * src_px_stride;
		src_v = src->band_v + i * src_px_stride;
		src_d = src->band_d + i * src_px_stride;
		for (j = left; j < right; ++j)
		{

			uint32_t h = (uint32_t)abs(src_h[j]);
			uint32_t v = (uint32_t)abs(src_v[j]);
			uint32_t d = (uint32_t)abs(src_d[j]);

			val = ((((((uint64_t)h * h) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * h) + add_before_shift_cub) >> shift_cub;

			accum_inner_h += val;

			val = ((((((uint64_t)v * v) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * v) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;

			val = ((((((uint64_t)d * d) + add_before_shift_sq[scale - 1]) >> shift_sq[scale - 1]) * d) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;
		}

		accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
	}
	double csf_h, csf_v, csf_d;

	/**
	 * All the results are converted to floating-point to calculate the scores
	 * For all scales the final shift is 3*shifts from dwt - total shifts done here
	 */
	csf_h = (double)(accum_h / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[0], 3);
	csf_v = (double)(accum_v / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[1], 3);
	csf_d = (double)(accum_d / pow(2, (accum_convert_float[scale - 1] - shift_inner_accum - shift_cub))) * pow(rfactor[2], 3);

	den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return (den_scale_h + den_scale_v + den_scale_d);
}

float integer_adm_cm_s(const integer_adm_dwt_band_t_s *src, const integer_adm_dwt_band_t_s *csf_f, const integer_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int flt_stride, int csf_a_stride, double border_factor, int scale)
{
	/* Take decouple_r as src and do dsf_s on decouple_r here to get csf_r */
	int16_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

	//rfactor is left shifted by 21 for rfactor[0,1] and by 23 for rfactor[2]
	uint16_t rfactor[3] = {36453,36453,49417};

	int32_t shift_xhsq = 29;
	int32_t shift_xvsq = 29;
	int32_t shift_xdsq = 30;

	int32_t add_before_shift_xhsq = 268435456;
	int32_t add_before_shift_xvsq = 268435456;
	int32_t add_before_shift_xdsq = 536870912;

	uint32_t shift_xhcub = (uint32_t)ceil(log2(w)-4);
	uint32_t add_before_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));

	uint32_t shift_xvcub = (uint32_t)ceil(log2(w)-4);
	uint32_t add_before_shift_xvcub = (uint32_t)pow(2, (shift_xvcub - 1));

	uint32_t shift_xdcub = (uint32_t)ceil(log2(w)-3);
	uint32_t add_before_shift_xdcub = (uint32_t)pow(2, (shift_xdcub - 1));

	uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
	uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

	int32_t shift_xhsub = 10;
	int32_t shift_xvsub = 10;
	int32_t shift_xdsub = 12;

	const int16_t *angles[3] = {csf_a->band_h, csf_a->band_v, csf_a->band_d};
	const int16_t *flt_angles[3] = {csf_f->band_h, csf_f->band_v, csf_f->band_d};

	int src_px_stride = src_stride / sizeof(float);
	int flt_px_stride = flt_stride / sizeof(float);
	int csf_px_stride = csf_a_stride / sizeof(float);

	int32_t xh, xv, xd;
	int32_t thr;
	int32_t xh_sq, xv_sq, xd_sq;

	int64_t val;
	int64_t accum_h = 0, accum_v = 0, accum_d = 0;
	int64_t accum_inner_h, accum_inner_v, accum_inner_d;
	float num_scale_h, num_scale_v, num_scale_d;

	/* The computation of the scales is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5;
	int top = h * border_factor - 0.5;
	int right = w - left;
	int bottom = h - top;

	int start_col = (left > 1) ? left : 1;
	int end_col = (right < (w - 1)) ? right : (w - 1);
	int start_row = (top > 1) ? top : 1;
	int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

	int i, j;

	/* i=0,j=0 */
	accum_inner_h = 0;
	accum_inner_v = 0;
	accum_inner_d = 0;
	if ((top <= 0) && (left <= 0))
	{
		xh =(int32_t) src->band_h[0] * rfactor[0];
		xv =(int32_t) src->band_v[0] * rfactor[1];
		xd =(int32_t) src->band_d[0] * rfactor[2];
		INTEGER_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_px_stride, &thr, w, h, 0, 0);

		//thr is shifted to make it's Q format equivalent to xh,xv,xd
		xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
		xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
		xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		//Shifted after squaring to make it Q32. xh,xv,xd max value is 22930*rfactor[i]
		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);

		/**
		 * max value of xh_sq and xv_sq is 1301381973 and that of xd_sq is 1195806729
		 *
		 * max(val before shift for h and v) is 9.995357299×10^17.
		 * 9.995357299×10^17 * 2^4 is close to 2^64.
		 * Hence shift is done based on width subtracting 4
		 *
		 * max(val before shift for d) is 1.355006643×10^18
		 * 1.355006643×10^18 * 2^3 is close to 2^64
		 * Hence shift is done based on width subtracting 3
		 */
		val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub; //max(val before shift) is 9.995357299×10^17. 9.995357299×10^17 * 2^4 is close to 2^64. Hence shift is done based on width subtracting 4
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
		accum_inner_d += val;
    }

	/* i=0, j */
	if (top <= 0) {
		for (j = start_col; j < end_col; ++j) {
			xh = src->band_h[j] * rfactor[0];
			xv = src->band_v[j] * rfactor[1];
			xd = src->band_d[j] * rfactor[2];
			INTEGER_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_px_stride, &thr, w, h, 0, j);

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

        }
    }

	/* i=0,j=w-1 */
	if ((top <= 0) && (right > (w - 1)))
    {
		xh = src->band_h[w - 1] * rfactor[0];
		xv = src->band_v[w - 1] * rfactor[1];
		xd = src->band_d[w - 1] * rfactor[2];
		INTEGER_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, 0, (w - 1));

		xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
		xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
		xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
		accum_inner_d += val;

    }
	//Shift is done based on height
	accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

	if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
	{
		for (i = start_row; i < end_row; ++i) {
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;
		    for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				INTEGER_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
				xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
				xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
				accum_inner_d += val;

		    }
		accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
	    }
	}
	else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
	{
		for (i = start_row; i < end_row; ++i) {
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;

			/* j = 0 */
			xh = src->band_h[i * src_px_stride] * rfactor[0];
			xv = src->band_v[i * src_px_stride] * rfactor[1];
			xd = src->band_d[i * src_px_stride] * rfactor[2];
			INTEGER_ADM_CM_THRESH_FIXED_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				INTEGER_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
				xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
				xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
				accum_inner_d += val;

            }
		accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
		accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
        }
    }
	else if ((left > 0) && (right > (w - 1))) /* Left border within frame, right outside */
	{
		for (i = start_row; i < end_row; ++i) {
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;
			/* j within frame */
			for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				INTEGER_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
				xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
				xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
				accum_inner_d += val;

            }
			/* j = w-1 */
			xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
			xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
			xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
			INTEGER_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
		}
	}
	else /* Both borders outside frame */
    {
		for (i = start_row; i < end_row; ++i) {
	        accum_inner_h = 0;
	        accum_inner_v = 0;
	        accum_inner_d = 0;

			/* j = 0 */
			xh = src->band_h[i * src_px_stride] * rfactor[0];
			xv = src->band_v[i * src_px_stride] * rfactor[1];
			xd = src->band_d[i * src_px_stride] * rfactor[2];
			INTEGER_ADM_CM_THRESH_FIXED_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				INTEGER_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
				xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
				xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
				accum_inner_d += val;

			}
			/* j = w-1 */
			xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
			xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
			xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
			INTEGER_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
	    }
	}
	accum_inner_h = 0;
	accum_inner_v = 0;
	accum_inner_d = 0;

	/* i=h-1,j=0 */
	if ((bottom > (h - 1)) && (left <= 0))
	{
		xh = src->band_h[(h - 1) * src_px_stride] * rfactor[0];
		xv = src->band_v[(h - 1) * src_px_stride] * rfactor[1];
		xd = src->band_d[(h - 1) * src_px_stride] * rfactor[2];
		INTEGER_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), 0);

		xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
		xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
		xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
		accum_inner_d += val;

	}

	/* i=h-1,j */
	if (bottom > (h - 1)) {
		for (j = start_col; j < end_col; ++j) {
			xh = src->band_h[(h - 1) * src_px_stride + j] * rfactor[0];
			xv = src->band_v[(h - 1) * src_px_stride + j] * rfactor[1];
			xd = src->band_d[(h - 1) * src_px_stride + j] * rfactor[2];
			INTEGER_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), j);

			xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
			xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
			xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
			accum_inner_d += val;

		}
	}

	/* i-h-1,j=w-1 */
	if ((bottom > (h - 1)) && (right > (w - 1)))
	{
		xh = src->band_h[(h - 1) * src_px_stride + w - 1] * rfactor[0];
		xv = src->band_v[(h - 1) * src_px_stride + w - 1] * rfactor[1];
		xd = src->band_d[(h - 1) * src_px_stride + w - 1] * rfactor[2];
		INTEGER_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), (w - 1));

		xh = abs(xh) - ((int32_t)(thr) << shift_xhsub);
		xv = abs(xv) - ((int32_t)(thr) << shift_xvsub);
		xd = abs(xd) - ((int32_t)(thr) << shift_xdsub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_xhsq) >> shift_xhsq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_xvsq) >> shift_xvsq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_xdsq) >> shift_xdsq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_xhcub) >> shift_xhcub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_xvcub) >> shift_xvcub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_xdcub) >> shift_xdcub;
		accum_inner_d += val;

    }
	accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

	/**
	 * For h and v total shifts pending from last stage is 6 rfactor[0,1] has 21 shifts
	 * => after cubing (6+21)*3=81 after squaring shifted by 29
	 * hence pending is 52-shift's done based on width and height
	 *
	 * For d total shifts pending from last stage is 6 rfactor[2] has 23 shifts
	 * => after cubing (6+23)*3=87 after squaring shifted by 30
	 * hence pending is 57-shift's done based on width and height
	 */
	float f_accum_h = (float) (accum_h/pow(2,(52-shift_xhcub-shift_inner_accum)));
	float f_accum_v = (float) (accum_v/pow(2,(52-shift_xvcub-shift_inner_accum)));
	float f_accum_d = (float) (accum_d/pow(2,(57-shift_xdcub-shift_inner_accum)));

	num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return (num_scale_h + num_scale_v + num_scale_d);
}

/**
 * Works similar to fixed-point function integer_adm_cm_s
 */
float integer_i4_adm_cm_s(const integer_i4_adm_dwt_band_t_s *src, const integer_i4_adm_dwt_band_t_s *csf_f, const integer_i4_adm_dwt_band_t_s *csf_a, int w, int h, int src_stride, int flt_stride, int csf_a_stride, double border_factor, int scale)
{
	/* Take decouple_r as src and do dsf_s on decouple_r here to get csf_r */
	int32_t *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
	float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
	float rfactor1[3] = {1.0f / factor1, 1.0f / factor1, 1.0f / factor2};

	uint32_t rfactor[3] = {(uint32_t)(rfactor1[0] * pow(2, 32)), (uint32_t)(rfactor1[1] * pow(2, 32)), (uint32_t)(rfactor1[2] * pow(2, 32))};

	int32_t shift_dst[3] = {28, 28, 28};
	int32_t shift_flt[3] = {32, 32, 32};

	int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

	int idx;
	for (idx = 0; idx < 3; idx++)
	{
		add_bef_shift_dst[idx] = (int32_t)pow(2, (shift_dst[idx] - 1));
		add_bef_shift_flt[idx] = (int32_t)pow(2, (shift_flt[idx] - 1));

	}

	uint32_t shift_cub = (uint32_t)ceil(log2(w));
	uint32_t add_before_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

	uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
	uint32_t add_before_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

	float final_shift[3] = {pow(2,(45-shift_cub-shift_inner_accum)),pow(2,(39-shift_cub-shift_inner_accum)), pow(2,(36-shift_cub-shift_inner_accum))};

	int32_t shift_sq = 30;

	int32_t add_before_shift_sq = 536870912; //2^29

	int32_t shift_sub = 0;

	const int32_t *angles[3] = {csf_a->band_h, csf_a->band_v, csf_a->band_d};
	const int32_t *flt_angles[3] = {csf_f->band_h, csf_f->band_v, csf_f->band_d};

	int src_px_stride = src_stride / sizeof(float);
	int flt_px_stride = flt_stride / sizeof(float);
	int csf_px_stride = csf_a_stride / sizeof(float);

	int32_t xh, xv, xd;
	int32_t thr;
	int32_t xh_sq, xv_sq, xd_sq;

	int64_t val;
	int64_t accum_h = 0, accum_v = 0, accum_d = 0;
	int64_t accum_inner_h, accum_inner_v, accum_inner_d;
	float num_scale_h, num_scale_v, num_scale_d;

	/* The computation of the scales is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5;
	int top = h * border_factor - 0.5;
	int right = w - left;
	int bottom = h - top;

	// printf("left %d right %d top %d bottom %d\n",left,right,top,bottom);
	// exit(0);
	int start_col = (left > 1) ? left : 1;
	int end_col = (right < (w - 1)) ? right : (w - 1);
	int start_row = (top > 1) ? top : 1;
	int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

	int i, j;

	/* i=0,j=0 */
	accum_inner_h = 0;
	accum_inner_v = 0;
	accum_inner_d = 0;
	if ((top <= 0) && (left <= 0))
	{
		xh = (int32_t)((((int64_t)src->band_h[0] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xv = (int32_t)((((int64_t)src->band_v[0] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xd = (int32_t)((((int64_t)src->band_d[0] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		INTEGER_I4_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_px_stride, &thr, w, h, 0, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

		xh = abs(xh) - (thr >> shift_sub); //Shifted to make it equivalent to xh
		xv = abs(xv) - (thr >> shift_sub);
		xd = abs(xd) - (thr >> shift_sub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
		accum_inner_d += val;

	}

	/* i=0, j */
	if (top <= 0)
	{
		for (j = start_col; j < end_col; ++j)
		{
			xh = (int32_t)((((int64_t)src->band_h[j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_px_stride, &thr, w, h, 0, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;
		}
	}

	/* i=0,j=w-1 */
	if ((top <= 0) && (right > (w - 1)))
	{
		xh = (int32_t)((((int64_t)src->band_h[w-1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xv = (int32_t)((((int64_t)src->band_v[w-1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xd = (int32_t)((((int64_t)src->band_d[w-1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		INTEGER_I4_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, 0, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

		xh = abs(xh) - (thr >> shift_sub);
		xv = abs(xv) - (thr >> shift_sub);
		xd = abs(xd) - (thr >> shift_sub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
		accum_inner_d += val;
	}

	accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

	if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
	{
		for (i = start_row; i < end_row; ++i)
		{
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;
			for (j = start_col; j < end_col; ++j)
			{

				xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

				xh = abs(xh) - (thr >> shift_sub);
				xv = abs(xv) - (thr >> shift_sub);
				xd = abs(xd) - (thr >> shift_sub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
				accum_inner_d += val;
			}
			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
		}
	}
	else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
	{
		for (i = start_row; i < end_row; ++i)
		{
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;

			/* j = 0 */
			xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j)
			{
				xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

				xh = abs(xh) - (thr >> shift_sub);
				xv = abs(xv) - (thr >> shift_sub);
				xd = abs(xd) - (thr >> shift_sub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
				accum_inner_d += val;
			}
			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
		}
	}
	else if ((left > 0) && (right > (w - 1))) /* Left border within frame, right outside */
	{
		for (i = start_row; i < end_row; ++i)
		{
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;
			/* j within frame */
			for (j = start_col; j < end_col; ++j)
			{
				xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

				xh = abs(xh) - (thr >> shift_sub);
				xv = abs(xv) - (thr >> shift_sub);
				xd = abs(xd) - (thr >> shift_sub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
				accum_inner_d += val;
			}
			/* j = w-1 */
			xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + w - 1] * rfactor[i * src_px_stride + w - 1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;

			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
		}
	}
	else /* Both borders outside frame */
	{
		for (i = start_row; i < end_row; ++i)
		{
			accum_inner_h = 0;
			accum_inner_v = 0;
			accum_inner_d = 0;

			/* j = 0 */
			xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j)
			{
				xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
				INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

				xh = abs(xh) - (thr >> shift_sub);
				xv = abs(xv) - (thr >> shift_sub);
				xd = abs(xd) - (thr >> shift_sub);

				xh = xh < 0 ? 0 : xh;
				xv = xv < 0 ? 0 : xv;
				xd = xd < 0 ? 0 : xd;

				xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
				xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
				xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
				val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
				accum_inner_h += val;
				val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
				accum_inner_v += val;
				val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
				accum_inner_d += val;
			}
			/* j = w-1 */
			xh = (int32_t)((((int64_t)src->band_h[i * src_px_stride + w - 1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[i * src_px_stride + w - 1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[i * src_px_stride + w - 1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_FIXED_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;

			accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
			accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;
		}
	}
	accum_inner_h = 0;
	accum_inner_v = 0;
	accum_inner_d = 0;

	/* i=h-1,j=0 */
	if ((bottom > (h - 1)) && (left <= 0))
	{
		xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		INTEGER_I4_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), 0, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

		xh = abs(xh) - (thr >> shift_sub);
		xv = abs(xv) - (thr >> shift_sub);
		xd = abs(xd) - (thr >> shift_sub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
		accum_inner_d += val;
	}

	/* i=h-1,j */
	if (bottom > (h - 1))
	{
		for (j = start_col; j < end_col; ++j)
		{
			xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride + j] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride + j] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride + j] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
			INTEGER_I4_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

			xh = abs(xh) - (thr >> shift_sub);
			xv = abs(xv) - (thr >> shift_sub);
			xd = abs(xd) - (thr >> shift_sub);

			xh = xh < 0 ? 0 : xh;
			xv = xv < 0 ? 0 : xv;
			xd = xd < 0 ? 0 : xd;

			xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
			xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
			xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
			val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
			accum_inner_h += val;
			val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
			accum_inner_v += val;
			val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
			accum_inner_d += val;
		}
	}

	/* i-h-1,j=w-1 */
	if ((bottom > (h - 1)) && (right > (w - 1)))
	{
		xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_px_stride + w - 1] * rfactor[0]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_px_stride + w - 1] * rfactor[1]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_px_stride + w - 1] * rfactor[2]) + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
		INTEGER_I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

		xh = abs(xh) - (thr >> shift_sub);
		xv = abs(xv) - (thr >> shift_sub);
		xd = abs(xd) - (thr >> shift_sub);

		xh = xh < 0 ? 0 : xh;
		xv = xv < 0 ? 0 : xv;
		xd = xd < 0 ? 0 : xd;

		xh_sq = (int32_t)((((int64_t)xh * xh) + add_before_shift_sq) >> shift_sq);
		xv_sq = (int32_t)((((int64_t)xv * xv) + add_before_shift_sq) >> shift_sq);
		xd_sq = (int32_t)((((int64_t)xd * xd) + add_before_shift_sq) >> shift_sq);
		val = (((int64_t)xh_sq * xh) + add_before_shift_cub) >> shift_cub;
		accum_inner_h += val;
		val = (((int64_t)xv_sq * xv) + add_before_shift_cub) >> shift_cub;
		accum_inner_v += val;
		val = (((int64_t)xd_sq * xd) + add_before_shift_cub) >> shift_cub;
		accum_inner_d += val;
	}
	accum_h += (accum_inner_h + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_v += (accum_inner_v + add_before_shift_inner_accum) >> shift_inner_accum;
	accum_d += (accum_inner_d + add_before_shift_inner_accum) >> shift_inner_accum;

	/**
	 * Converted to floating-point for calculating the final scores
	 * Final shifts is calculated from 3*(shifts_from_previous_stage(i.e src comes from dwt)+32)-total_shifts_done_in_this_function
	 */
	float f_accum_h = (float)(accum_h / final_shift[scale-1]);
	float f_accum_v = (float)(accum_v / final_shift[scale-1]);
	float f_accum_d = (float)(accum_d / final_shift[scale-1]);

	num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return (num_scale_h + num_scale_v + num_scale_d);
}

/*
 * This function just type casts the results of scale 0 such that next stage input is 32bits
 */
void i16_to_i32(const integer_adm_dwt_band_t_s *src, integer_i4_adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride)
{
	int i, j;
	int dst_px_stride = dst_stride / sizeof(float);

	for (i = 0; i < (h + 1) / 2; i++)
	{
		int32_t *dst_band_a_addr = &dst->band_a[i * dst_px_stride + 0];
		int16_t *src_band_a_addr = &src->band_a[i * dst_px_stride + 0];
		for (j = 0; j < (w + 1) / 2; ++j)
		{
			*(dst_band_a_addr++) = (int32_t)(*(src_band_a_addr++));
		}
	}
}

/**
 * Works similar to floating-point function adm_dwt2_s
 */
void integer_adm_dwt2_s(const int16_t *src, const integer_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, const int32_t *tmp_ref, int inp_size_bits)
{
	const int16_t *filter_lo = dwt2_db2_coeffs_lo_fix_s;
	const int16_t *filter_hi = dwt2_db2_coeffs_hi_fix_s;
	int fwidth = sizeof(dwt2_db2_coeffs_lo_fix_s) / sizeof(int16_t);

	int32_t add_bef_shift_round_VP = (int32_t) pow(2,(inp_size_bits-1));
	int32_t add_bef_shift_round_HP = 32768;
	int16_t shift_VerticalPass = inp_size_bits;
	int16_t shift_HorizontalPass = 16;

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	int16_t *tmplo = tmp_ref;
	int16_t *tmphi = tmplo + w;
	int16_t fcoeff_lo, fcoeff_hi, imgcoeff;
	int16_t s0, s1, s2, s3;
	int32_t accum;

	int i, j, fi, fj, ii, jj;
	int j0, j1, j2, j3;

	for (i = 0; i < (h + 1) / 2; ++i) {
		/* Vertical pass. */
		for (j = 0; j < w; ++j) {
			s0 = src[ind_y[0][i] * src_px_stride + j];
			s1 = src[ind_y[1][i] * src_px_stride + j];
			s2 = src[ind_y[2][i] * src_px_stride + j];
			s3 = src[ind_y[3][i] * src_px_stride + j];
			
			accum = 0;
			accum += (int32_t) filter_lo[0] * s0;
			accum += (int32_t) filter_lo[1] * s1;
			accum += (int32_t) filter_lo[2] * s2;
			accum += (int32_t) filter_lo[3] * s3;

			/**
			*  Accum is Q24 as imgcoeff1 is Q8(-128 to 127) fcoeff sums to -54822 to 54822. hence accum varies between -7017216 to 7017216 shifted by 8 to make tmp Q16(max(tmp is -27411 to 27411))
			*/
			tmplo[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;

			accum = 0;
			accum += (int32_t) filter_hi[0] * s0;
			accum += (int32_t) filter_hi[1] * s1;
			accum += (int32_t) filter_hi[2] * s2;
			accum += (int32_t) filter_hi[3] * s3;
			tmphi[j] = (accum + add_bef_shift_round_VP) >> shift_VerticalPass;
		}

		/* Horizontal pass (lo and hi). */
		for (j = 0; j < (w + 1) / 2; ++j) {

			j0 = ind_x[0][j];
			j1 = ind_x[1][j];
			j2 = ind_x[2][j];
			j3 = ind_x[3][j];
			s0 = tmplo[j0];
			s1 = tmplo[j1];
			s2 = tmplo[j2];
			s3 = tmplo[j3];
			
			accum = 0;
			accum += (int32_t) filter_lo[0] * s0;
			accum += (int32_t) filter_lo[1] * s1;
			accum += (int32_t) filter_lo[2] * s2;
			accum += (int32_t) filter_lo[3] * s3;
	        /**
             *  Accum is Q32 as imgcoeff1 is Q16(-27411 to 27411) fcoeff sums to -54822 to 54822. hence accum varies between -1502725842 to 1502725842 shifted by 16 to make tmp Q16(max(tmp is -22930 to 22930))
             */
			dst->band_a[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;
			
			accum = 0;
			accum += (int32_t) filter_hi[0] * s0;
			accum += (int32_t) filter_hi[1] * s1;
			accum += (int32_t) filter_hi[2] * s2;
			accum += (int32_t) filter_hi[3] * s3;
			dst->band_v[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;
			s0 = tmphi[j0];
			s1 = tmphi[j1];
			s2 = tmphi[j2];
			s3 = tmphi[j3];
			
			accum = 0;
			accum += (int32_t) filter_lo[0] * s0;
			accum += (int32_t) filter_lo[1] * s1;
			accum += (int32_t) filter_lo[2] * s2;
			accum += (int32_t) filter_lo[3] * s3;
			dst->band_h[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;
			
			accum = 0;
			accum += (int32_t) filter_hi[0] * s0;
			accum += (int32_t) filter_hi[1] * s1;
			accum += (int32_t) filter_hi[2] * s2;
			accum += (int32_t) filter_hi[3] * s3;
			dst->band_d[i * dst_px_stride + j] = (accum + add_bef_shift_round_HP) >> shift_HorizontalPass;

		}
	}
}

void integer_adm_dwt2_scale123_s(const int32_t *src, const integer_i4_adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride, int scale)
{
	const int16_t *filter_lo = dwt2_db2_coeffs_lo_fix_s;
	const int16_t *filter_hi = dwt2_db2_coeffs_hi_fix_s;
	int fwidth = sizeof(dwt2_db2_coeffs_lo_fix_s) / sizeof(int16_t);

	int32_t add_bef_shift_round_VP[3] = {0, 32768, 32768};
	int32_t add_bef_shift_round_HP[3] = {16384, 32768, 16384};
	int16_t shift_VerticalPass[3] = {0, 16, 16};
	int16_t shift_HorizontalPass[3] = {15, 16, 15};

	float f_shift[3] = {pow(2, 21), pow(2, 19), pow(2, 18)};

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	int32_t *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
	int32_t *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
	int16_t fcoeff_lo, fcoeff_hi;
	int32_t s0, s1, s2, s3;
	int64_t accum;

	int i, j, fi, fj, ii, jj;
	int j0, j1, j2, j3;

	for (i = 0; i < (h + 1) / 2; ++i)
	{
		/* Vertical pass. */
		for (j = 0; j < w; ++j)
		{
			s0 = src[ind_y[0][i] * src_px_stride + j];
			s1 = src[ind_y[1][i] * src_px_stride + j];
			s2 = src[ind_y[2][i] * src_px_stride + j];
			s3 = src[ind_y[3][i] * src_px_stride + j];

			accum = 0;
			accum += (int64_t)filter_lo[0] * s0;
			accum += (int64_t)filter_lo[1] * s1;
			accum += (int64_t)filter_lo[2] * s2;
			accum += (int64_t)filter_lo[3] * s3;
			/**
			 * max(src) for scale 1 is 22930
			 * max(src) for scale 2 is 2103119114
			 * max(src) for scale 3 is 1471681260
			 * abs(filtercoeff) sums up-to 54822
			 */
			tmplo[j] = (int32_t)((accum + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);

			accum = 0;
			accum += (int64_t)filter_hi[0] * s0;
			accum += (int64_t)filter_hi[1] * s1;
			accum += (int64_t)filter_hi[2] * s2;
			accum += (int64_t)filter_hi[3] * s3;
			tmphi[j] = (int32_t)((accum + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
		}

		/* Horizontal pass (lo and hi). */
		for (j = 0; j < (w + 1) / 2; ++j)
		{

			j0 = ind_x[0][j];
			j1 = ind_x[1][j];
			j2 = ind_x[2][j];
			j3 = ind_x[3][j];
			s0 = tmplo[j0];
			s1 = tmplo[j1];
			s2 = tmplo[j2];
			s3 = tmplo[j3];

			accum = 0;
			accum += (int64_t)filter_lo[0] * s0;
			accum += (int64_t)filter_lo[1] * s1;
			accum += (int64_t)filter_lo[2] * s2;
			accum += (int64_t)filter_lo[3] * s3;
			/**
			 * max(tmp) for scale 1 is 1257068460
			 * max(tmp) for scale 2 is 1759295594
			 * max(tmp) for scale 3 is 1231086884
			 * abs(filtercoeff) sums up-to 54822
			 *
			 * Shifts carried to next level for :
			 * scale 1 = 21
			 * scale 2 = 19
			 * scale 3 = 18
			 *
			 * max(scale 1 output) 2103119114
			 * max(scale 2 output) 1471681260
			 * max(scale 3 output) 2059651036
			 */
			dst->band_a[i * dst_px_stride + j] = (int32_t)((accum + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);


			accum = 0;
			accum += (int64_t)filter_hi[0] * s0;
			accum += (int64_t)filter_hi[1] * s1;
			accum += (int64_t)filter_hi[2] * s2;
			accum += (int64_t)filter_hi[3] * s3;
			dst->band_v[i * dst_px_stride + j] = (int32_t)((accum + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

			s0 = tmphi[j0];
			s1 = tmphi[j1];
			s2 = tmphi[j2];
			s3 = tmphi[j3];

			accum = 0;
			accum += (int64_t)filter_lo[0] * s0;
			accum += (int64_t)filter_lo[1] * s1;
			accum += (int64_t)filter_lo[2] * s2;
			accum += (int64_t)filter_lo[3] * s3;
			dst->band_h[i * dst_px_stride + j] = (int32_t)((accum + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

			accum = 0;
			accum += (int64_t)filter_hi[0] * s0;
			accum += (int64_t)filter_hi[1] * s1;
			accum += (int64_t)filter_hi[2] * s2;
			accum += (int64_t)filter_hi[3] * s3;
			dst->band_d[i * dst_px_stride + j] = (int32_t)((accum + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
		}
	}

	aligned_free(tmplo);
	aligned_free(tmphi);
}

void integer_adm_dwt2_scale123_combined_s(const int32_t *i4_curr_ref_scale_fixed, const integer_i4_adm_dwt_band_t_s *i4_ref_dwt2_fixed, const int32_t *i4_curr_dis_scale_fixed, const integer_i4_adm_dwt_band_t_s *i4_dis_dwt2_fixed, int **ind_y, int **ind_x, int w, int h, int curr_ref_stride, int curr_dis_stride, int dst_stride, int scale, const int32_t *tmp_ref)
{
	const int16_t *filter_lo = dwt2_db2_coeffs_lo_fix_s;
	const int16_t *filter_hi = dwt2_db2_coeffs_hi_fix_s;
	int fwidth = sizeof(dwt2_db2_coeffs_lo_fix_s) / sizeof(int16_t);

	int32_t add_bef_shift_round_VP[3] = { 0, 32768, 32768 };
	int32_t add_bef_shift_round_HP[3] = { 16384, 32768, 16384 };
	int16_t shift_VerticalPass[3] = { 0, 16, 16 };
	int16_t shift_HorizontalPass[3] = { 15, 16, 15 };

	float f_shift[3] = { pow(2, 21), pow(2, 19), pow(2, 18) };

	int src1_px_stride = curr_ref_stride / sizeof(float);
	int src2_px_stride = curr_dis_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	int32_t *tmplo_ref = tmp_ref;
	int32_t *tmphi_ref = tmplo_ref + w;
	int32_t *tmplo_dis = tmphi_ref + w;
	int32_t *tmphi_dis = tmplo_dis + w;
	int16_t fcoeff_lo, fcoeff_hi;
	int32_t s10, s11, s12, s13;

	int64_t accum_ref;

	int i, j, fi, fj, ii, jj;
	int j0, j1, j2, j3;

	for (i = 0; i < (h + 1) / 2; ++i)
	{
		/* Vertical pass. */
		for (j = 0; j < w; ++j)
		{
			s10 = i4_curr_ref_scale_fixed[ind_y[0][i] * src1_px_stride + j];
			s11 = i4_curr_ref_scale_fixed[ind_y[1][i] * src1_px_stride + j];
			s12 = i4_curr_ref_scale_fixed[ind_y[2][i] * src1_px_stride + j];
			s13 = i4_curr_ref_scale_fixed[ind_y[3][i] * src1_px_stride + j];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			tmplo_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			tmphi_ref[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);

			s10 = i4_curr_dis_scale_fixed[ind_y[0][i] * src2_px_stride + j];
			s11 = i4_curr_dis_scale_fixed[ind_y[1][i] * src2_px_stride + j];
			s12 = i4_curr_dis_scale_fixed[ind_y[2][i] * src2_px_stride + j];
			s13 = i4_curr_dis_scale_fixed[ind_y[3][i] * src2_px_stride + j];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			tmplo_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			tmphi_dis[j] = (int32_t)((accum_ref + add_bef_shift_round_VP[scale - 1]) >> shift_VerticalPass[scale - 1]);

		}

		/* Horizontal pass (lo and hi). */
		for (j = 0; j < (w + 1) / 2; ++j)
		{

			j0 = ind_x[0][j];
			j1 = ind_x[1][j];
			j2 = ind_x[2][j];
			j3 = ind_x[3][j];

			s10 = tmplo_ref[j0];
			s11 = tmplo_ref[j1];
			s12 = tmplo_ref[j2];
			s13 = tmplo_ref[j3];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			i4_ref_dwt2_fixed->band_a[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			i4_ref_dwt2_fixed->band_v[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			s10 = tmphi_ref[j0];
			s11 = tmphi_ref[j1];
			s12 = tmphi_ref[j2];
			s13 = tmphi_ref[j3];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			i4_ref_dwt2_fixed->band_h[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			i4_ref_dwt2_fixed->band_d[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);

			s10 = tmplo_dis[j0];
			s11 = tmplo_dis[j1];
			s12 = tmplo_dis[j2];
			s13 = tmplo_dis[j3];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			i4_dis_dwt2_fixed->band_a[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			i4_dis_dwt2_fixed->band_v[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			s10 = tmphi_dis[j0];
			s11 = tmphi_dis[j1];
			s12 = tmphi_dis[j2];
			s13 = tmphi_dis[j3];
			accum_ref = 0;
			accum_ref += (int64_t)filter_lo[0] * s10;
			accum_ref += (int64_t)filter_lo[1] * s11;
			accum_ref += (int64_t)filter_lo[2] * s12;
			accum_ref += (int64_t)filter_lo[3] * s13;
			i4_dis_dwt2_fixed->band_h[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
			accum_ref = 0;
			accum_ref += (int64_t)filter_hi[0] * s10;
			accum_ref += (int64_t)filter_hi[1] * s11;
			accum_ref += (int64_t)filter_hi[2] * s12;
			accum_ref += (int64_t)filter_hi[3] * s13;
			i4_dis_dwt2_fixed->band_d[i * dst_px_stride + j] = (int32_t)((accum_ref + add_bef_shift_round_HP[scale - 1]) >> shift_HorizontalPass[scale - 1]);
		}
    }
}
