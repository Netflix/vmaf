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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mem.h"
#include "adm_options.h"
#include "adm_tools.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef __SSE2__
#ifdef ADM_OPT_RECIP_DIVISION

#include <emmintrin.h>

static float rcp_s(float x)
{
    float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

#define DIVS(n, d) ((n) * rcp_s(d))
#endif //ADM_OPT_RECIP_DIVISION
#else
#define DIVS(n, d) ((n) / (d))
#endif // __SSE2__

static const float dwt2_db2_coeffs_lo_s[4] = { 0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921 };
static const float dwt2_db2_coeffs_hi_s[4] = { -0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690 };

static const double dwt2_db2_coeffs_lo_d[4] = { 0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921 };
static const double dwt2_db2_coeffs_hi_d[4] = { -0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690 };

#ifndef FLOAT_ONE_BY_30
#define FLOAT_ONE_BY_30	0.0333333351
#endif

#ifndef FLOAT_ONE_BY_15
#define FLOAT_ONE_BY_15 0.0666666701
#endif

float adm_sum_cube_s(const float *x, int w, int h, int stride, double border_factor)
{
    int px_stride = stride / sizeof(float);
    int left   = w * border_factor - 0.5;
    int top    = h * border_factor - 0.5;
    int right  = w - left;
    int bottom = h - top;

    int i, j;

    float val;
    float accum = 0;

    for (i = top; i < bottom; ++i) {
        float accum_inner = 0;

        for (j = left; j < right; ++j) {
            val = fabsf(x[i * px_stride + j]);

            accum_inner += val * val * val;
        }

        accum += accum_inner;
    }

    return powf(accum, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
}

void adm_decouple_s(const adm_dwt_band_t_s *ref, const adm_dwt_band_t_s *dis,
        const adm_dwt_band_t_s *r, const adm_dwt_band_t_s *a, int w, int h,
        int ref_stride, int dis_stride, int r_stride, int a_stride,
        double border_factor, double adm_enhn_gain_limit)
{
#ifdef ADM_OPT_AVOID_ATAN
	const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
	const float eps = 1e-30;

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

	float oh, ov, od, th, tv, td;
	float kh, kv, kd, rst_h, rst_v, rst_d;
#ifdef ADM_OPT_AVOID_ATAN
	float ot_dp, o_mag_sq, t_mag_sq;
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

			kh = DIVS(th, oh + eps);
			kv = DIVS(tv, ov + eps);
			kd = DIVS(td, od + eps);

			kh = kh < 0.0f ? 0.0f : (kh > 1.0f ? 1.0f : kh);
			kv = kv < 0.0f ? 0.0f : (kv > 1.0f ? 1.0f : kv);
			kd = kd < 0.0f ? 0.0f : (kd > 1.0f ? 1.0f : kd);

            rst_h = kh * oh;
            rst_v = kv * ov;
            rst_d = kd * od;
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
			ot_dp = oh * th + ov * tv;
			o_mag_sq = oh * oh + ov * ov;
			t_mag_sq = th * th + tv * tv;

			angle_flag = (ot_dp >= 0.0f) && (ot_dp * ot_dp >= cos_1deg_sq * o_mag_sq * t_mag_sq);
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
            /* ==== original ==== */
			// if (angle_flag) {
            //     rst_h = th;
            //     rst_v = tv;
            //     rst_d = td;
			// }

			/* ==== modification ==== */

            if (angle_flag && (rst_h > 0.0)) rst_h = MIN(rst_h * adm_enhn_gain_limit, th);
            if (angle_flag && (rst_h < 0.0)) rst_h = MAX(rst_h * adm_enhn_gain_limit, th);

            if (angle_flag && (rst_v > 0.0)) rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v < 0.0)) rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d > 0.0)) rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d < 0.0)) rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            /* == end of modification == */

			r->band_h[i * r_px_stride + j] = rst_h;
			r->band_v[i * r_px_stride + j] = rst_v;
			r->band_d[i * r_px_stride + j] = rst_d;

			a->band_h[i * a_px_stride + j] = th - rst_h;
			a->band_v[i * a_px_stride + j] = tv - rst_v;
			a->band_d[i * a_px_stride + j] = td - rst_d;
		}
	}
}

void adm_csf_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const adm_dwt_band_t_s *flt,
               int orig_h, int scale, int w, int h, int src_stride, int dst_stride, double border_factor,
               double adm_norm_view_dist, int adm_ref_display_height, int adm_csf_mode)
{
	(void)orig_h;
	(void)adm_csf_mode;

	const float *src_angles[3] = { src->band_h, src->band_v, src->band_d };
	float *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
	float *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

	const float *src_ptr;
	float *dst_ptr;
	float *flt_ptr;

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	// TODO: we will add more CSF functions here
	float factor1, factor2;
	factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
	factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
	float rfactor[3] = { factor1, factor1, factor2 };

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
	float dst_val;

	for (theta = 0; theta < 3; ++theta) {
		src_ptr = src_angles[theta];
		dst_ptr = dst_angles[theta];
		flt_ptr = flt_angles[theta];

		for (i = top; i < bottom; ++i) {
			src_offset = i * src_px_stride;
			dst_offset = i * dst_px_stride;

			for (j = left; j < right; ++j) {
				dst_val = rfactor[theta] * src_ptr[src_offset + j];
				dst_ptr[dst_offset + j] = dst_val;
				flt_ptr[dst_offset + j] = FLOAT_ONE_BY_30 * fabsf(dst_val);
			}
		}
	}
}

/* Combination of adm_csf_s and adm_sum_cube_s for csf_o based den_scale */
float adm_csf_den_scale_s(const adm_dwt_band_t_s *src, int orig_h, int scale,
                          int w, int h, int src_stride, double border_factor,
                          double adm_norm_view_dist, int adm_ref_display_height, int adm_csf_mode)
{
	(void)adm_csf_mode;
	(void)orig_h;

	float *src_h = src->band_h, *src_v = src->band_v, *src_d = src->band_d;

	int src_px_stride = src_stride / sizeof(float);

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	// TODO: we will add more CSF functions here
	float factor1, factor2;
	factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
	factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
	float rfactor[3] = { factor1, factor1, factor2 };

	float accum_h = 0, accum_v = 0, accum_d = 0;
	float accum_inner_h, accum_inner_v, accum_inner_d;
	float den_scale_h, den_scale_v, den_scale_d;

	float val;
	
	/* The computation of the denominator scales is not required for the regions which lie outside the frame borders */
	int left = w * border_factor - 0.5;
	int top = h * border_factor - 0.5;
	int right = w - left;
	int bottom = h - top;

	int i, j;

	for (i = top; i < bottom; ++i) {
		accum_inner_h = 0;
		accum_inner_v = 0;
		accum_inner_d = 0;
		src_h = src->band_h + i * src_px_stride;
		src_v = src->band_v + i * src_px_stride;
		src_d = src->band_d + i * src_px_stride;
		for (j = left; j < right; ++j) {
			float abs_csf_o_val_h = fabsf(rfactor[0] * src_h[j]);
			float abs_csf_o_val_v = fabsf(rfactor[1] * src_v[j]);
			float abs_csf_o_val_d = fabsf(rfactor[2] * src_d[j]);

			val = abs_csf_o_val_h * abs_csf_o_val_h * abs_csf_o_val_h;
			accum_inner_h += val;
			val = abs_csf_o_val_v * abs_csf_o_val_v * abs_csf_o_val_v;
			accum_inner_v += val;
			val = abs_csf_o_val_d * abs_csf_o_val_d * abs_csf_o_val_d;
			accum_inner_d += val;
		}

		accum_h += accum_inner_h;
		accum_v += accum_inner_v;
		accum_d += accum_inner_d;

	}

	den_scale_h = powf(accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_v = powf(accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	den_scale_d = powf(accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return(den_scale_h + den_scale_v + den_scale_d);

}

float adm_cm_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *csf_f,
               const adm_dwt_band_t_s *csf_a, int w, int h, int src_stride,
               int flt_stride, int csf_a_stride, double border_factor, int scale,
               double adm_norm_view_dist, int adm_ref_display_height, int adm_csf_mode)
{
	(void)flt_stride;
	(void)adm_csf_mode;

	// for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
	// 1 to 4 (from finest scale to coarsest scale).
	// TODO: we will add more CSF functions here
	float factor1, factor2;
	factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
	factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
	float rfactor[3] = { factor1, factor1, factor2 };

	const float *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
	const float *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

	int src_px_stride = src_stride / sizeof(float);
	int csf_px_stride = csf_a_stride / sizeof(float);

	float xh, xv, xd, thr;

	float val;
	float accum_h = 0, accum_v = 0, accum_d = 0;
	float accum_inner_h, accum_inner_v, accum_inner_d;
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
		xh = src->band_h[0] * rfactor[0];
		xv = src->band_v[0] * rfactor[1];
		xd = src->band_d[0] * rfactor[2];
		ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_px_stride, &thr, w, h, 0, 0);

		xh = fabsf(xh) - thr;
		xv = fabsf(xv) - thr;
		xd = fabsf(xd) - thr;

		xh = xh < 0.0f ? 0.0f : xh;
		xv = xv < 0.0f ? 0.0f : xv;
		xd = xd < 0.0f ? 0.0f : xd;

		val = (xh * xh * xh);
		accum_inner_h += val;
		val = (xv * xv * xv);
		accum_inner_v += val;
		val = (xd * xd * xd);
		accum_inner_d += val;

	}

	/* i=0, j */
	if (top <= 0) {
		for (j = start_col; j < end_col; ++j) {
			xh = src->band_h[j] * rfactor[0];
			xv = src->band_v[j] * rfactor[1];
			xd = src->band_d[j] * rfactor[2];
			ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_px_stride, &thr, w, h, 0, j);

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

		}
	}

	/* i=0,j=w-1 */
	if ((top <= 0) && (right > (w - 1)))
	{
		xh = src->band_h[w - 1] * rfactor[0];
		xv = src->band_v[w - 1] * rfactor[1];
		xd = src->band_d[w - 1] * rfactor[2];
		ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, 0, (w - 1));

		xh = fabsf(xh) - thr;
		xv = fabsf(xv) - thr;
		xd = fabsf(xd) - thr;

		xh = xh < 0.0f ? 0.0f : xh;
		xv = xv < 0.0f ? 0.0f : xv;
		xd = xd < 0.0f ? 0.0f : xd;

		val = (xh * xh * xh);
		accum_inner_h += val;
		val = (xv * xv * xv);
		accum_inner_v += val;
		val = (xd * xd * xd);
		accum_inner_d += val;

	}

	accum_h += accum_inner_h;
	accum_v += accum_inner_v;
	accum_d += accum_inner_d;

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
				ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

		}
			accum_h += accum_inner_h;
			accum_v += accum_inner_v;
			accum_d += accum_inner_d;
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
			ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);


			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = fabsf(xh) - thr;
				xv = fabsf(xv) - thr;
				xd = fabsf(xd) - thr;

				xh = xh < 0.0f ? 0.0f : xh;
				xv = xv < 0.0f ? 0.0f : xv;
				xd = xd < 0.0f ? 0.0f : xd;

				val = (xh * xh * xh);
				accum_inner_h += val;
				val = (xv * xv * xv);
				accum_inner_v += val;
				val = (xd * xd * xd);
				accum_inner_d += val;

			}
	accum_h += accum_inner_h;
	accum_v += accum_inner_v;
	accum_d += accum_inner_d;
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
				ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = fabsf(xh) - thr;
				xv = fabsf(xv) - thr;
				xd = fabsf(xd) - thr;

				xh = xh < 0.0f ? 0.0f : xh;
				xv = xv < 0.0f ? 0.0f : xv;
				xd = xd < 0.0f ? 0.0f : xd;

				val = (xh * xh * xh);
				accum_inner_h += val;
				val = (xv * xv * xv);
				accum_inner_v += val;
				val = (xd * xd * xd);
				accum_inner_d += val;

			}
			/* j = w-1 */
			xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
			xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
			xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
			ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

			accum_h += accum_inner_h;
			accum_v += accum_inner_v;
			accum_d += accum_inner_d;
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
			ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_px_stride, &thr, w, h, i, 0);

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

			/* j within frame */
			for (j = start_col; j < end_col; ++j) {
				xh = src->band_h[i * src_px_stride + j] * rfactor[0];
				xv = src->band_v[i * src_px_stride + j] * rfactor[1];
				xd = src->band_d[i * src_px_stride + j] * rfactor[2];
				ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_px_stride, &thr, w, h, i, j);

				xh = fabsf(xh) - thr;
				xv = fabsf(xv) - thr;
				xd = fabsf(xd) - thr;

				xh = xh < 0.0f ? 0.0f : xh;
				xv = xv < 0.0f ? 0.0f : xv;
				xd = xd < 0.0f ? 0.0f : xd;

				val = (xh * xh * xh);
				accum_inner_h += val;
				val = (xv * xv * xv);
				accum_inner_v += val;
				val = (xd * xd * xd);
				accum_inner_d += val;

			}
			/* j = w-1 */
			xh = src->band_h[i * src_px_stride + w - 1] * rfactor[0];
			xv = src->band_v[i * src_px_stride + w - 1] * rfactor[1];
			xd = src->band_d[i * src_px_stride + w - 1] * rfactor[2];
			ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, i, (w - 1));

		xh = fabsf(xh) - thr;
		xv = fabsf(xv) - thr;
		xd = fabsf(xd) - thr;

		xh = xh < 0.0f ? 0.0f : xh;
		xv = xv < 0.0f ? 0.0f : xv;
		xd = xd < 0.0f ? 0.0f : xd;

		val = (xh * xh * xh);
		accum_inner_h += val;
		val = (xv * xv * xv);
		accum_inner_v += val;
		val = (xd * xd * xd);
		accum_inner_d += val;

			accum_h += accum_inner_h;
			accum_v += accum_inner_v;
			accum_d += accum_inner_d;
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
		ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), 0);

		xh = fabsf(xh) - thr;
		xv = fabsf(xv) - thr;
		xd = fabsf(xd) - thr;

		xh = xh < 0.0f ? 0.0f : xh;
		xv = xv < 0.0f ? 0.0f : xv;
		xd = xd < 0.0f ? 0.0f : xd;

		val = (xh * xh * xh);
		accum_inner_h += val;
		val = (xv * xv * xv);
		accum_inner_v += val;
		val = (xd * xd * xd);
		accum_inner_d += val;

	}

	/* i=h-1,j */
	if (bottom > (h - 1)) {
		for (j = start_col; j < end_col; ++j) {
			xh = src->band_h[(h - 1) * src_px_stride + j] * rfactor[0];
			xv = src->band_v[(h - 1) * src_px_stride + j] * rfactor[1];
			xd = src->band_d[(h - 1) * src_px_stride + j] * rfactor[2];
			ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), j);

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

		}
	}

	/* i-h-1,j=w-1 */
	if ((bottom > (h - 1)) && (right > (w - 1)))
	{
		xh = src->band_h[(h - 1) * src_px_stride + w - 1] * rfactor[0];
		xv = src->band_v[(h - 1) * src_px_stride + w - 1] * rfactor[1];
		xd = src->band_d[(h - 1) * src_px_stride + w - 1] * rfactor[2];
		ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_px_stride, &thr, w, h, (h - 1), (w - 1));

			xh = fabsf(xh) - thr;
			xv = fabsf(xv) - thr;
			xd = fabsf(xd) - thr;

			xh = xh < 0.0f ? 0.0f : xh;
			xv = xv < 0.0f ? 0.0f : xv;
			xd = xd < 0.0f ? 0.0f : xd;

			val = (xh * xh * xh);
			accum_inner_h += val;
			val = (xv * xv * xv);
			accum_inner_v += val;
			val = (xd * xd * xd);
			accum_inner_d += val;

		}
		accum_h += accum_inner_h;
		accum_v += accum_inner_v;
		accum_d += accum_inner_d;
	


	num_scale_h = powf(accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_v = powf(accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
	num_scale_d = powf(accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

	return (num_scale_h + num_scale_v + num_scale_d);
}

// This function stores the imgcoeff values used in adm_dwt2_s in buffers, which reduces the control code cycles.
void dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h)
{
	int i, j;
	int ind0, ind1, ind2, ind3;
	/* Vertical pass */
	for (i = 0; i < (h + 1) / 2; ++i) { /* Index = 2 * i - 1 + fi */
		ind0 = 2 * i - 1;
		ind0 = (ind0 < 0) ? -ind0 : ((ind0 >= h) ? (2 * h - ind0 - 1) : ind0);
		ind1 = 2 * i;
		if (ind1 >= h) {
			ind1 = (2 * h - ind1 - 1);
		}
		ind2 = 2 * i + 1;
		if (ind2 >= h) {
			ind2 = (2 * h - ind2 - 1);
		}
		ind3 = 2 * i + 2;
		if (ind3 >= h) {
			ind3 = (2 * h - ind3 - 1);
		}
		src_ind_y[0][i] = ind0;
		src_ind_y[1][i] = ind1;
		src_ind_y[2][i] = ind2;
		src_ind_y[3][i] = ind3;
	}
	/* Horizontal pass */
	for (j = 0; j < (w + 1) / 2; ++j) { /* Index = 2 * j - 1 + fj */
		ind0 = 2 * j - 1;
		ind0 = (ind0 < 0) ? -ind0 : ((ind0 >= w) ? (2 * w - ind0 - 1) : ind0);
		ind1 = 2 * j;
		if (ind1 >= w) {
			ind1 = (2 * w - ind1 - 1);
		}
		ind2 = 2 * j + 1;
		if (ind2 >= w) {
			ind2 = (2 * w - ind2 - 1);
		}
		ind3 = 2 * j + 2;
		if (ind3 >= w) {
			ind3 = (2 * w - ind3 - 1);
		}
		src_ind_x[0][j] = ind0;
		src_ind_x[1][j] = ind1;
		src_ind_x[2][j] = ind2;
		src_ind_x[3][j] = ind3;
	}
}

void adm_dwt2_s(const float *src, const adm_dwt_band_t_s *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride)
{
	const float *filter_lo = dwt2_db2_coeffs_lo_s;
	const float *filter_hi = dwt2_db2_coeffs_hi_s;

	int src_px_stride = src_stride / sizeof(float);
	int dst_px_stride = dst_stride / sizeof(float);

	float *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
	float *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
	float s0, s1, s2, s3;
	float accum;

	int i, j;
	int j0, j1, j2, j3;

	for (i = 0; i < (h + 1) / 2; ++i) {
		/* Vertical pass. */
		for (j = 0; j < w; ++j) {
			s0 = src[ind_y[0][i] * src_px_stride + j];
			s1 = src[ind_y[1][i] * src_px_stride + j];
			s2 = src[ind_y[2][i] * src_px_stride + j];
			s3 = src[ind_y[3][i] * src_px_stride + j];
			
			accum = 0;
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			tmplo[j] = accum;
			
			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			tmphi[j] = accum;
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
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			dst->band_a[i * dst_px_stride + j] = accum;
			
			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			dst->band_v[i * dst_px_stride + j] = accum;
			s0 = tmphi[j0];
			s1 = tmphi[j1];
			s2 = tmphi[j2];
			s3 = tmphi[j3];
			
			accum = 0;
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			dst->band_h[i * dst_px_stride + j] = accum;
			
			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			dst->band_d[i * dst_px_stride + j] = accum;

		}
	}

	aligned_free(tmplo);
	aligned_free(tmphi);
}

void adm_dwt2_d(const double *src, const adm_dwt_band_t_d *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride)
{
	const double *filter_lo = dwt2_db2_coeffs_lo_d;
	const double *filter_hi = dwt2_db2_coeffs_hi_d;

	int src_px_stride = src_stride / sizeof(double);
	int dst_px_stride = dst_stride / sizeof(double);

	double *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(double) * w), MAX_ALIGN);
	double *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(double) * w), MAX_ALIGN);
	double s0, s1, s2, s3;
	double accum;

	int i, j;
	int j0, j1, j2, j3;

	for (i = 0; i < (h + 1) / 2; ++i) {
		/* Vertical pass. */
		for (j = 0; j < w; ++j) {
			s0 = src[ind_y[0][i] * src_px_stride + j];
			s1 = src[ind_y[1][i] * src_px_stride + j];
			s2 = src[ind_y[2][i] * src_px_stride + j];
			s3 = src[ind_y[3][i] * src_px_stride + j];

			accum = 0;
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			tmplo[j] = accum;

			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			tmphi[j] = accum;
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
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			dst->band_a[i * dst_px_stride + j] = accum;

			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			dst->band_v[i * dst_px_stride + j] = accum;
			s0 = tmphi[j0];
			s1 = tmphi[j1];
			s2 = tmphi[j2];
			s3 = tmphi[j3];

			accum = 0;
			accum += filter_lo[0] * s0;
			accum += filter_lo[1] * s1;
			accum += filter_lo[2] * s2;
			accum += filter_lo[3] * s3;
			dst->band_h[i * dst_px_stride + j] = accum;

			accum = 0;
			accum += filter_hi[0] * s0;
			accum += filter_hi[1] * s1;
			accum += filter_hi[2] * s2;
			accum += filter_hi[3] * s3;
			dst->band_d[i * dst_px_stride + j] = accum;

		}
	}

	aligned_free(tmplo);
	aligned_free(tmphi);
}

void adm_buffer_copy(const void *src, void *dst, int linewidth, int h, int src_stride, int dst_stride)
{
    const char *src_p = src;
    char *dst_p = dst;
    int i;

    for (i = 0; i < h; ++i) {
        memcpy(dst_p, src_p, linewidth);
        src_p += src_stride;
        dst_p += dst_stride;
    }
}
