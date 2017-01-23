/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
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
#include "common/alloc.h"
#include "adm_options.h"
#include "adm_tools.h"

#ifndef M_PI
  #define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

#ifdef ADM_OPT_RECIP_DIVISION

#include <emmintrin.h>

static float rcp_s(float x)
{
    float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

static double rcp_d(double x)
{
    __m128d xd = _mm_load_sd(&x);
    double xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_cvtsd_ss(_mm_setzero_ps(), xd)));

    xi = xi + xi * (1.0 - x * xi);
    xi = xi + xi * (1.0 - x * xi);

    return xi;
}

#define DIVS(n, d) ((n) * rcp_s(d))
#define DIVD(n, d) ((n) * rcp_d(d))

#else

#define DIVS(n, d) ((n) / (d))
#define DIVD(n, d) ((n) / (d))

#endif /* ADM_OPT_RECIP_DIVISION */

static const float dwt2_db2_coeffs_lo_s[4] = { 0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921 };
static const float dwt2_db2_coeffs_hi_s[4] = { -0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690 };

static const double dwt2_db2_coeffs_lo_d[4] = { 0x1.ee8dd4748ca11p-2, 0x1.ac4bdd6e3f184p-1, 0x1.cb0bf0b6b5b13p-3, -0x1.0907dc192d6ddp-3 };
static const double dwt2_db2_coeffs_hi_d[4] = { -0x1.0907dc192d6ddp-3, -0x1.cb0bf0b6b5b13p-3, 0x1.ac4bdd6e3f184p-1, -0x1.ee8dd4748ca11p-2 };

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

    return powf(accum, 1.0f / 3.0f);
}

double adm_sum_cube_d(const double *x, int w, int h, int stride, double border_factor)
{
    int px_stride = stride / sizeof(double);
    int left   = w * border_factor - 0.5;
    int top    = h * border_factor - 0.5;
    int right  = w - left;
    int bottom = h - top;

    int i, j;

    double val;
    double accum = 0;

    for (i = top; i < bottom; ++i) {
        double accum_inner = 0;

        for (j = left; j < right; ++j) {
            val = fabs(x[i * px_stride + j]);

            accum_inner += val * val * val;
        }

        accum += accum_inner;
    }

    return pow(accum, 1.0 / 3.0);
}

void adm_decouple_s(const adm_dwt_band_t_s *ref, const adm_dwt_band_t_s *dis, const adm_dwt_band_t_s *r, const adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride)
{
#ifdef ADM_OPT_AVOID_ATAN
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
    const float eps = 1e-30;

    int ref_px_stride = ref_stride / sizeof(float);
    int dis_px_stride = dis_stride / sizeof(float);
    int r_px_stride = r_stride / sizeof(float);
    int a_px_stride = a_stride / sizeof(float);

    float oh, ov, od, th, tv, td;
    float kh, kv, kd, tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
    float ot_dp, o_mag_sq, t_mag_sq;
#else
    float oa, ta, diff;
#endif
    int angle_flag;
    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
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

            tmph = kh * oh;
            tmpv = kv * ov;
            tmpd = kd * od;
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

void adm_decouple_d(const adm_dwt_band_t_d *ref, const adm_dwt_band_t_d *dis, const adm_dwt_band_t_d *r, const adm_dwt_band_t_d *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride)
{
#ifdef ADM_OPT_AVOID_ATAN
    const double cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
#endif
    const double eps = 1e-30;

    int ref_px_stride = ref_stride / sizeof(double);
    int dis_px_stride = dis_stride / sizeof(double);
    int r_px_stride = r_stride / sizeof(double);
    int a_px_stride = a_stride / sizeof(double);

    double oh, ov, od, th, tv, td;
    double kh, kv, kd, tmph, tmpv, tmpd;
#ifdef ADM_OPT_AVOID_ATAN
    double ot_dp, o_mag_sq, t_mag_sq;
#else
    double oa, ta, diff;
#endif
    int angle_flag;
    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            oh = ref->band_h[i * ref_px_stride + j];
            ov = ref->band_v[i * ref_px_stride + j];
            od = ref->band_d[i * ref_px_stride + j];
            th = dis->band_h[i * dis_px_stride + j];
            tv = dis->band_v[i * dis_px_stride + j];
            td = dis->band_d[i * dis_px_stride + j];

            kh = DIVD(th, oh + eps);
            kv = DIVD(tv, ov + eps);
            kd = DIVD(td, od + eps);

            kh = kh < 0.0 ? 0.0 : (kh > 1.0 ? 1.0 : kh);
            kv = kv < 0.0 ? 0.0 : (kv > 1.0 ? 1.0 : kv);
            kd = kd < 0.0 ? 0.0 : (kd > 1.0 ? 1.0 : kd);

            tmph = kh * oh;
            tmpv = kv * ov;
            tmpd = kd * od;
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

            angle_flag = (ot_dp >= 0.0) && (ot_dp * ot_dp >= cos_1deg_sq * o_mag_sq * t_mag_sq);
#else
            oa = atan(DIVD(ov, oh + eps));
            ta = atan(DIVD(tv, th + eps));

            if (oh < 0.0)
                oa += M_PI;
            if (th < 0.0)
                ta += M_PI;

            diff = fabs(oa - ta) * 180.0 / M_PI;
            angle_flag = diff < 1.0;
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

void adm_csf_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride)
{
    const float view_dis = 4;
    const float a = 0.31;
    const float b = 0.69;
    const float c = 0.29;

    const float *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    float *dst_angles[3]       = { dst->band_h, dst->band_v, dst->band_d };
    float p[3]                 = { 1, 1, -1 };

    const float *src_ptr;
    float *dst_ptr;

    float f1 = M_PI * orig_h * view_dis / (180 * 1 << (scale + 1));
    float f2;
    float factor;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    int i, j, theta;

    for (theta = 0; theta < 3; ++theta) {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];

        f2     = f1 / (0.15 * p[theta] + 0.85);
        factor = (a + b * f2) * exp(-c * f2);

        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                dst_ptr[i * dst_px_stride + j] = factor * src_ptr[i * src_px_stride + j];
            }
        }
    }
}

void adm_csf_d(const adm_dwt_band_t_d *src, const adm_dwt_band_t_d *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride)
{
    const double view_dis = 4;
    const double a = 0.31;
    const double b = 0.69;
    const double c = 0.29;

    const double *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    double *dst_angles[3]       = { dst->band_h, dst->band_v, dst->band_d };
    double p[3]                 = { 1, 1, -1 };

    const double *src_ptr;
    double *dst_ptr;

    double f1 = M_PI * orig_h * view_dis / (180 * 1 << (scale + 1));
    double f2;
    double factor;

    int src_px_stride = src_stride / sizeof(double);
    int dst_px_stride = dst_stride / sizeof(double);

    int i, j, theta;

    for (theta = 0; theta < 3; ++theta) {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];

        f2     = f1 / (0.15 * p[theta] + 0.85);
        factor = (a + b * f2) * exp(-c * f2);

        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                dst_ptr[i * dst_px_stride + j] = factor * src_ptr[i * src_px_stride + j];
            }
        }
    }
}

void adm_cm_thresh_s(const adm_dwt_band_t_s *src, float *dst, int w, int h, int src_stride, int dst_stride)
{
    const float *angles[3] = { src->band_h, src->band_v, src->band_d };
    const float *src_ptr;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float fcoeff, imgcoeff;

    int theta, i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Zero output row. */
        for (j = 0; j < w; ++j) {
            dst[i * dst_px_stride + j] = 0;
        }

        for (theta = 0; theta < 3; ++theta) {
            src_ptr = angles[theta];

            for (j = 0; j < w; ++j) {
                float accum = 0;

                /* Mean of three convolutions by [1 1 1; 1 2 1; 1 1 1]. */
                for (fi = 0; fi < 3; ++fi) {
                    for (fj = 0; fj < 3; ++fj) {
                        fcoeff = (fi == 1 && fj == 1) ? 1.0f / 15.0f : 1.0f / 30.0f;

                        ii = i - 1 + fi;
                        jj = j - 1 + fj;

                        /* Border handling by mirroring. */
                        if (ii < 0)
                            ii = -ii;
                        else if (ii >= h)
                            ii = 2 * h - ii - 1;
                        if (jj < 0)
                            jj = -jj;
                        else if (jj >= w)
                            jj = 2 * w - jj - 1;
                        imgcoeff = fabsf(src_ptr[ii * src_px_stride + jj]);

                        accum += fcoeff * imgcoeff;
                    }
                }

                dst[i * dst_px_stride + j] += accum;
            }
        }
    }
}

void adm_cm_thresh_d(const adm_dwt_band_t_d *src, double *dst, int w, int h, int src_stride, int dst_stride)
{
    const double *angles[3] = { src->band_h, src->band_v, src->band_d };
    const double *src_ptr;

    int src_px_stride = src_stride / sizeof(double);
    int dst_px_stride = dst_stride / sizeof(double);

    double fcoeff, imgcoeff;

    int theta, i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Zero output row. */
        for (j = 0; j < w; ++j) {
            dst[i * dst_px_stride + j] = 0;
        }

        for (theta = 0; theta < 3; ++theta) {
            src_ptr = angles[theta];

            for (j = 0; j < w; ++j) {
                double accum = 0;

                /* Mean of three convolutions by [1 1 1; 1 2 1; 1 1 1]. */
                for (fi = 0; fi < 3; ++fi) {
                    for (fj = 0; fj < 3; ++fj) {
                        fcoeff = (fi == 1 && fj == 1) ? 1.0 / 15.0 : 1.0 / 30.0;

                        ii = i - 1 + fi;
                        jj = j - 1 + fj;

                        /* Border handling by mirroring. */
                        if (ii < 0)
                            ii = -ii;
                        else if (ii >= h)
                            ii = 2 * h - ii - 1;
                        if (jj < 0)
                            jj = -jj;
                        else if (jj >= w)
                            jj = 2 * w - jj - 1;
                        imgcoeff = fabsf(src_ptr[ii * src_px_stride + jj]);

                        accum += fcoeff * imgcoeff;
                    }
                }

                dst[i * dst_px_stride + j] += accum;
            }
        }
    }
}

void adm_cm_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const float *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);
    int thresh_px_stride = thresh_stride / sizeof(float);

    float xh, xv, xd, thr;

    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            xh  = src->band_h[i * src_px_stride + j];
            xv  = src->band_v[i * src_px_stride + j];
            xd  = src->band_d[i * src_px_stride + j];
            thr = thresh[i * thresh_px_stride + j];

            xh = fabsf(xh) - thr;
            xv = fabsf(xv) - thr;
            xd = fabsf(xd) - thr;

            xh = xh < 0.0f ? 0.0f : xh;
            xv = xv < 0.0f ? 0.0f : xv;
            xd = xd < 0.0f ? 0.0f : xd;

            dst->band_h[i * dst_px_stride + j] = xh;
            dst->band_v[i * dst_px_stride + j] = xv;
            dst->band_d[i * dst_px_stride + j] = xd;
        }
    }
}

void adm_cm_d(const adm_dwt_band_t_d *src, const adm_dwt_band_t_d *dst, const double *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride)
{
    int src_px_stride = src_stride / sizeof(double);
    int dst_px_stride = dst_stride / sizeof(double);
    int thresh_px_stride = thresh_stride / sizeof(double);

    double xh, xv, xd, thr;

    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            xh  = src->band_h[i * src_px_stride + j];
            xv  = src->band_v[i * src_px_stride + j];
            xd  = src->band_d[i * src_px_stride + j];
            thr = thresh[i * thresh_px_stride + j];

            xh = fabs(xh) - thr;
            xv = fabs(xv) - thr;
            xd = fabs(xd) - thr;

            xh = xh < 0 ? 0 : xh;
            xv = xv < 0 ? 0 : xv;
            xd = xd < 0 ? 0 : xd;

            dst->band_h[i * dst_px_stride + j] = xh;
            dst->band_v[i * dst_px_stride + j] = xv;
            dst->band_d[i * dst_px_stride + j] = xd;
        }
    }
}

void adm_dwt2_s(const float *src, const adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride)
{
    const float *filter_lo = dwt2_db2_coeffs_lo_s;
    const float *filter_hi = dwt2_db2_coeffs_hi_s;
    int fwidth = sizeof(dwt2_db2_coeffs_lo_s) / sizeof(float);

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
    float fcoeff_lo, fcoeff_hi, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff_lo = filter_lo[fi];
                fcoeff_hi = filter_hi[fi];

                /* Border handling by mirroring. */
                ii = 2 * i - 1 + fi;

                if (ii < 0)
                    ii = -ii;
                else if (ii >= h)
                    ii = 2 * h - ii - 1;

                imgcoeff = src[ii * src_px_stride + j];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            tmplo[j] = accum_lo;
            tmphi[j] = accum_hi;
        }

        /* Horizontal pass (lo). */
        for (j = 0; j < (w + 1) / 2; ++j) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                /* Border handling by mirroring. */
                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmplo[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_a[i * dst_px_stride + j] = accum_lo;
            dst->band_v[i * dst_px_stride + j] = accum_hi;
        }

        /* Horizontal pass (hi). */
        for (j = 0; j < (w + 1) / 2; ++j) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                /* Border handling by mirroring. */
                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmphi[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_h[i * dst_px_stride + j] = accum_lo;
            dst->band_d[i * dst_px_stride + j] = accum_hi;
        }
    }

    aligned_free(tmplo);
    aligned_free(tmphi);
}

void adm_dwt2_d(const double *src, const adm_dwt_band_t_d *dst, int w, int h, int src_stride, int dst_stride)
{
    const double *filter_lo = dwt2_db2_coeffs_lo_d;
    const double *filter_hi = dwt2_db2_coeffs_hi_d;
    int fwidth = sizeof(dwt2_db2_coeffs_lo_d) / sizeof(double);

    int src_px_stride = src_stride / sizeof(double);
    int dst_px_stride = dst_stride / sizeof(double);

    double *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(double) * w), MAX_ALIGN);
    double *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(double) * w), MAX_ALIGN);
    double fcoeff_lo, fcoeff_hi, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            double accum_lo = 0;
            double accum_hi = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff_lo = filter_lo[fi];
                fcoeff_hi = filter_hi[fi];

                /* Border handling by mirroring. */
                ii = 2 * i - 1 + fi;

                if (ii < 0)
                    ii = -ii;
                else if (ii >= h)
                    ii = 2 * h - ii - 1;

                imgcoeff = src[ii * src_px_stride + j];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            tmplo[j] = accum_lo;
            tmphi[j] = accum_hi;
        }

        /* Horizontal pass (lo). */
        for (j = 0; j < (w + 1) / 2; ++j) {
            double accum_lo = 0;
            double accum_hi = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                /* Border handling by mirroring. */
                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmplo[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_a[i * dst_px_stride + j] = accum_lo;
            dst->band_v[i * dst_px_stride + j] = accum_hi;
        }

        /* Horizontal pass (hi). */
        for (j = 0; j < (w + 1) / 2; ++j) {
            double accum_lo = 0;
            double accum_hi = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                /* Border handling by mirroring. */
                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmphi[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_h[i * dst_px_stride + j] = accum_lo;
            dst->band_d[i * dst_px_stride + j] = accum_hi;
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
