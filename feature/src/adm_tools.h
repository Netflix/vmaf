#pragma once

#ifndef ADM_TOOLS_H_
#define ADM_TOOLS_H_

typedef struct adm_dwt_band_t_s {
	float *band_a; /* Low-pass V + low-pass H. */
	float *band_v; /* Low-pass V + high-pass H. */
	float *band_h; /* High-pass V + low-pass H. */
	float *band_d; /* High-pass V + high-pass H. */
} adm_dwt_band_t_s;

typedef struct adm_dwt_band_t_d {
	double *band_a; /* Low-pass V + low-pass H. */
	double *band_v; /* Low-pass V + high-pass H. */
	double *band_h; /* High-pass V + low-pass H. */
	double *band_d; /* High-pass V + high-pass H. */
} adm_dwt_band_t_d;

float adm_sum_cube_s(const float *x, int w, int h, int stride);
double adm_sum_cube_d(const double *x, int w, int h, int stride);

void adm_decouple_s(const adm_dwt_band_t_s *ref, const adm_dwt_band_t_s *dis, const adm_dwt_band_t_s *r, const adm_dwt_band_t_s *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride);
void adm_decouple_d(const adm_dwt_band_t_d *ref, const adm_dwt_band_t_d *dis, const adm_dwt_band_t_d *r, const adm_dwt_band_t_d *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride);

void adm_csf_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride);
void adm_csf_d(const adm_dwt_band_t_d *src, const adm_dwt_band_t_d *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride);

void adm_cm_thresh_s(const adm_dwt_band_t_s *src, float *dst, int w, int h, int src_stride, int dst_stride);
void adm_cm_thresh_d(const adm_dwt_band_t_d *src, double *dst, int w, int h, int src_stride, int dst_stride);

void adm_cm_s(const adm_dwt_band_t_s *src, const adm_dwt_band_t_s *dst, const float *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride);
void adm_cm_d(const adm_dwt_band_t_d *src, const adm_dwt_band_t_d *dst, const double *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride);

void adm_dwt2_s(const float *src, const adm_dwt_band_t_s *dst, int w, int h, int src_stride, int dst_stride);
void adm_dwt2_d(const double *src, const adm_dwt_band_t_d *dst, int w, int h, int src_stride, int dst_stride);

void adm_buffer_copy(const void *src, void *dst, int linewidth, int h, int src_stride, int dst_stride);

#endif /* ADM_TOOLS_H_ */
