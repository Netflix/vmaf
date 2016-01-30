#pragma once

#ifndef ANSNR_TOOLS_H_
#define ANSNR_TOOLS_H_

extern const float ansnr_filter1d_ref_s[3];
extern const double ansnr_filter1d_ref_d[3];

extern const float ansnr_filter1d_dis_s[5];
extern const double ansnr_filter1d_dis_d[5];

extern const int ansnr_filter1d_ref_width;
extern const int ansnr_filter1d_dis_width;

extern const float ansnr_filter2d_ref_s[3*3];
extern const double ansnr_filter2d_ref_d[3*3];

extern const float ansnr_filter2d_dis_s[5*5];
extern const double ansnr_filter2d_dis_d[5*5];

extern const int ansnr_filter2d_ref_width;
extern const int ansnr_filter2d_dis_width;

void ansnr_mse_s(const float *ref, const float *dis, float *sig, float *noise, int w, int h, int ref_stride, int dis_stride);
void ansnr_mse_d(const double *ref, const double *dis, double *sig, double *noise, int w, int h, int ref_stride, int dis_stride);

void ansnr_filter1d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth);
void ansnr_filter1d_d(const double *f, const double *src, double *dst, int w, int h, int src_stride, int dst_stride, int fwidth);

void ansnr_filter2d_s(const float *f, const float *src, float *dst, int w, int h, int src_stride, int dst_stride, int fwidth);
void ansnr_filter2d_d(const double *f, const double *src, double *dst, int w, int h, int src_stride, int dst_stride, int fwidth);

#endif /* ANSNR_TOOLS_H_ */
