#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

/*
 * All functions listed here expect a SYMMETRICAL filter.
 * All array arguments must be 32-byte aligned.
 *
 * filter - convolution kernel
 * filter_width - convolution width (including symmetric side)
 * src - input image
 * dst - output image
 * tmp - temporary array (at least same size as src, even for dec2 versions)
 * width - width of image
 * height - height of image
 * src_stride - distance between lines in src image (pixels, not bytes)
 * dst_stride - distance between lines in dst image (pixels, not bytes)
 */
void convolution_f32_c_s(const float *filter, int filter_width, const float *src, float *dst, float *tmp, int width, int height, int src_stride, int dst_stride);
void convolution_f32_c_d(const double *filter, int filter_width, const double *src, double *dst, double *tmp, int width, int height, int src_stride, int dst_stride);

#endif // CONVOLUTION_H_
