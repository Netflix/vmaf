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

#include "alignment.h"
#include "config.h"
#include "convolution.h"
#include "convolution_internal.h"
#include "cpu.h"

extern int vmaf_floorn(int, int);
extern int vmaf_ceiln(int, int);

void convolution_x_c_s(const float *filter, int filter_width, const float *src, float *dst, int width, int height, int src_stride, int dst_stride, int step)
{
	int radius = filter_width / 2;
	int borders_left = vmaf_ceiln(radius, step);
	int borders_right = vmaf_floorn(width - (filter_width - radius), step);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < borders_left; j += step) {
			dst[i * dst_stride + j / step] = convolution_edge_s(true, filter, filter_width, src, width, height, src_stride, i, j);
		}

		for (int j = borders_left; j < borders_right; j += step) {
			float accum = 0;
			for (int k = 0; k < filter_width; ++k) {
				accum += filter[k] * src[i * src_stride + j - radius + k];
			}
			dst[i * dst_stride + j / step] = accum;
		}

		for (int j = borders_right; j < width; j += step) {
			dst[i * dst_stride + j / step] = convolution_edge_s(true, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
}

void convolution_y_c_s(const float *filter, int filter_width, const float *src, float *dst, int width, int height, int src_stride, int dst_stride, int step)
{
	int radius = filter_width / 2;
	int borders_top = vmaf_ceiln(radius, step);
	int borders_bottom = vmaf_floorn(height - (filter_width - radius), step);

	for (int i = 0; i < borders_top; i += step) {
		for (int j = 0; j < width; ++j) {
			dst[(i / step) * dst_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
	for (int i = borders_top; i < borders_bottom; i += step) {
		for (int j = 0; j < width; ++j) {
			float accum = 0;
			for (int k = 0; k < filter_width; ++k) {
				accum += filter[k] * src[(i - radius + k) * src_stride + j];
			}
			dst[(i / step) * dst_stride + j] = accum;
		}
	}
	for (int i = borders_bottom; i < height; i += step) {
		for (int j = 0; j < width; ++j) {
			dst[(i / step) * dst_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
}

void convolution_f32_c_s(const float *filter, int filter_width, const float *src, float *dst, float *tmp, int width, int height, int src_stride, int dst_stride)
{
    /* if support avx */

#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        convolution_f32_avx_s(filter, filter_width, src, dst, tmp, width,
                              height, src_stride, dst_stride);
        return;
    }
#endif

    /* fall back */

	// convolve along y first then x
	convolution_y_c_s(filter, filter_width, src, tmp, width, height, src_stride, dst_stride, 1);
	convolution_x_c_s(filter, filter_width, tmp, dst, width, height, src_stride, dst_stride, 1);
}
