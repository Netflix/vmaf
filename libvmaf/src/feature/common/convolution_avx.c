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

#include <immintrin.h>
#include "alignment.h"
#include "convolution.h"
#include "convolution_internal.h"

void convolution_f32_avx_s_1d_h_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_v_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);
void convolution_f32_avx_s_1d_v_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);
void convolution_f32_avx_s_1d_v_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);

void convolution_f32_avx_s_1d_h_sq_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_sq_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_sq_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_v_sq_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);
void convolution_f32_avx_s_1d_v_sq_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);
void convolution_f32_avx_s_1d_v_sq_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end);

void convolution_f32_avx_s_1d_h_xy_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_xy_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_h_xy_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end);
void convolution_f32_avx_s_1d_v_xy_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end);
void convolution_f32_avx_s_1d_v_xy_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end);
void convolution_f32_avx_s_1d_v_xy_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end);

// Filter a single scanline.
static void convolution_f32_avx_s_1d_h_scanline(int N, const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{

	if (N == 5)
	{
		convolution_f32_avx_s_1d_h_scanline_5(filter, filter_width, src, dst, j_end);
	}
	else if (N == 9)
	{
		convolution_f32_avx_s_1d_h_scanline_9(filter, filter_width, src, dst, j_end);
	}
	else if (N == 17)
	{
		convolution_f32_avx_s_1d_h_scanline_17(filter, filter_width, src, dst, j_end);
	}
	else {

		int radius = filter_width / 2;

		for (int x = 0; x < filter_width; x += 9) {
			__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

			f0 = _mm256_setzero_ps();
			f1 = _mm256_setzero_ps();
			f2 = _mm256_setzero_ps();
			f3 = _mm256_setzero_ps();
			f5 = _mm256_setzero_ps();
			f6 = _mm256_setzero_ps();
			f7 = _mm256_setzero_ps();
			f8 = _mm256_setzero_ps();

			switch (filter_width - x) {
			default:
				f8 = _mm256_broadcast_ss(filter + x + 8);
				// fall through
			case 8:
				f7 = _mm256_broadcast_ss(filter + x + 7);
				// fall through
			case 7:
				f6 = _mm256_broadcast_ss(filter + x + 6);
				// fall through
			case 6:
				f5 = _mm256_broadcast_ss(filter + x + 5);
				// fall through
			case 5:
				f4 = _mm256_broadcast_ss(filter + x + 4);
				// fall through
			case 4:
				f3 = _mm256_broadcast_ss(filter + x + 3);
				// fall through
			case 3:
				f2 = _mm256_broadcast_ss(filter + x + 2);
				// fall through
			case 2:
				f1 = _mm256_broadcast_ss(filter + x + 1);
				// fall through
			case 1:
				f0 = _mm256_broadcast_ss(filter + x + 0);
				// fall through
			}

			for (int j = 0; j < j_end; j += 8) {
				__m256 accum = _mm256_setzero_ps();
				__m256 sum0, sum1, sum2, sum3;
				__m256 g;

				sum0 = _mm256_setzero_ps();
				sum1 = _mm256_setzero_ps();
				sum2 = _mm256_setzero_ps();
				sum3 = _mm256_setzero_ps();

				switch (filter_width - x) {
				default:
					g = _mm256_loadu_ps(src + j + x + 8);
					sum0 = _mm256_mul_ps(f8, g);
					// fall through
				case 8:
					g = _mm256_loadu_ps(src + j + x + 7);
					sum3 = _mm256_mul_ps(f7, g);
					// fall through
				case 7:
					g = _mm256_loadu_ps(src + j + x + 6);
					sum2 = _mm256_mul_ps(f6, g);
					// fall through
				case 6:
					g = _mm256_loadu_ps(src + j + x + 5);
					sum1 = _mm256_mul_ps(f5, g);
					// fall through
				case 5:
					g = _mm256_loadu_ps(src + j + x + 4);
					g = _mm256_mul_ps(f4, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				case 4:
					g = _mm256_loadu_ps(src + j + x + 3);
					g = _mm256_mul_ps(f3, g);
					sum3 = _mm256_add_ps(sum3, g);
					// fall through
				case 3:
					g = _mm256_loadu_ps(src + j + x + 2);
					g = _mm256_mul_ps(f2, g);
					sum2 = _mm256_add_ps(sum2, g);
					// fall through
				case 2:
					g = _mm256_loadu_ps(src + j + x + 1);
					g = _mm256_mul_ps(f1, g);
					sum1 = _mm256_add_ps(sum1, g);
					// fall through
				case 1:
					g = _mm256_loadu_ps(src + j + x + 0);
					g = _mm256_mul_ps(f0, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				}

				sum0 = _mm256_add_ps(sum0, sum2);
				sum1 = _mm256_add_ps(sum1, sum3);

				sum0 = _mm256_add_ps(sum0, sum1);
				accum = _mm256_add_ps(accum, sum0);

				if (x)
					accum = _mm256_add_ps(accum, _mm256_loadu_ps(dst + j + radius));

				_mm256_storeu_ps(dst + j + radius, accum);
			}
		}

	}
}

void convolution_f32_avx_s_1d_h_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	// Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 5);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 6);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 7);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src + j + 8);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_store_ps(dst + j + 8, accum); // radius = 8
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		float *dst_ptr = dst + j + 8; // radius = 8

		g = _mm256_loadu_ps(src + j + 9);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 10);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 11);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 12);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 13);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 14);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 15);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 16);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst_ptr), sum0);
		_mm256_store_ps(dst_ptr, sum0);
	}
}

void convolution_f32_avx_s_1d_h_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 5);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 6);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 7);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src + j + 8);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 4, accum); // radius = 4
	}
}

void convolution_f32_avx_s_1d_h_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 2, accum); // radius = 2
	}
}

// Filter a single scanline.
static void convolution_f32_avx_s_1d_v_scanline(int N, const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{

	if (N == 5)
	{
		convolution_f32_avx_s_1d_v_scanline_5(filter, filter_width, src, dst, src_stride, j_end);
	}
	else if (N == 9)
	{
		convolution_f32_avx_s_1d_v_scanline_9(filter, filter_width, src, dst, src_stride, j_end);
	}
	else if (N == 17)
	{
		convolution_f32_avx_s_1d_v_scanline_17(filter, filter_width, src, dst, src_stride, j_end);
	}
	else {

		int radius = filter_width / 2;
		src -= radius * src_stride;

		for (int y = 0; y < filter_width; y += 9) {
			__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

			f0 = _mm256_setzero_ps();
			f1 = _mm256_setzero_ps();
			f2 = _mm256_setzero_ps();
			f3 = _mm256_setzero_ps();
			f5 = _mm256_setzero_ps();
			f6 = _mm256_setzero_ps();
			f7 = _mm256_setzero_ps();
			f8 = _mm256_setzero_ps();

			switch (filter_width - y) {
			default:
				f8 = _mm256_broadcast_ss(filter + y + 8);
				// fall through
			case 8:
				f7 = _mm256_broadcast_ss(filter + y + 7);
				// fall through
			case 7:
				f6 = _mm256_broadcast_ss(filter + y + 6);
				// fall through
			case 6:
				f5 = _mm256_broadcast_ss(filter + y + 5);
				// fall through
			case 5:
				f4 = _mm256_broadcast_ss(filter + y + 4);
				// fall through
			case 4:
				f3 = _mm256_broadcast_ss(filter + y + 3);
				// fall through
			case 3:
				f2 = _mm256_broadcast_ss(filter + y + 2);
				// fall through
			case 2:
				f1 = _mm256_broadcast_ss(filter + y + 1);
				// fall through
			case 1:
				f0 = _mm256_broadcast_ss(filter + y + 0);
				// fall through
			}

			for (int j = 0; j < j_end; j += 8) {
				__m256 accum = _mm256_setzero_ps();
				__m256 sum0, sum1, sum2, sum3;
				__m256 g;

				sum0 = _mm256_setzero_ps();
				sum1 = _mm256_setzero_ps();
				sum2 = _mm256_setzero_ps();
				sum3 = _mm256_setzero_ps();

				switch (filter_width - y) {
				default:
					g = _mm256_load_ps(src + (y + 8) * src_stride + j);
					sum0 = _mm256_mul_ps(f8, g);
					// fall through
				case 8:
					g = _mm256_load_ps(src + (y + 7) * src_stride + j);
					sum3 = _mm256_mul_ps(f7, g);
					// fall through
				case 7:
					g = _mm256_load_ps(src + (y + 6) * src_stride + j);
					sum2 = _mm256_mul_ps(f6, g);
					// fall through
				case 6:
					g = _mm256_load_ps(src + (y + 5) * src_stride + j);
					sum1 = _mm256_mul_ps(f5, g);
					// fall through
				case 5:
					g = _mm256_load_ps(src + (y + 4) * src_stride + j);
					g = _mm256_mul_ps(f4, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				case 4:
					g = _mm256_load_ps(src + (y + 3) * src_stride + j);
					g = _mm256_mul_ps(f3, g);
					sum3 = _mm256_add_ps(sum3, g);
					// fall through
				case 3:
					g = _mm256_load_ps(src + (y + 2) * src_stride + j);
					g = _mm256_mul_ps(f2, g);
					sum2 = _mm256_add_ps(sum2, g);
					// fall through
				case 2:
					g = _mm256_load_ps(src + (y + 1) * src_stride + j);
					g = _mm256_mul_ps(f1, g);
					sum1 = _mm256_add_ps(sum1, g);
					// fall through
				case 1:
					g = _mm256_load_ps(src + (y + 0) * src_stride + j);
					g = _mm256_mul_ps(f0, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				}

				sum0 = _mm256_add_ps(sum0, sum2);
				sum1 = _mm256_add_ps(sum1, sum3);

				sum0 = _mm256_add_ps(sum0, sum1);
				accum = _mm256_add_ps(accum, sum0);

				if (y)
					accum = _mm256_add_ps(accum, _mm256_load_ps(dst + j));

				_mm256_store_ps(dst + j, accum);
			}
		}
	}
}

void convolution_f32_avx_s_1d_v_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src -= 8 * src_stride; // radius = 8

	// Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 5 * src_stride + j);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 6 * src_stride + j);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 7 * src_stride + j);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src + 8 * src_stride + j);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 9 * src_stride + j);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 10 * src_stride + j);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 11 * src_stride + j);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 12 * src_stride + j);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 13 * src_stride + j);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 14 * src_stride + j);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 15 * src_stride + j);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 16 * src_stride + j);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst + j), sum0);
		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src -= 4 * src_stride; // radius = 4

	// Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 5 * src_stride + j);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 6 * src_stride + j);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 7 * src_stride + j);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src + 8 * src_stride + j);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;
	src -= 2 * src_stride; // radius = 2

	// Evaluate filter taps 0-5
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d(
	int N,
	const float * RESTRICT filter,
	int filter_width,
	const float * RESTRICT src,
	float * RESTRICT dst,
	float * RESTRICT tmp,
	int width,
	int height,
	int src_stride,
	int dst_stride)
{
	int radius = filter_width / 2;
	int width_mod8 = vmaf_floorn(width, 8);
	int tmp_stride = vmaf_ceiln(width, 8);

	int i_vec_end = height - radius;
	int j_vec_end = width_mod8 - vmaf_ceiln(radius + 1, 8);

	// Vertical pass.
	for (int i = 0; i < radius; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
	for (int i = radius; i < i_vec_end; ++i) {
		convolution_f32_avx_s_1d_v_scanline(N, filter, filter_width, src + i * src_stride, tmp + i * tmp_stride, src_stride, width_mod8);

		for (int j = width_mod8; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
	for (int i = i_vec_end; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}

	// Horizontal pass.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < radius; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}

		convolution_f32_avx_s_1d_h_scanline(N, filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

		for (int j = j_vec_end + radius; j < width; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}
	}
}

void convolution_f32_avx_s(const float *filter, int filter_width, const float *src, float *dst, float *tmp, int width, int height, int src_stride, int dst_stride)
{
	switch (filter_width) {
	case 17:
		convolution_f32_avx_s_1d(17, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 9:
		convolution_f32_avx_s_1d(9, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 5:
		convolution_f32_avx_s_1d(5, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 3:
		convolution_f32_avx_s_1d(3, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	default:
		convolution_f32_avx_s_1d(0, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	}
}

void convolution_f32_avx_s_1d_h_sq_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	// Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 5);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 6);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 7);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src + j + 8);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_store_ps(dst + j + 8, accum); // radius = 8
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		float *dst_ptr = dst + j + 8; // radius = 8

		g = _mm256_loadu_ps(src + j + 9);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 10);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 11);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 12);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 13);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 14);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 15);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 16);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst_ptr), sum0);
		_mm256_store_ps(dst_ptr, sum0);
	}
}

void convolution_f32_avx_s_1d_h_sq_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src + j + 5);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src + j + 6);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src + j + 7);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src + j + 8);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 4, accum); // radius = 4
	}
}

void convolution_f32_avx_s_1d_h_sq_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_loadu_ps(src + j + 0);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src + j + 1);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src + j + 2);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src + j + 3);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src + j + 4);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 2, accum); // radius = 2
	}
}

// Filter a single scanline.
static void convolution_f32_avx_s_1d_v_sq_scanline(int N, const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{

	if (N == 5)
	{
		convolution_f32_avx_s_1d_v_sq_scanline_5(filter, filter_width, src, dst, src_stride, j_end);
	}
	else if (N == 9)
	{
		convolution_f32_avx_s_1d_v_sq_scanline_9(filter, filter_width, src, dst, src_stride, j_end);
	}
	else if (N == 17)
	{
		convolution_f32_avx_s_1d_v_sq_scanline_17(filter, filter_width, src, dst, src_stride, j_end);
	}
	else {

		int radius = filter_width / 2;
		src -= radius * src_stride;

		for (int y = 0; y < filter_width; y += 9) {
			__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

			f0 = _mm256_setzero_ps();
			f1 = _mm256_setzero_ps();
			f2 = _mm256_setzero_ps();
			f3 = _mm256_setzero_ps();
			f5 = _mm256_setzero_ps();
			f6 = _mm256_setzero_ps();
			f7 = _mm256_setzero_ps();
			f8 = _mm256_setzero_ps();

			switch (filter_width - y) {
			default:
				f8 = _mm256_broadcast_ss(filter + y + 8);
				// fall through
			case 8:
				f7 = _mm256_broadcast_ss(filter + y + 7);
				// fall through
			case 7:
				f6 = _mm256_broadcast_ss(filter + y + 6);
				// fall through
			case 6:
				f5 = _mm256_broadcast_ss(filter + y + 5);
				// fall through
			case 5:
				f4 = _mm256_broadcast_ss(filter + y + 4);
				// fall through
			case 4:
				f3 = _mm256_broadcast_ss(filter + y + 3);
				// fall through
			case 3:
				f2 = _mm256_broadcast_ss(filter + y + 2);
				// fall through
			case 2:
				f1 = _mm256_broadcast_ss(filter + y + 1);
				// fall through
			case 1:
				f0 = _mm256_broadcast_ss(filter + y + 0);
				// fall through
			}

			for (int j = 0; j < j_end; j += 8) {
				__m256 accum = _mm256_setzero_ps();
				__m256 sum0, sum1, sum2, sum3;
				__m256 g;

				sum0 = _mm256_setzero_ps();
				sum1 = _mm256_setzero_ps();
				sum2 = _mm256_setzero_ps();
				sum3 = _mm256_setzero_ps();

				switch (filter_width - y) {
				default:
					g = _mm256_load_ps(src + (y + 8) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					sum0 = _mm256_mul_ps(f8, g);
					// fall through
				case 8:
					g = _mm256_load_ps(src + (y + 7) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					sum3 = _mm256_mul_ps(f7, g);
					// fall through
				case 7:
					g = _mm256_load_ps(src + (y + 6) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					sum2 = _mm256_mul_ps(f6, g);
					// fall through
				case 6:
					g = _mm256_load_ps(src + (y + 5) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					sum1 = _mm256_mul_ps(f5, g);
					// fall through
				case 5:
					g = _mm256_load_ps(src + (y + 4) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					g = _mm256_mul_ps(f4, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				case 4:
					g = _mm256_load_ps(src + (y + 3) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					g = _mm256_mul_ps(f3, g);
					sum3 = _mm256_add_ps(sum3, g);
					// fall through
				case 3:
					g = _mm256_load_ps(src + (y + 2) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					g = _mm256_mul_ps(f2, g);
					sum2 = _mm256_add_ps(sum2, g);
					// fall through
				case 2:
					g = _mm256_load_ps(src + (y + 1) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					g = _mm256_mul_ps(f1, g);
					sum1 = _mm256_add_ps(sum1, g);
					// fall through
				case 1:
					g = _mm256_load_ps(src + (y + 0) * src_stride + j);
					g = _mm256_mul_ps(g, g);
					g = _mm256_mul_ps(f0, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				}

				sum0 = _mm256_add_ps(sum0, sum2);
				sum1 = _mm256_add_ps(sum1, sum3);

				sum0 = _mm256_add_ps(sum0, sum1);
				accum = _mm256_add_ps(accum, sum0);

				if (y)
					accum = _mm256_add_ps(accum, _mm256_load_ps(dst + j));

				_mm256_store_ps(dst + j, accum);
			}
		}
	}
}

void convolution_f32_avx_s_1d_v_sq_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src -= 8 * src_stride; // radius = 8

						   // Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 5 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 6 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 7 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src + 8 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 9 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 10 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 11 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 12 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 13 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 14 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 15 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 16 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst + j), sum0);
		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_sq_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src -= 4 * src_stride; // radius = 4

						   // Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src + 5 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src + 6 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src + 7 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src + 8 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_sq_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;
	src -= 2 * src_stride; // radius = 2

						   // Evaluate filter taps 0-5
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g;

		g = _mm256_load_ps(src + 0 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src + 1 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src + 2 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src + 3 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src + 4 * src_stride + j);
		g = _mm256_mul_ps(g, g);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_sq(
	int N,
	const float * RESTRICT filter,
	int filter_width,
	const float * RESTRICT src,
	float * RESTRICT dst,
	float * RESTRICT tmp,
	int width,
	int height,
	int src_stride,
	int dst_stride)
{
	int radius = filter_width / 2;
	int width_mod8 = vmaf_floorn(width, 8);
	int tmp_stride = vmaf_ceiln(width, 8);

	int i_vec_end = height - radius;
	int j_vec_end = width_mod8 - vmaf_ceiln(radius + 1, 8);

	// Vertical pass.
	for (int i = 0; i < radius; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_sq_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
	for (int i = radius; i < i_vec_end; ++i) {
		convolution_f32_avx_s_1d_v_sq_scanline(N, filter, filter_width, src + i * src_stride, tmp + i * tmp_stride, src_stride, width_mod8);

		for (int j = width_mod8; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_sq_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}
	for (int i = i_vec_end; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_sq_s(false, filter, filter_width, src, width, height, src_stride, i, j);
		}
	}

	// Horizontal pass.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < radius; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}

		convolution_f32_avx_s_1d_h_scanline(N, filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

		for (int j = j_vec_end + radius; j < width; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}
	}
}

void convolution_f32_avx_sq_s(const float *filter, int filter_width, const float *src, float *dst, float *tmp, int width, int height, int src_stride, int dst_stride)
{
	switch (filter_width) {
	case 17:
		convolution_f32_avx_s_1d_sq(17, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 9:
		convolution_f32_avx_s_1d_sq(9, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 5:
		convolution_f32_avx_s_1d_sq(5, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	case 3:
		convolution_f32_avx_s_1d_sq(3, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	default:
		convolution_f32_avx_s_1d_sq(0, filter, filter_width, src, dst, tmp, width, height, src_stride, dst_stride);
		break;
	}
}

void convolution_f32_avx_s_1d_h_xy_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	// Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_loadu_ps(src1 + j + 0);
		g2 = _mm256_loadu_ps(src2 + j + 0);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src1 + j + 1);
		g2 = _mm256_loadu_ps(src2 + j + 1);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src1 + j + 2);
		g2 = _mm256_loadu_ps(src2 + j + 2);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src1 + j + 3);
		g2 = _mm256_loadu_ps(src2 + j + 3);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src1 + j + 4);
		g2 = _mm256_loadu_ps(src2 + j + 4);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src1 + j + 5);
		g2 = _mm256_loadu_ps(src2 + j + 5);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src1 + j + 6);
		g2 = _mm256_loadu_ps(src2 + j + 6);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src1 + j + 7);
		g2 = _mm256_loadu_ps(src2 + j + 7);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src1 + j + 8);
		g2 = _mm256_loadu_ps(src2 + j + 8);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_store_ps(dst + j + 8, accum); // radius = 8
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		float *dst_ptr = dst + j + 8; // radius = 8

		g = _mm256_loadu_ps(src1 + j + 9);
		g2 = _mm256_loadu_ps(src2 + j + 9);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src1 + j + 10);
		g2 = _mm256_loadu_ps(src2 + j + 10);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src1 + j + 11);
		g2 = _mm256_loadu_ps(src2 + j + 11);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src1 + j + 12);
		g2 = _mm256_loadu_ps(src2 + j + 12);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src1 + j + 13);
		g2 = _mm256_loadu_ps(src2 + j + 13);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src1 + j + 14);
		g2 = _mm256_loadu_ps(src2 + j + 14);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src1 + j + 15);
		g2 = _mm256_loadu_ps(src2 + j + 15);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src1 + j + 16);
		g2 = _mm256_loadu_ps(src2 + j + 16);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst_ptr), sum0);
		_mm256_store_ps(dst_ptr, sum0);
	}
}

void convolution_f32_avx_s_1d_h_xy_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_loadu_ps(src1 + j + 0);
		g2 = _mm256_loadu_ps(src2 + j + 0);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src1 + j + 1);
		g2 = _mm256_loadu_ps(src2 + j + 1);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src1 + j + 2);
		g2 = _mm256_loadu_ps(src2 + j + 2);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src1 + j + 3);
		g2 = _mm256_loadu_ps(src2 + j + 3);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src1 + j + 4);
		g2 = _mm256_loadu_ps(src2 + j + 4);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_loadu_ps(src1 + j + 5);
		g2 = _mm256_loadu_ps(src2 + j + 5);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_loadu_ps(src1 + j + 6);
		g2 = _mm256_loadu_ps(src2 + j + 6);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_loadu_ps(src1 + j + 7);
		g2 = _mm256_loadu_ps(src2 + j + 7);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_loadu_ps(src1 + j + 8);
		g2 = _mm256_loadu_ps(src2 + j + 8);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 4, accum); // radius = 4
	}
}

void convolution_f32_avx_s_1d_h_xy_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;

	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 accum = _mm256_setzero_ps();
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_loadu_ps(src1 + j + 0);
		g2 = _mm256_loadu_ps(src2 + j + 0);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_loadu_ps(src1 + j + 1);
		g2 = _mm256_loadu_ps(src2 + j + 1);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_loadu_ps(src1 + j + 2);
		g2 = _mm256_loadu_ps(src2 + j + 2);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_loadu_ps(src1 + j + 3);
		g2 = _mm256_loadu_ps(src2 + j + 3);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_loadu_ps(src1 + j + 4);
		g2 = _mm256_loadu_ps(src2 + j + 4);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);
		accum = _mm256_add_ps(accum, sum0);

		_mm256_storeu_ps(dst + j + 2, accum); // radius = 2
	}
}

// Filter a single scanline.
static void convolution_f32_avx_s_1d_v_xy_scanline(int N, const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end)
{

	if (N == 5)
	{
		convolution_f32_avx_s_1d_v_xy_scanline_5(filter, filter_width, src1, src2, dst, src1_stride, src2_stride, j_end);
	}
	else if (N == 9)
	{
		convolution_f32_avx_s_1d_v_xy_scanline_9(filter, filter_width, src1, src2, dst, src1_stride, src2_stride, j_end);
	}
	else if (N == 17)
	{
		convolution_f32_avx_s_1d_v_xy_scanline_17(filter, filter_width, src1, src2, dst, src1_stride, src2_stride, j_end);
	}
	else {

		int radius = filter_width / 2;
		src1 -= radius * src1_stride;
		src2 -= radius * src2_stride;

		for (int y = 0; y < filter_width; y += 9) {
			__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;

			f0 = _mm256_setzero_ps();
			f1 = _mm256_setzero_ps();
			f2 = _mm256_setzero_ps();
			f3 = _mm256_setzero_ps();
			f5 = _mm256_setzero_ps();
			f6 = _mm256_setzero_ps();
			f7 = _mm256_setzero_ps();
			f8 = _mm256_setzero_ps();

			switch (filter_width - y) {
			default:
				f8 = _mm256_broadcast_ss(filter + y + 8);
				// fall through
			case 8:
				f7 = _mm256_broadcast_ss(filter + y + 7);
				// fall through
			case 7:
				f6 = _mm256_broadcast_ss(filter + y + 6);
				// fall through
			case 6:
				f5 = _mm256_broadcast_ss(filter + y + 5);
				// fall through
			case 5:
				f4 = _mm256_broadcast_ss(filter + y + 4);
				// fall through
			case 4:
				f3 = _mm256_broadcast_ss(filter + y + 3);
				// fall through
			case 3:
				f2 = _mm256_broadcast_ss(filter + y + 2);
				// fall through
			case 2:
				f1 = _mm256_broadcast_ss(filter + y + 1);
				// fall through
			case 1:
				f0 = _mm256_broadcast_ss(filter + y + 0);
				// fall through
			}

			for (int j = 0; j < j_end; j += 8) {
				__m256 accum = _mm256_setzero_ps();
				__m256 sum0, sum1, sum2, sum3;
				__m256 g, g2;

				sum0 = _mm256_setzero_ps();
				sum1 = _mm256_setzero_ps();
				sum2 = _mm256_setzero_ps();
				sum3 = _mm256_setzero_ps();

				switch (filter_width - y) {
				default:
					g = _mm256_load_ps(src1 + (y + 8) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 8) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					sum0 = _mm256_mul_ps(f8, g);
					// fall through
				case 8:
					g = _mm256_load_ps(src1 + (y + 7) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 7) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					sum3 = _mm256_mul_ps(f7, g);
					// fall through
				case 7:
					g = _mm256_load_ps(src1 + (y + 6) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 6) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					sum2 = _mm256_mul_ps(f6, g);
					// fall through
				case 6:
					g = _mm256_load_ps(src1 + (y + 5) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 5) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					sum1 = _mm256_mul_ps(f5, g);
					// fall through
				case 5:
					g = _mm256_load_ps(src1 + (y + 4) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 4) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					g = _mm256_mul_ps(f4, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				case 4:
					g = _mm256_load_ps(src1 + (y + 3) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 3) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					g = _mm256_mul_ps(f3, g);
					sum3 = _mm256_add_ps(sum3, g);
					// fall through
				case 3:
					g = _mm256_load_ps(src1 + (y + 2) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 2) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					g = _mm256_mul_ps(f2, g);
					sum2 = _mm256_add_ps(sum2, g);
					// fall through
				case 2:
					g = _mm256_load_ps(src1 + (y + 1) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 1) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					g = _mm256_mul_ps(f1, g);
					sum1 = _mm256_add_ps(sum1, g);
					// fall through
				case 1:
					g = _mm256_load_ps(src1 + (y + 0) * src1_stride + j);
					g2 = _mm256_load_ps(src2 + (y + 0) * src2_stride + j);
					g = _mm256_mul_ps(g, g2);
					g = _mm256_mul_ps(f0, g);
					sum0 = _mm256_add_ps(sum0, g);
					// fall through
				}

				sum0 = _mm256_add_ps(sum0, sum2);
				sum1 = _mm256_add_ps(sum1, sum3);

				sum0 = _mm256_add_ps(sum0, sum1);
				accum = _mm256_add_ps(accum, sum0);

				if (y)
					accum = _mm256_add_ps(accum, _mm256_load_ps(dst + j));

				_mm256_store_ps(dst + j, accum);
			}
		}
	}
}

void convolution_f32_avx_s_1d_v_xy_scanline_17(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src1 -= 8 * src1_stride; // radius = 8
	src2 -= 8 * src2_stride; // radius = 8

						   // Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_load_ps(src1 + 0 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 0 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src1 + 1 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 1 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src1 + 2 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 2 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src1 + 3 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 3 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src1 + 4 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 4 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src1 + 5 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 5 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src1 + 6 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 6 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src1 + 7 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 7 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src1 + 8 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 8 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}

	// Evaluate filter taps 9-16
	f0 = _mm256_broadcast_ss(filter + 9);
	f1 = _mm256_broadcast_ss(filter + 10);
	f2 = _mm256_broadcast_ss(filter + 11);
	f3 = _mm256_broadcast_ss(filter + 12);
	f4 = _mm256_broadcast_ss(filter + 13);
	f5 = _mm256_broadcast_ss(filter + 14);
	f6 = _mm256_broadcast_ss(filter + 15);
	f7 = _mm256_broadcast_ss(filter + 16);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_load_ps(src1 + 9 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 9 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src1 + 10 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 10 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src1 + 11 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 11 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src1 + 12 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 12 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src1 + 13 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 13 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src1 + 14 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 14 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src1 + 15 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 15 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src1 + 16 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 16 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		sum0 = _mm256_add_ps(_mm256_load_ps(dst + j), sum0);
		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_xy_scanline_9(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4, f5, f6, f7, f8;
	src1 -= 4 * src1_stride; // radius = 4
	src2 -= 4 * src2_stride; // radius = 4

						   // Evaluate filter taps 0-8
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);
	f5 = _mm256_broadcast_ss(filter + 5);
	f6 = _mm256_broadcast_ss(filter + 6);
	f7 = _mm256_broadcast_ss(filter + 7);
	f8 = _mm256_broadcast_ss(filter + 8);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_load_ps(src1 + 0 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 0 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src1 + 1 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 1 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src1 + 2 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 2 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src1 + 3 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 3 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src1 + 4 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 4 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		g = _mm256_load_ps(src1 + 5 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 5 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f5, g);
		sum1 = _mm256_add_ps(sum1, g);

		g = _mm256_load_ps(src1 + 6 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 6 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f6, g);
		sum2 = _mm256_add_ps(sum2, g);

		g = _mm256_load_ps(src1 + 7 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 7 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f7, g);
		sum3 = _mm256_add_ps(sum3, g);

		g = _mm256_load_ps(src1 + 8 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 8 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f8, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_v_xy_scanline_5(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end)
{
    (void) filter_width;

	__m256 f0, f1, f2, f3, f4;
	src1 -= 2 * src1_stride; // radius = 2
	src2 -= 2 * src2_stride; // radius = 2

						   // Evaluate filter taps 0-5
	f0 = _mm256_broadcast_ss(filter + 0);
	f1 = _mm256_broadcast_ss(filter + 1);
	f2 = _mm256_broadcast_ss(filter + 2);
	f3 = _mm256_broadcast_ss(filter + 3);
	f4 = _mm256_broadcast_ss(filter + 4);

	for (int j = 0; j < j_end; j += 8) {
		__m256 sum0, sum1, sum2, sum3;
		__m256 g, g2;

		g = _mm256_load_ps(src1 + 0 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 0 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f0, g);
		sum0 = g;

		g = _mm256_load_ps(src1 + 1 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 1 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f1, g);
		sum1 = g;

		g = _mm256_load_ps(src1 + 2 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 2 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f2, g);
		sum2 = g;

		g = _mm256_load_ps(src1 + 3 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 3 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f3, g);
		sum3 = g;

		g = _mm256_load_ps(src1 + 4 * src1_stride + j);
		g2 = _mm256_load_ps(src2 + 4 * src2_stride + j);
		g = _mm256_mul_ps(g, g2);
		g = _mm256_mul_ps(f4, g);
		sum0 = _mm256_add_ps(sum0, g);

		sum0 = _mm256_add_ps(sum0, sum2);
		sum1 = _mm256_add_ps(sum1, sum3);

		sum0 = _mm256_add_ps(sum0, sum1);

		_mm256_store_ps(dst + j, sum0);
	}
}

void convolution_f32_avx_s_1d_xy(
	int N,
	const float * RESTRICT filter,
	int filter_width,
	const float * RESTRICT src1,
	const float * RESTRICT src2,
	float * RESTRICT dst,
	float * RESTRICT tmp,
	int width,
	int height,
	int src1_stride,
	int src2_stride,
	int dst_stride)
{
	int radius = filter_width / 2;
	int width_mod8 = vmaf_floorn(width, 8);
	int tmp_stride = vmaf_ceiln(width, 8);

	int i_vec_end = height - radius;
	int j_vec_end = width_mod8 - vmaf_ceiln(radius + 1, 8);

	// Vertical pass.
	for (int i = 0; i < radius; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_xy_s(false, filter, filter_width, src1, src2, width, height, src1_stride, src2_stride, i, j);
		}
	}
	for (int i = radius; i < i_vec_end; ++i) {
		convolution_f32_avx_s_1d_v_xy_scanline(N, filter, filter_width, src1 + i * src1_stride, src2 + i * src2_stride, tmp + i * tmp_stride, src1_stride, src2_stride, width_mod8);

		for (int j = width_mod8; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_xy_s(false, filter, filter_width, src1, src2, width, height, src1_stride, src2_stride, i, j);
		}
	}
	for (int i = i_vec_end; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			tmp[i * tmp_stride + j] = convolution_edge_xy_s(false, filter, filter_width, src1, src2, width, height, src1_stride, src2_stride, i, j);
		}
	}

	// Horizontal pass.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < radius; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}

		convolution_f32_avx_s_1d_h_scanline(N, filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

		for (int j = j_vec_end + radius; j < width; ++j) {
			dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
		}
	}
}

void convolution_f32_avx_xy_s(const float *filter, int filter_width, const float *src1, const float *src2, float *dst, float *tmp, int width, int height, int src1_stride, int src2_stride, int dst_stride)
{
	switch (filter_width) {
	case 17:
		convolution_f32_avx_s_1d_xy(17, filter, filter_width, src1, src2, dst, tmp, width, height, src1_stride, src2_stride, dst_stride);
		break;
	case 9:
		convolution_f32_avx_s_1d_xy(9, filter, filter_width, src1, src2, dst, tmp, width, height, src1_stride, src2_stride, dst_stride);
		break;
	case 5:
		convolution_f32_avx_s_1d_xy(5, filter, filter_width, src1, src2, dst, tmp, width, height, src1_stride, src2_stride, dst_stride);
		break;
	case 3:
		convolution_f32_avx_s_1d_xy(3, filter, filter_width, src1, src2, dst, tmp, width, height, src1_stride, src2_stride, dst_stride);
		break;
	default:
		convolution_f32_avx_s_1d_xy(0, filter, filter_width, src1, src2, dst, tmp, width, height, src1_stride, src2_stride, dst_stride);
		break;
	}
}
