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

#define AVX_STEP (8)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))


void convolution_f32_avx_s_1d_h_scanline(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int j_end) {
    int radius = filter_width / 2;

    __m256 f[MAX_FWIDTH_AVX_CONV];

    for (int k = 0; k < filter_width; k++) {
        f[k] = _mm256_broadcast_ss(filter + k);
    }

    for (int j = 0; j < j_end; j += AVX_STEP) {
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < filter_width; k++) {
            __m256 g = _mm256_loadu_ps(src + j + k);
            g = _mm256_mul_ps(f[k], g);
            sum = _mm256_add_ps(sum, g);
        }

        _mm256_storeu_ps(dst + j + radius, sum);
    }
}

void convolution_f32_avx_s_1d_v_scanline(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end) {
    int radius = filter_width / 2;

    src -= radius * src_stride;

    __m256 f[MAX_FWIDTH_AVX_CONV];

    for (int k = 0; k < filter_width; k++) {
        f[k] = _mm256_broadcast_ss(filter + k);
    }

    for (int j = 0; j < j_end; j += AVX_STEP) {
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < filter_width; k++) {
            __m256 g = _mm256_load_ps(src + k * src_stride + j);
            g = _mm256_mul_ps(f[k], g);
            sum = _mm256_add_ps(sum, g);
        }

        _mm256_store_ps(dst + j, sum);
    }
}

void convolution_f32_avx_s_1d_v_sq_scanline(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, int src_stride, int j_end) {
    int radius = filter_width / 2;

    src -= radius * src_stride;

    __m256 f[MAX_FWIDTH_AVX_CONV];

    for (int k = 0; k < filter_width; k++) {
        f[k] = _mm256_broadcast_ss(filter + k);
    }

    for (int j = 0; j < j_end; j += AVX_STEP) {
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < filter_width; k++) {
            __m256 g = _mm256_load_ps(src + k * src_stride + j);
            g = _mm256_mul_ps(g, g);
            g = _mm256_mul_ps(f[k], g);
            sum = _mm256_add_ps(sum, g);
        }

        _mm256_store_ps(dst + j, sum);
    }
}

void convolution_f32_avx_s_1d_v_xy_scanline(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, int src1_stride, int src2_stride, int j_end)
{
    int radius = filter_width / 2;

    src1 -= radius * src1_stride;
    src2 -= radius * src2_stride;

    __m256 f[MAX_FWIDTH_AVX_CONV];

    for (int k = 0; k < filter_width; k++) {
        f[k] = _mm256_broadcast_ss(filter + k);
    }

    for (int j = 0; j < j_end; j += AVX_STEP) {
        __m256 sum = _mm256_setzero_ps();

        for (int k = 0; k < filter_width; k++) {
            __m256 g = _mm256_load_ps(src1 + k * src1_stride + j);
            __m256 g2 = _mm256_load_ps(src2 + k * src2_stride + j);
            g = _mm256_mul_ps(g, g2);
            g = _mm256_mul_ps(f[k], g);
            sum = _mm256_add_ps(sum, g);
        }

        _mm256_store_ps(dst + j, sum);
    }
}

void convolution_f32_avx_s(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int width, int height, int src_stride, int dst_stride) {    
    int radius = filter_width / 2;
    int width_floor_step = vmaf_floorn(width, AVX_STEP);
    int tmp_stride = vmaf_ceiln(width, AVX_STEP);

    int i_vec_end = height - radius;
    int j_vec_end = vmaf_floorn(width - radius, AVX_STEP);

    // Clamp edge-loop bounds so tiny width/height (< radius) cannot write outside
    // the tmp/dst buffers. i_vec_end is safe to raise in place: its two uses are the
    // vector loop's end bound and the trailing loop's start, and the raise only fires
    // when the raised value is <= radius, which leaves the vector loop
    // [radius, i_vec_end) empty. j_vec_end must NOT be raised
    // in place: it is also passed unmodified to convolution_f32_avx_s_1d_h_scanline,
    // whose j_end argument must stay <= vmaf_floorn(width - radius, AVX_STEP) or the
    // scanline itself writes out of bounds (it stores dst[j + radius .. j + radius +
    // AVX_STEP) for each j < j_end). So the horizontal trailing loop's clamped start
    // goes in a SEPARATE variable, j_trailing_start.
    int i_edge_end = MIN(radius, height);
    i_vec_end = MAX(i_vec_end, i_edge_end);
    int j_edge_end = MIN(radius, width);
    int j_trailing_start = MAX(j_vec_end, j_edge_end);

    // Vertical pass.
    for (int i = 0; i < i_edge_end; ++i) {
        for (int j = 0; j < width; ++j) {
            tmp[i * tmp_stride + j] = convolution_edge_s(false, filter, filter_width, src, width, height, src_stride, i, j);
        }
    }
    for (int i = radius; i < i_vec_end; ++i) {
        convolution_f32_avx_s_1d_v_scanline(filter, filter_width, src + i * src_stride, tmp + i * tmp_stride, src_stride, width_floor_step);

        for (int j = width_floor_step; j < width; ++j) {
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
        for (int j = 0; j < j_edge_end; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }

        convolution_f32_avx_s_1d_h_scanline(filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

        for (int j = j_trailing_start; j < width; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }
    }
}

void convolution_f32_avx_sq_s(const float * RESTRICT filter, int filter_width, const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int width, int height, int src_stride, int dst_stride) {
    int radius = filter_width / 2;
    int width_floor_step = vmaf_floorn(width, AVX_STEP);
    int tmp_stride = vmaf_ceiln(width, AVX_STEP);

    int i_vec_end = height - radius;
    int j_vec_end = vmaf_floorn(width - radius, AVX_STEP);

    // Clamp edge-loop bounds so tiny width/height (< radius) cannot write outside
    // the tmp/dst buffers. i_vec_end is safe to raise in place: its two uses are the
    // vector loop's end bound and the trailing loop's start, and the raise only fires
    // when the raised value is <= radius, which leaves the vector loop
    // [radius, i_vec_end) empty. j_vec_end must NOT be raised
    // in place: it is also passed unmodified to convolution_f32_avx_s_1d_h_scanline,
    // whose j_end argument must stay <= vmaf_floorn(width - radius, AVX_STEP) or the
    // scanline itself writes out of bounds (it stores dst[j + radius .. j + radius +
    // AVX_STEP) for each j < j_end). So the horizontal trailing loop's clamped start
    // goes in a SEPARATE variable, j_trailing_start.
    int i_edge_end = MIN(radius, height);
    i_vec_end = MAX(i_vec_end, i_edge_end);
    int j_edge_end = MIN(radius, width);
    int j_trailing_start = MAX(j_vec_end, j_edge_end);

    // Vertical pass.
    for (int i = 0; i < i_edge_end; ++i) {
        for (int j = 0; j < width; ++j) {
            tmp[i * tmp_stride + j] = convolution_edge_sq_s(false, filter, filter_width, src, width, height, src_stride, i, j);
        }
    }
    for (int i = radius; i < i_vec_end; ++i) {
        convolution_f32_avx_s_1d_v_sq_scanline(filter, filter_width, src + i * src_stride, tmp + i * tmp_stride, src_stride, width_floor_step);

        for (int j = width_floor_step; j < width; ++j) {
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
        for (int j = 0; j < j_edge_end; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }

        convolution_f32_avx_s_1d_h_scanline(filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

        for (int j = j_trailing_start; j < width; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }
    }
}

void convolution_f32_avx_xy_s(const float * RESTRICT filter, int filter_width, const float * RESTRICT src1, const float * RESTRICT src2, float * RESTRICT dst, float * RESTRICT tmp, int width, int height, int src1_stride, int src2_stride, int dst_stride)
{
    int radius = filter_width / 2;
    int width_floor_step = vmaf_floorn(width, AVX_STEP);
    int tmp_stride = vmaf_ceiln(width, AVX_STEP);

    int i_vec_end = height - radius;
    int j_vec_end = vmaf_floorn(width - radius, AVX_STEP);

    // Clamp edge-loop bounds so tiny width/height (< radius) cannot write outside
    // the tmp/dst buffers. i_vec_end is safe to raise in place: its two uses are the
    // vector loop's end bound and the trailing loop's start, and the raise only fires
    // when the raised value is <= radius, which leaves the vector loop
    // [radius, i_vec_end) empty. j_vec_end must NOT be raised
    // in place: it is also passed unmodified to convolution_f32_avx_s_1d_h_scanline,
    // whose j_end argument must stay <= vmaf_floorn(width - radius, AVX_STEP) or the
    // scanline itself writes out of bounds (it stores dst[j + radius .. j + radius +
    // AVX_STEP) for each j < j_end). So the horizontal trailing loop's clamped start
    // goes in a SEPARATE variable, j_trailing_start.
    int i_edge_end = MIN(radius, height);
    i_vec_end = MAX(i_vec_end, i_edge_end);
    int j_edge_end = MIN(radius, width);
    int j_trailing_start = MAX(j_vec_end, j_edge_end);

    // Vertical pass.
    for (int i = 0; i < i_edge_end; ++i) {
        for (int j = 0; j < width; ++j) {
            tmp[i * tmp_stride + j] = convolution_edge_xy_s(false, filter, filter_width, src1, src2, width, height, src1_stride, src2_stride, i, j);
        }
    }
    for (int i = radius; i < i_vec_end; ++i) {
        convolution_f32_avx_s_1d_v_xy_scanline(filter, filter_width, src1 + i * src1_stride, src2 + i * src2_stride, tmp + i * tmp_stride, src1_stride, src2_stride, width_floor_step);

        for (int j = width_floor_step; j < width; ++j) {
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
        for (int j = 0; j < j_edge_end; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }

        convolution_f32_avx_s_1d_h_scanline(filter, filter_width, tmp + i * tmp_stride, dst + i * dst_stride, j_vec_end);

        for (int j = j_trailing_start; j < width; ++j) {
            dst[i * dst_stride + j] = convolution_edge_s(true, filter, filter_width, tmp, width, height, tmp_stride, i, j);
        }
    }
}