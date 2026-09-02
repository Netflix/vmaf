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
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "libvmaf/picture.h"
#include "cambi_avx2.h"
#include "cambi.h"

// Picks every other uint16 in place: dst[i, j] = src[2i, 2j] for j in [0, width).
// Reads always stay ahead of writes both within a row and across rows, so
// in-place operation is safe. Vectorizes via uint32-mask + packus + permute.
void decimate_avx2(VmafPicture *image, unsigned width, unsigned height) {
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    const __m256i mask = _mm256_set1_epi32(0xFFFF);
    for (unsigned i = 0; i < height; i++) {
        const uint16_t *src = &data[2 * i * stride];
        uint16_t *dst = &data[i * stride];
        unsigned j = 0;
        for (; j + 16 <= width; j += 16) {
            __m256i v1 = _mm256_loadu_si256((const __m256i*)&src[2 * j]);
            __m256i v2 = _mm256_loadu_si256((const __m256i*)&src[2 * j + 16]);
            v1 = _mm256_and_si256(v1, mask);  // keep low uint16 of each uint32 pair
            v2 = _mm256_and_si256(v2, mask);
            __m256i packed = _mm256_packus_epi32(v1, v2);  // values <= 0xFFFF, no actual saturation
            packed = _mm256_permute4x64_epi64(packed, 0xD8);  // 0,2,1,3 to fix lane order
            _mm256_storeu_si256((__m256i*)&dst[j], packed);
        }
        for (; j < width; j++) {
            dst[j] = src[2 * j];
        }
    }
}

// mode3 returns the duplicate among (a, b, c) if any pair matches, otherwise
// the unsigned min. Vectorized over 16 uint16 lanes.
static inline __m256i mode3_avx2(__m256i a, __m256i b, __m256i c) {
    __m256i ab_eq = _mm256_cmpeq_epi16(a, b);
    __m256i ac_eq = _mm256_cmpeq_epi16(a, c);
    __m256i bc_eq = _mm256_cmpeq_epi16(b, c);
    __m256i a_dup = _mm256_or_si256(ab_eq, ac_eq);
    __m256i min_abc = _mm256_min_epu16(_mm256_min_epu16(a, b), c);
    // result = a_dup ? a : (bc_eq ? b : min_abc).  blendv_epi8 works at 16-bit
    // granularity here because cmpeq_epi16 produces consistent byte patterns
    // (both bytes 0xFF or 0x00) within each 16-bit lane.
    __m256i res = _mm256_blendv_epi8(min_abc, b, bc_eq);
    res = _mm256_blendv_epi8(res, a, a_dup);
    return res;
}

static inline uint16_t mode3_scalar(uint16_t a, uint16_t b, uint16_t c) {
    if (a == b || a == c) return a;
    if (b == c) return b;
    uint16_t ab = a < b ? a : b;
    return ab < c ? ab : c;
}

void filter_mode_avx2(const VmafPicture *image, int width, int height, uint16_t *buffer) {
    uint16_t *data = image->data[0];
    ptrdiff_t stride = image->stride[0] >> 1;
    int curr_line = 0;
    for (int i = 0; i < height; i++) {
        // Horizontal pass: buffer[curr_line] = mode3(data[j-1], data[j], data[j+1])
        // for j in [1, width-2]; first and last column copied verbatim.
        buffer[curr_line * width + 0] = data[i * stride + 0];
        int j = 1;
        // Vector loop: each iter writes buffer[j..j+15], which requires
        // j+15 <= width-2 (the last valid mode3 column), i.e. j+16 < width-1.
        for (; j + 16 < width - 1; j += 16) {
            __m256i a = _mm256_loadu_si256((const __m256i*)&data[i * stride + j - 1]);
            __m256i b = _mm256_loadu_si256((const __m256i*)&data[i * stride + j]);
            __m256i c = _mm256_loadu_si256((const __m256i*)&data[i * stride + j + 1]);
            _mm256_storeu_si256((__m256i*)&buffer[curr_line * width + j], mode3_avx2(a, b, c));
        }
        for (; j < width - 1; j++) {
            buffer[curr_line * width + j] = mode3_scalar(
                data[i * stride + j - 1], data[i * stride + j], data[i * stride + j + 1]);
        }
        buffer[curr_line * width + width - 1] = data[i * stride + width - 1];

        // Vertical pass: data[i-1] = mode3(buffer[0], buffer[1], buffer[2])
        if (i > 1) {
            int j2 = 0;
            for (; j2 + 16 <= width; j2 += 16) {
                __m256i a = _mm256_loadu_si256((const __m256i*)&buffer[0 * width + j2]);
                __m256i b = _mm256_loadu_si256((const __m256i*)&buffer[1 * width + j2]);
                __m256i c = _mm256_loadu_si256((const __m256i*)&buffer[2 * width + j2]);
                _mm256_storeu_si256((__m256i*)&data[(i - 1) * stride + j2], mode3_avx2(a, b, c));
            }
            for (; j2 < width; j2++) {
                data[(i - 1) * stride + j2] = mode3_scalar(
                    buffer[0 * width + j2], buffer[1 * width + j2], buffer[2 * width + j2]);
            }
        }
        curr_line = (curr_line + 1 == 3 ? 0 : curr_line + 1);
    }
}

void cambi_increment_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_add_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    } 
}

void cambi_decrement_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_sub_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    } 
}

void calculate_c_values_row_avx2(
    float *c_values, const uint16_t *histograms, const uint16_t *image, const uint16_t *mask,
    int row, int width, ptrdiff_t stride,
    const uint16_t num_diffs, const uint16_t *tvi_thresholds, uint16_t vlt_luma,
    const int *diff_weights, const int *all_diffs,
    const float *reciprocal_lut)
{
    int v_lo_signed_sc = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed_sc > 0 ? (uint16_t)v_lo_signed_sc : 0;
    uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    const __m256i col_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i width_v = _mm256_set1_epi32(width);
    const __m256i num_diffs_v = _mm256_set1_epi32(num_diffs);
    const __m256i vlt_luma_v = _mm256_set1_epi32(vlt_luma);
    const __m256i lo16_mask = _mm256_set1_epi32(0xFFFF);
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i band_offset_v = _mm256_set1_epi32((int)num_diffs + (int)v_band_base);
    const __m256i band_max_v = _mm256_set1_epi32((int)v_band_size - 1);
    const __m256i zero = _mm256_setzero_si256();

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row  = &mask[row * stride];
    float *c_row = &c_values[row * width];

    int col = 0;
    // Vector loop: require at least one extra column past the chunk so the
    // 4-byte vpgatherdd at the worst-case lane never overruns the histogram buffer.
    for (; col + 8 < width; col += 8) {
        // Load 8 mask values, promote to int32, build active mask.
        __m128i mask16 = _mm_loadu_si128((const __m128i*)&mask_row[col]);
        __m256i mask32 = _mm256_cvtepu16_epi32(mask16);
        __m256i mask_active = _mm256_cmpgt_epi32(mask32, zero);

        // Skip the entire chunk if no lane is active.
        if (_mm256_testz_si256(mask_active, mask_active)) {
            _mm256_storeu_ps(&c_row[col], _mm256_setzero_ps());
            continue;
        }

        // value = image[col + lane] + num_diffs  (adjusted space, used for TVI/vlt checks)
        __m128i img16 = _mm_loadu_si128((const __m128i*)&image_row[col]);
        __m256i value_v = _mm256_add_epi32(_mm256_cvtepu16_epi32(img16), num_diffs_v);

        // compact_v = value_v - band_offset, clamped to [0, band_max] for safe gathers
        __m256i compact_v = _mm256_sub_epi32(value_v, band_offset_v);
        compact_v = _mm256_max_epi32(compact_v, zero);
        compact_v = _mm256_min_epi32(compact_v, band_max_v);

        __m256i col_v = _mm256_add_epi32(_mm256_set1_epi32(col), col_base);

        // p_0 = histograms[compact_v * width + col + lane]
        __m256i p0_idx = _mm256_add_epi32(_mm256_mullo_epi32(compact_v, width_v), col_v);
        __m256i p0 = _mm256_and_si256(
            _mm256_i32gather_epi32((const int*)histograms, p0_idx, 2), lo16_mask);

        __m256 c_value = _mm256_setzero_ps();

        for (int d = 0; d < num_diffs; d++) {
            int delta_plus = all_diffs[num_diffs + d + 1];
            int delta_minus = all_diffs[num_diffs - d - 1];
            int weight = diff_weights[d];
            int tvi_thresh = tvi_thresholds[d];

            // pred_a = (value <= tvi_thresh) = NOT (value > tvi_thresh)
            __m256i pred_a_neg = _mm256_cmpgt_epi32(value_v, _mm256_set1_epi32(tvi_thresh));
            __m256i pred_a = _mm256_xor_si256(pred_a_neg, all_ones);

            __m256i value_plus = _mm256_add_epi32(value_v, _mm256_set1_epi32(delta_plus));
            __m256i pred_b = _mm256_cmpgt_epi32(value_plus, vlt_luma_v);
            __m256i predicate = _mm256_and_si256(pred_a, pred_b);

            if (_mm256_testz_si256(predicate, predicate)) continue;

            // compact p1/p2 indices, clamped for safe gathers; track OOB lanes to zero them
            __m256i compact_plus_raw = _mm256_add_epi32(compact_v, _mm256_set1_epi32(delta_plus));
            __m256i compact_plus = _mm256_min_epi32(compact_plus_raw, band_max_v);

            __m256i compact_minus_raw = _mm256_add_epi32(compact_v, _mm256_set1_epi32(delta_minus));
            __m256i p2_inbounds = _mm256_cmpgt_epi32(compact_minus_raw, _mm256_set1_epi32(-1));
            __m256i compact_minus = _mm256_max_epi32(compact_minus_raw, zero);

            // p_1 / p_2 gathers
            __m256i p1_idx = _mm256_add_epi32(_mm256_mullo_epi32(compact_plus, width_v), col_v);
            __m256i p1 = _mm256_and_si256(
                _mm256_i32gather_epi32((const int*)histograms, p1_idx, 2), lo16_mask);

            __m256i p2_idx = _mm256_add_epi32(_mm256_mullo_epi32(compact_minus, width_v), col_v);
            __m256i p2 = _mm256_and_si256(
                _mm256_i32gather_epi32((const int*)histograms, p2_idx, 2), lo16_mask);
            p2 = _mm256_and_si256(p2, p2_inbounds);

            __m256i p_max = _mm256_max_epu32(p1, p2);
            __m256i denom = _mm256_add_epi32(p_max, p0);

            // num = (float)(weight * p_0 * p_max), all uint16-bounded so int32 mul fits
            __m256i num_int = _mm256_mullo_epi32(_mm256_set1_epi32(weight),
                                                  _mm256_mullo_epi32(p0, p_max));
            __m256 num_f = _mm256_cvtepi32_ps(num_int);

            // rcp = reciprocal_lut[denom]; LUT is small, hot in L1
            __m256 rcp = _mm256_i32gather_ps(reciprocal_lut, denom, 4);

            __m256 val = _mm256_mul_ps(num_f, rcp);
            // mask off lanes where predicate is false
            val = _mm256_and_ps(val, _mm256_castsi256_ps(predicate));
            c_value = _mm256_max_ps(c_value, val);
        }

        // Apply mask: lanes with mask == 0 keep 0
        c_value = _mm256_and_ps(c_value, _mm256_castsi256_ps(mask_active));
        _mm256_storeu_ps(&c_row[col], c_value);
    }

    // Scalar tail
    for (; col < width; col++) {
        if (mask_row[col]) {
            uint16_t value = (uint16_t)(image_row[col] + num_diffs);
            int compact_v_signed = (int)image_row[col] - (int)v_band_base;
            if ((unsigned)compact_v_signed >= v_band_size) {
                c_row[col] = 0.0f;
                continue;
            }
            uint16_t compact_v_sc = (uint16_t)compact_v_signed;
            uint16_t p_0 = histograms[compact_v_sc * width + col];
            float c_v = 0.0f;
            for (int d = 0; d < num_diffs; d++) {
                if ((value <= tvi_thresholds[d]) && ((value + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
                    int idx1 = compact_v_signed + all_diffs[num_diffs + d + 1];
                    int idx2 = compact_v_signed + all_diffs[num_diffs - d - 1];
                    uint16_t p_1 = histograms[idx1 * width + col];
                    uint16_t p_2 = (idx2 >= 0) ? histograms[idx2 * width + col] : 0;
                    uint16_t p_max = (p_1 > p_2) ? p_1 : p_2;
                    float val = (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
                    if (val > c_v) c_v = val;
                }
            }
            c_row[col] = c_v;
        } else {
            c_row[col] = 0.0f;
        }
    }
}

void get_derivative_data_for_row_avx2(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride) {
    // For the last row, we only compute horizontal derivatives
    if (row == height - 1) {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i result = _mm256_cmpeq_epi16(vals1, vals2);
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(ones, result));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] = (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    }
    else {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i horiz_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i horiz_vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i horiz_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(horiz_vals1, horiz_vals2));
            __m256i vert_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vert_vals2 = _mm256_loadu_si256((__m256i*) &image_data[(row + 1) * stride + col]);
            __m256i vert_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(vert_vals1, vert_vals2));
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(horiz_result, vert_result));
        }
        for (; col < width; col++) {
            bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative = image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] =  horizontal_derivative && vertical_derivative;
        }
    }
}

void calculate_c_values_avx2(VmafPicture *pic, const VmafPicture *mask_pic,
                             float *c_values, uint16_t *histograms, uint16_t window_size,
                             const uint16_t num_diffs, const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                             const int *diff_weights, const int *all_diffs, int width, int height) {
    uint16_t pad_size = window_size >> 1;

    int v_lo_signed = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    uint16_t v_band_size = tvi_for_diff[num_diffs - 1] + 1 - v_band_base;

    uint16_t *image = pic->data[0];
    uint16_t *mask = mask_pic->data[0];
    ptrdiff_t stride = pic->stride[0] >> 1;

    memset(c_values, 0.0, sizeof(float) * width * height);
    memset(histograms, 0, width * v_band_size * sizeof(uint16_t));

    for (int i = 0; i < pad_size; i++) {
        for (int j = 0; j < pad_size; j++) {
            update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
        }
        for (int j = pad_size; j < width - pad_size - 1; j++) {
            update_histogram_add_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
        }
        for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
            update_histogram_add_edge_first_pass(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
        }
    }

    for (int i = 0; i < pad_size + 1; i++) {
        if (i + pad_size < height) {
            for (int j = 0; j < pad_size; j++) {
                update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
            }
            for (int j = pad_size; j < width - pad_size - 1; j++) {
                update_histogram_add(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
            }
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
                update_histogram_add_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_increment_range_avx2);
            }
        }
        calculate_c_values_row_avx2(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, vlt_luma, diff_weights, all_diffs, reciprocal_lut);
    }
    for (int i = pad_size + 1; i < height - pad_size; i++) {
        for (int j = 0; j < pad_size; j++)
            uh_slide_edge(histograms, image, mask, i, j, width, stride, pad_size, v_band_base, v_band_size, cambi_increment_range_avx2, cambi_decrement_range_avx2);
        for (int j = pad_size; j < width - pad_size - 1; j++)
            uh_slide(histograms, image, mask, i, j, width, stride, pad_size, v_band_base, v_band_size, cambi_increment_range_avx2, cambi_decrement_range_avx2);
        for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++)
            uh_slide_edge(histograms, image, mask, i, j, width, stride, pad_size, v_band_base, v_band_size, cambi_increment_range_avx2, cambi_decrement_range_avx2);
        calculate_c_values_row_avx2(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, vlt_luma, diff_weights, all_diffs, reciprocal_lut);
    }
    for (int i = height - pad_size; i < height; i++) {
        if (i - pad_size - 1 >= 0) {
            for (int j = 0; j < pad_size; j++) {
                update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_decrement_range_avx2);
            }
            for (int j = pad_size; j < width - pad_size - 1; j++) {
                update_histogram_subtract(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_decrement_range_avx2);
            }
            for (int j = MAX(width - pad_size - 1, pad_size); j < width; j++) {
                update_histogram_subtract_edge(histograms, image, mask, i, j, width, stride, pad_size, num_diffs, v_band_base, v_band_size, cambi_decrement_range_avx2);
            }
        }
        calculate_c_values_row_avx2(c_values, histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, vlt_luma, diff_weights, all_diffs, reciprocal_lut);
    }
}