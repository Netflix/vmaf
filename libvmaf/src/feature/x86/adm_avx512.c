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

#include "feature/integer_adm.h"
#include "adm_avx512.h"
#include <immintrin.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define COS_1DEG_SQ cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0)

#if 1

#define print_128_16(a) \
{ \
    printf("%hd ", _mm_extract_epi16(a, 0)); \
    printf("%hd ", _mm_extract_epi16(a, 1)); \
    printf("%hd ", _mm_extract_epi16(a, 2)); \
    printf("%hd ", _mm_extract_epi16(a, 3)); \
    printf("%hd ", _mm_extract_epi16(a, 4)); \
    printf("%hd ", _mm_extract_epi16(a, 5)); \
    printf("%hd ", _mm_extract_epi16(a, 6)); \
    printf("%hd ", _mm_extract_epi16(a, 7)); \
}

#define print_128_32(a) \
{ \
    printf("%d ", _mm_extract_epi32(a, 0)); \
    printf("%d ", _mm_extract_epi32(a, 1)); \
    printf("%d ", _mm_extract_epi32(a, 2)); \
    printf("%d ", _mm_extract_epi32(a, 3)); \
}
#define print_128_ps(a) \
{ \
    printf("%0.31f ", _mm256_cvtss_f32(_mm256_castps128_ps256(a))); \
    printf("%0.31f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x1)))); \
    printf("%0.31f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x2)))); \
    printf("%0.31f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x3)))); \
}
#define print_128_pd(a) \
{ \
    printf("%0.31lf ", _mm256_cvtsd_f64(_mm256_castpd128_pd256(a))); \
    printf("%0.31lf ", _mm256_cvtsd_f64(_mm256_castpd128_pd256(_mm_permute_pd(a, 0x1)))); \
}
#define print_128_32u(a) \
{ \
    printf("%u ", _mm_extract_epi32(a, 0)); \
    printf("%u ", _mm_extract_epi32(a, 1)); \
    printf("%u ", _mm_extract_epi32(a, 2)); \
    printf("%u ", _mm_extract_epi32(a, 3)); \
}
#define print_128_64(a) \
{ \
    printf("%lld ", _mm_extract_epi64(a, 0)); \
    printf("%lld ", _mm_extract_epi64(a, 1)); \
}
#define print_128_x(a) \
{ \
    printf("%x ", _mm_extract_epi32(a, 0)); \
    printf("%x ", _mm_extract_epi32(a, 1)); \
    printf("%x ", _mm_extract_epi32(a, 2)); \
    printf("%x ", _mm_extract_epi32(a, 3)); \
}
#define print_256_x(a) \
{ \
    print_128_x(_mm256_extracti128_si256(a,0)); \
    print_128_x(_mm256_extracti128_si256(a,1)); \
}

#define print_256_16(a) \
{ \
    print_128_16(_mm256_extracti128_si256(a,0)); \
    print_128_16(_mm256_extracti128_si256(a,1)); \
}
#define print_256_32(a) \
{ \
    print_128_32(_mm256_extracti32x4_epi32(a,0)); \
    print_128_32(_mm256_extracti32x4_epi32(a,1)); \
}
#define print_256_ps(a) \
{ \
    print_128_ps(_mm256_extractf32x4_ps(a,0)); \
    print_128_ps(_mm256_extractf32x4_ps(a,1)); \
}
#define print_256_pd(a) \
{ \
    print_128_pd(_mm256_extractf128_pd(a,0)); \
    print_128_pd(_mm256_extractf128_pd(a,1)); \
}
#define print_256_32u(a) \
{ \
    print_128_32u(_mm256_extracti128_si256(a,0)); \
    print_128_32u(_mm256_extracti128_si256(a,1)); \
}
#define print_256_64(a) \
{ \
    print_128_64(_mm256_extracti128_si256(a,0)); \
    print_128_64(_mm256_extracti128_si256(a,1)); \
}

#define print_512_16(a) \
{ \
    print_256_16(_mm512_extracti64x4_epi64(a,0)); \
    print_256_16(_mm512_extracti64x4_epi64(a,1)); \
}

#define print_512_32(a) \
{ \
    print_256_32(_mm512_extracti64x4_epi64(a,0)); \
    print_256_32(_mm512_extracti64x4_epi64(a,1)); \
}
#define print_512_x(a) \
{ \
    print_256_x(_mm512_extracti64x4_epi64(a,0)); \
    print_256_x(_mm512_extracti64x4_epi64(a,1)); \
}
#define print_512_64(a) \
{ \
    print_256_64(_mm512_extracti64x4_epi64(a,0)); \
    print_256_64(_mm512_extracti64x4_epi64(a,1)); \
}
#define print_512_ps(a) \
{ \
    print_256_ps(_mm512_extractf32x8_ps(a,0)); \
    print_256_ps(_mm512_extractf32x8_ps(a,1)); \
}
#define print_512_pd(a) \
{ \
    print_256_pd(_mm512_extractf64x4_pd(a,0)); \
    print_256_pd(_mm512_extractf64x4_pd(a,1)); \
}

#define shift15_64b_signExt_256(a, r) \
{ \
    r = _mm256_add_epi64( _mm256_srli_epi64(a, 15) , _mm256_and_si256(a, _mm256_set1_epi64x(0xFFFE000000000000))); \
}
#define shift15_64b_signExt_512(a, r) \
{ \
    r = _mm512_add_epi64( _mm512_srli_epi64(a, 15) , _mm512_and_si512(a, _mm512_set1_epi64(0xFFFE000000000000))); \
}
#endif


// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define ADM_CM_THRESH_S_0_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_0_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1];  \
		*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[j - 1];\
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define ADM_CM_THRESH_S_I_J(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[j]))+ 2048)>>12);\
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

#define ADM_CM_THRESH_S_I_J_avx512(angles,flt_angles,src_stride,accum,w,h,i,j,sum) \
{ \
    __m512i perm1 = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1); \
    __m512i perm2 = _mm512_set_epi32(1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2); \
    __m512i const_2048_32b = _mm512_set1_epi32(2048); \
    int16_t *src_ptr0 = angles[0]; \
    int16_t *flt_ptr0 = flt_angles[0]; \
    int16_t *src_ptr1 = angles[1]; \
    int16_t *flt_ptr1 = flt_angles[1]; \
    int16_t *src_ptr2 = angles[2]; \
    int16_t *flt_ptr2 = flt_angles[2]; \
    src_ptr0 += (src_stride * (i - 1)); \
    flt_ptr0 += (src_stride * (i - 1)); \
    src_ptr1 += (src_stride * (i - 1)); \
    flt_ptr1 += (src_stride * (i - 1)); \
    src_ptr2 += (src_stride * (i - 1)); \
    flt_ptr2 += (src_stride * (i - 1)); \
    __m512i flt00 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
    __m512i flt10 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
    __m512i flt20 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
    src_ptr0 += src_stride; \
    flt_ptr0 += src_stride; \
    src_ptr1 += src_stride; \
    flt_ptr1 += src_stride; \
    src_ptr2 += src_stride; \
    flt_ptr2 += src_stride; \
    __m512i src01 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_ptr0 + j - 1))); \
    __m512i src11 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_ptr1 + j - 1))); \
    __m512i src21 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_ptr2 + j - 1))); \
    __m512i flt01 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
    __m512i flt11 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
    __m512i flt21 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
    __m512i one_by_15 = _mm512_set1_epi32(ONE_BY_15); \
    src01 = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_abs_epi32(src01), one_by_15), const_2048_32b), 12); \
    src11 = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_abs_epi32(src11), one_by_15), const_2048_32b), 12); \
    src21 = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_abs_epi32(src21), one_by_15), const_2048_32b), 12); \
    src01 = _mm512_sub_epi32(src01, flt01); \
    src11 = _mm512_sub_epi32(src11, flt11); \
    src21 = _mm512_sub_epi32(src21, flt21); \
    __m512i sum0 = _mm512_add_epi32(flt00, flt01); \
    __m512i sum1 = _mm512_add_epi32(flt10, flt11); \
    __m512i sum2 = _mm512_add_epi32(flt20, flt21); \
    src_ptr0 += src_stride; \
    src_ptr1 += src_stride; \
    src_ptr2 += src_stride; \
    flt_ptr0 += src_stride; \
    flt_ptr1 += src_stride; \
    flt_ptr2 += src_stride; \
    __m512i flt02 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
    __m512i flt12 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
    __m512i flt22 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
    sum0 = _mm512_add_epi32(sum0, flt02); \
    sum1 = _mm512_add_epi32(sum1, flt12); \
    sum2 = _mm512_add_epi32(sum2, flt22); \
    __m512i tmp0 = sum0; \
    __m512i tmp1 = sum1; \
    __m512i tmp2 = sum2; \
    sum0 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm1, tmp0), sum0); \
    sum1 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm1, tmp1), sum1); \
    sum2 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm1, tmp2), sum2); \
    sum0 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm2, tmp0), sum0); \
    sum1 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm2, tmp1), sum1); \
    sum2 = _mm512_add_epi32(_mm512_permutexvar_epi32(perm2, tmp2), sum2); \
    sum0 = _mm512_add_epi32(sum0, _mm512_permutexvar_epi32(perm1, src01)); \
    sum1 = _mm512_add_epi32(sum1, _mm512_permutexvar_epi32(perm1, src11)); \
    sum2 = _mm512_add_epi32(sum2, _mm512_permutexvar_epi32(perm1, src21)); \
    __m512i mask_end = _mm512_set_epi64(0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF); \
    sum0 = _mm512_add_epi32(sum0, sum1); \
    sum0 = _mm512_add_epi32(sum0, sum2); \
    sum0 = _mm512_and_si512(mask_end, sum0); \
    _mm512_storeu_si512((__m512i*)sum, sum0); \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define ADM_CM_THRESH_S_I_0(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[0]))+ 2048)>>12);\
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int16_t *src_ptr = angles[theta]; \
		int16_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i-1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int16_t)(((ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ 2048)>>12);\
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 0: indices y: 1,0,1, x: 1,0,1  for Fixed-point
#define I4_ADM_CM_THRESH_S_0_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs( src_ptr[0]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[1]; \
		sum += flt_ptr[src_stride + 1]; \
		sum += flt_ptr[src_stride]; \
		sum += flt_ptr[src_stride + 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = w-1: indices y: 1,0,1, x: w-2, w-1, w-1
#define I4_ADM_CM_THRESH_S_0_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[src_stride + w - 2]; \
		sum += flt_ptr[src_stride + w - 1]; \
		sum += flt_ptr[src_stride + w - 1]; \
		*accum += sum; \
	} \
}

// i = 0, j = 1, ..., w-2: indices y: 1,0,1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_0_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[src_stride + j - 1]; \
		sum += flt_ptr[src_stride + j]; \
		sum += flt_ptr[src_stride + j + 1];  \
		*accum += sum; \
	} \
}

// i = h-1, j = 0: indices y: h-2,h-1,h-1, x: 1,0,1
#define I4_ADM_CM_THRESH_S_H_M_1_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
	    int32_t sum = 0; \
	    int32_t *src_ptr = angles[theta]; \
	    int32_t *flt_ptr = flt_angles[theta]; \
	    src_ptr += (src_stride * (h - 2)); \
	    flt_ptr += (src_stride * (h - 2)); \
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[0]; \
	    sum += flt_ptr[1]; \
	    src_ptr += src_stride; \
	    flt_ptr += src_stride; \
	    sum += flt_ptr[1]; \
	    sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[1]; \
	    sum += flt_ptr[0]; \
	    sum += flt_ptr[1]; \
	    *accum += sum; \
	} \
}

// i = h-1, j = w-1: indices y: h-2,h-1,h-1, x: w-2, w-1, w-1
#define I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

// i = h-1, j = 1, ..., w-2: indices y: h-2,h-1,h-1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_H_M_1_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (h - 2)); \
		flt_ptr += (src_stride * (h - 2)); \
		sum += flt_ptr[j - 1];\
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

#define I4_ADM_CM_THRESH_S_I_J_avx512(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift,sum) \
{ \
        __m512i perm1 = _mm512_set_epi64(0, 7, 6, 5, 4, 3, 2, 1); \
        __m512i perm2 = _mm512_set_epi64(1, 0, 7, 6, 5, 4, 3, 2); \
        __m512i add_bef_shift_512 = _mm512_set1_epi64(add_bef_shift); \
        int32_t *src_ptr0 = angles[0]; \
        int32_t *flt_ptr0 = flt_angles[0]; \
        int32_t *src_ptr1 = angles[1]; \
        int32_t *flt_ptr1 = flt_angles[1]; \
        int32_t *src_ptr2 = angles[2]; \
        int32_t *flt_ptr2 = flt_angles[2]; \
        src_ptr0 += (src_stride * (i - 1)); \
        flt_ptr0 += (src_stride * (i - 1)); \
        src_ptr1 += (src_stride * (i - 1)); \
        flt_ptr1 += (src_stride * (i - 1)); \
        src_ptr2 += (src_stride * (i - 1)); \
        flt_ptr2 += (src_stride * (i - 1)); \
        __m512i flt00 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
        __m512i flt10 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
        __m512i flt20 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
        src_ptr0 += src_stride; \
        flt_ptr0 += src_stride; \
        src_ptr1 += src_stride; \
        flt_ptr1 += src_stride; \
        src_ptr2 += src_stride; \
        flt_ptr2 += src_stride; \
        __m512i src01 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src_ptr0 + j - 1))); \
        __m512i src11 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src_ptr1 + j - 1))); \
        __m512i src21 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src_ptr2 + j - 1))); \
        __m512i flt01 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
        __m512i flt11 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
        __m512i flt21 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
        __m512i i4_one_by_15 = _mm512_set1_epi64(I4_ONE_BY_15); \
        src01 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(_mm512_abs_epi64(src01), i4_one_by_15), add_bef_shift_512), shift); \
        src11 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(_mm512_abs_epi64(src11), i4_one_by_15), add_bef_shift_512), shift); \
        src21 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(_mm512_abs_epi64(src21), i4_one_by_15), add_bef_shift_512), shift); \
        src01 = _mm512_sub_epi64(src01, flt01); \
        src11 = _mm512_sub_epi64(src11, flt11); \
        src21 = _mm512_sub_epi64(src21, flt21); \
        __m512i sum0 = _mm512_add_epi64(flt00, flt01); \
        __m512i sum1 = _mm512_add_epi64(flt10, flt11); \
        __m512i sum2 = _mm512_add_epi64(flt20, flt21); \
        src_ptr0 += src_stride; \
        src_ptr1 += src_stride; \
        src_ptr2 += src_stride; \
        flt_ptr0 += src_stride; \
        flt_ptr1 += src_stride; \
        flt_ptr2 += src_stride; \
        __m512i flt02 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr0 + j - 1))); \
        __m512i flt12 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr1 + j - 1))); \
        __m512i flt22 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(flt_ptr2 + j - 1))); \
        sum0 = _mm512_add_epi64(sum0, flt02); \
        sum1 = _mm512_add_epi64(sum1, flt12); \
        sum2 = _mm512_add_epi64(sum2, flt22); \
        __m512i tmp0 = sum0; \
        __m512i tmp1 = sum1; \
        __m512i tmp2 = sum2; \
        sum0 = _mm512_add_epi64(sum0, _mm512_permutexvar_epi64(perm1, tmp0)); \
        sum1 = _mm512_add_epi64(sum1, _mm512_permutexvar_epi64(perm1, tmp1)); \
        sum2 = _mm512_add_epi64(sum2, _mm512_permutexvar_epi64(perm1, tmp2)); \
        sum0 = _mm512_add_epi64(sum0, _mm512_permutexvar_epi64(perm2, tmp0)); \
        sum1 = _mm512_add_epi64(sum1, _mm512_permutexvar_epi64(perm2, tmp1)); \
        sum2 = _mm512_add_epi64(sum2, _mm512_permutexvar_epi64(perm2, tmp2)); \
        sum0 = _mm512_add_epi64(sum0, _mm512_permutexvar_epi64(perm1, src01)); \
        sum1 = _mm512_add_epi64(sum1, _mm512_permutexvar_epi64(perm1, src11)); \
        sum2 = _mm512_add_epi64(sum2, _mm512_permutexvar_epi64(perm1, src21)); \
        __m512i mask_end = _mm512_set_epi64(0x0, 0x0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF); \
        sum0 = _mm512_add_epi64(sum0, sum1); \
        sum0 = _mm512_add_epi64(sum0, sum2); \
        sum0 = _mm512_and_si512(mask_end, sum0); \
        _mm512_storeu_si512((__m512i*)sum, sum0); \
    }

// i = 1,..,h-2, j = 1,..,w-2: indices y: i-1,i,i+1, x: j-1,j,j+1
#define I4_ADM_CM_THRESH_S_I_J(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t sum = 0; \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[j]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[j + 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[j - 1]; \
		sum += flt_ptr[j]; \
		sum += flt_ptr[j + 1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = 0: indices y: i-1,i,i+1, x: 1,0,1
#define I4_ADM_CM_THRESH_S_I_0(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i - 1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[0]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[1]; \
		sum += flt_ptr[0]; \
		sum += flt_ptr[1]; \
		*accum += sum; \
	} \
}

// i = 1,..,h-2, j = w-1: indices y: i-1,i,i+1, x: w-2,w-1,w-1
#define I4_ADM_CM_THRESH_S_I_W_M_1(angles,flt_angles,src_stride,accum,w,h,i,j,add_bef_shift,shift) \
{ \
	*accum = 0; \
	for (int theta = 0; theta < 3; ++theta) { \
		int32_t *src_ptr = angles[theta]; \
		int32_t *flt_ptr = flt_angles[theta]; \
		int32_t sum = 0; \
		src_ptr += (src_stride * (i-1)); \
		flt_ptr += (src_stride * (i - 1)); \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t) src_ptr[w - 1]))+ add_bef_shift)>>shift);\
		sum += flt_ptr[w - 1]; \
		src_ptr += src_stride; \
		flt_ptr += src_stride; \
		sum += flt_ptr[w - 2]; \
		sum += flt_ptr[w - 1]; \
		sum += flt_ptr[w - 1]; \
		*accum += sum; \
	} \
}

#define ADM_CM_ACCUM_ROUND(x, thr, shift_xsub, x_sq, add_shift_xsq, shift_xsq, val, \
                           add_shift_xcub, shift_xcub, accum_inner) \
{ \
    x = abs(x) - ((int32_t)(thr) << shift_xsub); \
    x = x < 0 ? 0 : x; \
    x_sq = (int32_t)((((int64_t)x * x) + add_shift_xsq) >> shift_xsq); \
    val = (((int64_t)x_sq * x) + add_shift_xcub) >> shift_xcub; \
    accum_inner += val; \
}

#define ADM_CM_ACCUM_ROUND_avx512(x, thr, shift_xsub, x_sq, add_shift_xsq, shift_xsq, val, \
                           add_shift_xcub, shift_xcub, accum_inner_lo, accum_inner_hi) \
{ \
    x = _mm512_sub_epi32(_mm512_abs_epi32(x), _mm512_slli_epi32(thr, shift_xsub)); \
    x = _mm512_max_epi32(x, _mm512_setzero_si512()); \
    __m512i x_sq_lo = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(x, x), _mm512_set1_epi64(add_shift_xsq)), shift_xsq); \
    __m512i x_sq_hi = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(_mm512_srli_epi64(x, 32), _mm512_srli_epi64(x, 32)), _mm512_set1_epi64(add_shift_xsq)), shift_xsq); \
    x_sq_lo = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(x_sq_lo, x), _mm512_set1_epi64(add_shift_xcub)), shift_xcub); \
    x_sq_hi = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(x_sq_hi, _mm512_srli_epi64(x, 32)), _mm512_set1_epi64(add_shift_xcub)), shift_xcub); \
    accum_inner_lo = _mm512_add_epi64(accum_inner_lo, x_sq_lo); \
    accum_inner_hi = _mm512_add_epi64(accum_inner_hi, x_sq_hi); \
}

#define I4_ADM_CM_ACCUM_ROUND(x, thr, shift_sub, x_sq, add_shift_sq, shift_sq, val, \
                              add_shift_cub, shift_cub, accum_inner)    \
{ \
    x = abs(x) - (thr >> shift_sub); \
    x = x < 0 ? 0 : x; \
    x_sq = (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq); \
    val = (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub; \
    accum_inner += val; \
}

#define I4_ADM_CM_ACCUM_ROUND_avx512(x, thr, shift_xsub, x_sq, add_shift_xsq, shift_xsq, val, \
                           add_shift_xcub, shift_xcub, accum_inner) \
{ \
    x = _mm512_sub_epi64(_mm512_abs_epi64(x), _mm512_srli_epi64(thr, shift_xsub)); \
    x = _mm512_max_epi64(x, _mm512_setzero_si512()); \
    __m512i x_sq = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(x, x), _mm512_set1_epi64(add_shift_xsq)), shift_xsq); \
    x_sq = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(x_sq, x), _mm512_set1_epi64(add_shift_xcub)), shift_xcub); \
    accum_inner = _mm512_add_epi64(accum_inner, x_sq); \
}

void adm_decouple_avx512(AdmBuffer *buf, int w, int h, int stride,
                         double adm_enhn_gain_limit, int32_t* adm_div_lookup)
{
    const float const_cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);

    const adm_dwt_band_t *ref = &buf->ref_dwt2;
    const adm_dwt_band_t *dis = &buf->dis_dwt2;
    const adm_dwt_band_t *r = &buf->decouple_r;
    const adm_dwt_band_t *a = &buf->decouple_a;

    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int64_t ot_dp, o_mag_sq, t_mag_sq;

    int right_mod16 = right - (right % 16);
    __m512 inv_32768 = _mm512_set1_ps((float)1/32768);
    __m512 inv_4096 = _mm512_set1_ps((float)0.000244140625);
    __m512 inv_64 = _mm512_set1_ps((float)1/64);
#if CALC_ANGLE1
    __m512 cos_1deg_sq = _mm512_set1_ps(const_cos_1deg_sq);
#endif    
    __m512d cos_1deg_sq_pd = _mm512_set1_pd((float)const_cos_1deg_sq);

    for (int i = top; i < bottom; ++i) {
        for (int j = left; j < right_mod16; j+=16) {
            __m512i oh = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(ref->band_h + i * stride + j)));
            __m512i ov = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(ref->band_v + i * stride + j)));
            __m512i od = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(ref->band_d + i * stride + j)));
            __m512i th = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(dis->band_h + i * stride + j)));
            __m512i tv = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(dis->band_v + i * stride + j)));
            __m512i td = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(dis->band_d + i * stride + j)));

            __m512i oh_ov = _mm512_or_si512(_mm512_and_si512(oh, _mm512_set1_epi32(0xFFFF)), _mm512_slli_epi32(ov, 16));
            __m512i th_tv = _mm512_or_si512(_mm512_and_si512(th, _mm512_set1_epi32(0xFFFF)), _mm512_slli_epi32(tv, 16));

            __m512i o_mag_sq = _mm512_madd_epi16(oh_ov, oh_ov);
            __m512i ot_dp = _mm512_madd_epi16(oh_ov, th_tv);
            __m512i t_mag_sq = _mm512_madd_epi16(th_tv, th_tv);

            __m512 ot_dp_ps = _mm512_mul_ps(inv_4096, _mm512_cvtepi32_ps(ot_dp));
            __m512 o_mag_sq_ps = _mm512_mul_ps(inv_4096, _mm512_cvtepi32_ps(o_mag_sq));
            __m512 t_mag_sq_ps = _mm512_mul_ps(inv_4096, _mm512_cvtepi32_ps(t_mag_sq));

#if CALC_ANGLE1
            __mmask16 gt_0 = _mm512_cmp_ps_mask(ot_dp_ps, _mm512_setzero_ps(), 13);
            __m512 ot_dp_sq = _mm512_mul_ps(ot_dp_ps, ot_dp_ps);
            __m512 o_mag_sq_t_mag_sq = _mm512_mul_ps(o_mag_sq_ps, t_mag_sq_ps);            
            __m512 ot_mag_sq_cos1 = _mm512_mul_ps(o_mag_sq_t_mag_sq, cos_1deg_sq);
            __mmask16 cmp_ot_cosot_mag = _mm512_cmp_ps_mask(ot_dp_sq, ot_mag_sq_cos1, 13);
            __mmask16 angle_flag = _kand_mask16(gt_0, cmp_ot_cosot_mag);
#else
            __mmask16 gt_0 = _mm512_cmp_ps_mask(ot_dp_ps, _mm512_setzero_ps(), 13);
            __m512d ot_dp_pd_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(ot_dp_ps));
            __m512d ot_dp_pd_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ot_dp_ps, 1));
            __m512d ot_dp_sq_lo = _mm512_mul_pd(ot_dp_pd_lo, ot_dp_pd_lo);
            __m512d ot_dp_sq_hi = _mm512_mul_pd(ot_dp_pd_hi, ot_dp_pd_hi);
            __m512d o_mag_sq_pd_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(o_mag_sq_ps));
            __m512d o_mag_sq_pd_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(o_mag_sq_ps, 1));
            __m512d t_mag_sq_pd_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(t_mag_sq_ps));
            __m512d t_mag_sq_pd_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(t_mag_sq_ps, 1));
            __m512d o_mag_sq_t_mag_sq_lo = _mm512_mul_pd(o_mag_sq_pd_lo, t_mag_sq_pd_lo);
            __m512d o_mag_sq_t_mag_sq_hi = _mm512_mul_pd(o_mag_sq_pd_hi, t_mag_sq_pd_hi);
            __m512d ot_mag_sq_cos1_lo = _mm512_mul_pd(o_mag_sq_t_mag_sq_lo, cos_1deg_sq_pd);
            __m512d ot_mag_sq_cos1_hi = _mm512_mul_pd(o_mag_sq_t_mag_sq_hi, cos_1deg_sq_pd);
            __mmask8 cmp_ot_cosot_mag_lo = _mm512_cmp_pd_mask(ot_dp_sq_lo, ot_mag_sq_cos1_lo, 13);
            __mmask8 cmp_ot_cosot_mag_hi = _mm512_cmp_pd_mask(ot_dp_sq_hi, ot_mag_sq_cos1_hi, 13);
            __mmask16 angle_flag = _kand_mask16(gt_0, cmp_ot_cosot_mag_lo + (cmp_ot_cosot_mag_hi << 8));
#endif
      
            __m512i oh_div = _mm512_i32gather_epi32(_mm512_add_epi32(oh, _mm512_set1_epi32(32768)), adm_div_lookup, 4);
            __m512i ov_div = _mm512_i32gather_epi32(_mm512_add_epi32(ov, _mm512_set1_epi32(32768)), adm_div_lookup, 4);
            __m512i od_div = _mm512_i32gather_epi32(_mm512_add_epi32(od, _mm512_set1_epi32(32768)), adm_div_lookup, 4);

            __m512i oh_div_th_lo = _mm512_mul_epi32(oh_div, th);
            __m512i oh_div_th_hi = _mm512_mul_epi32(_mm512_srli_epi64(oh_div, 32), _mm512_srli_epi64(th, 32));
            __m512i ov_div_th_lo = _mm512_mul_epi32(ov_div, tv);
            __m512i ov_div_th_hi = _mm512_mul_epi32(_mm512_srli_epi64(ov_div, 32), _mm512_srli_epi64(tv, 32));
            __m512i od_div_th_lo = _mm512_mul_epi32(od_div, td);
            __m512i od_div_th_hi = _mm512_mul_epi32(_mm512_srli_epi64(od_div, 32), _mm512_srli_epi64(td, 32));

            oh_div_th_lo = _mm512_add_epi64(oh_div_th_lo, _mm512_set1_epi32(32768));
            oh_div_th_hi = _mm512_add_epi64(oh_div_th_hi, _mm512_set1_epi32(32768));
            od_div_th_lo = _mm512_add_epi64(od_div_th_lo, _mm512_set1_epi32(32768));
            od_div_th_hi = _mm512_add_epi64(od_div_th_hi, _mm512_set1_epi32(32768));
            ov_div_th_lo = _mm512_add_epi64(ov_div_th_lo, _mm512_set1_epi32(32768));
            ov_div_th_hi = _mm512_add_epi64(ov_div_th_hi, _mm512_set1_epi32(32768));

            shift15_64b_signExt_512(oh_div_th_lo, oh_div_th_lo);
            shift15_64b_signExt_512(oh_div_th_hi, oh_div_th_hi);
            shift15_64b_signExt_512(ov_div_th_lo, ov_div_th_lo);
            shift15_64b_signExt_512(ov_div_th_hi, ov_div_th_hi);
            shift15_64b_signExt_512(od_div_th_lo, od_div_th_lo);
            shift15_64b_signExt_512(od_div_th_hi, od_div_th_hi);

            __m512i tmp_kh = _mm512_or_si512(_mm512_and_si512(oh_div_th_lo, _mm512_set1_epi64(0xFFFFFFFF)), _mm512_slli_epi64(oh_div_th_hi, 32));
            __m512i tmp_kv = _mm512_or_si512(_mm512_and_si512(ov_div_th_lo, _mm512_set1_epi64(0xFFFFFFFF)), _mm512_slli_epi64(ov_div_th_hi, 32));
            __m512i tmp_kd = _mm512_or_si512(_mm512_and_si512(od_div_th_lo, _mm512_set1_epi64(0xFFFFFFFF)), _mm512_slli_epi64(od_div_th_hi, 32));

            __mmask16 eqz_oh = _mm512_cmp_epi32_mask(oh, _mm512_setzero_si512(), 0);
            __mmask16 eqz_ov = _mm512_cmp_epi32_mask(ov, _mm512_setzero_si512(), 0);
            __mmask16 eqz_od = _mm512_cmp_epi32_mask(od, _mm512_setzero_si512(), 0);

            tmp_kh = _mm512_mask_blend_epi32(eqz_oh, tmp_kh, _mm512_set1_epi32(32768));
            tmp_kv = _mm512_mask_blend_epi32(eqz_ov, tmp_kv, _mm512_set1_epi32(32768));
            tmp_kd = _mm512_mask_blend_epi32(eqz_od, tmp_kd, _mm512_set1_epi32(32768));

            tmp_kh = _mm512_max_epi32(tmp_kh, _mm512_setzero_si512());
            tmp_kh = _mm512_min_epi32(tmp_kh, _mm512_set1_epi32(32768));
            tmp_kv = _mm512_max_epi32(tmp_kv, _mm512_setzero_si512());
            tmp_kv = _mm512_min_epi32(tmp_kv, _mm512_set1_epi32(32768));
            tmp_kd = _mm512_max_epi32(tmp_kd, _mm512_setzero_si512());
            tmp_kd = _mm512_min_epi32(tmp_kd, _mm512_set1_epi32(32768));

            __m512i rst_h = _mm512_mullo_epi32(tmp_kh, oh);
            __m512i rst_v = _mm512_mullo_epi32(tmp_kv, ov);
            __m512i rst_d = _mm512_mullo_epi32(tmp_kd, od);

            __m512i const_16384_32b = _mm512_set1_epi32(16384);
            rst_h = _mm512_add_epi32(rst_h, const_16384_32b);
            rst_v = _mm512_add_epi32(rst_v, const_16384_32b);
            rst_d = _mm512_add_epi32(rst_d, const_16384_32b);

            rst_h = _mm512_srai_epi32(rst_h, 15);
            rst_v = _mm512_srai_epi32(rst_v, 15);
            rst_d = _mm512_srai_epi32(rst_d, 15);

            __m512 kh_inv_32768 = _mm512_mul_ps(inv_32768, _mm512_cvtepi32_ps(tmp_kh));
            __m512 oh_inv_64 = _mm512_mul_ps(inv_64, _mm512_cvtepi32_ps(oh));
            __m512 rst_h_f = _mm512_mul_ps(kh_inv_32768, oh_inv_64);
            
            __m512 kv_inv_32768 = _mm512_mul_ps(inv_32768, _mm512_cvtepi32_ps(tmp_kv));
            __m512 ov_inv_64 = _mm512_mul_ps(inv_64, _mm512_cvtepi32_ps(ov));
            __m512 rst_v_f = _mm512_mul_ps(kv_inv_32768, ov_inv_64);

            __m512 kd_inv_32768 = _mm512_mul_ps(inv_32768, _mm512_cvtepi32_ps(tmp_kd));
            __m512 od_inv_64 = _mm512_mul_ps(inv_64, _mm512_cvtepi32_ps(od));
            __m512 rst_d_f = _mm512_mul_ps(kd_inv_32768, od_inv_64);

            __mmask16 gt0_rst_h_f = _mm512_cmp_ps_mask(rst_h_f, _mm512_setzero_ps(), 14);
            __mmask16 lt0_rst_h_f = _mm512_cmp_ps_mask(rst_h_f, _mm512_setzero_ps(), 1);
            __mmask16 gt0_rst_v_f = _mm512_cmp_ps_mask(rst_v_f, _mm512_setzero_ps(), 14);
            __mmask16 lt0_rst_v_f = _mm512_cmp_ps_mask(rst_v_f, _mm512_setzero_ps(), 1);
            __mmask16 gt0_rst_d_f = _mm512_cmp_ps_mask(rst_d_f, _mm512_setzero_ps(), 14);
            __mmask16 lt0_rst_d_f = _mm512_cmp_ps_mask(rst_d_f, _mm512_setzero_ps(), 1);

            __m512d adm_gain_d = _mm512_set1_pd(adm_enhn_gain_limit);
            __m512d rst_h_gainlo_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_h, 0)), adm_gain_d);
            __m512d rst_h_gainhi_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_h, 1)), adm_gain_d);
            __m512i rst_h_gain = _mm512_inserti32x8(_mm512_castsi256_si512(_mm512_cvtpd_epi32(rst_h_gainlo_d)), _mm512_cvtpd_epi32(rst_h_gainhi_d),1);
            __m512d rst_v_gainlo_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_v, 0)), adm_gain_d);
            __m512d rst_v_gainhi_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_v, 1)), adm_gain_d);
            __m512i rst_v_gain = _mm512_inserti32x8(_mm512_castsi256_si512(_mm512_cvtpd_epi32(rst_v_gainlo_d)), _mm512_cvtpd_epi32(rst_v_gainhi_d),1);
            __m512d rst_d_gainlo_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_d, 0)), adm_gain_d);
            __m512d rst_d_gainhi_d = _mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(rst_d, 1)), adm_gain_d);
            __m512i rst_d_gain = _mm512_inserti32x8(_mm512_castsi256_si512(_mm512_cvtpd_epi32(rst_d_gainlo_d)), _mm512_cvtpd_epi32(rst_d_gainhi_d),1);
 
            __m512i h_min = _mm512_min_epi32(rst_h_gain, th);
            __m512i v_min = _mm512_min_epi32(rst_v_gain, tv);
            __m512i d_min = _mm512_min_epi32(rst_d_gain, td);
            
            __m512i h_max = _mm512_max_epi32(rst_h_gain, th);
            __m512i v_max = _mm512_max_epi32(rst_v_gain, tv);
            __m512i d_max = _mm512_max_epi32(rst_d_gain, td);

            h_min = _mm512_mask_blend_epi32(gt0_rst_h_f, _mm512_setzero_epi32(), h_min);
            h_max = _mm512_mask_blend_epi32(lt0_rst_h_f, _mm512_setzero_epi32(), h_max);
            v_min = _mm512_mask_blend_epi32(gt0_rst_v_f, _mm512_setzero_epi32(), v_min);
            v_max = _mm512_mask_blend_epi32(lt0_rst_v_f, _mm512_setzero_epi32(), v_max);
            d_min = _mm512_mask_blend_epi32(gt0_rst_d_f, _mm512_setzero_epi32(), d_min);
            d_max = _mm512_mask_blend_epi32(lt0_rst_d_f, _mm512_setzero_epi32(), d_max);

            __mmask16 mask_min_max_h = _kor_mask16(gt0_rst_h_f, lt0_rst_h_f);
            __mmask16 mask_min_max_v = _kor_mask16(gt0_rst_v_f, lt0_rst_v_f);
            __mmask16 mask_min_max_d = _kor_mask16(gt0_rst_d_f, lt0_rst_d_f);

            __m512i h_min_max = _mm512_or_si512(h_min, h_max);
            __m512i v_min_max = _mm512_or_si512(v_min, v_max);
            __m512i d_min_max = _mm512_or_si512(d_min, d_max);

            h_min_max = _mm512_mask_blend_epi32(mask_min_max_h, rst_h, h_min_max);
            v_min_max = _mm512_mask_blend_epi32(mask_min_max_v, rst_v, v_min_max);
            d_min_max = _mm512_mask_blend_epi32(mask_min_max_d, rst_d, d_min_max);

            rst_h = _mm512_mask_blend_epi32(angle_flag, rst_h, h_min_max);
            rst_v = _mm512_mask_blend_epi32(angle_flag, rst_v, v_min_max);
            rst_d = _mm512_mask_blend_epi32(angle_flag, rst_d, d_min_max);

            th = _mm512_sub_epi32(th, rst_h);
            tv = _mm512_sub_epi32(tv, rst_v);
            td = _mm512_sub_epi32(td, rst_d);

            __m512i perm = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
            rst_h = _mm512_permutexvar_epi16(perm, rst_h);
            rst_v = _mm512_permutexvar_epi16(perm, rst_v);
            rst_d = _mm512_permutexvar_epi16(perm, rst_d);
            th = _mm512_permutexvar_epi16(perm, th);
            tv = _mm512_permutexvar_epi16(perm, tv);
            td = _mm512_permutexvar_epi16(perm, td);

            _mm256_storeu_si256((__m256i*)(r->band_h + i * stride + j), _mm512_castsi512_si256(rst_h));
            _mm256_storeu_si256((__m256i*)(r->band_v + i * stride + j), _mm512_castsi512_si256(rst_v));
            _mm256_storeu_si256((__m256i*)(r->band_d + i * stride + j), _mm512_castsi512_si256(rst_d));
            _mm256_storeu_si256((__m256i*)(a->band_h + i * stride + j), _mm512_castsi512_si256(th));
            _mm256_storeu_si256((__m256i*)(a->band_v + i * stride + j), _mm512_castsi512_si256(tv));
            _mm256_storeu_si256((__m256i*)(a->band_d + i * stride + j), _mm512_castsi512_si256(td));
        }  

        for (int j = right_mod16; j < right; j++)
        {
            int16_t oh = ref->band_h[i * stride + j];
            int16_t ov = ref->band_v[i * stride + j];
            int16_t od = ref->band_d[i * stride + j];
            int16_t th = dis->band_h[i * stride + j];
            int16_t tv = dis->band_v[i * stride + j];
            int16_t td = dis->band_d[i * stride + j];
            int16_t rst_h, rst_v, rst_d;

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
             * angle_flag is calculated in floating-point by converting fixed-point variables back to
             * floating-point
             */

            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    const_cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

            /**
             * Division th/oh is carried using lookup table and converted to multiplication
             */

            int32_t tmp_kh = (oh == 0) ?
                32768 : (((int64_t)adm_div_lookup[oh + 32768] * th) + 16384) >> 15;
            int32_t tmp_kv = (ov == 0) ?
                32768 : (((int64_t)adm_div_lookup[ov + 32768] * tv) + 16384) >> 15;
            int32_t tmp_kd = (od == 0) ?
                32768 : (((int64_t)adm_div_lookup[od + 32768] * td) + 16384) >> 15;

            int32_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
            int32_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
            int32_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

            /**
             * kh,kv,kd are in Q15 type and oh,ov,od are in Q16 type hence shifted by
             * 15 to make result Q16
             */
            rst_h = ((kh * oh) + 16384) >> 15;
            rst_v = ((kv * ov) + 16384) >> 15;
            rst_d = ((kd * od) + 16384) >> 15;

            const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
            const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
            const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

            if (angle_flag && (rst_h_f > 0.)) rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
            if (angle_flag && (rst_h_f < 0.)) rst_h = MAX((rst_h * adm_enhn_gain_limit), th);

            if (angle_flag && (rst_v_f > 0.)) rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v_f < 0.)) rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d_f > 0.)) rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d_f < 0.)) rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            r->band_h[i * stride + j] = rst_h;
            r->band_v[i * stride + j] = rst_v;
            r->band_d[i * stride + j] = rst_d;

            a->band_h[i * stride + j] = th - rst_h;
            a->band_v[i * stride + j] = tv - rst_v;
            a->band_d[i * stride + j] = td - rst_d;
            
        }
    }        
}

static inline uint16_t get_best15_from32(uint32_t temp, int *x)
{
    int k = __builtin_clz(temp);    //built in for intel
    k = 17 - k;
    temp = (temp + (1 << (k - 1))) >> k;
    *x = k;
    return temp;
}

#define calc_angle(ot_dp, o_mag_sq, t_mag_sq, angle_flag, i) \
{ \
angle_flag += ((((float)ot_dp / 4096.0) >= 0.0f) && \
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >= \
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0))) << i; \
} \

void adm_decouple_s123_avx512(AdmBuffer *buf, int w, int h, int stride,
                              double adm_enhn_gain_limit, int32_t* adm_div_lookup)
{
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);

    const i4_adm_dwt_band_t *ref = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *dis = &buf->i4_dis_dwt2;
    const i4_adm_dwt_band_t *r = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *a = &buf->i4_decouple_a;
    /* The computation of the score is not required for the regions
    which lie outside the frame borders */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int64_t ot_dp, o_mag_sq, t_mag_sq;

    int right_mod16 = right - (right % 16);
    
#if CALC_ANGLE2
    __m512d inv_4096 = _mm512_set1_pd((float)1/4096);
#endif
    __m512i perm_for_mul = _mm512_set_epi32(15, 7, 14, 6, 13, 5,  12, 4, 11, 3, 10, 2, 9, 1, 8, 0);

    for (int i = top; i < bottom; ++i)
    {
        for (int j = left; j < right_mod16; j+=16)
        {
                __m512i oh = _mm512_loadu_si512((__m512i*)(ref->band_h + i * stride + j));
                __m512i ov = _mm512_loadu_si512((__m512i*)(ref->band_v + i * stride + j));
                __m512i od = _mm512_loadu_si512((__m512i*)(ref->band_d + i * stride + j));
                __m512i th = _mm512_loadu_si512((__m512i*)(dis->band_h + i * stride + j));
                __m512i tv = _mm512_loadu_si512((__m512i*)(dis->band_v + i * stride + j));
                __m512i td = _mm512_loadu_si512((__m512i*)(dis->band_d + i * stride + j));

                __m512i oh_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(oh,0));
                __m512i oh_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(oh,1));                
                __m512i ov_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ov,0));
                __m512i ov_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ov,1));
                __m512i od_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(od,0));
                __m512i od_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(od,1));
                __m512i th_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(th,0));
                __m512i th_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(th,1));
                __m512i tv_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(tv,0));
                __m512i tv_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(tv,1));
                __m512i td_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(td,0));
                __m512i td_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(td,1));

                __m512i oh_th_lo = _mm512_mul_epi32(oh_lo, th_lo);
                __m512i oh_th_hi = _mm512_mul_epi32(oh_hi, th_hi);
                __m512i ov_tv_lo = _mm512_mul_epi32(ov_lo, tv_lo);
                __m512i ov_tv_hi = _mm512_mul_epi32(ov_hi, tv_hi);
                __m512i oh_oh_lo = _mm512_mul_epi32(oh_lo, oh_lo);
                __m512i oh_oh_hi = _mm512_mul_epi32(oh_hi, oh_hi);
                __m512i ov_ov_lo = _mm512_mul_epi32(ov_lo, ov_lo);
                __m512i ov_ov_hi = _mm512_mul_epi32(ov_hi, ov_hi);
                __m512i th_th_lo = _mm512_mul_epi32(th_lo, th_lo);
                __m512i th_th_hi = _mm512_mul_epi32(th_hi, th_hi);
                __m512i tv_tv_lo = _mm512_mul_epi32(tv_lo, tv_lo);
                __m512i tv_tv_hi = _mm512_mul_epi32(tv_hi, tv_hi);

                __m512i ot_dp_lo = _mm512_add_epi64(oh_th_lo, ov_tv_lo);
                __m512i ot_dp_hi = _mm512_add_epi64(oh_th_hi, ov_tv_hi);
                __m512i o_mag_sq_lo = _mm512_add_epi64(oh_oh_lo, ov_ov_lo);
                __m512i o_mag_sq_hi = _mm512_add_epi64(oh_oh_hi, ov_ov_hi);
                __m512i t_mag_sq_lo = _mm512_add_epi64(th_th_lo, tv_tv_lo);
                __m512i t_mag_sq_hi = _mm512_add_epi64(th_th_hi, tv_tv_hi);

#if CALC_ANGLE2
                __m512d ot_dp_pd_lo = _mm512_cvtepi64_pd(ot_dp_lo);
                __m512d ot_dp_pd_hi = _mm512_cvtepi64_pd(ot_dp_hi);
                __m512d o_mag_sq_pd_lo = _mm512_cvtepi64_pd(o_mag_sq_lo);
                __m512d o_mag_sq_pd_hi = _mm512_cvtepi64_pd(o_mag_sq_hi);
                __m512d t_mag_sq_pd_lo = _mm512_cvtepi64_pd(t_mag_sq_lo);
                __m512d t_mag_sq_pd_hi = _mm512_cvtepi64_pd(t_mag_sq_hi);

                ot_dp_pd_lo = _mm512_mul_pd(ot_dp_pd_lo, inv_4096);
                ot_dp_pd_hi = _mm512_mul_pd(ot_dp_pd_hi, inv_4096);
                o_mag_sq_pd_lo = _mm512_mul_pd(o_mag_sq_pd_lo, inv_4096);
                o_mag_sq_pd_hi = _mm512_mul_pd(o_mag_sq_pd_hi, inv_4096);
                t_mag_sq_pd_lo = _mm512_mul_pd(t_mag_sq_pd_lo, inv_4096);
                t_mag_sq_pd_hi = _mm512_mul_pd(t_mag_sq_pd_hi, inv_4096);               

                __mmask8 gt_lo_0 = _mm512_cmp_pd_mask(ot_dp_pd_lo, _mm512_setzero_pd(), 13);
                __mmask8 gt_hi_0 = _mm512_cmp_pd_mask(ot_dp_pd_hi, _mm512_setzero_pd(), 13);

                __m512d ot_dp_sq_pd_lo = _mm512_mul_pd(ot_dp_pd_lo, ot_dp_pd_lo);
                __m512d ot_dp_sq_pd_hi = _mm512_mul_pd(ot_dp_pd_hi, ot_dp_pd_hi);

                __m512d o_mag_sq_t_mag_sq_lo = _mm512_mul_pd(o_mag_sq_pd_lo, t_mag_sq_pd_lo);
                __m512d o_mag_sq_t_mag_sq_hi = _mm512_mul_pd(o_mag_sq_pd_hi, t_mag_sq_pd_hi);
                __m512d ot_mag_sq_cos1_lo = _mm512_mul_pd(o_mag_sq_t_mag_sq_lo, _mm512_set1_pd(cos_1deg_sq));
                __m512d ot_mag_sq_cos1_hi = _mm512_mul_pd(o_mag_sq_t_mag_sq_hi, _mm512_set1_pd(cos_1deg_sq));
                __mmask8 cmp_ot_cosot_mag_lo = _mm512_cmp_pd_mask(ot_dp_sq_pd_lo, ot_mag_sq_cos1_lo, 13);
                __mmask8 cmp_ot_cosot_mag_hi = _mm512_cmp_pd_mask(ot_dp_sq_pd_hi, ot_mag_sq_cos1_hi, 13);
                
                __mmask8 angle_flag_lo = _kand_mask8(gt_lo_0, cmp_ot_cosot_mag_lo);
                __mmask8 angle_flag_hi = _kand_mask8(gt_hi_0, cmp_ot_cosot_mag_hi);
                
#else
                __mmask8 angle_flag_lo = 0;
                __mmask8 angle_flag_hi = 0;

                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 0), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 0), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 0), 0), angle_flag_lo, 0);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 0), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 0), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 0), 1), angle_flag_lo, 1);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 0), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 0), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 0), 2), angle_flag_lo, 2);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 0), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 0), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 0), 3), angle_flag_lo, 3);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 1), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 1), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 1), 0), angle_flag_lo, 4);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 1), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 1), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 1), 1), angle_flag_lo, 5);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 1), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 1), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 1), 2), angle_flag_lo, 6);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_lo, 1), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_lo, 1), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_lo, 1), 3), angle_flag_lo, 7);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 0), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 0), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 0), 0), angle_flag_hi, 0);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 0), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 0), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 0), 1), angle_flag_hi, 1);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 0), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 0), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 0), 2), angle_flag_hi, 2);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 0), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 0), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 0), 3), angle_flag_hi, 3);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 1), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 1), 0), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 1), 0), angle_flag_hi, 4);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 1), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 1), 1), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 1), 1), angle_flag_hi, 5);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 1), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 1), 2), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 1), 2), angle_flag_hi, 6);
                calc_angle(_mm256_extract_epi64(_mm512_extracti64x4_epi64(ot_dp_hi, 1), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(o_mag_sq_hi, 1), 3), _mm256_extract_epi64(_mm512_extracti64x4_epi64(t_mag_sq_hi, 1), 3), angle_flag_hi, 7);

                // int angle_flag3 = angle_flag_lo + (angle_flag_hi << 8);
#endif

                __m512i abs_oh = _mm512_abs_epi32(oh);
                __m512i abs_ov = _mm512_abs_epi32(ov);
                __m512i abs_od = _mm512_abs_epi32(od);

                __mmask16 kh_sign = _mm512_cmp_epi32_mask(oh, _mm512_setzero_si512(), 5);
                __mmask16 kv_sign = _mm512_cmp_epi32_mask(ov, _mm512_setzero_si512(), 5);
                __mmask16 kd_sign = _mm512_cmp_epi32_mask(od, _mm512_setzero_si512(), 5);

                __m512i kh_shift = _mm512_lzcnt_epi32(abs_oh);
                __m512i kv_shift = _mm512_lzcnt_epi32(abs_ov);
                __m512i kd_shift = _mm512_lzcnt_epi32(abs_od);

                kh_shift = _mm512_sub_epi32(_mm512_set1_epi32(17), kh_shift);
                kv_shift = _mm512_sub_epi32(_mm512_set1_epi32(17), kv_shift);
                kd_shift = _mm512_sub_epi32(_mm512_set1_epi32(17), kd_shift);

                kh_shift = _mm512_max_epi32(kh_shift, _mm512_setzero_si512());
                kd_shift = _mm512_max_epi32(kd_shift, _mm512_setzero_si512());
                kv_shift = _mm512_max_epi32(kv_shift, _mm512_setzero_si512());

                __m512i add_shift_oh = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_sub_epi32(kh_shift, _mm512_set1_epi32(1)));
                __m512i add_shift_ov = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_sub_epi32(kv_shift, _mm512_set1_epi32(1)));
                __m512i add_shift_od = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_sub_epi32(kd_shift, _mm512_set1_epi32(1)));

                __m512i tmp_kh = _mm512_srlv_epi32(_mm512_add_epi32(abs_oh, add_shift_oh), kh_shift);
                __m512i tmp_kv = _mm512_srlv_epi32(_mm512_add_epi32(abs_ov, add_shift_ov), kv_shift);
                __m512i tmp_kd = _mm512_srlv_epi32(_mm512_add_epi32(abs_od, add_shift_od), kd_shift);

                __mmask16 abs_oh_lt_32k = _mm512_cmp_epi32_mask(abs_oh, _mm512_set1_epi32(32768), 1);
                __mmask16 abs_ov_lt_32k = _mm512_cmp_epi32_mask(abs_ov, _mm512_set1_epi32(32768), 1);
                __mmask16 abs_od_lt_32k = _mm512_cmp_epi32_mask(abs_od, _mm512_set1_epi32(32768), 1); 

                __m512i kh_msb = _mm512_mask_blend_epi32(abs_oh_lt_32k, tmp_kh, abs_oh);
                __m512i kv_msb = _mm512_mask_blend_epi32(abs_ov_lt_32k, tmp_kv, abs_ov);
                __m512i kd_msb = _mm512_mask_blend_epi32(abs_od_lt_32k, tmp_kd, abs_od);

                kh_msb = _mm512_permutexvar_epi32(perm_for_mul, kh_msb);
                kv_msb = _mm512_permutexvar_epi32(perm_for_mul, kv_msb);
                kd_msb = _mm512_permutexvar_epi32(perm_for_mul, kd_msb);

                __m512i th_mul = _mm512_mullo_epi32(th, _mm512_set1_epi32(-1));
                __m512i tv_mul = _mm512_mullo_epi32(tv, _mm512_set1_epi32(-1));
                __m512i td_mul = _mm512_mullo_epi32(td, _mm512_set1_epi32(-1));

                th_mul = _mm512_mask_blend_epi32(kh_sign, th_mul, th);
                tv_mul = _mm512_mask_blend_epi32(kv_sign, tv_mul, tv);
                td_mul = _mm512_mask_blend_epi32(kd_sign, td_mul, td);

                th_mul = _mm512_permutexvar_epi32(perm_for_mul, th_mul);
                tv_mul = _mm512_permutexvar_epi32(perm_for_mul, tv_mul);
                td_mul = _mm512_permutexvar_epi32(perm_for_mul, td_mul);

                __m512i oh_div = _mm512_i32gather_epi32(_mm512_add_epi32(kh_msb, _mm512_set1_epi32(32768)), adm_div_lookup, 4);
                __m512i ov_div = _mm512_i32gather_epi32(_mm512_add_epi32(kv_msb, _mm512_set1_epi32(32768)), adm_div_lookup, 4);
                __m512i od_div = _mm512_i32gather_epi32(_mm512_add_epi32(kd_msb, _mm512_set1_epi32(32768)), adm_div_lookup, 4);

                __m512i kh_shift_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kh_shift, 0));
                __m512i kh_shift_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kh_shift, 1));
                __m512i kv_shift_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kv_shift, 0));
                __m512i kv_shift_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kv_shift, 1));
                __m512i kd_shift_lo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kd_shift, 0));
                __m512i kd_shift_hi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(kd_shift, 1));

                __m512i add_kh_lo = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kh_shift_lo, _mm512_set1_epi64(14)));
                __m512i add_kh_hi = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kh_shift_hi, _mm512_set1_epi64(14)));
                __m512i add_kv_lo = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kv_shift_lo, _mm512_set1_epi64(14)));
                __m512i add_kv_hi = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kv_shift_hi, _mm512_set1_epi64(14)));
                __m512i add_kd_lo = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kd_shift_lo, _mm512_set1_epi64(14)));
                __m512i add_kd_hi = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_add_epi64(kd_shift_hi, _mm512_set1_epi64(14)));

                __m512i oh_div_th_lo = _mm512_add_epi64(_mm512_mul_epi32(oh_div, th_mul), add_kh_lo);
                __m512i oh_div_th_hi = _mm512_add_epi64(_mm512_mul_epi32(_mm512_srli_epi64(oh_div, 32), _mm512_srli_epi64(th_mul, 32)), add_kh_hi);
                __m512i ov_div_th_lo = _mm512_add_epi64(_mm512_mul_epi32(ov_div, tv_mul), add_kv_lo);
                __m512i ov_div_th_hi = _mm512_add_epi64(_mm512_mul_epi32(_mm512_srli_epi64(ov_div, 32), _mm512_srli_epi64(tv_mul, 32)), add_kv_hi);
                __m512i od_div_th_lo = _mm512_add_epi64(_mm512_mul_epi32(od_div, td_mul), add_kd_lo);
                __m512i od_div_th_hi = _mm512_add_epi64(_mm512_mul_epi32(_mm512_srli_epi64(od_div, 32), _mm512_srli_epi64(td_mul, 32)), add_kd_hi);

                __m512i kh_shift_lo_p_15 = _mm512_add_epi64(kh_shift_lo, _mm512_set1_epi64(15));
                __m512i kh_shift_hi_p_15 = _mm512_add_epi64(kh_shift_hi, _mm512_set1_epi64(15));
                __m512i kv_shift_lo_p_15 = _mm512_add_epi64(kv_shift_lo, _mm512_set1_epi64(15));
                __m512i kv_shift_hi_p_15 = _mm512_add_epi64(kv_shift_hi, _mm512_set1_epi64(15));
                __m512i kd_shift_lo_p_15 = _mm512_add_epi64(kd_shift_lo, _mm512_set1_epi64(15));
                __m512i kd_shift_hi_p_15 = _mm512_add_epi64(kd_shift_hi, _mm512_set1_epi64(15));

                __mmask8 kh_sign_lo = _mm512_cmp_epi64_mask(oh_div_th_lo, _mm512_setzero_si512(), 5);
                __mmask8 kh_sign_hi = _mm512_cmp_epi64_mask(oh_div_th_hi, _mm512_setzero_si512(), 5);
                __mmask8 kv_sign_lo = _mm512_cmp_epi64_mask(ov_div_th_lo, _mm512_setzero_si512(), 5);
                __mmask8 kv_sign_hi = _mm512_cmp_epi64_mask(ov_div_th_hi, _mm512_setzero_si512(), 5);
                __mmask8 kd_sign_lo = _mm512_cmp_epi64_mask(od_div_th_lo, _mm512_setzero_si512(), 5);
                __mmask8 kd_sign_hi = _mm512_cmp_epi64_mask(od_div_th_hi, _mm512_setzero_si512(), 5);
                
                oh_div_th_lo = _mm512_srlv_epi64(oh_div_th_lo, kh_shift_lo_p_15);
                oh_div_th_hi = _mm512_srlv_epi64(oh_div_th_hi, kh_shift_hi_p_15);
                ov_div_th_lo = _mm512_srlv_epi64(ov_div_th_lo, kv_shift_lo_p_15);
                ov_div_th_hi = _mm512_srlv_epi64(ov_div_th_hi, kv_shift_hi_p_15);
                od_div_th_lo = _mm512_srlv_epi64(od_div_th_lo, kd_shift_lo_p_15);
                od_div_th_hi = _mm512_srlv_epi64(od_div_th_hi, kd_shift_hi_p_15);    

                __m512i msb_kh_lo = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kh_shift_lo_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
                __m512i msb_kh_hi = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kh_shift_hi_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
                __m512i msb_kv_lo = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kv_shift_lo_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
                __m512i msb_kv_hi = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kv_shift_hi_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
                __m512i msb_kd_lo = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kd_shift_lo_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
                __m512i msb_kd_hi = _mm512_andnot_si512(_mm512_srlv_epi64(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF), kd_shift_hi_p_15), _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));

                oh_div_th_lo = _mm512_or_si512(oh_div_th_lo, _mm512_mask_blend_epi64(kh_sign_lo, msb_kh_lo, _mm512_setzero_si512()));
                oh_div_th_hi = _mm512_or_si512(oh_div_th_hi, _mm512_mask_blend_epi64(kh_sign_hi, msb_kh_hi, _mm512_setzero_si512()));
                ov_div_th_lo = _mm512_or_si512(ov_div_th_lo, _mm512_mask_blend_epi64(kv_sign_lo, msb_kv_lo, _mm512_setzero_si512()));
                ov_div_th_hi = _mm512_or_si512(ov_div_th_hi, _mm512_mask_blend_epi64(kv_sign_hi, msb_kv_hi, _mm512_setzero_si512()));
                od_div_th_lo = _mm512_or_si512(od_div_th_lo, _mm512_mask_blend_epi64(kd_sign_lo, msb_kd_lo, _mm512_setzero_si512()));
                od_div_th_hi = _mm512_or_si512(od_div_th_hi, _mm512_mask_blend_epi64(kd_sign_hi, msb_kd_hi, _mm512_setzero_si512()));

                __m512i tmp_kh_lo = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(oh_lo, _mm512_setzero_si512(), 0), oh_div_th_lo, _mm512_set1_epi64(32768));
                __m512i tmp_kh_hi = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(oh_hi, _mm512_setzero_si512(), 0), oh_div_th_hi, _mm512_set1_epi64(32768));
                __m512i tmp_kd_lo = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(od_lo, _mm512_setzero_si512(), 0), od_div_th_lo, _mm512_set1_epi64(32768));
                __m512i tmp_kd_hi = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(od_hi, _mm512_setzero_si512(), 0), od_div_th_hi, _mm512_set1_epi64(32768));
                __m512i tmp_kv_lo = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(ov_lo, _mm512_setzero_si512(), 0), ov_div_th_lo, _mm512_set1_epi64(32768));
                __m512i tmp_kv_hi = _mm512_mask_blend_epi64(_mm512_cmp_epi64_mask(ov_hi, _mm512_setzero_si512(), 0), ov_div_th_hi, _mm512_set1_epi64(32768));

                tmp_kh_lo = _mm512_min_epi64(_mm512_max_epi64(tmp_kh_lo, _mm512_setzero_si512()), _mm512_set1_epi64(32768));
                tmp_kh_hi = _mm512_min_epi64(_mm512_max_epi64(tmp_kh_hi, _mm512_setzero_si512()), _mm512_set1_epi64(32768));
                tmp_kd_lo = _mm512_min_epi64(_mm512_max_epi64(tmp_kd_lo, _mm512_setzero_si512()), _mm512_set1_epi64(32768));
                tmp_kd_hi = _mm512_min_epi64(_mm512_max_epi64(tmp_kd_hi, _mm512_setzero_si512()), _mm512_set1_epi64(32768));
                tmp_kv_lo = _mm512_min_epi64(_mm512_max_epi64(tmp_kv_lo, _mm512_setzero_si512()), _mm512_set1_epi64(32768));
                tmp_kv_hi = _mm512_min_epi64(_mm512_max_epi64(tmp_kv_hi, _mm512_setzero_si512()), _mm512_set1_epi64(32768));

                __m512i const_16384_64b = _mm512_set1_epi64(16384);

                __m512i rst_h_lo = _mm512_add_epi64(_mm512_mul_epi32(tmp_kh_lo, oh_lo), const_16384_64b);
                __m512i rst_h_hi = _mm512_add_epi64(_mm512_mul_epi32(tmp_kh_hi, oh_hi), const_16384_64b);
                __m512i rst_d_lo = _mm512_add_epi64(_mm512_mul_epi32(tmp_kd_lo, od_lo), const_16384_64b);
                __m512i rst_d_hi = _mm512_add_epi64(_mm512_mul_epi32(tmp_kd_hi, od_hi), const_16384_64b);
                __m512i rst_v_lo = _mm512_add_epi64(_mm512_mul_epi32(tmp_kv_lo, ov_lo), const_16384_64b);
                __m512i rst_v_hi = _mm512_add_epi64(_mm512_mul_epi32(tmp_kv_hi, ov_hi), const_16384_64b);

                shift15_64b_signExt_512(rst_h_lo, rst_h_lo);
                shift15_64b_signExt_512(rst_h_hi, rst_h_hi);
                // rst_h_hi = _mm512_srai_epi64(rst_h_hi, 15);
                shift15_64b_signExt_512(rst_d_lo, rst_d_lo);
                shift15_64b_signExt_512(rst_d_hi, rst_d_hi);
                shift15_64b_signExt_512(rst_v_lo, rst_v_lo);
                shift15_64b_signExt_512(rst_v_hi, rst_v_hi);

                __m512d inv_32768 = _mm512_set1_pd((double)1/32768);
                __m512d inv_64 = _mm512_set1_pd((double)1/64);

                __m512d rst_h_f_lo = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kh_lo), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(oh_lo), inv_64));
                __m512d rst_h_f_hi = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kh_hi), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(oh_hi), inv_64));
                __m512d rst_d_f_lo = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kd_lo), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(od_lo), inv_64));
                __m512d rst_d_f_hi = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kd_hi), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(od_hi), inv_64));
                __m512d rst_v_f_lo = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kv_lo), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(ov_lo), inv_64));
                __m512d rst_v_f_hi = _mm512_mul_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(tmp_kv_hi), inv_32768), _mm512_mul_pd(_mm512_cvtepi64_pd(ov_hi), inv_64));

                __m512i const_adm_enhn_gain = _mm512_set1_epi64((int)adm_enhn_gain_limit);

                __m512i rst_h_gain_lo = _mm512_mul_epi32(rst_h_lo, const_adm_enhn_gain);
                __m512i rst_h_gain_hi = _mm512_mul_epi32(rst_h_hi, const_adm_enhn_gain);
                __m512i rst_v_gain_lo = _mm512_mul_epi32(rst_v_lo, const_adm_enhn_gain);
                __m512i rst_v_gain_hi = _mm512_mul_epi32(rst_v_hi, const_adm_enhn_gain);
                __m512i rst_d_gain_lo = _mm512_mul_epi32(rst_d_lo, const_adm_enhn_gain);
                __m512i rst_d_gain_hi = _mm512_mul_epi32(rst_d_hi, const_adm_enhn_gain);

                __m512i h_min_lo = _mm512_min_epi64(rst_h_gain_lo, th_lo);
                __m512i h_min_hi = _mm512_min_epi64(rst_h_gain_hi, th_hi);
                __m512i h_max_lo = _mm512_max_epi64(rst_h_gain_lo, th_lo);
                __m512i h_max_hi = _mm512_max_epi64(rst_h_gain_hi, th_hi);
                __m512i v_min_lo = _mm512_min_epi64(rst_v_gain_lo, tv_lo);
                __m512i v_min_hi = _mm512_min_epi64(rst_v_gain_hi, tv_hi);
                __m512i v_max_lo = _mm512_max_epi64(rst_v_gain_lo, tv_lo);
                __m512i v_max_hi = _mm512_max_epi64(rst_v_gain_hi, tv_hi);
                __m512i d_min_lo = _mm512_min_epi64(rst_d_gain_lo, td_lo);
                __m512i d_min_hi = _mm512_min_epi64(rst_d_gain_hi, td_hi);
                __m512i d_max_lo = _mm512_max_epi64(rst_d_gain_lo, td_lo);
                __m512i d_max_hi = _mm512_max_epi64(rst_d_gain_hi, td_hi);

                __mmask8 gt0_rst_h_f_lo = _mm512_cmp_pd_mask(rst_h_f_lo, _mm512_setzero_pd(), 14);
                __mmask8 gt0_rst_h_f_hi = _mm512_cmp_pd_mask(rst_h_f_hi, _mm512_setzero_pd(), 14);
                __mmask8 lt0_rst_h_f_lo = _mm512_cmp_pd_mask(rst_h_f_lo, _mm512_setzero_pd(), 1);
                __mmask8 lt0_rst_h_f_hi = _mm512_cmp_pd_mask(rst_h_f_hi, _mm512_setzero_pd(), 1);
                
                __mmask8 gt0_rst_v_f_lo = _mm512_cmp_pd_mask(rst_v_f_lo, _mm512_setzero_pd(), 14);
                __mmask8 gt0_rst_v_f_hi = _mm512_cmp_pd_mask(rst_v_f_hi, _mm512_setzero_pd(), 14);
                __mmask8 lt0_rst_v_f_lo = _mm512_cmp_pd_mask(rst_v_f_lo, _mm512_setzero_pd(), 1);
                __mmask8 lt0_rst_v_f_hi = _mm512_cmp_pd_mask(rst_v_f_hi, _mm512_setzero_pd(), 1);

                __mmask8 gt0_rst_d_f_lo = _mm512_cmp_pd_mask(rst_d_f_lo, _mm512_setzero_pd(), 14);
                __mmask8 gt0_rst_d_f_hi = _mm512_cmp_pd_mask(rst_d_f_hi, _mm512_setzero_pd(), 14);
                __mmask8 lt0_rst_d_f_lo = _mm512_cmp_pd_mask(rst_d_f_lo, _mm512_setzero_pd(), 1);
                __mmask8 lt0_rst_d_f_hi = _mm512_cmp_pd_mask(rst_d_f_hi, _mm512_setzero_pd(), 1);

                h_min_lo = _mm512_mask_blend_epi64(gt0_rst_h_f_lo, _mm512_setzero_epi32(), h_min_lo);
                h_min_hi = _mm512_mask_blend_epi64(gt0_rst_h_f_hi, _mm512_setzero_epi32(), h_min_hi);
                h_max_lo = _mm512_mask_blend_epi64(lt0_rst_h_f_lo, _mm512_setzero_epi32(), h_max_lo);
                h_max_hi = _mm512_mask_blend_epi64(lt0_rst_h_f_hi, _mm512_setzero_epi32(), h_max_hi);
                v_min_lo = _mm512_mask_blend_epi64(gt0_rst_v_f_lo, _mm512_setzero_epi32(), v_min_lo);
                v_min_hi = _mm512_mask_blend_epi64(gt0_rst_v_f_hi, _mm512_setzero_epi32(), v_min_hi);
                v_max_lo = _mm512_mask_blend_epi64(lt0_rst_v_f_lo, _mm512_setzero_epi32(), v_max_lo);
                v_max_hi = _mm512_mask_blend_epi64(lt0_rst_v_f_hi, _mm512_setzero_epi32(), v_max_hi);
                d_min_lo = _mm512_mask_blend_epi64(gt0_rst_d_f_lo, _mm512_setzero_epi32(), d_min_lo);
                d_min_hi = _mm512_mask_blend_epi64(gt0_rst_d_f_hi, _mm512_setzero_epi32(), d_min_hi);
                d_max_lo = _mm512_mask_blend_epi64(lt0_rst_d_f_lo, _mm512_setzero_epi32(), d_max_lo);
                d_max_hi = _mm512_mask_blend_epi64(lt0_rst_d_f_hi, _mm512_setzero_epi32(), d_max_hi);

                __mmask8 mask_min_max_h_lo = _kor_mask8(gt0_rst_h_f_lo, lt0_rst_h_f_lo);
                __mmask8 mask_min_max_h_hi = _kor_mask8(gt0_rst_h_f_hi, lt0_rst_h_f_hi);
                __mmask8 mask_min_max_v_lo = _kor_mask8(gt0_rst_v_f_lo, lt0_rst_v_f_lo);
                __mmask8 mask_min_max_v_hi = _kor_mask8(gt0_rst_v_f_hi, lt0_rst_v_f_hi);
                __mmask8 mask_min_max_d_lo = _kor_mask8(gt0_rst_d_f_lo, lt0_rst_d_f_lo);
                __mmask8 mask_min_max_d_hi = _kor_mask8(gt0_rst_d_f_hi, lt0_rst_d_f_hi);
            
                __m512i h_min_max_lo = _mm512_or_si512(h_min_lo, h_max_lo);
                __m512i h_min_max_hi = _mm512_or_si512(h_min_hi, h_max_hi);
                __m512i v_min_max_lo = _mm512_or_si512(v_min_lo, v_max_lo);
                __m512i v_min_max_hi = _mm512_or_si512(v_min_hi, v_max_hi);
                __m512i d_min_max_lo = _mm512_or_si512(d_min_lo, d_max_lo);
                __m512i d_min_max_hi = _mm512_or_si512(d_min_hi, d_max_hi);
                
                h_min_max_lo = _mm512_mask_blend_epi64(mask_min_max_h_lo, rst_h_lo, h_min_max_lo);
                h_min_max_hi = _mm512_mask_blend_epi64(mask_min_max_h_hi, rst_h_hi, h_min_max_hi);
                v_min_max_lo = _mm512_mask_blend_epi64(mask_min_max_v_lo, rst_v_lo, v_min_max_lo);
                v_min_max_hi = _mm512_mask_blend_epi64(mask_min_max_v_hi, rst_v_hi, v_min_max_hi);
                d_min_max_lo = _mm512_mask_blend_epi64(mask_min_max_d_lo, rst_d_lo, d_min_max_lo);
                d_min_max_hi = _mm512_mask_blend_epi64(mask_min_max_d_hi, rst_d_hi, d_min_max_hi);

                rst_h_lo = _mm512_mask_blend_epi64(angle_flag_lo, rst_h_lo, h_min_max_lo);
                rst_h_hi = _mm512_mask_blend_epi64(angle_flag_hi, rst_h_hi, h_min_max_hi);
                rst_v_lo = _mm512_mask_blend_epi64(angle_flag_lo, rst_v_lo, v_min_max_lo);
                rst_v_hi = _mm512_mask_blend_epi64(angle_flag_hi, rst_v_hi, v_min_max_hi);
                rst_d_lo = _mm512_mask_blend_epi64(angle_flag_lo, rst_d_lo, d_min_max_lo);
                rst_d_hi = _mm512_mask_blend_epi64(angle_flag_hi, rst_d_hi, d_min_max_hi);

                th_lo = _mm512_sub_epi64(th_lo, rst_h_lo);
                th_hi = _mm512_sub_epi64(th_hi, rst_h_hi);
                tv_lo = _mm512_sub_epi64(tv_lo, rst_v_lo);
                tv_hi = _mm512_sub_epi64(tv_hi, rst_v_hi);
                td_lo = _mm512_sub_epi64(td_lo, rst_d_lo);
                td_hi = _mm512_sub_epi64(td_hi, rst_d_hi);

                _mm512_storeu_si512((__m512i*)(r->band_h + i * stride + j), _mm512_permutex2var_epi32(rst_h_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , rst_h_hi));
                _mm512_storeu_si512((__m512i*)(r->band_v + i * stride + j), _mm512_permutex2var_epi32(rst_v_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , rst_v_hi));
                _mm512_storeu_si512((__m512i*)(r->band_d + i * stride + j), _mm512_permutex2var_epi32(rst_d_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , rst_d_hi));
                _mm512_storeu_si512((__m512i*)(a->band_h + i * stride + j), _mm512_permutex2var_epi32(th_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , th_hi));
                _mm512_storeu_si512((__m512i*)(a->band_v + i * stride + j), _mm512_permutex2var_epi32(tv_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , tv_hi));
                _mm512_storeu_si512((__m512i*)(a->band_d + i * stride + j), _mm512_permutex2var_epi32(td_lo, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0) , td_hi));
        }

        for(int j = right_mod16; j < right; j++)
        {
            int32_t oh = ref->band_h[i * stride + j];
            int32_t ov = ref->band_v[i * stride + j];
            int32_t od = ref->band_d[i * stride + j];
            int32_t th = dis->band_h[i * stride + j];
            int32_t tv = dis->band_v[i * stride + j];
            int32_t td = dis->band_d[i * stride + j];
            int32_t rst_h, rst_v, rst_d;

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

            int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
                (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                    cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));

            /**
             * Division th/oh is carried using lookup table and converted to multiplication
             * int64 / int32 is converted to multiplication using following method
             * num /den :
             * DenAbs = Abs(den)
             * MSBDen = MSB(DenAbs)     (gives position of first 1 bit form msb side)
             * If (DenAbs < (1 << 15))
             *      Round = (1<<14)
             *      Score = (num *  div_lookup[den] + Round ) >> 15
             * else
             *      RoundD  = (1<< (16 - MSBDen))
             *      Round   = (1<< (14 + (17 - MSBDen))
             *      Score   = (num * div_lookup[(DenAbs + RoundD )>>(17 - MSBDen)]*sign(Denominator) + Round)
             *                  >> ((15 + (17 - MSBDen))
             */

            int32_t kh_shift = 0;
            int32_t kv_shift = 0;
            int32_t kd_shift = 0;

            uint32_t abs_oh = abs(oh);
            uint32_t abs_ov = abs(ov);
            uint32_t abs_od = abs(od);

            int8_t kh_sign = (oh < 0 ? -1 : 1);
            int8_t kv_sign = (ov < 0 ? -1 : 1);
            int8_t kd_sign = (od < 0 ? -1 : 1);

            uint16_t kh_msb = (abs_oh < (32768) ? abs_oh : get_best15_from32(abs_oh, &kh_shift));
            uint16_t kv_msb = (abs_ov < (32768) ? abs_ov : get_best15_from32(abs_ov, &kv_shift));
            uint16_t kd_msb = (abs_od < (32768) ? abs_od : get_best15_from32(abs_od, &kd_shift));

            int64_t tmp_kh = (oh == 0) ? 32768 : (((int64_t)adm_div_lookup[kh_msb + 32768] * th)*(kh_sign) +
                (1 << (14 + kh_shift))) >> (15 + kh_shift);
            int64_t tmp_kv = (ov == 0) ? 32768 : (((int64_t)adm_div_lookup[kv_msb + 32768] * tv)*(kv_sign) +
                (1 << (14 + kv_shift))) >> (15 + kv_shift);
            int64_t tmp_kd = (od == 0) ? 32768 : (((int64_t)adm_div_lookup[kd_msb + 32768] * td)*(kd_sign) +
                (1 << (14 + kd_shift))) >> (15 + kd_shift);

            int64_t kh = tmp_kh < 0 ? 0 : (tmp_kh > 32768 ? 32768 : tmp_kh);
            int64_t kv = tmp_kv < 0 ? 0 : (tmp_kv > 32768 ? 32768 : tmp_kv);
            int64_t kd = tmp_kd < 0 ? 0 : (tmp_kd > 32768 ? 32768 : tmp_kd);

            rst_h = ((kh * oh) + 16384) >> 15;
            rst_v = ((kv * ov) + 16384) >> 15;
            rst_d = ((kd * od) + 16384) >> 15;

            const float rst_h_f = ((float)kh / 32768) * ((float)oh / 64);
            const float rst_v_f = ((float)kv / 32768) * ((float)ov / 64);
            const float rst_d_f = ((float)kd / 32768) * ((float)od / 64);

            if (angle_flag && (rst_h_f > 0.)) rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
            if (angle_flag && (rst_h_f < 0.)) rst_h = MAX((rst_h * adm_enhn_gain_limit), th);

            if (angle_flag && (rst_v_f > 0.)) rst_v = MIN(rst_v * adm_enhn_gain_limit, tv);
            if (angle_flag && (rst_v_f < 0.)) rst_v = MAX(rst_v * adm_enhn_gain_limit, tv);

            if (angle_flag && (rst_d_f > 0.)) rst_d = MIN(rst_d * adm_enhn_gain_limit, td);
            if (angle_flag && (rst_d_f < 0.)) rst_d = MAX(rst_d * adm_enhn_gain_limit, td);

            r->band_h[i * stride + j] = rst_h;
            r->band_v[i * stride + j] = rst_v;
            r->band_d[i * stride + j] = rst_d;

            a->band_h[i * stride + j] = th - rst_h;
            a->band_v[i * stride + j] = tv - rst_v;
            a->band_d[i * stride + j] = td - rst_d;
        }
    }
}

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static inline float
dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
        double adm_norm_view_dist, int adm_ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree
    // of visual angle. This should be 56.55
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0, lambda + 1)*params->f0*params->g[theta] / r);
    float Q = 2.0*params->a*pow(10.0, params->k*temp*temp) /
        dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

float adm_cm_avx512(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                    double adm_norm_view_dist, int adm_ref_display_height)
{
    const adm_dwt_band_t *src   = &buf->decouple_r;
    const adm_dwt_band_t *csf_f = &buf->csf_f;
    const adm_dwt_band_t *csf_a = &buf->csf_a;

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    // 0 is scale zero passed to dwt_quant_step

    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    /**
     * rfactor is converted to fixed-point for scale0 and stored in i_rfactor
     * multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2].
     * For adm_norm_view_dist 3.0 and adm_ref_display_height 1080,
     * i_rfactor is around { 36453,36453,49417 }
     */
    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    }
    else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t) (rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t) (rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t) (rfactor1[2] * pow2_23);
    }

    const int32_t shift_xhsq = 29;
    const int32_t shift_xvsq = 29;
    const int32_t shift_xdsq = 30;
    const int32_t add_shift_xhsq = 268435456;
    const int32_t add_shift_xvsq = 268435456;
    const int32_t add_shift_xdsq = 536870912;

    const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xhcub = (uint32_t)pow(2, (shift_xhcub - 1));

    const uint32_t shift_xvcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t add_shift_xvcub = (uint32_t)pow(2, (shift_xvcub - 1));

    const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
    const uint32_t add_shift_xdcub = (uint32_t)pow(2, (shift_xdcub - 1));

    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    const uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    const int32_t shift_xhsub = 10;
    const int32_t shift_xvsub = 10;
    const int32_t shift_xdsub = 12;

    int16_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int16_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    /* The computation of the scales is not required for the regions which lie
     * outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int end_col_mod14 = end_col - ((end_col - start_col) % 14);

    int i, j;
    int64_t val;
    int32_t xh, xv, xd, thr;
    int32_t xh_sq, xv_sq, xd_sq;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;

    __m512i thr_512;
    __m512i xh_512, xv_512, xd_512;
    __m512i accum_inner_h_lo_512, accum_inner_h_hi_512, accum_inner_v_lo_512, \
    accum_inner_v_hi_512, accum_inner_d_lo_512, accum_inner_d_hi_512;

    /* i=0,j=0 */
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)src->band_h[0] * i_rfactor[0];
        xv = (int32_t)src->band_v[0] * i_rfactor[1];
        xd = (int32_t)src->band_d[0] * i_rfactor[2];
        ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_a_stride, &thr, w, h, 0, 0);

        //thr is shifted to make it's Q format equivalent to xh,xv,xd

        /**
         * max value of xh_sq and xv_sq is 1301381973 and that of xd_sq is 1195806729
         *
         * max(val before shift for h and v) is 9.995357299 * —10^17.
         * 9.995357299 * —10^17 * 2^4 is close to 2^64.
         * Hence shift is done based on width subtracting 4
         *
         * max(val before shift for d) is 1.355006643 * —10^18
         * 1.355006643 * —10^18 * 2^3 is close to 2^64
         * Hence shift is done based on width subtracting 3
         */
        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }

    /* i=0, j */
    if (top <= 0) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[j] * i_rfactor[0];
            xv = src->band_v[j] * i_rfactor[1];
            xd = src->band_d[j] * i_rfactor[2];
            ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_a_stride, &thr, w, h, 0, j);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);
        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = src->band_h[w - 1] * i_rfactor[0];
        xv = src->band_v[w - 1] * i_rfactor[1];
        xd = src->band_d[w - 1] * i_rfactor[2];
        ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, 0, (w - 1));

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }
    //Shift is done based on height
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        __m512i i_rfactor0 = _mm512_maskz_set1_epi32(0x3FFF, i_rfactor[0]);
        __m512i i_rfactor1 = _mm512_maskz_set1_epi32(0x3FFF, i_rfactor[1]);
        __m512i i_rfactor2 = _mm512_maskz_set1_epi32(0x3FFF, i_rfactor[2]);  
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            accum_inner_h_lo_512 = accum_inner_h_hi_512 = _mm512_setzero_si512();
            accum_inner_v_lo_512 = accum_inner_v_hi_512 = _mm512_setzero_si512();
            accum_inner_d_lo_512 = accum_inner_d_hi_512 = _mm512_setzero_si512();

            for (j = start_col; j < end_col_mod14; j += 14) {
                xh_512 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src->band_h + i * src_stride + j)));
                xv_512 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src->band_v + i * src_stride + j)));
                xd_512 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src->band_d + i * src_stride + j)));

                xh_512 = _mm512_mullo_epi32(xh_512, i_rfactor0);
                xv_512 = _mm512_mullo_epi32(xv_512, i_rfactor1);
                xd_512 = _mm512_mullo_epi32(xd_512, i_rfactor2);

                ADM_CM_THRESH_S_I_J_avx512(angles, flt_angles, csf_a_stride, &thr, w, h, i, j, &thr_512);

                ADM_CM_ACCUM_ROUND_avx512(xh_512, thr_512, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                add_shift_xhcub, shift_xhcub, accum_inner_h_lo_512, accum_inner_h_hi_512);
                ADM_CM_ACCUM_ROUND_avx512(xv_512, thr_512, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                add_shift_xvcub, shift_xvcub, accum_inner_v_lo_512, accum_inner_v_hi_512);
                ADM_CM_ACCUM_ROUND_avx512(xd_512, thr_512, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                add_shift_xdcub, shift_xdcub, accum_inner_d_lo_512, accum_inner_d_hi_512);
            }            
            accum_inner_h_lo_512 = _mm512_add_epi64(accum_inner_h_lo_512, accum_inner_h_hi_512);
            __m256i r4_h = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_h_lo_512), _mm512_extracti64x4_epi64(accum_inner_h_lo_512, 1));
            __m128i r2_h = _mm_add_epi64(_mm256_castsi256_si128(r4_h), _mm256_extracti128_si256(r4_h, 1));
            int64_t res_h = r2_h[0] + r2_h[1];

            accum_inner_v_lo_512 = _mm512_add_epi64(accum_inner_v_lo_512, accum_inner_v_hi_512);
            __m256i r4_v = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_v_lo_512), _mm512_extracti64x4_epi64(accum_inner_v_lo_512, 1));
            __m128i r2_v = _mm_add_epi64(_mm256_castsi256_si128(r4_v), _mm256_extracti128_si256(r4_v, 1));
            int64_t res_v = r2_v[0] + r2_v[1];
            
            accum_inner_d_lo_512 = _mm512_add_epi64(accum_inner_d_lo_512, accum_inner_d_hi_512);
            __m256i r4_d = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_d_lo_512), _mm512_extracti64x4_epi64(accum_inner_d_lo_512, 1));
            __m128i r2_d = _mm_add_epi64(_mm256_castsi256_si128(r4_d), _mm256_extracti128_si256(r4_d, 1));
            int64_t res_d = r2_d[0] + r2_d[1];

            for (j = end_col_mod14; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];

                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);
                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                    
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }

            accum_h += (accum_inner_h + res_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + res_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + res_d + add_shift_inner_accum) >> shift_inner_accum;
        }        
    }
    else if ((left <= 0) && (right <= (w - 1))) /* Right border within frame, left outside */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_stride] * i_rfactor[0];
            xv = src->band_v[i * src_stride] * i_rfactor[1];
            xd = src->band_d[i * src_stride] * i_rfactor[2];
            ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            /* j = w-1 */
            xh = src->band_h[i * src_stride + w - 1] * i_rfactor[0];
            xv = src->band_v[i * src_stride + w - 1] * i_rfactor[1];
            xd = src->band_d[i * src_stride + w - 1] * i_rfactor[2];
            ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1));

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

        }
    }
    else /* Both borders outside frame */
    {
        for (i = start_row; i < end_row; ++i) {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;

            /* j = 0 */
            xh = src->band_h[i * src_stride] * i_rfactor[0];
            xv = src->band_v[i * src_stride] * i_rfactor[1];
            xd = src->band_d[i * src_stride] * i_rfactor[2];
            ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j) {
                xh = src->band_h[i * src_stride + j] * i_rfactor[0];
                xv = src->band_v[i * src_stride + j] * i_rfactor[1];
                xd = src->band_d[i * src_stride + j] * i_rfactor[2];
                ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j);

                ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                                   add_shift_xhcub, shift_xhcub, accum_inner_h);
                ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                                   add_shift_xvcub, shift_xvcub, accum_inner_v);
                ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                                   add_shift_xdcub, shift_xdcub, accum_inner_d);
            }
            /* j = w-1 */
            xh = src->band_h[i * src_stride + w - 1] * i_rfactor[0];
            xv = src->band_v[i * src_stride + w - 1] * i_rfactor[1];
            xd = src->band_d[i * src_stride + w - 1] * i_rfactor[2];
            ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1));

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = src->band_h[(h - 1) * src_stride] * i_rfactor[0];
        xv = src->band_v[(h - 1) * src_stride] * i_rfactor[1];
        xd = src->band_d[(h - 1) * src_stride] * i_rfactor[2];
        ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), 0);

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }

    /* i=h-1,j */
    if (bottom > (h - 1)) {
        for (j = start_col; j < end_col; ++j) {
            xh = src->band_h[(h - 1) * src_stride + j] * i_rfactor[0];
            xv = src->band_v[(h - 1) * src_stride + j] * i_rfactor[1];
            xd = src->band_d[(h - 1) * src_stride + j] * i_rfactor[2];
            ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), j);

            ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                               add_shift_xhcub, shift_xhcub, accum_inner_h);
            ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                               add_shift_xvcub, shift_xvcub, accum_inner_v);
            ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                               add_shift_xdcub, shift_xdcub, accum_inner_d);
        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = src->band_h[(h - 1) * src_stride + w - 1] * i_rfactor[0];
        xv = src->band_v[(h - 1) * src_stride + w - 1] * i_rfactor[1];
        xd = src->band_d[(h - 1) * src_stride + w - 1] * i_rfactor[2];
        ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h,
            (h - 1), (w - 1));

        ADM_CM_ACCUM_ROUND(xh, thr, shift_xhsub, xh_sq, add_shift_xhsq, shift_xhsq, val,
                           add_shift_xhcub, shift_xhcub, accum_inner_h);
        ADM_CM_ACCUM_ROUND(xv, thr, shift_xvsub, xv_sq, add_shift_xvsq, shift_xvsq, val,
                           add_shift_xvcub, shift_xvcub, accum_inner_v);
        ADM_CM_ACCUM_ROUND(xd, thr, shift_xdsub, xd_sq, add_shift_xdsq, shift_xdsq, val,
                           add_shift_xdcub, shift_xdcub, accum_inner_d);
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    /**
     * For h and v total shifts pending from last stage is 6 rfactor[0,1] has 21 shifts
     * => after cubing (6+21)*3=81 after squaring shifted by 29
     * hence pending is 52-shift's done based on width and height
     *
     * For d total shifts pending from last stage is 6 rfactor[2] has 23 shifts
     * => after cubing (6+23)*3=87 after squaring shifted by 30
     * hence pending is 57-shift's done based on width and height
     */

    float f_accum_h = (float)((float)accum_h / pow(2, (52 - shift_xhcub - shift_inner_accum)));
    float f_accum_v = (float)(accum_v / pow(2, (52 - shift_xvcub - shift_inner_accum)));
    float f_accum_d = (float)(accum_d / pow(2, (57 - shift_xdcub - shift_inner_accum)));

    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) *
                        (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

float i4_adm_cm_avx512(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride, int scale,
                       double adm_norm_view_dist, int adm_ref_display_height)
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_r;
    const i4_adm_dwt_band_t *csf_f = &buf->i4_csf_f;
    const i4_adm_dwt_band_t *csf_a = &buf->i4_csf_a;

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    const uint32_t rfactor[3] = { (uint32_t)(rfactor1[0] * pow(2, 32)),
                                  (uint32_t)(rfactor1[1] * pow(2, 32)),
                                  (uint32_t)(rfactor1[2] * pow(2, 32)) };

    const uint32_t shift_dst[3] = { 28, 28, 28 };
    const uint32_t shift_flt[3] = { 32, 32, 32 };
    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    for (unsigned idx = 0; idx < 3; ++idx) {
        add_bef_shift_dst[idx] = (1u << (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1));

    }

    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

    uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    uint32_t add_shift_inner_accum = (uint32_t)pow(2, (shift_inner_accum - 1));

    float final_shift[3] = { pow(2,(45 - shift_cub - shift_inner_accum)),
                             pow(2,(39 - shift_cub - shift_inner_accum)),
                             pow(2,(36 - shift_cub - shift_inner_accum)) };

    const int32_t shift_sq = 30;
    const int32_t add_shift_sq = 536870912; //2^29
    const int32_t shift_sub = 0;
    int32_t *angles[3] = { csf_a->band_h, csf_a->band_v, csf_a->band_d };
    int32_t *flt_angles[3] = { csf_f->band_h, csf_f->band_v, csf_f->band_d };

    /* The computation of the scales is not required for the regions which lie
     * outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    const int start_col = (left > 1) ? left : 1;
    const int end_col = (right < (w - 1)) ? right : (w - 1);
    const int start_row = (top > 1) ? top : 1;
    const int end_row = (bottom < (h - 1)) ? bottom : (h - 1);

    int end_col_mod6 = end_col - ((end_col - start_col) % 6);

    int i, j;
    int32_t xh, xv, xd, thr;
    int32_t xh_sq, xv_sq, xd_sq;
    int64_t val;
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    int64_t accum_inner_h = 0, accum_inner_v = 0, accum_inner_d = 0;

    __m512i thr_512;
    __m512i xh_512, xv_512, xd_512;
    __m512i accum_inner_h_512, accum_inner_v_512, accum_inner_d_512;

    /* i=0,j=0 */
    if ((top <= 0) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[0] * rfactor[0]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[0] * rfactor[1]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[0] * rfactor[2]) + add_bef_shift_dst[scale - 1])
            >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_0_0(angles, flt_angles, csf_a_stride, &thr, w, h, 0, 0,
                                       add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    /* i=0, j */
    if (top <= 0)
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[j] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[j] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[j] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_0_J(angles, flt_angles, csf_a_stride, &thr, w, h,
                                           0, j, add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);
        }
    }

    /* i=0,j=w-1 */
    if ((top <= 0) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[w - 1] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[w - 1] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[w - 1] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_0_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, 0, (w - 1),
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    if ((left > 0) && (right <= (w - 1))) /* Completely within frame */
    {
        for (i = start_row; i < end_row; ++i)
        {
            accum_inner_h = 0;
            accum_inner_v = 0;
            accum_inner_d = 0;
            
            accum_inner_h_512 = _mm512_setzero_si512();
            accum_inner_v_512 = _mm512_setzero_si512();
            accum_inner_d_512 = _mm512_setzero_si512();

            for (j = start_col; j < end_col_mod6; j+=6)
            {
                xh_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src->band_h + i * src_stride + j)));
                xv_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src->band_v + i * src_stride + j)));
                xd_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(src->band_d + i * src_stride + j)));
                
                __m512i rfactor0 = _mm512_maskz_set1_epi64(0x3F, rfactor[0]);
                __m512i rfactor1 = _mm512_maskz_set1_epi64(0x3F, rfactor[1]);
                __m512i rfactor2 = _mm512_maskz_set1_epi64(0x3F, rfactor[2]);
                __m512i add_shift = _mm512_set1_epi64(add_bef_shift_dst[scale - 1]);

                xh_512 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(xh_512, rfactor0), add_shift), shift_dst[scale - 1]);
                xv_512 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(xv_512, rfactor1), add_shift), shift_dst[scale - 1]);
                xd_512 = _mm512_srai_epi64(_mm512_add_epi64(_mm512_mul_epi32(xd_512, rfactor2), add_shift), shift_dst[scale - 1]);
                
                I4_ADM_CM_THRESH_S_I_J_avx512(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                                add_bef_shift_flt[scale - 1], shift_flt[scale - 1], &thr_512);

                I4_ADM_CM_ACCUM_ROUND_avx512(xh_512, thr_512, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h_512);
                
                I4_ADM_CM_ACCUM_ROUND_avx512(xv_512, thr_512, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v_512);

                I4_ADM_CM_ACCUM_ROUND_avx512(xd_512, thr_512, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d_512);
            }

            __m256i r4_h = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_h_512), _mm512_extracti64x4_epi64(accum_inner_h_512, 1));
            __m128i r2_h = _mm_add_epi64(_mm256_castsi256_si128(r4_h), _mm256_extracti128_si256(r4_h, 1));
            int64_t res_h = r2_h[0] + r2_h[1];

            __m256i r4_v = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_v_512), _mm512_extracti64x4_epi64(accum_inner_v_512, 1));
            __m128i r2_v = _mm_add_epi64(_mm256_castsi256_si128(r4_v), _mm256_extracti128_si256(r4_v, 1));
            int64_t res_v = r2_v[0] + r2_v[1];
            
            __m256i r4_d = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_d_512), _mm512_extracti64x4_epi64(accum_inner_d_512, 1));
            __m128i r2_d = _mm_add_epi64(_mm256_castsi256_si128(r4_d), _mm256_extracti128_si256(r4_d, 1));
            int64_t res_d = r2_d[0] + r2_d[1];

            for (j = end_col_mod6; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }

            accum_h += (accum_inner_h + res_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + res_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + res_d + add_shift_inner_accum) >> shift_inner_accum;
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
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride + w - 1] * rfactor[i * src_stride + w - 1])
                + add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1),
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
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
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_0(angles, flt_angles, csf_a_stride, &thr, w, h, i, 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            /* j within frame */
            for (j = start_col; j < end_col; ++j)
            {
                xh = (int32_t)((((int64_t)src->band_h[i * src_stride + j] * rfactor[0]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xv = (int32_t)((((int64_t)src->band_v[i * src_stride + j] * rfactor[1]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                xd = (int32_t)((((int64_t)src->band_d[i * src_stride + j] * rfactor[2]) +
                    add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
                I4_ADM_CM_THRESH_S_I_J(angles, flt_angles, csf_a_stride, &thr, w, h, i, j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

                I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_h);
                I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_v);
                I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                      add_shift_cub, shift_cub, accum_inner_d);
            }
            /* j = w-1 */
            xh = (int32_t)((((int64_t)src->band_h[i * src_stride + w - 1] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[i * src_stride + w - 1] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[i * src_stride + w - 1] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_I_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, i, (w - 1),
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);

            accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;
        }
    }
    accum_inner_h = 0;
    accum_inner_v = 0;
    accum_inner_d = 0;

    /* i=h-1,j=0 */
    if ((bottom > (h - 1)) && (left <= 0))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_H_M_1_0(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), 0,
                                           add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }

    /* i=h-1,j */
    if (bottom > (h - 1))
    {
        for (j = start_col; j < end_col; ++j)
        {
            xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride + j] * rfactor[0]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride + j] * rfactor[1]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride + j] * rfactor[2]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            I4_ADM_CM_THRESH_S_H_M_1_J(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1), j,
                                               add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

            I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_h);
            I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_v);
            I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                                  add_shift_cub, shift_cub, accum_inner_d);
        }
    }

    /* i-h-1,j=w-1 */
    if ((bottom > (h - 1)) && (right > (w - 1)))
    {
        xh = (int32_t)((((int64_t)src->band_h[(h - 1) * src_stride + w - 1] * rfactor[0]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xv = (int32_t)((((int64_t)src->band_v[(h - 1) * src_stride + w - 1] * rfactor[1]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        xd = (int32_t)((((int64_t)src->band_d[(h - 1) * src_stride + w - 1] * rfactor[2]) +
            add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
        I4_ADM_CM_THRESH_S_H_M_1_W_M_1(angles, flt_angles, csf_a_stride, &thr, w, h, (h - 1),
                                               (w - 1), add_bef_shift_flt[scale - 1], shift_flt[scale - 1]);

        I4_ADM_CM_ACCUM_ROUND(xh, thr, shift_sub, xh_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_h);
        I4_ADM_CM_ACCUM_ROUND(xv, thr, shift_sub, xv_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_v);
        I4_ADM_CM_ACCUM_ROUND(xd, thr, shift_sub, xd_sq, add_shift_sq, shift_sq, val,
                              add_shift_cub, shift_cub, accum_inner_d);
    }
    accum_h += (accum_inner_h + add_shift_inner_accum) >> shift_inner_accum;
    accum_v += (accum_inner_v + add_shift_inner_accum) >> shift_inner_accum;
    accum_d += (accum_inner_d + add_shift_inner_accum) >> shift_inner_accum;

    /**
     * Converted to floating-point for calculating the final scores
     * Final shifts is calculated from 3*(shifts_from_previous_stage(i.e src comes from dwt)+32)-total_shifts_done_in_this_function
     */
    float f_accum_h = (float)(accum_h / final_shift[scale - 1]);
    float f_accum_v = (float)(accum_v / final_shift[scale - 1]);
    float f_accum_d = (float)(accum_d / final_shift[scale - 1]);

    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

void adm_dwt2_s123_combined_avx512(const int32_t *i4_ref_scale, const int32_t *i4_curr_dis,
                                   AdmBuffer *buf, int w, int h, int ref_stride,
                                   int dis_stride, int dst_stride, int scale)
{
    const i4_adm_dwt_band_t *i4_ref_dwt2 = &buf->i4_ref_dwt2;
    const i4_adm_dwt_band_t *i4_dis_dwt2 = &buf->i4_dis_dwt2;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int32_t add_bef_shift_round_VP[3] = { 0, 32768, 32768 };
    const int32_t add_bef_shift_round_HP[3] = { 16384, 32768, 16384 };
    const int16_t shift_VerticalPass[3] = { 0, 16, 16 };
    const int16_t shift_HorizontalPass[3] = { 15, 16, 15 };

    int32_t *tmplo_ref = buf->tmp_ref;
    int32_t *tmphi_ref = tmplo_ref + w;
    int32_t *tmplo_dis = tmphi_ref + w;
    int32_t *tmphi_dis = tmplo_dis + w;
    int32_t s10, s11, s12, s13;

    int64_t accum_ref;

    int shift_VP =  shift_VerticalPass[scale - 1];
    int shift_HP =  shift_HorizontalPass[scale - 1];
    int add_bef_shift_VP = add_bef_shift_round_VP[scale - 1];
    int add_bef_shift_HP = add_bef_shift_round_HP[scale - 1];

    __m512i accum_ref_lo_512, accum_ref_hi_512, accum_dis_lo_512, accum_dis_hi_512;
    __m512i f0_lo = _mm512_set1_epi64(filter_lo[0]);
    __m512i f1_lo = _mm512_set1_epi64(filter_lo[1]);
    __m512i f2_lo = _mm512_set1_epi64(filter_lo[2]);
    __m512i f3_lo = _mm512_set1_epi64(filter_lo[3]);

    __m512i f0_hi = _mm512_set1_epi64(filter_hi[0]);
    __m512i f1_hi = _mm512_set1_epi64(filter_hi[1]);
    __m512i f2_hi = _mm512_set1_epi64(filter_hi[2]);
    __m512i f3_hi = _mm512_set1_epi64(filter_hi[3]);
    __m512i add_bef_shift_round_VP_512 = _mm512_set1_epi64(add_bef_shift_round_VP[scale - 1]);
    __m512i add_bef_shift_round_HP_512 = _mm512_set1_epi64(add_bef_shift_round_HP[scale - 1]);

    
    int w_mod8 = (w  - (w  % 8));
    int half_w_mod8 = ((w + 1) / 2) - ((((w + 1) / 2) - 1) % 8);
    
    for (int i = 0; i < (h + 1) / 2; ++i)
    {
        /* Vertical pass. */       
        for (int j = 0; j < w_mod8; j+=8)
        {
            __m512i ref10_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_ref_scale + (ind_y[0][i] * ref_stride) + j)));
            __m512i ref11_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_ref_scale + (ind_y[1][i] * ref_stride) + j)));
            __m512i ref12_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_ref_scale + (ind_y[2][i] * ref_stride) + j)));
            __m512i ref13_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_ref_scale + (ind_y[3][i] * ref_stride) + j)));

            __m512i dis10_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_curr_dis + (ind_y[0][i] * dis_stride) + j)));
            __m512i dis11_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_curr_dis + (ind_y[1][i] * dis_stride) + j)));
            __m512i dis12_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_curr_dis + (ind_y[2][i] * dis_stride) + j)));
            __m512i dis13_512 = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)(i4_curr_dis + (ind_y[3][i] * dis_stride) + j)));            

            __m512i ref10_lo = _mm512_mul_epi32(ref10_512, f0_lo);
            __m512i ref11_lo = _mm512_mul_epi32(ref11_512, f1_lo);
            __m512i ref12_lo = _mm512_mul_epi32(ref12_512, f2_lo);
            __m512i ref13_lo = _mm512_mul_epi32(ref13_512, f3_lo);

            __m512i ref10_hi = _mm512_mul_epi32(ref10_512, f0_hi);
            __m512i ref11_hi = _mm512_mul_epi32(ref11_512, f1_hi);
            __m512i ref12_hi = _mm512_mul_epi32(ref12_512, f2_hi);
            __m512i ref13_hi = _mm512_mul_epi32(ref13_512, f3_hi);

            __m512i dis10_lo = _mm512_mul_epi32(dis10_512, f0_lo);
            __m512i dis11_lo = _mm512_mul_epi32(dis11_512, f1_lo);
            __m512i dis12_lo = _mm512_mul_epi32(dis12_512, f2_lo);
            __m512i dis13_lo = _mm512_mul_epi32(dis13_512, f3_lo);

            __m512i dis10_hi = _mm512_mul_epi32(dis10_512, f0_hi);
            __m512i dis11_hi = _mm512_mul_epi32(dis11_512, f1_hi);
            __m512i dis12_hi = _mm512_mul_epi32(dis12_512, f2_hi);
            __m512i dis13_hi = _mm512_mul_epi32(dis13_512, f3_hi);

            accum_ref_lo_512 = _mm512_add_epi64(ref10_lo, ref11_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref12_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref13_lo);

            accum_ref_hi_512 = _mm512_add_epi64(ref10_hi, ref11_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref12_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref13_hi);

            accum_dis_lo_512 = _mm512_add_epi64(dis10_lo, dis11_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis12_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis13_lo);

            accum_dis_hi_512 = _mm512_add_epi64(dis10_hi, dis11_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis12_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis13_hi);

            accum_ref_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_lo_512, add_bef_shift_round_VP_512), shift_VP);
            accum_ref_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_hi_512, add_bef_shift_round_VP_512), shift_VP);
            accum_dis_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_lo_512, add_bef_shift_round_VP_512), shift_VP);
            accum_dis_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_hi_512, add_bef_shift_round_VP_512), shift_VP);

            _mm256_storeu_si256((__m256i*)(tmplo_ref + j), _mm512_cvtepi64_epi32(accum_ref_lo_512));
            _mm256_storeu_si256((__m256i*)(tmphi_ref + j), _mm512_cvtepi64_epi32(accum_ref_hi_512));
            _mm256_storeu_si256((__m256i*)(tmplo_dis + j), _mm512_cvtepi64_epi32(accum_dis_lo_512));
            _mm256_storeu_si256((__m256i*)(tmphi_dis + j), _mm512_cvtepi64_epi32(accum_dis_hi_512));
        }

        for (int j = w_mod8; j < w; ++j)
        {            
            s10 = i4_ref_scale[ind_y[0][i] * ref_stride + j];
            s11 = i4_ref_scale[ind_y[1][i] * ref_stride + j];
            s12 = i4_ref_scale[ind_y[2][i] * ref_stride + j];
            s13 = i4_ref_scale[ind_y[3][i] * ref_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_ref[j] = (int32_t)((accum_ref + add_bef_shift_VP) >> shift_VP);
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_ref[j] = (int32_t)((accum_ref + add_bef_shift_VP) >> shift_VP);

            s10 = i4_curr_dis[ind_y[0][i] * dis_stride + j];
            s11 = i4_curr_dis[ind_y[1][i] * dis_stride + j];
            s12 = i4_curr_dis[ind_y[2][i] * dis_stride + j];
            s13 = i4_curr_dis[ind_y[3][i] * dis_stride + j];
            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            tmplo_dis[j] = (int32_t)((accum_ref + add_bef_shift_VP) >> shift_VP);
            
            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            tmphi_dis[j] = (int32_t)((accum_ref + add_bef_shift_VP) >> shift_VP);
        }
    
        // j == 0
        {
            int j = 0;
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            s10 = tmplo_ref[j0];
            s11 = tmplo_ref[j1];
            s12 = tmplo_ref[j2];
            s13 = tmplo_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmphi_ref[j0];
            s11 = tmphi_ref[j1];
            s12 = tmphi_ref[j2];
            s13 = tmphi_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmplo_dis[j0];
            s11 = tmplo_dis[j1];
            s12 = tmplo_dis[j2];
            s13 = tmplo_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmphi_dis[j0];
            s11 = tmphi_dis[j1];
            s12 = tmphi_dis[j2];
            s13 = tmphi_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);
        }

        /* Horizontal pass (lo and hi). */        
        for (int j = 1; j < half_w_mod8; j+=8)
        {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            __m512i ref10_lo = _mm512_loadu_si512((__m512i*)(tmplo_ref + j0));
            __m512i ref11_lo = _mm512_loadu_si512((__m512i*)(tmplo_ref + j1));
            __m512i ref12_lo = _mm512_loadu_si512((__m512i*)(tmplo_ref + j2));
            __m512i ref13_lo = _mm512_loadu_si512((__m512i*)(tmplo_ref + j3));

            __m512i ref10_lo_f0_lo = _mm512_mul_epi32(ref10_lo, f0_lo);
            __m512i ref11_lo_f1_lo = _mm512_mul_epi32(ref11_lo, f1_lo);
            __m512i ref12_lo_f2_lo = _mm512_mul_epi32(ref12_lo, f2_lo);
            __m512i ref13_lo_f3_lo = _mm512_mul_epi32(ref13_lo, f3_lo);

            __m512i ref10_lo_f0_hi = _mm512_mul_epi32(ref10_lo, f0_hi);
            __m512i ref11_lo_f1_hi = _mm512_mul_epi32(ref11_lo, f1_hi);
            __m512i ref12_lo_f2_hi = _mm512_mul_epi32(ref12_lo, f2_hi);
            __m512i ref13_lo_f3_hi = _mm512_mul_epi32(ref13_lo, f3_hi);

            accum_ref_lo_512 = _mm512_add_epi64(ref10_lo_f0_lo, ref11_lo_f1_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref12_lo_f2_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref13_lo_f3_lo);

            accum_ref_hi_512 = _mm512_add_epi64(ref10_lo_f0_hi, ref11_lo_f1_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref12_lo_f2_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref13_lo_f3_hi);

            accum_ref_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_lo_512, add_bef_shift_round_HP_512), shift_HP);
            accum_ref_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_hi_512, add_bef_shift_round_HP_512), shift_HP);

            _mm256_storeu_si256((__m256i*)(i4_ref_dwt2->band_a + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_ref_lo_512));
            _mm256_storeu_si256((__m256i*)(i4_ref_dwt2->band_v + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_ref_hi_512));

            __m512i ref10_hi = _mm512_loadu_si512((__m512i*)(tmphi_ref + j0));
            __m512i ref11_hi = _mm512_loadu_si512((__m512i*)(tmphi_ref + j1));
            __m512i ref12_hi = _mm512_loadu_si512((__m512i*)(tmphi_ref + j2));
            __m512i ref13_hi = _mm512_loadu_si512((__m512i*)(tmphi_ref + j3));

            __m512i ref10_hi_f0_lo = _mm512_mul_epi32(ref10_hi, f0_lo);
            __m512i ref11_hi_f1_lo = _mm512_mul_epi32(ref11_hi, f1_lo);
            __m512i ref12_hi_f2_lo = _mm512_mul_epi32(ref12_hi, f2_lo);
            __m512i ref13_hi_f3_lo = _mm512_mul_epi32(ref13_hi, f3_lo);

            __m512i ref10_hi_f0_hi = _mm512_mul_epi32(ref10_hi, f0_hi);
            __m512i ref11_hi_f1_hi = _mm512_mul_epi32(ref11_hi, f1_hi);
            __m512i ref12_hi_f2_hi = _mm512_mul_epi32(ref12_hi, f2_hi);
            __m512i ref13_hi_f3_hi = _mm512_mul_epi32(ref13_hi, f3_hi);

            accum_ref_lo_512 = _mm512_add_epi64(ref10_hi_f0_lo, ref11_hi_f1_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref12_hi_f2_lo);
            accum_ref_lo_512 = _mm512_add_epi64(accum_ref_lo_512, ref13_hi_f3_lo);

            accum_ref_hi_512 = _mm512_add_epi64(ref10_hi_f0_hi, ref11_hi_f1_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref12_hi_f2_hi);
            accum_ref_hi_512 = _mm512_add_epi64(accum_ref_hi_512, ref13_hi_f3_hi);

            accum_ref_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_lo_512, add_bef_shift_round_HP_512), shift_HorizontalPass[scale - 1]);
            accum_ref_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_ref_hi_512, add_bef_shift_round_HP_512), shift_HorizontalPass[scale - 1]);

            _mm256_storeu_si256((__m256i*)(i4_ref_dwt2->band_h + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_ref_lo_512));
            _mm256_storeu_si256((__m256i*)(i4_ref_dwt2->band_d + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_ref_hi_512));

            __m512i dis10_lo = _mm512_loadu_si512((__m512i*)(tmplo_dis + j0));
            __m512i dis11_lo = _mm512_loadu_si512((__m512i*)(tmplo_dis + j1));
            __m512i dis12_lo = _mm512_loadu_si512((__m512i*)(tmplo_dis + j2));
            __m512i dis13_lo = _mm512_loadu_si512((__m512i*)(tmplo_dis + j3));

            __m512i dis10_lo_f0_lo = _mm512_mul_epi32(dis10_lo, f0_lo);
            __m512i dis11_lo_f1_lo = _mm512_mul_epi32(dis11_lo, f1_lo);
            __m512i dis12_lo_f2_lo = _mm512_mul_epi32(dis12_lo, f2_lo);
            __m512i dis13_lo_f3_lo = _mm512_mul_epi32(dis13_lo, f3_lo);

            __m512i dis10_lo_f0_hi = _mm512_mul_epi32(dis10_lo, f0_hi);
            __m512i dis11_lo_f1_hi = _mm512_mul_epi32(dis11_lo, f1_hi);
            __m512i dis12_lo_f2_hi = _mm512_mul_epi32(dis12_lo, f2_hi);
            __m512i dis13_lo_f3_hi = _mm512_mul_epi32(dis13_lo, f3_hi);

            accum_dis_lo_512 = _mm512_add_epi64(dis10_lo_f0_lo, dis11_lo_f1_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis12_lo_f2_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis13_lo_f3_lo);

            accum_dis_hi_512 = _mm512_add_epi64(dis10_lo_f0_hi, dis11_lo_f1_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis12_lo_f2_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis13_lo_f3_hi);

            accum_dis_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_lo_512, add_bef_shift_round_HP_512), shift_HorizontalPass[scale - 1]);
            accum_dis_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_hi_512, add_bef_shift_round_HP_512), shift_HorizontalPass[scale - 1]);

            _mm256_storeu_si256((__m256i*)(i4_dis_dwt2->band_a + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_dis_lo_512));
            _mm256_storeu_si256((__m256i*)(i4_dis_dwt2->band_v + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_dis_hi_512));

            __m512i dis10_hi = _mm512_loadu_si512((__m512i*)(tmphi_dis + j0));
            __m512i dis11_hi = _mm512_loadu_si512((__m512i*)(tmphi_dis + j1));
            __m512i dis12_hi = _mm512_loadu_si512((__m512i*)(tmphi_dis + j2));
            __m512i dis13_hi = _mm512_loadu_si512((__m512i*)(tmphi_dis + j3));

            __m512i dis10_hi_f0_lo = _mm512_mul_epi32(dis10_hi, f0_lo);
            __m512i dis11_hi_f1_lo = _mm512_mul_epi32(dis11_hi, f1_lo);
            __m512i dis12_hi_f2_lo = _mm512_mul_epi32(dis12_hi, f2_lo);
            __m512i dis13_hi_f3_lo = _mm512_mul_epi32(dis13_hi, f3_lo);

            __m512i dis10_hi_f0_hi = _mm512_mul_epi32(dis10_hi, f0_hi);
            __m512i dis11_hi_f1_hi = _mm512_mul_epi32(dis11_hi, f1_hi);
            __m512i dis12_hi_f2_hi = _mm512_mul_epi32(dis12_hi, f2_hi);
            __m512i dis13_hi_f3_hi = _mm512_mul_epi32(dis13_hi, f3_hi);

            accum_dis_lo_512 = _mm512_add_epi64(dis10_hi_f0_lo, dis11_hi_f1_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis12_hi_f2_lo);
            accum_dis_lo_512 = _mm512_add_epi64(accum_dis_lo_512, dis13_hi_f3_lo);

            accum_dis_hi_512 = _mm512_add_epi64(dis10_hi_f0_hi, dis11_hi_f1_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis12_hi_f2_hi);
            accum_dis_hi_512 = _mm512_add_epi64(accum_dis_hi_512, dis13_hi_f3_hi);

            accum_dis_lo_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_lo_512, add_bef_shift_round_HP_512), shift_HP);
            accum_dis_hi_512 = _mm512_srai_epi64(_mm512_add_epi64(accum_dis_hi_512, add_bef_shift_round_HP_512), shift_HP);

            _mm256_storeu_si256((__m256i*)(i4_dis_dwt2->band_h + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_dis_lo_512));
            _mm256_storeu_si256((__m256i*)(i4_dis_dwt2->band_d + (i * dst_stride) + j), _mm512_cvtepi64_epi32(accum_dis_hi_512));
        }

        for (int j = half_w_mod8; j < (w + 1) / 2; ++j)
        {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            s10 = tmplo_ref[j0];
            s11 = tmplo_ref[j1];
            s12 = tmplo_ref[j2];
            s13 = tmplo_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmphi_ref[j0];
            s11 = tmphi_ref[j1];
            s12 = tmphi_ref[j2];
            s13 = tmphi_ref[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_ref_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_ref_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmplo_dis[j0];
            s11 = tmplo_dis[j1];
            s12 = tmplo_dis[j2];
            s13 = tmplo_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_a[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_v[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            s10 = tmphi_dis[j0];
            s11 = tmphi_dis[j1];
            s12 = tmphi_dis[j2];
            s13 = tmphi_dis[j3];

            accum_ref = 0;
            accum_ref += (int64_t)filter_lo[0] * s10;
            accum_ref += (int64_t)filter_lo[1] * s11;
            accum_ref += (int64_t)filter_lo[2] * s12;
            accum_ref += (int64_t)filter_lo[3] * s13;
            i4_dis_dwt2->band_h[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);

            accum_ref = 0;
            accum_ref += (int64_t)filter_hi[0] * s10;
            accum_ref += (int64_t)filter_hi[1] * s11;
            accum_ref += (int64_t)filter_hi[2] * s12;
            accum_ref += (int64_t)filter_hi[3] * s13;
            i4_dis_dwt2->band_d[i * dst_stride + j] = (int32_t)((accum_ref + add_bef_shift_HP) >> shift_HP);
        }
    }
}

void adm_dwt2_16_avx512(const uint16_t *src, const adm_dwt_band_t *dst, AdmBuffer *buf, int w, int h,
                        int src_stride, int dst_stride, int inp_size_bits)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_VP = inp_size_bits;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 1 << (inp_size_bits - 1);
    const int32_t add_shift_HP = 32768;

    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    __m512i f01_lo = _mm512_set1_epi32(filter_lo[0] + (uint32_t)(filter_lo[1] << 16) /* + (1 << 16) */);
    __m512i f23_lo = _mm512_set1_epi32(filter_lo[2] + (uint32_t)(filter_lo[3] << 16) /* + (1 << 16) */);
    __m512i f01_hi = _mm512_set1_epi32(filter_hi[0] + (uint32_t)(filter_hi[1] << 16) + (1 << 16));
    __m512i f23_hi = _mm512_set1_epi32(filter_hi[2] + (uint32_t)(filter_hi[3] << 16) /*+ (1 << 16)*/);

    __m512i accum0, accum0_lo, accum0_hi;
    //__m512i norm_lo = _mm512_set1_epi32((int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP);
    //__m512i norm_hi = _mm512_set1_epi32((int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP);

    //int w_mod32 = (w  - (w  % 32));
    int half_w_mod64 = ((w + 1) / 2) - ((((w + 1) / 2) - 1) % 64);

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
	    /*
        for (int j = 0; j < w_mod32; j+=32) {
            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            __m512i u_s0_512 = _mm512_loadu_si512((__m512i*)(src + (ind_y[0][i] * src_stride + j)));
            __m512i u_s1_512 = _mm512_loadu_si512((__m512i*)(src + (ind_y[1][i] * src_stride + j)));
            __m512i u_s2_512 = _mm512_loadu_si512((__m512i*)(src + (ind_y[2][i] * src_stride + j)));
            __m512i u_s3_512 = _mm512_loadu_si512((__m512i*)(src + (ind_y[3][i] * src_stride + j)));

            __m512i s0_s1_lo = _mm512_unpacklo_epi16(u_s0_512, u_s1_512);
            __m512i s0_s1_hi = _mm512_unpackhi_epi16(u_s0_512, u_s1_512);
            __m512i s2_s3_lo = _mm512_unpacklo_epi16(u_s2_512, u_s3_512);
            __m512i s2_s3_hi = _mm512_unpackhi_epi16(u_s2_512, u_s3_512);

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0_s1_lo, f01_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_s1_hi, f01_lo));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2_s3_lo, f23_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_s3_hi, f23_lo));

            accum0_lo = _mm512_sub_epi32(accum0_lo, norm_lo);
            accum0_hi = _mm512_sub_epi32(accum0_hi, norm_lo);
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_VP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_VP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_VP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_VP);

            accum0 = _mm512_packs_epi32(accum0_lo, accum0_hi);

            _mm512_storeu_si512((__m512i*)(tmplo + j), accum0);

            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0_s1_lo, f01_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_s1_hi, f01_hi));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2_s3_lo, f23_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_s3_hi, f23_hi));

            accum0_lo = _mm512_sub_epi32(accum0_lo, norm_hi);
            accum0_hi = _mm512_sub_epi32(accum0_hi, norm_hi);
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_VP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_VP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_VP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_VP);

            accum0 = _mm512_packs_epi32(accum0_lo, accum0_hi);

            _mm512_storeu_si512((__m512i*)(tmphi + j), accum0);
        }
	*/

        for (int j = 0; j < w; ++j) {
            uint16_t u_s0 = src[ind_y[0][i] * src_stride + j];
            uint16_t u_s1 = src[ind_y[1][i] * src_stride + j];
            uint16_t u_s2 = src[ind_y[2][i] * src_stride + j];
            uint16_t u_s3 = src[ind_y[3][i] * src_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP;

            tmplo[j] = (accum + add_shift_VP) >> shift_VP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP;

            tmphi[j] = (accum + add_shift_VP) >> shift_VP;
        }

        /* Horizontal pass (lo and hi). */
        for (int j = 0; j < 1; ++j) {            
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }
            
        for (int j = 1; j < half_w_mod64; j += 32) {            
            int j0 = ind_x[0][j];
            int j2 = ind_x[2][j];
            int j16 = ind_x[0][j + 16];
            int j18 = ind_x[2][j + 16];

            __m512i s0 = _mm512_loadu_si512((__m512i*)(tmplo + j0));
            __m512i s2 = _mm512_loadu_si512((__m512i*)(tmplo + j2));
            __m512i s0_32 = _mm512_loadu_si512((__m512i*)(tmplo + j16));
            __m512i s2_32 = _mm512_loadu_si512((__m512i*)(tmplo + j18));

            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0, f01_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_32, f01_lo));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2, f23_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_32, f23_lo));
            
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_HP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_HP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_HP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_HP);

            accum0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(accum0_lo)), _mm512_cvtepi32_epi16(accum0_hi), 1);

            _mm512_storeu_si512((__m512i*)(dst->band_a + i * dst_stride + j), accum0);

            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0, f01_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_32, f01_hi));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2, f23_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_32, f23_hi));
     
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_HP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_HP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_HP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_HP);
            accum0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(accum0_lo)), _mm512_cvtepi32_epi16(accum0_hi), 1);
            _mm512_storeu_si512((__m512i*)(dst->band_v + i * dst_stride + j), accum0);

            s0 = _mm512_loadu_si512((__m512i*)(tmphi + j0));
            s2 = _mm512_loadu_si512((__m512i*)(tmphi + j2));
            s0_32 = _mm512_loadu_si512((__m512i*)(tmphi + j16)); // PV
            s2_32 = _mm512_loadu_si512((__m512i*)(tmphi + j18));

            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0, f01_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_32, f01_lo));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2, f23_lo));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_32, f23_lo));

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_HP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_HP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_HP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_HP);
            accum0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(accum0_lo)), _mm512_cvtepi32_epi16(accum0_hi), 1);
            _mm512_storeu_si512((__m512i*)(dst->band_h + i * dst_stride + j), accum0);

            accum0_lo = _mm512_setzero_si512();
            accum0_hi = _mm512_setzero_si512();

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s0, f01_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s0_32, f01_hi));
            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_madd_epi16(s2, f23_hi));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_madd_epi16(s2_32, f23_hi));

            accum0_lo = _mm512_add_epi32(accum0_lo, _mm512_set1_epi32(add_shift_HP));
            accum0_hi = _mm512_add_epi32(accum0_hi, _mm512_set1_epi32(add_shift_HP));
            accum0_lo = _mm512_srai_epi32(accum0_lo, shift_HP);
            accum0_hi = _mm512_srai_epi32(accum0_hi, shift_HP);
            accum0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(accum0_lo)), _mm512_cvtepi32_epi16(accum0_hi), 1);
            _mm512_storeu_si512((__m512i*)(dst->band_d + i * dst_stride + j), accum0);
        }

        for (int j = half_w_mod64; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }
        
    }
}

void adm_dwt2_8_avx512(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_VP = 8;
    const int16_t shift_HP = 16;
    const int32_t add_shift_VP = 128;
    const int32_t add_shift_HP = 32768;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    __m512i dwt2_db2_coeffs_lo_sum_const = _mm512_set1_epi32(5931776);
    __m512i fl0 =
        _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_lo));
    __m512i fl1 =
        _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_lo + 2)));
    __m512i fh0 =
        _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_hi));
    __m512i fh1 =
        _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_hi + 2)));
    __m512i add_shift_VP_vex = _mm512_set1_epi32(128);
    __m512i pad_register = _mm512_setzero_si512();
    __m512i add_shift_HP_vex = _mm512_set1_epi32(32768);

    int w_mod_32 = (w >> 5) << 5;
    int half_w_mod_32 = ((w + 1) / 2) - ((((w + 1) / 2) - 1) % 32);

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */
        for (int j = 0; j < w_mod_32; j = j + 32) {
            __m512i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
            accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                _mm512_setzero_si512();
            __m512i s0, s1, s2, s3;

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(src + (ind_y[0][i] * src_stride) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(src + (ind_y[1][i] * src_stride) + j)));
            s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(src + (ind_y[2][i] * src_stride) + j)));
            s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(src + (ind_y[3][i] * src_stride) + j)));

            __m512i s0lo = _mm512_unpacklo_epi16(s0, s1);
            __m512i s0hi = _mm512_unpackhi_epi16(s0, s1);

            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s0lo, fl0));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s0hi, fl0));

            __m512i s1lo = _mm512_unpacklo_epi16(s2, s3);
            __m512i s1hi = _mm512_unpackhi_epi16(s2, s3);

            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s1lo, fl1));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s1hi, fl1));

            accum_mu2_lo =
                _mm512_sub_epi32(accum_mu2_lo, dwt2_db2_coeffs_lo_sum_const);
            accum_mu2_hi =
                _mm512_sub_epi32(accum_mu2_hi, dwt2_db2_coeffs_lo_sum_const);

            accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo, add_shift_VP_vex);
            accum_mu2_lo = _mm512_srli_epi32(accum_mu2_lo, 0x08);
            accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi, add_shift_VP_vex);
            accum_mu2_hi = _mm512_srli_epi32(accum_mu2_hi, 0x08);
            
            accum_mu2_lo = _mm512_mask_blend_epi16(0xAAAAAAAA, accum_mu2_lo, pad_register);
            accum_mu2_hi = _mm512_mask_blend_epi16(0xAAAAAAAA, accum_mu2_hi, pad_register);

            accum_mu2_hi = _mm512_packus_epi32(accum_mu2_lo, accum_mu2_hi);
            
            _mm512_storeu_si512((__m512i *)(tmplo + j), accum_mu2_hi);

            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s0lo, fh0));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s0hi, fh0));
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s1lo, fh1));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s1hi, fh1));

            accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo, add_shift_VP_vex);
            accum_mu1_lo = _mm512_srli_epi32(accum_mu1_lo, 0x08);
            accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi, add_shift_VP_vex);
            accum_mu1_hi = _mm512_srli_epi32(accum_mu1_hi, 0x08);

            accum_mu1_lo = _mm512_mask_blend_epi16(0xAAAAAAAA, accum_mu1_lo, pad_register);
            accum_mu1_hi = _mm512_mask_blend_epi16(0xAAAAAAAA, accum_mu1_hi, pad_register);
            accum_mu1_hi = _mm512_packus_epi32(accum_mu1_lo, accum_mu1_hi);
            _mm512_storeu_si512((__m512i *)(tmphi + j), accum_mu1_hi);            
        }

        for (int j = w_mod_32; j < w; ++j) {
            uint16_t u_s0 = src[ind_y[0][i] * src_stride + j];
            uint16_t u_s1 = src[ind_y[1][i] * src_stride + j];
            uint16_t u_s2 = src[ind_y[2][i] * src_stride + j];
            uint16_t u_s3 = src[ind_y[3][i] * src_stride + j];

            accum = 0;
            accum += (int32_t)filter_lo[0] * (int32_t)u_s0;
            accum += (int32_t)filter_lo[1] * (int32_t)u_s1;
            accum += (int32_t)filter_lo[2] * (int32_t)u_s2;
            accum += (int32_t)filter_lo[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_lo_sum * add_shift_VP;

            tmplo[j] = (accum + add_shift_VP) >> shift_VP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * (int32_t)u_s0;
            accum += (int32_t)filter_hi[1] * (int32_t)u_s1;
            accum += (int32_t)filter_hi[2] * (int32_t)u_s2;
            accum += (int32_t)filter_hi[3] * (int32_t)u_s3;

            /* normalizing is done for range from(0 to N) to (-N/2 to N/2) */
            accum -= (int32_t)dwt2_db2_coeffs_hi_sum * add_shift_VP;

            tmphi[j] = (accum + add_shift_VP) >> shift_VP;
        }
        
        int j0 = ind_x[0][0];
        int j1 = ind_x[1][0];
        int j2 = ind_x[2][0];
        int j3 = ind_x[3][0];

        int16_t s0 = tmplo[j0];
        int16_t s1 = tmplo[j1];
        int16_t s2 = tmplo[j2];
        int16_t s3 = tmplo[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_a[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_v[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        s0 = tmphi[j0];
        s1 = tmphi[j1];
        s2 = tmphi[j2];
        s3 = tmphi[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_h[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_d[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        for (int j = 1; j < half_w_mod_32; j = j + 32) {
            {
                __m512i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm512_setzero_si512();

                __m512i s00, s22, s33, s44;

                s00 = _mm512_loadu_si512((__m512i *)(tmplo + ind_x[0][j]));
                s22 = _mm512_loadu_si512((__m512i *)(tmplo + ind_x[2][j]));
                s33 = _mm512_loadu_si512((__m512i *)(tmplo + 32 + ind_x[0][j]));
                s44 = _mm512_loadu_si512((__m512i *)(tmplo + 32 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm512_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm512_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm512_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), accum_mu2_hi);
                _mm512_storeu_si512(
                    (__m512i *)(dst->band_a + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm512_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm512_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm512_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), accum_mu1_hi);
                _mm512_storeu_si512(
                    (__m512i *)(dst->band_v + i * dst_stride + j),
                    accum_mu1_hi);
            }

            {
                __m512i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm512_setzero_si512();

                __m512i s00, s22, s33, s44;

                __m512i add_shift_HP_vex = _mm512_set1_epi32(32768);

                s00 = _mm512_loadu_si512((__m512i *)(tmphi + ind_x[0][j]));
                s22 = _mm512_loadu_si512((__m512i *)(tmphi + ind_x[2][j]));
                s33 = _mm512_loadu_si512((__m512i *)(tmphi + 32 + ind_x[0][j]));
                s44 = _mm512_loadu_si512((__m512i *)(tmphi + 32 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm512_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm512_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm512_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), accum_mu2_hi);
                _mm512_storeu_si512(
                    (__m512i *)(dst->band_h + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm512_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm512_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm512_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), accum_mu1_hi);
                _mm512_storeu_si512(
                    (__m512i *)(dst->band_d + i * dst_stride + j),
                    accum_mu1_hi);
            }
        }

        for (int j = half_w_mod_32; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            int16_t s0 = tmplo[j0];
            int16_t s1 = tmplo[j1];
            int16_t s2 = tmplo[j2];
            int16_t s3 = tmplo[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
 
            dst->band_a[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;

            dst->band_v[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            s0 = tmphi[j0];
            s1 = tmphi[j1];
            s2 = tmphi[j2];
            s3 = tmphi[j3];

            accum = 0;
            accum += (int32_t)filter_lo[0] * s0;
            accum += (int32_t)filter_lo[1] * s1;
            accum += (int32_t)filter_lo[2] * s2;
            accum += (int32_t)filter_lo[3] * s3;
            dst->band_h[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;

            accum = 0;
            accum += (int32_t)filter_hi[0] * s0;
            accum += (int32_t)filter_hi[1] * s1;
            accum += (int32_t)filter_hi[2] * s2;
            accum += (int32_t)filter_hi[3] * s3;
            dst->band_d[i * dst_stride + j] = (accum + add_shift_HP) >> shift_HP;
        }

    }
}

void adm_csf_avx512(AdmBuffer *buf, int w, int h, int stride,
                    double adm_norm_view_dist, int adm_ref_display_height)
{
    const adm_dwt_band_t *src = &buf->decouple_a;
    const adm_dwt_band_t *dst = &buf->csf_a;
    const adm_dwt_band_t *flt = &buf->csf_f;

    const int16_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int16_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int16_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    // 0 is scale zero passed to dwt_quant_step

    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    /**
     * rfactor is converted to fixed-point for scale0 and stored in i_rfactor
     * multiplied by 2^21 for rfactor[0,1] and by 2^23 for rfactor[2].
     * For adm_norm_view_dist 3.0 and adm_ref_display_height 1080,
     * i_rfactor is around { 36453,36453,49417 }
     */
    uint16_t i_rfactor[3];
    if (fabs(adm_norm_view_dist * adm_ref_display_height - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    }
    else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t) (rfactor1[0] * pow2_21);
        i_rfactor[1] = (uint16_t) (rfactor1[1] * pow2_21);
        i_rfactor[2] = (uint16_t) (rfactor1[2] * pow2_23);
    }

    /**
     * Shifts pending from previous stage is 6
     * hence variables multiplied by i_rfactor[0,1] has to be shifted by 21+6=27 to convert
     * into floating-point. But shifted by 15 to make it Q16
     * and variables multiplied by i_factor[2] has to be shifted by 23+6=29 to convert into
     * floating-point. But shifted by 17 to make it Q16
     * Hence remaining shifts after shifting by i_shifts is 12 to make it equivalent to
     * floating-point
     */
    uint8_t i_shifts[3] = { 15,15,17 };
    uint16_t i_shiftsadd[3] = { 16384, 16384, 65535 };
    uint16_t FIX_ONE_BY_30 = 4369; //(1/30)*2^17
    /* The computation of the csf values is not required for the regions which
     *lie outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int right_mod_16 = right - ((right - left ) % 16);

    for (int i = top; i < bottom; ++i) {
        int src_offset = i * stride;
        int dst_offset = i * stride;

        for (int j = left; j < right_mod_16; j+=16) {
            __m512i src0 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_angles[0] + src_offset + j)));
            __m512i r_factor0 = _mm512_set1_epi32(i_rfactor[0]);
            __m512i dst_val0 = _mm512_mullo_epi32(src0, r_factor0);
            __m512i i16_dst_val0 = _mm512_srai_epi32(_mm512_add_epi32(dst_val0, _mm512_set1_epi32(i_shiftsadd[0])), i_shifts[0]);
            _mm256_storeu_si256((__m256i*)(dst_angles[0] + dst_offset + j), _mm512_cvtepi32_epi16(i16_dst_val0));
            __m512i flt0 = _mm512_mullo_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(i16_dst_val0));
            flt0 = _mm512_srli_epi32(_mm512_add_epi32(flt0, _mm512_set1_epi32(2048)), 12);
            _mm256_storeu_si256((__m256i*)(flt_angles[0] + dst_offset + j), _mm512_cvtepi32_epi16(flt0));            

            __m512i src1 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_angles[1] + src_offset + j)));
            __m512i r_factor1 = _mm512_set1_epi32(i_rfactor[1]);
            __m512i dst_val1 = _mm512_mullo_epi32(src1, r_factor1);
            __m512i i16_dst_val1 = _mm512_srai_epi32(_mm512_add_epi32(dst_val1, _mm512_set1_epi32(i_shiftsadd[1])), i_shifts[1]);
            _mm256_storeu_si256((__m256i*)(dst_angles[1] + dst_offset + j), _mm512_cvtepi32_epi16(i16_dst_val1));
            __m512i flt1 = _mm512_mullo_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(i16_dst_val1));
            flt1 = _mm512_srli_epi32(_mm512_add_epi32(flt1, _mm512_set1_epi32(2048)), 12);
            _mm256_storeu_si256((__m256i*)(flt_angles[1] + dst_offset + j), _mm512_cvtepi32_epi16(flt1));

            __m512i src2 = _mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(src_angles[2] + src_offset + j)));
            __m512i r_factor2 = _mm512_set1_epi32(i_rfactor[2]);
            __m512i dst_val2 = _mm512_mullo_epi32(src2, r_factor2);
            __m512i i16_dst_val2 = _mm512_srai_epi32(_mm512_add_epi32(dst_val2, _mm512_set1_epi32(i_shiftsadd[2])), i_shifts[2]);
            _mm256_storeu_si256((__m256i*)(dst_angles[2] + dst_offset + j), _mm512_cvtepi32_epi16(i16_dst_val2));
            __m512i flt2 = _mm512_mullo_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(i16_dst_val2));
            flt2 = _mm512_srli_epi32(_mm512_add_epi32(flt2, _mm512_set1_epi32(2048)), 12);
            _mm256_storeu_si256((__m256i*)(flt_angles[2] + dst_offset + j), _mm512_cvtepi32_epi16(flt2));
        }
        
        for (int j = right_mod_16; j < right; ++j) {
            
            int32_t dst_val0 = i_rfactor[0] * (int32_t)src_angles[0][src_offset + j];
            int16_t i16_dst_val0 = ((int16_t)((dst_val0 + i_shiftsadd[0]) >> i_shifts[0]));
            dst_angles[0][dst_offset + j] = i16_dst_val0;
            flt_angles[0][dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val0))
                + 2048) >> 12));

            int32_t dst_val1 = i_rfactor[1] * (int32_t)src_angles[1][src_offset + j];
            int16_t i16_dst_val1 = ((int16_t)((dst_val1 + i_shiftsadd[1]) >> i_shifts[1]));
            dst_angles[1][dst_offset + j] = i16_dst_val1;
            flt_angles[1][dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val1))
                + 2048) >> 12));

            int32_t dst_val2 = i_rfactor[2] * (int32_t)src_angles[2][src_offset + j];
            int16_t i16_dst_val2 = ((int16_t)((dst_val2 + i_shiftsadd[2]) >> i_shifts[2]));
            dst_angles[2][dst_offset + j] = i16_dst_val2;
            flt_angles[2][dst_offset + j] = ((int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val2))
                + 2048) >> 12));
        }
    }
}

void i4_adm_csf_avx512(AdmBuffer *buf, int scale, int w, int h, int stride,
                       double adm_norm_view_dist, int adm_ref_display_height)
{
    const i4_adm_dwt_band_t *src = &buf->i4_decouple_a;
    const i4_adm_dwt_band_t *dst = &buf->i4_csf_a;
    const i4_adm_dwt_band_t *flt = &buf->i4_csf_f;

    const int32_t *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    int32_t *dst_angles[3] = { dst->band_h, dst->band_v, dst->band_d };
    int32_t *flt_angles[3] = { flt->band_h, flt->band_v, flt->band_d };

    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor1[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    //i_rfactor in fixed-point
    const double pow2_32 = pow(2, 32);
    const uint32_t i_rfactor[3] = { (uint32_t)(rfactor1[0] * pow2_32),
                                    (uint32_t)(rfactor1[1] * pow2_32),
                                    (uint32_t)(rfactor1[2] * pow2_32) };

    const uint32_t FIX_ONE_BY_30 = 143165577;
    const uint32_t shift_dst[3] = { 28, 28, 28 };
    const uint32_t shift_flt[3] = { 32, 32, 32 };
    int32_t add_bef_shift_dst[3], add_bef_shift_flt[3];

    for (unsigned idx = 0; idx < 3; ++idx) {
        add_bef_shift_dst[idx] = (1u << (shift_dst[idx] - 1));
        add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1));
    }

    /* The computation of the csf values is not required for the regions
     * which lie outside the frame borders
     */
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
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

    int right_mod_16 = right - ((right - left ) % 16);

    for (int i = top; i < bottom; ++i)
    {
        int src_offset = i * stride;
        int dst_offset = i * stride;

        for (int j = left; j < right_mod_16; j+=16)
        {
            __m512i src0 = _mm512_loadu_si512((__m512i*)(src_angles[0] + src_offset + j));
            __m512i r_factor0 = _mm512_set1_epi32(i_rfactor[0]);
            __m512i dst_val0_lo = _mm512_mul_epi32(src0, r_factor0);
            __m512i dst_val0_hi = _mm512_mul_epi32(_mm512_srli_epi64(src0, 32), r_factor0);

            dst_val0_lo = _mm512_srai_epi64(_mm512_add_epi64(dst_val0_lo, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);
            dst_val0_hi = _mm512_srai_epi64(_mm512_add_epi64(dst_val0_hi, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);

            dst_val0_lo = _mm512_permutex2var_epi32(dst_val0_lo, _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), dst_val0_hi);
            _mm512_storeu_si512((__m512i*)(dst_angles[0] + dst_offset + j), dst_val0_lo);

            __m512i flt0_lo = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(dst_val0_lo));
            __m512i flt0_hi = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_srli_epi64(_mm512_abs_epi32(dst_val0_lo), 32));

            flt0_lo = _mm512_srli_epi64(_mm512_add_epi64(flt0_lo, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]); 
            flt0_hi = _mm512_srli_epi64(_mm512_add_epi64(flt0_hi, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]);

            flt0_lo = _mm512_permutex2var_epi32(flt0_lo, _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), flt0_hi);
            _mm512_storeu_si512((__m512i*)(flt_angles[0] + dst_offset + j), flt0_lo);

            __m512i src1 = _mm512_loadu_si512((__m512i*)(src_angles[1] + src_offset + j));
            __m512i r_factor1 = _mm512_set1_epi32(i_rfactor[1]);
            __m512i dst_val1_lo = _mm512_mul_epi32(src1, r_factor1);
            __m512i dst_val1_hi = _mm512_mul_epi32(_mm512_srli_epi64(src1, 32), r_factor1);

            dst_val1_lo = _mm512_srai_epi64(_mm512_add_epi64(dst_val1_lo, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);
            dst_val1_hi = _mm512_srai_epi64(_mm512_add_epi64(dst_val1_hi, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);
            dst_val1_lo = _mm512_permutex2var_epi32(dst_val1_lo,  _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), dst_val1_hi);
            _mm512_storeu_si512((__m512i*)(dst_angles[1] + dst_offset + j), dst_val1_lo);

            __m512i flt1_lo = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(dst_val1_lo));
            __m512i flt1_hi = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_srli_epi64(_mm512_abs_epi32(dst_val1_lo), 32));
            flt1_lo = _mm512_srli_epi64(_mm512_add_epi64(flt1_lo, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]);
            flt1_hi = _mm512_srli_epi64(_mm512_add_epi64(flt1_hi, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]);

            flt1_lo = _mm512_permutex2var_epi32(flt1_lo, _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), flt1_hi);
            _mm512_storeu_si512((__m512i*)(flt_angles[1] + dst_offset + j), flt1_lo);

            __m512i src2 = _mm512_loadu_si512((__m512i*)(src_angles[2] + src_offset + j));
            __m512i r_factor2 = _mm512_set1_epi32(i_rfactor[2]);
            __m512i dst_val2_lo = _mm512_mul_epi32(src2, r_factor2);
            __m512i dst_val2_hi = _mm512_mul_epi32(_mm512_srli_epi64(src2, 32), r_factor2);

            dst_val2_lo = _mm512_srai_epi64(_mm512_add_epi64(dst_val2_lo, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);
            dst_val2_hi = _mm512_srai_epi64(_mm512_add_epi64(dst_val2_hi, _mm512_set1_epi64(add_bef_shift_dst[scale - 1])), shift_dst[scale - 1]);
            dst_val2_lo = _mm512_permutex2var_epi32(dst_val2_lo,  _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), dst_val2_hi);
            _mm512_storeu_si512((__m512i*)(dst_angles[2] + dst_offset + j), dst_val2_lo);

            __m512i flt2_lo = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_abs_epi32(dst_val2_lo));
            __m512i flt2_hi = _mm512_mul_epi32(_mm512_set1_epi32(FIX_ONE_BY_30), _mm512_srli_epi64(_mm512_abs_epi32(dst_val2_lo), 32));
            flt2_lo = _mm512_srli_epi64(_mm512_add_epi64(flt2_lo, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]);
            flt2_hi = _mm512_srli_epi64(_mm512_add_epi64(flt2_hi, _mm512_set1_epi64(add_bef_shift_flt[scale - 1])), shift_flt[scale - 1]);

            flt2_lo = _mm512_permutex2var_epi32(flt2_lo, _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), flt2_hi);
            _mm512_storeu_si512((__m512i*)(flt_angles[2] + dst_offset + j), flt2_lo);
        }

        for (int j = right_mod_16; j < right; ++j)
        {
            int32_t dst_val0 = (int32_t)(((i_rfactor[0] * (int64_t)src_angles[0][src_offset + j]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            dst_angles[0][dst_offset + j] = dst_val0;
            flt_angles[0][dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val0)) +
                add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);

            int32_t dst_val1 = (int32_t)(((i_rfactor[1] * (int64_t)src_angles[1][src_offset + j]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            dst_angles[1][dst_offset + j] = dst_val1;
            flt_angles[1][dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val1)) +
                add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);

            int32_t dst_val2 = (int32_t)(((i_rfactor[2] * (int64_t)src_angles[2][src_offset + j]) +
                add_bef_shift_dst[scale - 1]) >> shift_dst[scale - 1]);
            dst_angles[2][dst_offset + j] = dst_val2;
            flt_angles[2][dst_offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val2)) +
                add_bef_shift_flt[scale - 1]) >> shift_flt[scale - 1]);
        }
    }
}

float adm_csf_den_scale_avx512(const adm_dwt_band_t *src, int w, int h,
                               int src_stride,
                               double adm_norm_view_dist, int adm_ref_display_height)
{
    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 1, adm_norm_view_dist, adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], 0, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;

    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    int32_t shift_accum = (int32_t)ceil(log2((bottom - top)*(right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    int32_t add_shift_accum =
        shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;

    /**
     * The rfactor is multiplied at the end after cubing
     * Because d+ = (a[i]^3)*(r^3)
     * is equivalent to d+=a[i]^3 and d=d*(r^3)
     */
    int16_t *src_h = src->band_h + top * src_stride;
    int16_t *src_v = src->band_v + top * src_stride;
    int16_t *src_d = src->band_d + top * src_stride;

    int right_mod_16 = right - ((right - left) % 16);

    for (int i = top; i < bottom; ++i) {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        __m512i accum_inner_h_lo, accum_inner_h_hi, accum_inner_v_lo, accum_inner_v_hi, accum_inner_d_lo, accum_inner_d_hi;
        accum_inner_h_lo = accum_inner_h_hi = accum_inner_v_lo = accum_inner_v_hi = accum_inner_d_lo = accum_inner_d_hi = _mm512_setzero_si512();

        for (int j = left; j < right_mod_16; j += 16) {
            __m512i h = _mm512_cvtepu16_epi32(_mm256_abs_epi16(_mm256_loadu_si256((__m256i*)(src_h + j))));
            __m512i v = _mm512_cvtepu16_epi32(_mm256_abs_epi16(_mm256_loadu_si256((__m256i*)(src_v + j))));
            __m512i d = _mm512_cvtepu16_epi32(_mm256_abs_epi16(_mm256_loadu_si256((__m256i*)(src_d + j))));

            __m512i h_sq = _mm512_mullo_epi32(h, h);
            __m512i h_cu_lo = _mm512_mul_epu32(h_sq, h);
            __m512i h_cu_hi = _mm512_mul_epu32(_mm512_srli_epi64(h_sq, 32), _mm512_srli_epi64(h, 32));
            accum_inner_h_lo = _mm512_add_epi64(accum_inner_h_lo, h_cu_lo);
            accum_inner_h_hi = _mm512_add_epi64(accum_inner_h_hi, h_cu_hi);

            __m512i v_sq = _mm512_mullo_epi32(v, v);
            __m512i v_cu_lo = _mm512_mul_epu32(v_sq, v);
            __m512i v_cu_hi = _mm512_mul_epu32(_mm512_srli_epi64(v_sq, 32), _mm512_srli_epi64(v, 32));
            accum_inner_v_lo = _mm512_add_epi64(accum_inner_v_lo, v_cu_lo);
            accum_inner_v_hi = _mm512_add_epi64(accum_inner_v_hi, v_cu_hi);

            __m512i d_sq = _mm512_mullo_epi32(d, d);
            __m512i d_cu_lo = _mm512_mul_epu32(d_sq, d);
            __m512i d_cu_hi = _mm512_mul_epu32(_mm512_srli_epi64(d_sq, 32), _mm512_srli_epi64(d, 32));
            accum_inner_d_lo = _mm512_add_epi64(accum_inner_d_lo, d_cu_lo);
            accum_inner_d_hi = _mm512_add_epi64(accum_inner_d_hi, d_cu_hi);
        }

        accum_inner_h_lo = _mm512_add_epi64(accum_inner_h_lo, accum_inner_h_hi);
        __m256i h_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_h_lo), _mm512_extracti64x4_epi64(accum_inner_h_lo, 1));
        __m128i h_r2 = _mm_add_epi64(_mm256_castsi256_si128(h_r4), _mm256_extracti64x2_epi64(h_r4, 1));
        uint64_t h_r1 = h_r2[0] + h_r2[1];
        
        accum_inner_v_lo = _mm512_add_epi64(accum_inner_v_lo, accum_inner_v_hi);
        __m256i v_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_v_lo), _mm512_extracti64x4_epi64(accum_inner_v_lo, 1));
        __m128i v_r2 = _mm_add_epi64(_mm256_castsi256_si128(v_r4), _mm256_extracti64x2_epi64(v_r4, 1));
        uint64_t v_r1 = v_r2[0] + v_r2[1];

        accum_inner_d_lo = _mm512_add_epi64(accum_inner_d_lo, accum_inner_d_hi);
        __m256i d_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_d_lo), _mm512_extracti64x4_epi64(accum_inner_d_lo, 1));
        __m128i d_r2 = _mm_add_epi64(_mm256_castsi256_si128(d_r4), _mm256_extracti64x2_epi64(d_r4, 1));
        uint64_t d_r1 = d_r2[0] + d_r2[1];

        for (int j = right_mod_16; j < right; ++j) {
            uint16_t h = (uint16_t)abs(src_h[j]);
            uint16_t v = (uint16_t)abs(src_v[j]);
            uint16_t d = (uint16_t)abs(src_d[j]);

            uint64_t val = ((uint64_t)h * h) * h;
            accum_inner_h += val;
            val = ((uint64_t)v * v) * v;
            accum_inner_v += val;
            val = ((uint64_t)d * d) * d;
            accum_inner_d += val;
        }

        /**
         * max_value of h^3, v^3, d^3 is 1.205624776 * —10^13
         * accum_h can hold till 1.844674407 * —10^19
         * accum_h's maximum is reached when it is 2^20 * max(h^3)
         * Therefore the accum_h,v,d is shifted based on width and height subtracted by 20
         */
        accum_h += (accum_inner_h + h_r1 + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + v_r1 + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + d_r1 + add_shift_accum) >> shift_accum;
        src_h += src_stride;
        src_v += src_stride;
        src_d += src_stride;
    }

    /**
     * rfactor is multiplied after cubing
     * accum_h,v,d is converted to floating-point for score calculation
     * 6bits are yet to be shifted from previous stage that is after dwt hence
     * after cubing 18bits are to shifted
     * Hence final shift is 18-shift_accum
     */
    double shift_csf = pow(2, (18 - shift_accum));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return(den_scale_h + den_scale_v + den_scale_d);

}

float adm_csf_den_s123_avx512(const i4_adm_dwt_band_t *src, int scale, int w, int h,
                              int src_stride,
                              double adm_norm_view_dist, int adm_ref_display_height)
{
    // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
    // 1 to 4 (from finest scale to coarsest scale).
    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist, adm_ref_display_height);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist, adm_ref_display_height);
    const float rfactor[3] = { 1.0f / factor1, 1.0f / factor1, 1.0f / factor2 };

    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    const uint32_t shift_sq[3] = { 31, 30, 31 };
    const uint32_t accum_convert_float[3] = { 32, 27, 23 };
    const uint32_t add_shift_sq[3] =
        { 1u << shift_sq[0], 1u << shift_sq[1], 1u << shift_sq[2] };

    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    uint32_t add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));
    uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
    uint32_t add_shift_accum = (uint32_t)pow(2, (shift_accum - 1));

    int32_t *src_h = src->band_h + top * src_stride;
    int32_t *src_v = src->band_v + top * src_stride;
    int32_t *src_d = src->band_d + top * src_stride;

    int right_mod_8 = right - ((right - left) % 8);
    for (int i = top; i < bottom; ++i)
    {
        uint64_t accum_inner_h = 0;
        uint64_t accum_inner_v = 0;
        uint64_t accum_inner_d = 0;
        __m512i accum_inner_h_512, accum_inner_v_512, accum_inner_d_512;
        accum_inner_h_512 = accum_inner_v_512 = accum_inner_d_512 = _mm512_setzero_si512();
        for (int j = left; j < right_mod_8; j+=8)
        {
            __m512i h = _mm512_cvtepu32_epi64(_mm256_abs_epi32(_mm256_loadu_si256((__m256i*)(src_h + j))));
            __m512i v = _mm512_cvtepu32_epi64(_mm256_abs_epi32(_mm256_loadu_si256((__m256i*)(src_v + j))));
            __m512i d = _mm512_cvtepu32_epi64(_mm256_abs_epi32(_mm256_loadu_si256((__m256i*)(src_d + j))));

            __m512i h_sq = _mm512_add_epi64(_mm512_mul_epu32(h, h), _mm512_set1_epi64(add_shift_sq[scale - 1]));
            h_sq = _mm512_srli_epi64(h_sq, shift_sq[scale - 1]);
            __m512i h_cu = _mm512_add_epi64(_mm512_mullo_epi64(h_sq, h), _mm512_set1_epi64(add_shift_cub));
            h_cu = _mm512_srli_epi64(h_cu, shift_cub);
            accum_inner_h_512 = _mm512_add_epi64(accum_inner_h_512, h_cu);

            __m512i v_sq = _mm512_add_epi64(_mm512_mul_epu32(v, v), _mm512_set1_epi64(add_shift_sq[scale - 1]));
            v_sq = _mm512_srli_epi64(v_sq, shift_sq[scale - 1]);
            __m512i v_cu = _mm512_add_epi64(_mm512_mullo_epi64(v_sq, v), _mm512_set1_epi64(add_shift_cub));
            v_cu = _mm512_srli_epi64(v_cu, shift_cub);
            accum_inner_v_512 = _mm512_add_epi64(accum_inner_v_512, v_cu);

            __m512i d_sq = _mm512_add_epi64(_mm512_mul_epu32(d, d), _mm512_set1_epi64(add_shift_sq[scale - 1]));
            d_sq = _mm512_srli_epi64(d_sq, shift_sq[scale - 1]);
            __m512i d_cu = _mm512_add_epi64(_mm512_mullo_epi64(d_sq, d), _mm512_set1_epi64(add_shift_cub));
            d_cu = _mm512_srli_epi64(d_cu, shift_cub);
            accum_inner_d_512 = _mm512_add_epi64(accum_inner_d_512, d_cu);
        }
        __m256i h_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_h_512), _mm512_extracti64x4_epi64(accum_inner_h_512, 1));
        __m128i h_r2 = _mm_add_epi64(_mm256_castsi256_si128(h_r4), _mm256_extracti64x2_epi64(h_r4, 1));
        uint64_t h_r1 = h_r2[0] + h_r2[1];

        __m256i d_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_d_512), _mm512_extracti64x4_epi64(accum_inner_d_512, 1));
        __m128i d_r2 = _mm_add_epi64(_mm256_castsi256_si128(d_r4), _mm256_extracti64x2_epi64(d_r4, 1));
        uint64_t d_r1 = d_r2[0] + d_r2[1];

        __m256i v_r4 = _mm256_add_epi64(_mm512_castsi512_si256(accum_inner_v_512), _mm512_extracti64x4_epi64(accum_inner_v_512, 1));
        __m128i v_r2 = _mm_add_epi64(_mm256_castsi256_si128(v_r4), _mm256_extracti64x2_epi64(v_r4, 1));
        uint64_t v_r1 = v_r2[0] + v_r2[1];

        for (int j = right_mod_8; j < right; ++j)
        {
            uint32_t h = (uint32_t)abs(src_h[j]);
            uint32_t v = (uint32_t)abs(src_v[j]);
            uint32_t d = (uint32_t)abs(src_d[j]);

            uint64_t val = ((((((uint64_t)h * h) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * h) + add_shift_cub) >> shift_cub;

            accum_inner_h += val;


            val = ((((((uint64_t)v * v) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * v) + add_shift_cub) >> shift_cub;
            accum_inner_v += val;

            val = ((((((uint64_t)d * d) + add_shift_sq[scale - 1]) >>
                shift_sq[scale - 1]) * d) + add_shift_cub) >> shift_cub;
            accum_inner_d += val;

        }

        accum_h += (accum_inner_h + h_r1 + add_shift_accum) >> shift_accum;
        accum_v += (accum_inner_v + v_r1 + add_shift_accum) >> shift_accum;
        accum_d += (accum_inner_d + d_r1 + add_shift_accum) >> shift_accum;

        src_h += src_stride;
        src_v += src_stride;
        src_d += src_stride;
    }
    /**
     * All the results are converted to floating-point to calculate the scores
     * For all scales the final shift is 3*shifts from dwt - total shifts done here
     */
    double shift_csf = pow(2, (accum_convert_float[scale - 1] - shift_accum - shift_cub));
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return (den_scale_h + den_scale_v + den_scale_d);
}
