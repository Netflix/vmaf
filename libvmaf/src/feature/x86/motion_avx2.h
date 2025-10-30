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

#ifndef X86_AVX2_MOTION_H_
#define X86_AVX2_MOTION_H_

#include <stdint.h>

void y_convolution_8_avx2(void *src, uint16_t *dst, unsigned width,
                unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride,
                unsigned inp_size_bits);

void y_convolution_16_avx2(void *src, uint16_t *dst, unsigned width,
                 unsigned height, ptrdiff_t src_stride,
                 ptrdiff_t dst_stride, unsigned inp_size_bits);

void x_convolution_16_avx2(const uint16_t *src, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride);

void sad_avx2(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad);

#endif /* X86_AVX2_MOTION_H_ */
