/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#ifndef X86_AVX2_PSNR_H_
#define X86_AVX2_PSNR_H_

#include <stdint.h>

uint32_t psnr_sse_line_8_avx2(const uint8_t *ref, const uint8_t *dis,
                               unsigned w);

uint64_t psnr_sse_line_16_avx2(const uint16_t *ref, const uint16_t *dis,
                                unsigned w);

#endif /* X86_AVX2_PSNR_H_ */
