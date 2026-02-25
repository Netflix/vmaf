/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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

#ifndef X86_AVX512_MOTION_V2_H_
#define X86_AVX512_MOTION_V2_H_

#include <stddef.h>
#include <stdint.h>

uint64_t motion_score_pipeline_8_avx512(const uint8_t *prev, ptrdiff_t prev_stride,
                                        const uint8_t *cur, ptrdiff_t cur_stride,
                                        int32_t *y_row, unsigned w, unsigned h,
                                        unsigned bpc);

uint64_t motion_score_pipeline_16_avx512(const uint8_t *prev, ptrdiff_t prev_stride,
                                         const uint8_t *cur, ptrdiff_t cur_stride,
                                         int32_t *y_row, unsigned w, unsigned h,
                                         unsigned bpc);

#endif /* X86_AVX512_MOTION_V2_H_ */
