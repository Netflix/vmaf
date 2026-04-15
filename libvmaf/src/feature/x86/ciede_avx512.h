/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#ifndef X86_AVX512_CIEDE_H_
#define X86_AVX512_CIEDE_H_

#include <stdint.h>

void ciede_preprocess_8_avx512(const uint8_t *y_buf, const uint8_t *u_buf,
                                const uint8_t *v_buf, float *out_y,
                                float *out_u, float *out_v, int w);

void ciede_preprocess_16_avx512(const uint16_t *y_buf, const uint16_t *u_buf,
                                 const uint16_t *v_buf, float *out_y,
                                 float *out_u, float *out_v, int w);

#endif /* X86_AVX512_CIEDE_H_ */
