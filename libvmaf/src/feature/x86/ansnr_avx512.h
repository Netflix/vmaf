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

#ifndef X86_AVX512_ANSNR_H_
#define X86_AVX512_ANSNR_H_

void ansnr_mse_line_avx512(const float *ref, const float *dis,
                            float *sig_accum, float *noise_accum, int w);

#endif /* X86_AVX512_ANSNR_H_ */
