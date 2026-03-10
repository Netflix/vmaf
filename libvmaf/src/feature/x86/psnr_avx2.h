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

#ifndef PSNR_AVX2_H_
#define PSNR_AVX2_H_

#include <stddef.h>
#include <stdint.h>

void psnr_sse_8_avx2(const uint8_t *ref, const uint8_t *dis,
                     unsigned w, unsigned h,
                     ptrdiff_t stride_ref, ptrdiff_t stride_dis,
                     uint64_t *sse);

#endif /* PSNR_AVX2_H_ */
