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

#ifndef X86_AVX512_SSIM_H_
#define X86_AVX512_SSIM_H_

void ssim_precompute_avx512(const float *ref, const float *cmp,
                             float *ref_sq, float *cmp_sq,
                             float *ref_cmp, int n);

void ssim_variance_avx512(float *ref_sigma_sqd, float *cmp_sigma_sqd,
                           float *sigma_both, const float *ref_mu,
                           const float *cmp_mu, int n);

void ssim_accumulate_avx512(const float *ref_mu, const float *cmp_mu,
                             const float *ref_sigma_sqd,
                             const float *cmp_sigma_sqd,
                             const float *sigma_both, int n,
                             float C1, float C2, float C3,
                             double *ssim_sum, double *l_sum,
                             double *c_sum, double *s_sum);

#endif /* X86_AVX512_SSIM_H_ */
