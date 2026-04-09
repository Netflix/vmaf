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

#ifndef IQA_SSIM_SIMD_H_
#define IQA_SSIM_SIMD_H_

typedef void (*ssim_precompute_fn)(const float *ref, const float *cmp,
                                    float *ref_sq, float *cmp_sq,
                                    float *ref_cmp, int n);

typedef void (*ssim_variance_fn)(float *ref_sigma_sqd, float *cmp_sigma_sqd,
                                  float *sigma_both, const float *ref_mu,
                                  const float *cmp_mu, int n);

typedef void (*ssim_accumulate_fn)(const float *ref_mu, const float *cmp_mu,
                                    const float *ref_sigma_sqd,
                                    const float *cmp_sigma_sqd,
                                    const float *sigma_both, int n,
                                    float C1, float C2, float C3,
                                    double *ssim_sum, double *l_sum,
                                    double *c_sum, double *s_sum);

void _iqa_ssim_set_dispatch(ssim_precompute_fn precompute,
                             ssim_variance_fn variance,
                             ssim_accumulate_fn accumulate);

#endif /* IQA_SSIM_SIMD_H_ */
