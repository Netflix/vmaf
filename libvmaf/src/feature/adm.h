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

int compute_adm(const float *ref, const float *dis, int w, int h,
                int ref_stride, int dis_stride, double *score,
                double *score_num, double *score_den, double *scores,
                double border_factor, double adm_enhn_gain_limit,
                double adm_norm_view_dist, int adm_ref_display_height,
                int adm_csf_mode);
