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

float vmaf_image_sad_c(const float *img1, const float *img2, int width, int height, int img1_stride, int img2_stride, int motion_add_scale1);
int compute_motion(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, int motion_decimate);
