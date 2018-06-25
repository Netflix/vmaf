/**
 *
 *  Copyright 2016-2018 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef LIBVMAF_H_
#define LIBVMAF_H_

#ifdef __cplusplus
extern "C" {
#endif

int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data),
				 void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip, int disable_avx, int enable_transform, int phone_model, int do_psnr,
				 int do_ssim, int do_ms_ssim, char *pool_method, int n_thread, int n_subsample, int enable_conf_interval);

#ifdef __cplusplus
}
#endif

#endif /* _LIBVMAF_H */
