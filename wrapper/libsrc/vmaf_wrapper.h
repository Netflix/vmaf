/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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

#ifndef VMAFWRAPPER_H_
#define VMAFWRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

double RunVmaf1(char* fmt, int width, int height, int (*read_frame)(float *ref_buf, int *ref_stride, float *main_buf, int *main_stride, double *score, void *user_data), const char *model_path, void *user_data,
	           const char *log_path, const char *log_fmt,
	           int disable_clip, int enable_transform,
	           int do_psnr, int do_ssim, int do_ms_ssim,
	           const char *pool_method);

#ifdef __cplusplus
}
#endif
#endif
