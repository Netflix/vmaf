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

#ifndef COMPUTE_VMAF_H_
#define COMPUTE_VMAF_H_

#ifdef __cplusplus
extern "C" {
#endif

 /**
  *
  *
  * **compute_vmaf** - Run VMAF on a series of frames from the supplied `read_frame` function
  *
  *
  * `read_frame` takes the following arguments:
  * - `float *ref_data` : A pointer to a floating point array of at least width * height elements,
  *   which the read_frame function is to fill with the luminance data from the reference video.
  *   The values must be between 0.0 to 255.0 (NOT 1.0).
  *
  * - `float *main_data` : A pointer to a floating point array of at least width * height elements,
  *   which the read_frame function is to fill with the luminance data from the video to test.
  *   The values must be between 0.0 to 255.0 (NOT 1.0).
  *
  * - `float *temp_data` : A pointer to a floating point array of at least width * height elements.
  *   This is provided as a convenience for the user, and can be filled with any intermediate
  *   values. The read_frame can ignore this if not needed.
  *
  * - `int stride_byte` : The number of bytes required between consecutive rows in
  *   the buffers written to `ref_data` and `main_data`.
  *
  * - `void *user_data` : The same `user_data` pointer that the user provides to `compute_vmaf`.
  *
  * Other arguments to `compute_vmaf`:
  * - `void *user_data` : Pointer to user-specific data structure that can be used to ensure `read_frame` is
  *   provided with information of the image (width, height) and where to read it from
  *   (e.g. file pointers, iteration counters).
  *
  * Most other parameters can be deduced from the usage printout of `vmafossexec`.
  * The rest can be safely be left at 0 unless the user has specific requirements
  */

#ifdef __GNUC__
__attribute__ ((deprecated))
#elif defined(_MSC_VER)
__declspec(deprecated)
#else
#pragma message("compute_vmaf() is deprecated")
#endif
int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data),
				 void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip, int disable_avx, int enable_transform, int phone_model, int do_psnr,
				 int do_ssim, int do_ms_ssim, char *pool_method, int n_thread, int n_subsample, int enable_conf_interval);

#ifdef __cplusplus
}
#endif

#endif /* COMPUTE_VMAF_H */
