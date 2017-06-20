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

#ifndef LIBVMAF_H_
#define LIBVMAF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

double compute_vmaf(char* fmt, int width, int height, int (*read_frame)(float *ref_data, int *ref_stride, float *main_data, int *main_stride, double *score), char *model_path);


#ifdef __cplusplus
}
#endif

#endif /* _LIBVMAF_H */
