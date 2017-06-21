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

#include "libvmaf.h"
#include "vmaf_wrapper.h"
#include <cstdio>
#include <cstdint>
#include "cpu.h"

extern "C" {

enum vmaf_cpu cpu; // global

double compute_vmaf(char* fmt, int width, int height, int (*read_frame)(float *ref_data, int *ref_stride, float *main_data, int *main_stride, double *score, void *user_data), char *model_path, void *user_data)
	{
		char *log_path = NULL;
		char *log_fmt = NULL;
		int disable_clip = 0;
		int disable_avx = 0;
		int enable_transform = 0;
		int do_psnr = 1;
		int do_ssim = 1;
		int do_ms_ssim = 1;
		char *pool_method = 0;
		int *ref_buf;

		cpu = cpu_autodetect();

		double score = RunVmaf1(fmt, width, height, read_frame, model_path, user_data, log_path, log_fmt, disable_clip, enable_transform, do_psnr, do_ssim, do_ms_ssim, pool_method);
		
		return score;

	}

}
