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

double compute_vmaf(char* fmt, int width, int height, int (*read_frame)(float *ref_data, int *ref_stride, float *main_data, int *main_stride, double *score, void *user_data), char *model_path, char *log_path, char *log_fmt, int disable_clip, int disable_avx, int enable_transform, int phone_model, int do_psnr, int do_ssim, int do_ms_ssim, char *pool_method, void *user_data)
	{
		if(enable_transform || phone_model){
			enable_transform = 1;
		}
		cpu = cpu_autodetect();

        if (disable_avx)
        {
            cpu = VMAF_CPU_NONE;
        }

		double score = RunVmaf1(fmt, width, height, read_frame, model_path, user_data, log_path, log_fmt, disable_clip, enable_transform, do_psnr, do_ssim, do_ms_ssim, pool_method);
		
		return score;

	}

}
