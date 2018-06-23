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

#include "libvmaf.h"
#include "vmaf.h"
#include <cstdio>
#include "cpu.h"

extern "C" {

enum vmaf_cpu cpu; // global

int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride_byte, void *user_data),
				 void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip, int disable_avx, int enable_transform, int phone_model, int do_psnr,
				 int do_ssim, int do_ms_ssim, char *pool_method, int n_thread, int n_subsample, int enable_conf_interval)
	{
		bool d_c = false;
		bool d_a = false;
		bool e_t = false;
		bool d_p = false;
		bool d_s = false;
		bool d_m_s = false;	

		if(enable_transform || phone_model){
			e_t = true;
		}
		if(disable_clip){
			d_c = true;
		}
		if(disable_avx){
			d_a = true;
		}
		if(do_psnr){
			d_p = true;
		}
		if(do_ssim){
			d_s = true;
		}
		if(do_ms_ssim){
			d_m_s = true;
		}
		
		cpu = cpu_autodetect();

        if (disable_avx)
        {
            cpu = VMAF_CPU_NONE;
        }

        try {
            double score = RunVmaf(fmt, width, height, read_frame, user_data, model_path, log_path, log_fmt, d_c, e_t, d_p, d_s, d_m_s, pool_method, n_thread, n_subsample, enable_conf_interval);
            *vmaf_score = score;
            return 0;
        }
        catch (VmafException& e)
        {
            printf("Caught VmafException: %s\n", e.what());
            return -2;
        }
        catch (std::runtime_error& e)
        {
            printf("Caught runtime_error: %s\n", e.what());
            return -3;
        }
        catch (std::logic_error& e)
        {
            printf("Caught logic_error: %s\n", e.what());
            return -4;
        }
    }

}
