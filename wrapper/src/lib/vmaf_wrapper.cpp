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

#include "vmaf_lib.h"
#include "vmaf_wrapper.h"
#include <cstdint>

extern "C" {

	double RunVmaf1(char* fmt, int width, int height, int (*read_frame)(float *ref_buf, int *ref_stride, float *main_buf, int *main_stride, double *score), const char *model_path,const char *log_path, const char *log_fmt,
	           int disable_clip, int enable_transform,
	           int do_psnr, int do_ssim, int do_ms_ssim,
	           const char *pool_method){   
		
		bool d_c = false;
		bool disable_avx = false;
		bool e_t = false;
		bool d_p = false;
		bool d_s = false;
		bool d_m_s = false;

		if(disable_clip){
			d_c = true;	
		}
		if(enable_transform){
			e_t = true;	
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

		try
		{
		    double score = RunVmaf(fmt, width, height, read_frame, model_path, log_path, log_fmt, d_c, e_t, d_p, d_s, d_m_s, pool_method);
		    return score;
		}
		catch (const std::exception &e)
		{
		    fprintf(stderr, "Error: %s\n", e.what());
		    return -1;
		}
	

	}


}
