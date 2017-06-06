#include "libvmaf.h"
#include "vmaf_wrapper.h"
#include <cstdio>

extern "C" {

	double compute_vmaf(char* fmt, int width, int height, char *ref_path, char *dis_path, char *model_path)
	{
		//printf("under libvmaf\n");   
		char *log_path = NULL;
		char *log_fmt = NULL;
		int disable_clip = 0;
		int disable_avx = 0;
		int enable_transform = 0;
		int do_psnr = 0;
		int do_ssim = 0;
		int do_ms_ssim = 0;
		char *pool_method = 0;

		double score = RunVmaf1(fmt, width, height, ref_path, dis_path, model_path, log_path, log_fmt, disable_clip, enable_transform, do_psnr, do_ssim, do_ms_ssim, pool_method);

		return score;

	}

}
