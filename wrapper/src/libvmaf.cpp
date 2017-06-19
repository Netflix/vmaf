#include "libvmaf.h"
#include "vmaf_wrapper.h"
#include <cstdio>
#include <cstdint>
#include "cpu.h"

extern "C" {

enum vmaf_cpu cpu; // global

double compute_vmaf(char* fmt, int width, int height, int (*read_frame)(float *ref_data, int *ref_stride, float *main_data, int *main_stride, double *score), char *model_path)
	{
		printf("under libvmaf\n");   
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

		double score = RunVmaf1(fmt, width, height, read_frame, model_path, log_path, log_fmt, disable_clip, enable_transform, do_psnr, do_ssim, do_ms_ssim, pool_method);

		return 0.0;

	}

}
