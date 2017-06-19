#ifndef VMAFWRAPPER_H_
#define VMAFWRAPPER_H_


#ifdef __cplusplus
extern "C" {
#endif

double RunVmaf1(char* fmt, int width, int height, int (*read_frame)(float *ref_buf, int *ref_stride, float *main_buf, int *main_stride, double *score), const char *model_path,
	           const char *log_path, const char *log_fmt,
	           int disable_clip, int enable_transform,
	           int do_psnr, int do_ssim, int do_ms_ssim,
	           const char *pool_method);



#ifdef __cplusplus
}
#endif
#endif
