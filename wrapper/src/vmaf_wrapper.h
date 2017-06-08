#ifndef VMAFWRAPPER_H_
#define VMAFWRAPPER_H_


#ifdef __cplusplus
extern "C" {
#endif

typedef struct Asset Asset;

Asset* newAsset();

int get_width(Asset* v);
int get_height(Asset* v);
const char* get_ref_path(Asset* v);
const char* get_dis_path(Asset* v);
const char* get_fmt(Asset* v);

void deleteAsset(Asset* v);

double RunVmaf1(const char* fmt, int width, int height,
	           const uint8_t *ref, const uint8_t *main, const char *model_path,
	           const char *log_path, const char *log_fmt,
	           int disable_clip, int enable_transform,
	           int do_psnr, int do_ssim, int do_ms_ssim,
	           const char *pool_method);



#ifdef __cplusplus
}
#endif
#endif
