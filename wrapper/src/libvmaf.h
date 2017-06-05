#ifndef LIBVMAF_H_
#define LIBVMAF_H_

#ifdef __cplusplus
extern "C" {
#endif

double compute_vmaf(char* fmt, int width, int height, char *ref_path, char *dis_path, char *model_path);

#ifdef __cplusplus
}
#endif

#endif /* _LIBVMAF_H */
