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
