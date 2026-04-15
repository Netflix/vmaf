/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_DNN_ORT_BACKEND_H_
#define LIBVMAF_DNN_ORT_BACKEND_H_

#include <stddef.h>

#include "libvmaf/dnn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafOrtSession VmafOrtSession;

int  vmaf_ort_open (VmafOrtSession **out, const char *onnx_path,
                    const VmafDnnConfig *cfg);
int  vmaf_ort_infer(VmafOrtSession *sess,
                    const float *input, const int64_t *input_shape,
                    size_t input_rank,
                    float *output, size_t output_capacity,
                    size_t *output_written);
/**
 * Fetch the static shape of input 0. @p max_rank caps the number of dims
 * written; actual rank is returned via @p out_rank. Dimensions marked dynamic
 * (symbolic) in the model are reported as -1. Returns 0 on success.
 */
int  vmaf_ort_input_shape(VmafOrtSession *sess,
                          int64_t *out_shape, size_t max_rank,
                          size_t *out_rank);
void vmaf_ort_close(VmafOrtSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_ORT_BACKEND_H_ */
