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
void vmaf_ort_close(VmafOrtSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_ORT_BACKEND_H_ */
