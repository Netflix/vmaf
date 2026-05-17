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

int vmaf_ort_open(VmafOrtSession **out, const char *onnx_path, const VmafDnnConfig *cfg);
int vmaf_ort_infer(VmafOrtSession *sess, const float *input, const int64_t *input_shape,
                   size_t input_rank, float *output, size_t output_capacity,
                   size_t *output_written);

/**
 * Multi-input / multi-output inference. @p inputs carries @p n_inputs
 * descriptors; each binds by ONNX input name when @p in_names[i] is
 * non-NULL, else positionally. Same convention for outputs. All tensors
 * are float32. On -ENOSPC (output buffer too small) the @p out_written
 * entry for the short buffer is set to the required element count.
 */
typedef struct {
    const char *name;
    const float *data;
    const int64_t *shape;
    size_t rank;
} VmafOrtTensorIn;

typedef struct {
    const char *name;
    float *data;
    size_t capacity;
    size_t written;
} VmafOrtTensorOut;

int vmaf_ort_run(VmafOrtSession *sess, const VmafOrtTensorIn *inputs, size_t n_inputs,
                 VmafOrtTensorOut *outputs, size_t n_outputs);

/** Number of graph inputs / outputs in the loaded model. */
int vmaf_ort_io_count(VmafOrtSession *sess, size_t *n_inputs, size_t *n_outputs);

/**
 * Maximum number of graph inputs or outputs supported by vmaf_ort_run.
 * Callers with more IO must use a dedicated path; the limit covers every
 * VMAF model shipped to date (max observed: 6).
 */
#define VMAF_ORT_MAX_IO 8

/**
 * Fetch the static shape of input 0. @p max_rank caps the number of dims
 * written; actual rank is returned via @p out_rank. Dimensions marked dynamic
 * (symbolic) in the model are reported as -1. Returns 0 on success.
 */
int vmaf_ort_input_shape(VmafOrtSession *sess, int64_t *out_shape, size_t max_rank,
                         size_t *out_rank);

/**
 * Name of the ONNX Runtime execution provider that actually attached to
 * the session. Populated at open time; owned by the session. Stable
 * strings: "CPU", "CUDA", "OpenVINO:GPU", "OpenVINO:CPU", "OpenVINO:NPU",
 * "CoreML", "CoreML:ANE", "CoreML:GPU", "CoreML:CPU", "ROCm". Returns NULL
 * if @p sess is NULL.
 */
const char *vmaf_ort_attached_ep(const VmafOrtSession *sess);

void vmaf_ort_close(VmafOrtSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_ORT_BACKEND_H_ */
