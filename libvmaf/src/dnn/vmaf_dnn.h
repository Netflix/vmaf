/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *
 *  Licensed under the BSD+Patent License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      https://opensource.org/licenses/BSDplusPatent
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

/**
 * @file vmaf_dnn.h
 * @brief Internal DNN inference surface for libvmaf (ONNX Runtime backed).
 *
 * This header is intentionally minimal. It is a stable seam for the
 * feature-extractor hookup; the implementation is gated behind
 * `-Denable_dnn=true` and is off by default.
 */

#ifndef LIBVMAF_DNN_VMAF_DNN_H_
#define LIBVMAF_DNN_VMAF_DNN_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafDnnSession VmafDnnSession;

typedef enum {
    VMAF_DNN_PROVIDER_CPU = 0,
    VMAF_DNN_PROVIDER_CUDA,
    VMAF_DNN_PROVIDER_DML,    /* DirectML (Windows)        */
    VMAF_DNN_PROVIDER_COREML, /* CoreML (Apple)            */
    VMAF_DNN_PROVIDER_SYCL,   /* via oneDNN / OpenVINO EP  */
} VmafDnnProvider;

/**
 * Open an ONNX model and create an inference session.
 * @return 0 on success, negative errno on failure.
 */
int vmaf_dnn_open(VmafDnnSession **out, const char *path,
                  VmafDnnProvider provider);

/**
 * Run one forward pass. @p features is a contiguous float32 vector of length
 * @p n_features; the scalar score is written to @p score.
 */
int vmaf_dnn_infer(VmafDnnSession *sess, const float *features,
                   size_t n_features, double *score);

/** Close the session and free all associated resources. */
void vmaf_dnn_close(VmafDnnSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_VMAF_DNN_H_ */
