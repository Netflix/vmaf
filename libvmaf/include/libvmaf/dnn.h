/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Licensed under the BSD+Patent License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      https://opensource.org/licenses/BSDplusPatent
 */

/**
 * @file dnn.h
 * @brief Public DNN surface — load/execute tiny ONNX models alongside SVM models.
 *
 * All functions return 0 on success and a negative errno on failure. When
 * libvmaf was built with `-Denable_dnn=false`, `vmaf_dnn_available()`
 * returns 0 and every other entry point returns -ENOSYS.
 */

#ifndef __VMAF_DNN_H__
#define __VMAF_DNN_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VmafDnnDevice {
    VMAF_DNN_DEVICE_AUTO     = 0,
    VMAF_DNN_DEVICE_CPU      = 1,
    VMAF_DNN_DEVICE_CUDA     = 2,
    VMAF_DNN_DEVICE_OPENVINO = 3,   /**< covers SYCL / oneAPI / Intel GPU */
    VMAF_DNN_DEVICE_ROCM     = 4,
} VmafDnnDevice;

typedef struct VmafDnnConfig {
    VmafDnnDevice device;
    int           device_index;     /**< multi-GPU index; 0 for single-GPU/CPU */
    int           threads;          /**< CPU EP intra-op threads; 0 = ORT default */
    bool          fp16_io;          /**< request fp16 tensors when supported */
} VmafDnnConfig;

/**
 * Returns 1 if libvmaf was built with DNN support (-Denable_dnn=true) and
 * ONNX Runtime is linked, 0 otherwise.
 */
int vmaf_dnn_available(void);

/**
 * Attach a tiny ONNX model (C1 / C2) to @p ctx. The model is registered
 * alongside any SVM models and participates in the same per-frame pipeline.
 *
 * @param ctx        live VmafContext (from vmaf_init())
 * @param onnx_path  filesystem path to a .onnx file; must be a regular file
 * @param cfg        optional device config; NULL uses VMAF_DNN_DEVICE_AUTO
 *
 * @return 0 on success, -ENOSYS if built without DNN support, -EINVAL on bad
 *         args, -ENOENT if the path does not exist, -E2BIG if the file is
 *         larger than VMAF_MAX_MODEL_BYTES (default 50 MB).
 */
int vmaf_use_tiny_model(VmafContext *ctx,
                        const char *onnx_path,
                        const VmafDnnConfig *cfg);

/**
 * Standalone DNN session for filter-style inference (learned pre-processing,
 * C3). Unlike vmaf_use_tiny_model() this path does NOT need a VmafContext —
 * intended for consumers that want luma-in / luma-out without scoring.
 */
typedef struct VmafDnnSession VmafDnnSession;

/**
 * Open a session against @p onnx_path. Applies the same size-cap + allowlist
 * validation as vmaf_use_tiny_model().
 */
int vmaf_dnn_session_open(VmafDnnSession **out,
                          const char *onnx_path,
                          const VmafDnnConfig *cfg);

/**
 * Run one luma-in / luma-out pass. The model's input must be NCHW
 * [1, 1, H, W] float32. Input luma is normalised to [0,1] (and mean/std
 * from the sidecar if available); output is denormalised, rounded, and
 * clamped to [0, 255].
 *
 * @return 0 on success, -ENOTSUP if the model shape is not luma-only,
 *         -ERANGE if @p w/@p h don't match the model's static input shape.
 */
int vmaf_dnn_session_run_luma8(VmafDnnSession *sess,
                               const uint8_t *in,  size_t in_stride,
                               int w, int h,
                               uint8_t *out, size_t out_stride);

void vmaf_dnn_session_close(VmafDnnSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_DNN_H__ */
