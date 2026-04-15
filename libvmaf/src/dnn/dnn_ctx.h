/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Internal bridge between the public `dnn.h` surface (dnn_api.c) and the
 *  VmafContext-owning translation unit (libvmaf.c). Keeps VmafContext opaque
 *  to the DNN module while letting dnn_api.c hand off the opened ORT session
 *  for per-frame inference.
 */

#ifndef LIBVMAF_DNN_DNN_CTX_H_
#define LIBVMAF_DNN_DNN_CTX_H_

#include <stddef.h>
#include <stdint.h>

#include "libvmaf/libvmaf.h"

#include "model_loader.h"
#include "ort_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Attach an opened ORT session to the VmafContext. Ownership of @p sess,
 * @p meta, and @p feature_name (strdup'd) is transferred to the context; the
 * caller must not close/free them on success. On failure the caller retains
 * ownership.
 *
 * @param in_shape  static input shape reported by ORT (rank dims).
 * @param in_rank   rank of in_shape. Must be 3 (CHW) or 4 (NCHW).
 *
 * Currently accepts NCHW shape [1, 1, H, W] — single-channel luma. Other
 * shapes are rejected with -ENOTSUP so the failure is visible, not silent.
 */
int vmaf_ctx_dnn_attach(VmafContext *ctx,
                        VmafOrtSession *sess,
                        const VmafModelSidecar *meta,
                        const int64_t *in_shape, size_t in_rank,
                        const char *feature_name);

/** Returns 1 if a tiny model is attached to @p ctx, else 0. */
int vmaf_ctx_dnn_has_session(const VmafContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_DNN_CTX_H_ */
