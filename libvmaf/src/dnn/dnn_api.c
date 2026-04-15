/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Public `libvmaf/dnn.h` entry points. Thin wrapper that:
 *    1. validates the path + file (size cap, regular-file, sidecar)
 *    2. opens an ORT session via ort_backend.c
 *    3. registers the session with the VmafContext so per-frame extraction
 *       dispatches to the DNN runtime alongside SVM models.
 *
 *  When built with -Denable_dnn=false, this TU compiles a stub that returns
 *  -ENOSYS from every entry point so consumers degrade gracefully.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/dnn.h"

#include "model_loader.h"
#include "ort_backend.h"

int vmaf_dnn_available(void)
{
#if defined(VMAF_HAVE_DNN) && VMAF_HAVE_DNN
    return 1;
#else
    return 0;
#endif
}

int vmaf_use_tiny_model(VmafContext *ctx,
                        const char *onnx_path,
                        const VmafDnnConfig *cfg)
{
#if defined(VMAF_HAVE_DNN) && VMAF_HAVE_DNN
    if (!ctx || !onnx_path) return -EINVAL;

    size_t max_bytes = VMAF_DNN_DEFAULT_MAX_BYTES;
    const char *env = getenv("VMAF_MAX_MODEL_BYTES");
    if (env && *env) {
        char *endp = NULL;
        unsigned long v = strtoul(env, &endp, 10);
        if (endp && *endp == '\0' && v > 0) {
            max_bytes = (size_t) v;
        }
    }
    int rc = vmaf_dnn_validate_onnx(onnx_path, max_bytes);
    if (rc < 0) return rc;

    VmafModelSidecar meta;
    rc = vmaf_dnn_sidecar_load(onnx_path, &meta);
    /* Missing sidecar is not fatal — we only need it for NR/FR disambiguation
     * and pretty-printing. Lack of a sidecar defaults to FR. */
    if (rc < 0 && rc != -ENOENT) {
        return rc;
    }

    VmafOrtSession *sess = NULL;
    rc = vmaf_ort_open(&sess, onnx_path, cfg);
    if (rc < 0) {
        vmaf_dnn_sidecar_free(&meta);
        return rc;
    }

    /* TODO(phase-3k): attach `sess` to `ctx` so the per-frame loop in
     * libvmaf/src/libvmaf.c invokes vmaf_ort_infer() and publishes the
     * result as a `tiny_model` JSON block. Until that wiring lands, the
     * session is opened and closed here to prove the load/validate path. */
    vmaf_ort_close(sess);
    vmaf_dnn_sidecar_free(&meta);
    return 0;
#else
    (void) ctx; (void) onnx_path; (void) cfg;
    return -ENOSYS;
#endif
}
