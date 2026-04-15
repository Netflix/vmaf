/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Public `libvmaf/dnn.h` entry points. Thin wrapper that:
 *    1. validates the path + file (size cap, regular-file, sidecar)
 *    2. opens an ORT session via ort_backend.c
 *    3. queries the input shape and hands ownership to libvmaf.c via the
 *       `dnn_ctx` bridge, so per-frame inference runs alongside SVM models.
 *
 *  When built with -Denable_dnn=false, this TU compiles a stub that returns
 *  -ENOSYS from every entry point so consumers degrade gracefully.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/dnn.h"

#include "dnn_ctx.h"
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
    memset(&meta, 0, sizeof(meta));
    bool have_meta = false;
    rc = vmaf_dnn_sidecar_load(onnx_path, &meta);
    /* Missing sidecar is not fatal — we only need it for NR/FR disambiguation
     * and pretty-printing. Lack of a sidecar defaults to FR. */
    if (rc < 0 && rc != -ENOENT) {
        return rc;
    }
    if (rc == 0) have_meta = true;

    VmafOrtSession *sess = NULL;
    rc = vmaf_ort_open(&sess, onnx_path, cfg);
    if (rc < 0) {
        if (have_meta) vmaf_dnn_sidecar_free(&meta);
        return rc;
    }

    int64_t in_shape[4] = { 0 };
    size_t in_rank = 0;
    rc = vmaf_ort_input_shape(sess, in_shape, 4u, &in_rank);
    if (rc < 0) {
        vmaf_ort_close(sess);
        if (have_meta) vmaf_dnn_sidecar_free(&meta);
        return rc;
    }

    const char *feature_name =
        (have_meta && meta.name && *meta.name) ? meta.name : "vmaf_tiny_model";

    rc = vmaf_ctx_dnn_attach(ctx, sess,
                             have_meta ? &meta : NULL,
                             in_shape, in_rank,
                             feature_name);
    if (rc < 0) {
        vmaf_ort_close(sess);
        if (have_meta) vmaf_dnn_sidecar_free(&meta);
        return rc;
    }
    /* Ownership transferred — do NOT close sess / free meta here. */
    return 0;
#else
    (void) ctx; (void) onnx_path; (void) cfg;
    return -ENOSYS;
#endif
}
