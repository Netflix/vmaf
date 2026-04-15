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
#include "tensor_io.h"

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

#if defined(VMAF_HAVE_DNN) && VMAF_HAVE_DNN

struct VmafDnnSession {
    VmafOrtSession  *ort;
    VmafModelSidecar meta;
    bool             has_sidecar;
    int              w;
    int              h;
    float           *in_buf;    /* w*h floats */
    float           *out_buf;   /* w*h floats */
};

int vmaf_dnn_session_open(VmafDnnSession **out, const char *onnx_path,
                          const VmafDnnConfig *cfg)
{
    if (!out || !onnx_path) return -EINVAL;

    size_t max_bytes = VMAF_DNN_DEFAULT_MAX_BYTES;
    const char *env = getenv("VMAF_MAX_MODEL_BYTES");
    if (env && *env) {
        char *endp = NULL;
        unsigned long v = strtoul(env, &endp, 10);
        if (endp && *endp == '\0' && v > 0) max_bytes = (size_t) v;
    }
    int rc = vmaf_dnn_validate_onnx(onnx_path, max_bytes);
    if (rc < 0) return rc;

    VmafDnnSession *s = (VmafDnnSession *) calloc(1, sizeof(*s));
    if (!s) return -ENOMEM;

    rc = vmaf_dnn_sidecar_load(onnx_path, &s->meta);
    if (rc == 0) s->has_sidecar = true;
    else if (rc != -ENOENT) { free(s); return rc; }

    rc = vmaf_ort_open(&s->ort, onnx_path, cfg);
    if (rc < 0) {
        if (s->has_sidecar) vmaf_dnn_sidecar_free(&s->meta);
        free(s);
        return rc;
    }

    int64_t shape[4] = { 0 };
    size_t rank = 0;
    rc = vmaf_ort_input_shape(s->ort, shape, 4u, &rank);
    if (rc < 0) goto fail;

    if (rank != 4 || shape[0] != 1 || shape[1] != 1 ||
        shape[2] <= 0 || shape[3] <= 0) {
        rc = -ENOTSUP;
        goto fail;
    }
    s->h = (int) shape[2];
    s->w = (int) shape[3];

    const size_t n = (size_t) s->w * (size_t) s->h;
    s->in_buf  = (float *) calloc(n, sizeof(float));
    s->out_buf = (float *) calloc(n, sizeof(float));
    if (!s->in_buf || !s->out_buf) { rc = -ENOMEM; goto fail; }

    *out = s;
    return 0;

fail:
    vmaf_dnn_session_close(s);
    return rc;
}

int vmaf_dnn_session_run_luma8(VmafDnnSession *sess,
                               const uint8_t *in, size_t in_stride,
                               int w, int h,
                               uint8_t *out, size_t out_stride)
{
    if (!sess || !in || !out) return -EINVAL;
    if (w != sess->w || h != sess->h) return -ERANGE;

    const float *mean = NULL;
    const float *std  = NULL;
    float m = 0.f, s_ = 1.f;
    if (sess->has_sidecar && sess->meta.has_norm && sess->meta.norm_std > 0.f) {
        m = sess->meta.norm_mean;
        s_ = sess->meta.norm_std;
        mean = &m; std = &s_;
    }

    int rc = vmaf_tensor_from_luma(in, in_stride, w, h,
                                   VMAF_TENSOR_LAYOUT_NCHW,
                                   VMAF_TENSOR_DTYPE_F32,
                                   mean, std, sess->in_buf);
    if (rc < 0) return rc;

    const int64_t shape[4] = { 1, 1, h, w };
    const size_t n = (size_t) w * (size_t) h;
    size_t written = 0;
    rc = vmaf_ort_infer(sess->ort, sess->in_buf, shape, 4,
                        sess->out_buf, n, &written);
    if (rc < 0) return rc;
    if (written != n) return -ENOTSUP;

    return vmaf_tensor_to_luma(sess->out_buf,
                               VMAF_TENSOR_LAYOUT_NCHW,
                               VMAF_TENSOR_DTYPE_F32,
                               w, h, mean, std,
                               out, out_stride);
}

void vmaf_dnn_session_close(VmafDnnSession *sess)
{
    if (!sess) return;
    if (sess->ort) vmaf_ort_close(sess->ort);
    if (sess->has_sidecar) vmaf_dnn_sidecar_free(&sess->meta);
    free(sess->in_buf);
    free(sess->out_buf);
    free(sess);
}

#else  /* !VMAF_HAVE_DNN */

struct VmafDnnSession { int _unused; };

int vmaf_dnn_session_open(VmafDnnSession **out, const char *onnx_path,
                          const VmafDnnConfig *cfg)
{
    (void) out; (void) onnx_path; (void) cfg;
    return -ENOSYS;
}

int vmaf_dnn_session_run_luma8(VmafDnnSession *sess,
                               const uint8_t *in, size_t in_stride,
                               int w, int h,
                               uint8_t *out, size_t out_stride)
{
    (void) sess; (void) in; (void) in_stride; (void) w; (void) h;
    (void) out; (void) out_stride;
    return -ENOSYS;
}

void vmaf_dnn_session_close(VmafDnnSession *sess)
{
    (void) sess;
}

#endif /* VMAF_HAVE_DNN */
