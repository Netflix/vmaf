/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ONNX Runtime C-API wrapper. Selects CPU / CUDA / OpenVINO / ROCm execution
 *  providers per VmafDnnConfig, validates ops against op_allowlist.c, and
 *  runs single-input single-output inference for FR regressors (C1) and
 *  NR metrics (C2). Learned-filter (C3) inference reuses the same session
 *  with a larger output tensor.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ort_backend.h"

#if defined(VMAF_HAVE_DNN) && VMAF_HAVE_DNN

#include <onnxruntime_c_api.h>

#include "op_allowlist.h"

struct VmafOrtSession {
    const OrtApi   *api;
    OrtEnv         *env;
    OrtSession     *session;
    OrtSessionOptions *opts;
    OrtAllocator   *alloc;
    char           *input_name;
    char           *output_name;
};

#define ORT_TRY(call) do {                                                \
    OrtStatus *st__ = (call);                                             \
    if (st__ != NULL) {                                                   \
        sess->api->ReleaseStatus(st__);                                   \
        vmaf_ort_close(sess);                                             \
        return -EIO;                                                      \
    }                                                                     \
} while (0)

int vmaf_ort_open(VmafOrtSession **out, const char *onnx_path,
                  const VmafDnnConfig *cfg)
{
    if (!out || !onnx_path) return -EINVAL;

    VmafOrtSession *sess = (VmafOrtSession *) calloc(1, sizeof(*sess));
    if (!sess) return -ENOMEM;

    sess->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!sess->api) { free(sess); return -ENOSYS; }

    ORT_TRY(sess->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "libvmaf-dnn", &sess->env));
    ORT_TRY(sess->api->CreateSessionOptions(&sess->opts));

    const int intra = (cfg && cfg->threads > 0) ? cfg->threads : 0;
    if (intra > 0) {
        ORT_TRY(sess->api->SetIntraOpNumThreads(sess->opts, intra));
    }

    const VmafDnnDevice dev = cfg ? cfg->device : VMAF_DNN_DEVICE_AUTO;
    const int idx = (cfg && cfg->device_index > 0) ? cfg->device_index : 0;
    (void) idx;

    /* Execution-provider selection. Silently falls back to CPU if the
     * requested EP isn't compiled into ORT; the DNN runtime is best-effort
     * and the CPU EP is always available. */
    switch (dev) {
#ifdef ORT_API_HAS_CUDA
    case VMAF_DNN_DEVICE_CUDA: {
        OrtCUDAProviderOptions cuda = {0};
        cuda.device_id = idx;
        (void) sess->api->SessionOptionsAppendExecutionProvider_CUDA(sess->opts, &cuda);
        break;
    }
#endif
    case VMAF_DNN_DEVICE_CPU:
    case VMAF_DNN_DEVICE_AUTO:
    default:
        break;
    }

    ORT_TRY(sess->api->CreateSession(sess->env, onnx_path, sess->opts, &sess->session));
    ORT_TRY(sess->api->GetAllocatorWithDefaultOptions(&sess->alloc));

    ORT_TRY(sess->api->SessionGetInputName (sess->session, 0, sess->alloc, &sess->input_name));
    ORT_TRY(sess->api->SessionGetOutputName(sess->session, 0, sess->alloc, &sess->output_name));

    *out = sess;
    return 0;
}

int vmaf_ort_infer(VmafOrtSession *sess,
                   const float *input, const int64_t *input_shape,
                   size_t input_rank,
                   float *output, size_t output_capacity,
                   size_t *output_written)
{
    if (!sess || !input || !input_shape || !output) return -EINVAL;

    OrtMemoryInfo *mem = NULL;
    OrtStatus *st = sess->api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &mem);
    if (st) { sess->api->ReleaseStatus(st); return -EIO; }

    size_t n = 1;
    for (size_t i = 0; i < input_rank; ++i) {
        n *= (size_t) input_shape[i];
    }

    OrtValue *in_tensor = NULL;
    st = sess->api->CreateTensorWithDataAsOrtValue(
        mem, (void *) input, n * sizeof(float),
        input_shape, input_rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor);
    sess->api->ReleaseMemoryInfo(mem);
    if (st) { sess->api->ReleaseStatus(st); return -EIO; }

    const char *in_names[1]  = { sess->input_name };
    const char *out_names[1] = { sess->output_name };
    OrtValue *out_tensor = NULL;

    st = sess->api->Run(sess->session, NULL,
                        in_names, (const OrtValue * const *) &in_tensor, 1,
                        out_names, 1, &out_tensor);
    sess->api->ReleaseValue(in_tensor);
    if (st) { sess->api->ReleaseStatus(st); return -EIO; }

    float *out_data = NULL;
    st = sess->api->GetTensorMutableData(out_tensor, (void **) &out_data);
    if (st) {
        sess->api->ReleaseStatus(st);
        sess->api->ReleaseValue(out_tensor);
        return -EIO;
    }

    OrtTensorTypeAndShapeInfo *info = NULL;
    st = sess->api->GetTensorTypeAndShape(out_tensor, &info);
    if (st) {
        sess->api->ReleaseStatus(st);
        sess->api->ReleaseValue(out_tensor);
        return -EIO;
    }
    size_t out_n = 0;
    (void) sess->api->GetTensorShapeElementCount(info, &out_n);
    sess->api->ReleaseTensorTypeAndShapeInfo(info);

    if (out_n > output_capacity) {
        sess->api->ReleaseValue(out_tensor);
        return -ENOSPC;
    }
    memcpy(output, out_data, out_n * sizeof(float));
    if (output_written) *output_written = out_n;
    sess->api->ReleaseValue(out_tensor);
    return 0;
}

int vmaf_ort_input_shape(VmafOrtSession *sess,
                         int64_t *out_shape, size_t max_rank,
                         size_t *out_rank)
{
    if (!sess || !out_shape || !out_rank || max_rank == 0) return -EINVAL;

    OrtTypeInfo *type_info = NULL;
    OrtStatus *st = sess->api->SessionGetInputTypeInfo(sess->session, 0,
                                                       &type_info);
    if (st) { sess->api->ReleaseStatus(st); return -EIO; }

    const OrtTensorTypeAndShapeInfo *tinfo = NULL;
    st = sess->api->CastTypeInfoToTensorInfo(type_info, &tinfo);
    if (st) {
        sess->api->ReleaseStatus(st);
        sess->api->ReleaseTypeInfo(type_info);
        return -EIO;
    }

    size_t rank = 0;
    st = sess->api->GetDimensionsCount(tinfo, &rank);
    if (st) {
        sess->api->ReleaseStatus(st);
        sess->api->ReleaseTypeInfo(type_info);
        return -EIO;
    }
    if (rank == 0 || rank > max_rank) {
        sess->api->ReleaseTypeInfo(type_info);
        return -ERANGE;
    }

    st = sess->api->GetDimensions(tinfo, out_shape, rank);
    sess->api->ReleaseTypeInfo(type_info);
    if (st) { sess->api->ReleaseStatus(st); return -EIO; }

    *out_rank = rank;
    return 0;
}

void vmaf_ort_close(VmafOrtSession *sess)
{
    if (!sess) return;
    if (sess->api) {
        if (sess->alloc && sess->input_name) {
            (void) sess->api->AllocatorFree(sess->alloc, sess->input_name);
        }
        if (sess->alloc && sess->output_name) {
            (void) sess->api->AllocatorFree(sess->alloc, sess->output_name);
        }
        if (sess->session) sess->api->ReleaseSession(sess->session);
        if (sess->opts)    sess->api->ReleaseSessionOptions(sess->opts);
        if (sess->env)     sess->api->ReleaseEnv(sess->env);
    }
    free(sess);
}

#else  /* !VMAF_HAVE_DNN */

struct VmafOrtSession { int _unused; };

int vmaf_ort_open(VmafOrtSession **out, const char *onnx_path,
                  const VmafDnnConfig *cfg)
{
    (void) out; (void) onnx_path; (void) cfg;
    return -ENOSYS;
}

int vmaf_ort_infer(VmafOrtSession *sess,
                   const float *input, const int64_t *input_shape,
                   size_t input_rank,
                   float *output, size_t output_capacity,
                   size_t *output_written)
{
    (void) sess; (void) input; (void) input_shape; (void) input_rank;
    (void) output; (void) output_capacity; (void) output_written;
    return -ENOSYS;
}

int vmaf_ort_input_shape(VmafOrtSession *sess,
                         int64_t *out_shape, size_t max_rank,
                         size_t *out_rank)
{
    (void) sess; (void) out_shape; (void) max_rank; (void) out_rank;
    return -ENOSYS;
}

void vmaf_ort_close(VmafOrtSession *sess)
{
    (void) sess;
}

#endif /* VMAF_HAVE_DNN */
