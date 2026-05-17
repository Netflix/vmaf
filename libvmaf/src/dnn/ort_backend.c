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

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ort_backend.h"
#include "ort_backend_internal.h"

#if defined(VMAF_HAVE_DNN) && VMAF_HAVE_DNN

#include <onnxruntime_c_api.h>

#include "op_allowlist.h"

/* Maximum number of model inputs or outputs supported by vmaf_ort_run().
 * All current tiny-AI models are 1-in/1-out (NR) or 5-in/1-out (FR);
 * 8 leaves headroom without triggering VLA (banned by Power-of-10). */
#define VMAF_ORT_MAX_IO 8

struct VmafOrtSession {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtSessionOptions *opts;
    OrtAllocator *alloc;
    /* input_name / output_name mirror names[0] for the single-input legacy
     * path (vmaf_ort_infer). names/n_inputs/n_outputs cover the full graph
     * IO for vmaf_ort_run. */
    char *input_name;
    char *output_name;
    char **input_names;
    char **output_names;
    size_t n_inputs;
    size_t n_outputs;
    /* Per-IO element type (ONNXTensorElementDataType). Populated at open
     * time so vmaf_ort_infer/run can decide whether to emit fp32 or fp16
     * tensors to ORT without re-querying the model each call. */
    int *input_elem_types;
    int *output_elem_types;
    bool fp16_io;
    /* EP actually attached, for diagnostics. "CPU" when nothing else bound. */
    const char *ep_name;
};

/* ------------------------------------------------------------------ */
/* Portable IEEE 754 half-precision conversion.                        */
/* Avoids depending on _Float16 / F16C intrinsics so the DNN backend    */
/* still builds on hosts without hardware fp16 support (older SSE      */
/* x86_64, ARMv7). Handles inf / nan / subnormals / overflow.          */
/* ------------------------------------------------------------------ */

static uint16_t fp32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 31) & 0x1u;
    const int32_t exp_f = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp_f == 128) {
        /* inf / nan */
        return (uint16_t)((sign << 15) | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp_f > 15) {
        /* overflow → inf */
        return (uint16_t)((sign << 15) | 0x7C00u);
    }
    if (exp_f < -24) {
        /* underflow → ±0 */
        return (uint16_t)(sign << 15);
    }
    if (exp_f < -14) {
        /* subnormal half */
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(-exp_f - 14) + 13u;
        return (uint16_t)((sign << 15) | (mant >> shift));
    }
    const uint16_t h_exp = (uint16_t)(exp_f + 15);
    const uint16_t h_mant = (uint16_t)(mant >> 13);
    return (uint16_t)((sign << 15) | ((uint16_t)(h_exp << 10)) | h_mant);
}

static float fp16_to_fp32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp_h = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t x;

    if (exp_h == 0u) {
        if (mant == 0u) {
            x = sign << 31;
        } else {
            /* Subnormal half. Loop normalises the mantissa: each shift
             * moves the leading 1 toward bit 10. The loop body counts
             * one shift more than the leading-zero distance (the
             * iteration that *places* the implicit 1 also runs), so the
             * fp32 biased exponent is (127 - 15 - e), not (127 - 14 - e).
             * The earlier formula doubled the magnitude of every
             * subnormal (e.g. 0x03FF → 1.22e-4 instead of 6.10e-5). */
            int32_t e = -1;
            do {
                mant <<= 1;
                ++e;
            } while ((mant & 0x400u) == 0u);
            mant &= 0x3FFu;
            x = (sign << 31) | (uint32_t)((127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp_h == 31u) {
        x = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        x = (sign << 31) | ((exp_h + 112u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

/* ------------------------------------------------------------------ */
/* Execution-provider selection.                                       */
/* ORT's generic SessionOptionsAppendExecutionProvider returns         */
/* non-null OrtStatus when the requested EP isn't registered in this   */
/* ORT build — we treat that as "EP unavailable, try next" and fall    */
/* through to the CPU EP, which is always linked.                      */
/* ------------------------------------------------------------------ */

static int try_append_ep_generic(struct VmafOrtSession *sess, const char *name,
                                 const char *const *keys, const char *const *values, size_t nk)
{
    OrtStatus *st =
        sess->api->SessionOptionsAppendExecutionProvider(sess->opts, name, keys, values, nk);
    if (st != NULL) {
        sess->api->ReleaseStatus(st);
        return -ENOSYS;
    }
    return 0;
}

static int try_append_cuda(struct VmafOrtSession *sess, int device_index)
{
    OrtCUDAProviderOptions cuda = {0};
    cuda.device_id = device_index;
    OrtStatus *st = sess->api->SessionOptionsAppendExecutionProvider_CUDA(sess->opts, &cuda);
    if (st != NULL) {
        sess->api->ReleaseStatus(st);
        return -ENOSYS;
    }
    return 0;
}

static int try_append_openvino(struct VmafOrtSession *sess, const char *device_type, bool fp16_io)
{
    const char *keys[2];
    const char *values[2];
    size_t nk = 0u;
    keys[nk] = "device_type";
    values[nk] = device_type;
    ++nk;
    if (fp16_io) {
        keys[nk] = "precision";
        values[nk] = "FP16";
        ++nk;
    }
    return try_append_ep_generic(sess, "OpenVINOExecutionProvider", keys, values, nk);
}

static int try_append_rocm(struct VmafOrtSession *sess)
{
    return try_append_ep_generic(sess, "ROCMExecutionProvider", NULL, NULL, 0u);
}

/* Apple CoreML execution provider. Targets the Apple Neural Engine (ANE),
 * Metal-backed GPU, and CPU on M-series and Intel Macs. The CoreML EP key
 * "MLComputeUnits" pins a single compute unit; valid values are
 * "ALL" (auto-route — default when key is absent), "CPUOnly",
 * "CPUAndGPU", "CPUAndNeuralEngine".
 *
 * API surface used here is the generic
 * `SessionOptionsAppendExecutionProvider("CoreMLExecutionProvider", ...)`
 * key/value form rather than the older
 * `OrtSessionOptionsAppendExecutionProvider_CoreML(opts, uint32_t flags)`
 * factory in `coreml_provider_factory.h`. The generic form needs no extra
 * header and degrades cleanly on Linux ORT builds (no CoreML EP linked
 * → non-null OrtStatus → -ENOSYS → CPU fallback).
 *
 * Reference: ONNX Runtime CoreML Execution Provider documentation,
 *   https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
 *   (accessed 2026-05-09).
 *
 * @p compute_units may be NULL (omits the key — CoreML picks any unit) or
 * one of the strings above. @p fp16_io adds "ModelFormat=NeuralNetwork"
 * is intentionally NOT set; the EP defaults to ML Program format which
 * supports fp16 weight precision via the model itself, not via an EP
 * option. */
static int try_append_coreml(struct VmafOrtSession *sess, const char *compute_units)
{
    const char *keys[1];
    const char *values[1];
    size_t nk = 0u;
    if (compute_units != NULL) {
        keys[nk] = "MLComputeUnits";
        values[nk] = compute_units;
        ++nk;
    }
    return try_append_ep_generic(sess, "CoreMLExecutionProvider", keys, values, nk);
}

#define ORT_TRY(call)                                                                              \
    do {                                                                                           \
        OrtStatus *st__ = (call);                                                                  \
        if (st__ != NULL) {                                                                        \
            sess->api->ReleaseStatus(st__);                                                        \
            vmaf_ort_close(sess);                                                                  \
            return -EIO;                                                                           \
        }                                                                                          \
    } while (0)

int vmaf_ort_open(VmafOrtSession **out, const char *onnx_path, const VmafDnnConfig *cfg)
{
    if (!out || !onnx_path)
        return -EINVAL;
    assert(out != NULL);
    assert(onnx_path != NULL);

    VmafOrtSession *sess = (VmafOrtSession *)calloc(1, sizeof(*sess));
    if (!sess)
        return -ENOMEM;

    sess->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!sess->api) {
        free(sess);
        return -ENOSYS;
    }

    ORT_TRY(sess->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "libvmaf-dnn", &sess->env));
    ORT_TRY(sess->api->CreateSessionOptions(&sess->opts));

    const int intra = (cfg && cfg->threads > 0) ? cfg->threads : 0;
    if (intra > 0) {
        ORT_TRY(sess->api->SetIntraOpNumThreads(sess->opts, intra));
    }

    const VmafDnnDevice dev = cfg ? cfg->device : VMAF_DNN_DEVICE_AUTO;
    const int idx = (cfg && cfg->device_index > 0) ? cfg->device_index : 0;
    sess->fp16_io = (cfg != NULL) && cfg->fp16_io;

    /* Execution-provider selection.
     *
     * AUTO: try CUDA → OpenVINO (GPU then CPU) → ROCm → CPU. The first EP
     * whose append call returns NULL OrtStatus wins; EPs absent from the
     * ORT build return non-null and we fall through. The CPU EP is always
     * linked, so the final fall-through never fails.
     *
     * Explicit device: try only the requested EP; on failure the session
     * silently downgrades to CPU. Callers that need to know which EP
     * actually bound can check sess->ep_name via vmaf_ort_attached_ep()
     * (exposed for diagnostics / tests).
     */
    /* OpenVINO EP option set is documented at:
     *   https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
     *   (accessed 2026-05-08).
     * device_type values understood by OpenVINOExecutionProvider include
     * "CPU", "GPU" (alias for GPU.0), "GPU.0", "GPU.1", and "NPU". The NPU
     * value targets the Intel AI-PC neural processing unit on Meteor /
     * Lunar / Arrow Lake silicon. */
    sess->ep_name = "CPU";
    switch (dev) {
    case VMAF_DNN_DEVICE_CUDA:
        if (try_append_cuda(sess, idx) == 0)
            sess->ep_name = "CUDA";
        break;
    case VMAF_DNN_DEVICE_OPENVINO:
        if (try_append_openvino(sess, "GPU", sess->fp16_io) == 0) {
            sess->ep_name = "OpenVINO:GPU";
        } else if (try_append_openvino(sess, "CPU", sess->fp16_io) == 0) {
            sess->ep_name = "OpenVINO:CPU";
        }
        break;
    case VMAF_DNN_DEVICE_OPENVINO_NPU:
        if (try_append_openvino(sess, "NPU", sess->fp16_io) == 0)
            sess->ep_name = "OpenVINO:NPU";
        break;
    case VMAF_DNN_DEVICE_OPENVINO_CPU:
        if (try_append_openvino(sess, "CPU", sess->fp16_io) == 0)
            sess->ep_name = "OpenVINO:CPU";
        break;
    case VMAF_DNN_DEVICE_OPENVINO_GPU:
        if (try_append_openvino(sess, "GPU", sess->fp16_io) == 0)
            sess->ep_name = "OpenVINO:GPU";
        break;
    case VMAF_DNN_DEVICE_ROCM:
        if (try_append_rocm(sess) == 0)
            sess->ep_name = "ROCm";
        break;
    /* CoreML EP variants. The unscoped CoreML selector lets the EP
     * auto-route across compute units; the explicit ANE/GPU/CPU
     * selectors pin a single MLComputeUnits value. On non-Apple hosts
     * (e.g. Linux CI runners) the CoreML EP is absent from the linked
     * ORT and try_append_coreml returns -ENOSYS — the session keeps
     * the default ep_name="CPU" and CreateSession runs on the CPU EP. */
    case VMAF_DNN_DEVICE_COREML:
        if (try_append_coreml(sess, NULL) == 0)
            sess->ep_name = "CoreML";
        break;
    case VMAF_DNN_DEVICE_COREML_ANE:
        if (try_append_coreml(sess, "CPUAndNeuralEngine") == 0)
            sess->ep_name = "CoreML:ANE";
        break;
    case VMAF_DNN_DEVICE_COREML_GPU:
        if (try_append_coreml(sess, "CPUAndGPU") == 0)
            sess->ep_name = "CoreML:GPU";
        break;
    case VMAF_DNN_DEVICE_COREML_CPU:
        if (try_append_coreml(sess, "CPUOnly") == 0)
            sess->ep_name = "CoreML:CPU";
        break;
    case VMAF_DNN_DEVICE_AUTO:
        if (try_append_cuda(sess, idx) == 0) {
            sess->ep_name = "CUDA";
        } else if (try_append_openvino(sess, "GPU", sess->fp16_io) == 0) {
            sess->ep_name = "OpenVINO:GPU";
        } else if (try_append_rocm(sess) == 0) {
            sess->ep_name = "ROCm";
        } else if (try_append_coreml(sess, NULL) == 0) {
            /* CoreML is last in the AUTO chain because the explicit
             * --tiny-device=coreml-ane selector is the recommended
             * Apple-silicon entry point for highest perf-per-watt;
             * AUTO picks CoreML only when no discrete-GPU EP is
             * available (typical on M-series Macs). The unscoped
             * variant lets CoreML pick any compute unit — see the
             * ANE perf note in docs/ai/inference.md. */
            sess->ep_name = "CoreML";
        }
        /* NPU is intentionally NOT in the AUTO chain. NPU has surprising
         * latency floors on small graphs (power-state-transition cost
         * dominates sub-ms inferences) and is opt-in only via the
         * explicit `--tiny-device openvino-npu` selector. */
        break;
    case VMAF_DNN_DEVICE_CPU:
    default:
        break;
    }

    /* Two-stage session creation with non-CPU → CPU fallback.
     *
     * try_append_<EP> returning success only proves ORT *registered* the EP;
     * actual hardware initialisation happens inside CreateSession. If no
     * matching device is present (CUDA EP registered without an NVIDIA GPU,
     * OpenVINO without an OV-supported runtime, etc.) CreateSession returns
     * a non-null OrtStatus. Without this fallback, that condition would
     * surface as -EIO and the entire session_open would fail — even though
     * the CPU EP is always linked and could have served the request.
     *
     * Behaviour change on the happy path: when CUDA / OpenVINO / ROCm
     * actually work, nothing changes. The fallback only fires when the
     * non-CPU EP attached but its hardware is unavailable, which previously
     * was a hard failure. Callers that need to detect the degraded mode can
     * check vmaf_ort_attached_ep() — it now returns "CPU" after fallback.
     * See ADR-0113. */
    OrtStatus *create_st =
        sess->api->CreateSession(sess->env, onnx_path, sess->opts, &sess->session);
    if (create_st != NULL) {
        sess->api->ReleaseStatus(create_st);
        if (strcmp(sess->ep_name, "CPU") != 0) {
            /* Recreate session_options with no non-CPU EPs and retry. */
            sess->api->ReleaseSessionOptions(sess->opts);
            sess->opts = NULL;
            OrtStatus *opts_st = sess->api->CreateSessionOptions(&sess->opts);
            if (opts_st != NULL) {
                sess->api->ReleaseStatus(opts_st);
                vmaf_ort_close(sess);
                return -EIO;
            }
            const int intra_retry = (cfg && cfg->threads > 0) ? cfg->threads : 0;
            if (intra_retry > 0) {
                OrtStatus *t_st = sess->api->SetIntraOpNumThreads(sess->opts, intra_retry);
                if (t_st != NULL)
                    sess->api->ReleaseStatus(t_st);
            }
            sess->ep_name = "CPU";
            OrtStatus *retry_st =
                sess->api->CreateSession(sess->env, onnx_path, sess->opts, &sess->session);
            if (retry_st != NULL) {
                sess->api->ReleaseStatus(retry_st);
                vmaf_ort_close(sess);
                return -EIO;
            }
        } else {
            vmaf_ort_close(sess);
            return -EIO;
        }
    }
    ORT_TRY(sess->api->GetAllocatorWithDefaultOptions(&sess->alloc));

    size_t ni = 0, no = 0;
    ORT_TRY(sess->api->SessionGetInputCount(sess->session, &ni));
    ORT_TRY(sess->api->SessionGetOutputCount(sess->session, &no));
    if (ni == 0 || no == 0) {
        vmaf_ort_close(sess);
        return -EINVAL;
    }
    sess->n_inputs = ni;
    sess->n_outputs = no;
    sess->input_names = (char **)calloc(ni, sizeof(char *));
    sess->output_names = (char **)calloc(no, sizeof(char *));
    if (!sess->input_names || !sess->output_names) {
        vmaf_ort_close(sess);
        return -ENOMEM;
    }
    for (size_t i = 0; i < ni; ++i) {
        ORT_TRY(
            sess->api->SessionGetInputName(sess->session, i, sess->alloc, &sess->input_names[i]));
    }
    for (size_t i = 0; i < no; ++i) {
        ORT_TRY(
            sess->api->SessionGetOutputName(sess->session, i, sess->alloc, &sess->output_names[i]));
    }
    /* legacy single-IO pointers alias position 0 */
    sess->input_name = sess->input_names[0];
    sess->output_name = sess->output_names[0];

    /* Cache per-IO element types so the run path can decide fp32 vs fp16
     * tensor creation without re-querying the model every call. */
    sess->input_elem_types = (int *)calloc(ni, sizeof(int));
    sess->output_elem_types = (int *)calloc(no, sizeof(int));
    if (!sess->input_elem_types || !sess->output_elem_types) {
        vmaf_ort_close(sess);
        return -ENOMEM;
    }
    for (size_t i = 0; i < ni; ++i) {
        OrtTypeInfo *ti = NULL;
        ORT_TRY(sess->api->SessionGetInputTypeInfo(sess->session, i, &ti));
        const OrtTensorTypeAndShapeInfo *tinfo = NULL;
        OrtStatus *cst = sess->api->CastTypeInfoToTensorInfo(ti, &tinfo);
        if (cst == NULL && tinfo != NULL) {
            ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            (void)sess->api->GetTensorElementType(tinfo, &et);
            sess->input_elem_types[i] = (int)et;
        } else if (cst != NULL) {
            sess->api->ReleaseStatus(cst);
        }
        sess->api->ReleaseTypeInfo(ti);
    }
    for (size_t i = 0; i < no; ++i) {
        OrtTypeInfo *ti = NULL;
        ORT_TRY(sess->api->SessionGetOutputTypeInfo(sess->session, i, &ti));
        const OrtTensorTypeAndShapeInfo *tinfo = NULL;
        OrtStatus *cst = sess->api->CastTypeInfoToTensorInfo(ti, &tinfo);
        if (cst == NULL && tinfo != NULL) {
            ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            (void)sess->api->GetTensorElementType(tinfo, &et);
            sess->output_elem_types[i] = (int)et;
        } else if (cst != NULL) {
            sess->api->ReleaseStatus(cst);
        }
        sess->api->ReleaseTypeInfo(ti);
    }

    *out = sess;
    return 0;
}

/* Build an OrtValue from caller-supplied fp32 data. When the model declares
 * FLOAT16 at this slot and fp16_io is enabled, we emit a scratch fp16 tensor
 * (ownership returned via @p scratch_out so the caller can free it after
 * Run()); otherwise the caller's buffer is wrapped directly as fp32. */
static int build_input_tensor(VmafOrtSession *sess, OrtMemoryInfo *mem, size_t slot,
                              const float *data, const int64_t *shape, size_t rank,
                              OrtValue **tensor_out, void **scratch_out)
{
    size_t n = 1;
    for (size_t d = 0; d < rank; ++d) {
        n *= (size_t)shape[d];
    }

    const bool want_fp16 =
        sess->fp16_io && sess->input_elem_types[slot] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

    if (want_fp16) {
        uint16_t *half = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!half)
            return -ENOMEM;
        for (size_t i = 0; i < n; ++i)
            half[i] = fp32_to_fp16(data[i]);
        OrtStatus *st = sess->api->CreateTensorWithDataAsOrtValue(
            mem, half, n * sizeof(uint16_t), shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            tensor_out);
        if (st) {
            sess->api->ReleaseStatus(st);
            free(half);
            return -EIO;
        }
        *scratch_out = half;
        return 0;
    }

    OrtStatus *st =
        sess->api->CreateTensorWithDataAsOrtValue(mem, (void *)data, n * sizeof(float), shape, rank,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_out);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }
    *scratch_out = NULL;
    return 0;
}

/* Copy out an OrtValue into the caller's fp32 buffer. Detects the actual
 * tensor element type from the OrtValue — if fp16, casts back to fp32. */
static int copy_output_tensor(VmafOrtSession *sess, OrtValue *tensor, float *dst, size_t capacity,
                              size_t *written)
{
    OrtTensorTypeAndShapeInfo *info = NULL;
    OrtStatus *st = sess->api->GetTensorTypeAndShape(tensor, &info);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }
    size_t out_n = 0;
    (void)sess->api->GetTensorShapeElementCount(info, &out_n);
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    (void)sess->api->GetTensorElementType(info, &et);
    sess->api->ReleaseTensorTypeAndShapeInfo(info);

    if (written)
        *written = out_n;
    if (out_n > capacity)
        return -ENOSPC;

    void *raw = NULL;
    st = sess->api->GetTensorMutableData(tensor, &raw);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < out_n; ++i)
            dst[i] = fp16_to_fp32(src[i]);
    } else {
        memcpy(dst, raw, out_n * sizeof(float));
    }
    return 0;
}

int vmaf_ort_infer(VmafOrtSession *sess, const float *input, const int64_t *input_shape,
                   size_t input_rank, float *output, size_t output_capacity, size_t *output_written)
{
    if (!sess || !input || !input_shape || !output)
        return -EINVAL;

    OrtMemoryInfo *mem = NULL;
    OrtStatus *st = sess->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }

    OrtValue *in_tensor = NULL;
    void *in_scratch = NULL;
    int rc =
        build_input_tensor(sess, mem, 0u, input, input_shape, input_rank, &in_tensor, &in_scratch);
    sess->api->ReleaseMemoryInfo(mem);
    if (rc != 0)
        return rc;

    const char *in_names[1] = {sess->input_name};
    const char *out_names[1] = {sess->output_name};
    OrtValue *out_tensor = NULL;

    st = sess->api->Run(sess->session, NULL, in_names, (const OrtValue *const *)&in_tensor, 1,
                        out_names, 1, &out_tensor);
    sess->api->ReleaseValue(in_tensor);
    free(in_scratch);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }

    size_t produced = 0;
    rc = copy_output_tensor(sess, out_tensor, output, output_capacity, &produced);
    sess->api->ReleaseValue(out_tensor);
    if (output_written)
        *output_written = produced;
    return rc;
}

int vmaf_ort_input_shape(VmafOrtSession *sess, int64_t *out_shape, size_t max_rank,
                         size_t *out_rank)
{
    if (!sess || !out_shape || !out_rank || max_rank == 0)
        return -EINVAL;

    OrtTypeInfo *type_info = NULL;
    OrtStatus *st = sess->api->SessionGetInputTypeInfo(sess->session, 0, &type_info);
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }

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
    if (st) {
        sess->api->ReleaseStatus(st);
        return -EIO;
    }

    *out_rank = rank;
    return 0;
}

void vmaf_ort_close(VmafOrtSession *sess)
{
    if (!sess)
        return;
    assert(sess != NULL);
    if (sess->api) {
        assert(sess->api != NULL);
        if (sess->alloc && sess->input_names) {
            assert(sess->n_inputs > 0u);
            for (size_t i = 0; i < sess->n_inputs; ++i) {
                if (sess->input_names[i])
                    (void)sess->api->AllocatorFree(sess->alloc, sess->input_names[i]);
            }
        }
        if (sess->alloc && sess->output_names) {
            assert(sess->n_outputs > 0u);
            for (size_t i = 0; i < sess->n_outputs; ++i) {
                if (sess->output_names[i])
                    (void)sess->api->AllocatorFree(sess->alloc, sess->output_names[i]);
            }
        }
        if (sess->session)
            sess->api->ReleaseSession(sess->session);
        if (sess->opts)
            sess->api->ReleaseSessionOptions(sess->opts);
        if (sess->env)
            sess->api->ReleaseEnv(sess->env);
    }
    free(sess->input_names);
    free(sess->output_names);
    free(sess->input_elem_types);
    free(sess->output_elem_types);
    free(sess);
}

const char *vmaf_ort_attached_ep(const VmafOrtSession *sess)
{
    if (!sess)
        return NULL;
    return sess->ep_name;
}

int vmaf_ort_io_count(VmafOrtSession *sess, size_t *n_inputs, size_t *n_outputs)
{
    if (!sess || !n_inputs || !n_outputs)
        return -EINVAL;
    *n_inputs = sess->n_inputs;
    *n_outputs = sess->n_outputs;
    return 0;
}

/* Resolve a user-supplied input/output name against the session's name
 * table. NULL name → positional fallback at @p pos. Returns the const
 * char* used by ORT (owned by the session) or NULL on lookup failure. */
static const char *resolve_name(char **table, size_t count, const char *name, size_t pos)
{
    if (name == NULL) {
        if (pos >= count)
            return NULL;
        return table[pos];
    }
    for (size_t i = 0; i < count; ++i) {
        if (table[i] && strcmp(table[i], name) == 0)
            return table[i];
    }
    return NULL;
}

int vmaf_ort_run(VmafOrtSession *sess, const VmafOrtTensorIn *inputs, size_t n_inputs,
                 VmafOrtTensorOut *outputs, size_t n_outputs)
{
    if (!sess || !inputs || !outputs || n_inputs == 0u || n_outputs == 0u)
        return -EINVAL;
    assert(sess != NULL);
    assert(inputs != NULL);
    assert(outputs != NULL);
    if (n_inputs != sess->n_inputs || n_outputs != sess->n_outputs)
        return -EINVAL;

    OrtMemoryInfo *mem = NULL;
    OrtStatus *st0 = sess->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
    if (st0) {
        sess->api->ReleaseStatus(st0);
        return -EIO;
    }

    /* Bounds guard: n_inputs/n_outputs were already validated against
     * sess->n_inputs/n_outputs above; this catches any future caller that
     * bypasses that path and prevents out-of-bounds writes into the fixed-size
     * stack arrays below (replacing the per-call calloc/free pairs — F3-B). */
    if (n_inputs > VMAF_ORT_MAX_IO || n_outputs > VMAF_ORT_MAX_IO) {
        sess->api->ReleaseMemoryInfo(mem);
        return -EINVAL;
    }

    const char *in_names[VMAF_ORT_MAX_IO];
    const char *out_names[VMAF_ORT_MAX_IO];
    OrtValue *in_vals[VMAF_ORT_MAX_IO];
    OrtValue *out_vals[VMAF_ORT_MAX_IO];
    void *in_scratch[VMAF_ORT_MAX_IO];
    memset(in_names, 0, n_inputs * sizeof(in_names[0]));
    memset(out_names, 0, n_outputs * sizeof(out_names[0]));
    memset(in_vals, 0, n_inputs * sizeof(in_vals[0]));
    memset(out_vals, 0, n_outputs * sizeof(out_vals[0]));
    memset(in_scratch, 0, n_inputs * sizeof(in_scratch[0]));

    int rc = 0;
    for (size_t i = 0; i < n_inputs; ++i) {
        if (!inputs[i].data || !inputs[i].shape || inputs[i].rank == 0u) {
            rc = -EINVAL;
            goto cleanup;
        }
        in_names[i] = resolve_name(sess->input_names, sess->n_inputs, inputs[i].name, i);
        if (!in_names[i]) {
            rc = -EINVAL;
            goto cleanup;
        }
        for (size_t d = 0; d < inputs[i].rank; ++d) {
            if (inputs[i].shape[d] <= 0) {
                rc = -EINVAL;
                goto cleanup;
            }
        }
        int brc = build_input_tensor(sess, mem, i, inputs[i].data, inputs[i].shape, inputs[i].rank,
                                     &in_vals[i], &in_scratch[i]);
        if (brc != 0) {
            rc = brc;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n_outputs; ++i) {
        if (!outputs[i].data) {
            rc = -EINVAL;
            goto cleanup;
        }
        out_names[i] = resolve_name(sess->output_names, sess->n_outputs, outputs[i].name, i);
        if (!out_names[i]) {
            rc = -EINVAL;
            goto cleanup;
        }
    }

    OrtStatus *st_run =
        sess->api->Run(sess->session, NULL, in_names, (const OrtValue *const *)in_vals, n_inputs,
                       out_names, n_outputs, out_vals);
    if (st_run) {
        sess->api->ReleaseStatus(st_run);
        rc = -EIO;
        goto cleanup;
    }

    for (size_t i = 0; i < n_outputs; ++i) {
        size_t produced = 0;
        int cpy =
            copy_output_tensor(sess, out_vals[i], outputs[i].data, outputs[i].capacity, &produced);
        outputs[i].written = produced;
        if (cpy == -ENOSPC) {
            /* Short buffer — record the required count and keep going so all
             * outputs get their `written` populated, but propagate -ENOSPC. */
            rc = -ENOSPC;
        } else if (cpy != 0) {
            rc = cpy;
            goto cleanup;
        }
    }

cleanup:
    for (size_t i = 0; i < n_inputs; ++i) {
        if (in_vals[i])
            sess->api->ReleaseValue(in_vals[i]);
        free(in_scratch[i]);
    }
    for (size_t i = 0; i < n_outputs; ++i) {
        if (out_vals[i])
            sess->api->ReleaseValue(out_vals[i]);
    }
    sess->api->ReleaseMemoryInfo(mem);
    /* in_names, out_names, in_vals, out_vals, in_scratch are stack arrays —
     * no free() needed (replaced calloc per F3-B). */
    return rc;
}

/* Internal-test entry points (declared in ort_backend_internal.h). Thin
 * wrappers preserve the static qualifier on the originals so production
 * call sites remain fully inlinable while the test binary can still
 * exercise the helpers directly. See ADR-0112. */
uint16_t vmaf_ort_internal_fp32_to_fp16(float f)
{
    return fp32_to_fp16(f);
}

float vmaf_ort_internal_fp16_to_fp32(uint16_t h)
{
    return fp16_to_fp32(h);
}

const char *vmaf_ort_internal_resolve_name(char **table, size_t count, const char *name, size_t pos)
{
    return resolve_name(table, count, name, pos);
}

#else /* !VMAF_HAVE_DNN */

struct VmafOrtSession {
    int _unused;
};

int vmaf_ort_open(VmafOrtSession **out, const char *onnx_path, const VmafDnnConfig *cfg)
{
    (void)out;
    (void)onnx_path;
    (void)cfg;
    return -ENOSYS;
}

/* NOLINTBEGIN(readability-non-const-parameter)
 * Stub signatures must match the real-ORT path declared in the header. */
int vmaf_ort_infer(VmafOrtSession *sess, const float *input, const int64_t *input_shape,
                   size_t input_rank, float *output, size_t output_capacity, size_t *output_written)
{
    (void)sess;
    (void)input;
    (void)input_shape;
    (void)input_rank;
    (void)output;
    (void)output_capacity;
    (void)output_written;
    return -ENOSYS;
}

int vmaf_ort_input_shape(VmafOrtSession *sess, int64_t *out_shape, size_t max_rank,
                         size_t *out_rank)
{
    (void)sess;
    (void)out_shape;
    (void)max_rank;
    (void)out_rank;
    return -ENOSYS;
}

int vmaf_ort_io_count(VmafOrtSession *sess, size_t *n_inputs, size_t *n_outputs)
{
    (void)sess;
    (void)n_inputs;
    (void)n_outputs;
    return -ENOSYS;
}

int vmaf_ort_run(VmafOrtSession *sess, const VmafOrtTensorIn *inputs, size_t n_inputs,
                 VmafOrtTensorOut *outputs, size_t n_outputs)
{
    (void)sess;
    (void)inputs;
    (void)n_inputs;
    (void)outputs;
    (void)n_outputs;
    return -ENOSYS;
}
/* NOLINTEND(readability-non-const-parameter) */

void vmaf_ort_close(VmafOrtSession *sess)
{
    (void)sess;
}

const char *vmaf_ort_attached_ep(const VmafOrtSession *sess)
{
    (void)sess;
    return NULL;
}

/* Internal-test stubs for the DNN-disabled build. The real wrappers live
 * in the VMAF_HAVE_DNN branch above; here they exist purely so
 * test_ort_internals.c links cleanly on stub builds. The test bodies
 * short-circuit via vmaf_dnn_available() before invoking these. */
uint16_t vmaf_ort_internal_fp32_to_fp16(float f)
{
    (void)f;
    return 0u;
}

float vmaf_ort_internal_fp16_to_fp32(uint16_t h)
{
    (void)h;
    return 0.0f;
}

const char *vmaf_ort_internal_resolve_name(char **table, size_t count, const char *name, size_t pos)
{
    (void)table;
    (void)count;
    (void)name;
    (void)pos;
    return NULL;
}

#endif /* VMAF_HAVE_DNN */
