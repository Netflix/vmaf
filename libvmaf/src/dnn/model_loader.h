/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_DNN_MODEL_LOADER_H_
#define LIBVMAF_DNN_MODEL_LOADER_H_

#include <stdbool.h>
#include <stddef.h>

#include "libvmaf/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Defaults; override via env VMAF_MAX_MODEL_BYTES (50 MB). */
#define VMAF_DNN_DEFAULT_MAX_BYTES ((size_t) 50u * 1024u * 1024u)

typedef struct VmafModelSidecar {
    VmafModelKind kind;            /**< mirrors sidecar "kind" field */
    int           opset;
    char         *name;            /**< owned */
    char         *input_name;      /**< owned */
    char         *output_name;     /**< owned */
    float         norm_mean;
    float         norm_std;
    bool          has_norm;
    float         expected_min;
    float         expected_max;
    bool          has_range;
} VmafModelSidecar;

/** Byte-identical magic check. Returns VMAF_MODEL_KIND_SVM for libsvm json/pkl,
 *  DNN_FR/DNN_NR for ONNX (kind refined from sidecar), or -1 on unknown. */
int vmaf_dnn_sniff_kind(const char *path);

/** Loads `<path>.json` next to the ONNX file. Returns 0 on success. */
int vmaf_dnn_sidecar_load(const char *onnx_path, VmafModelSidecar *out);
void vmaf_dnn_sidecar_free(VmafModelSidecar *s);

/** Validate an ONNX file on disk: size cap + operator allowlist. */
int vmaf_dnn_validate_onnx(const char *path, size_t max_bytes);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_MODEL_LOADER_H_ */
