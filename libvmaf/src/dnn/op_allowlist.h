/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

/**
 * @file op_allowlist.h
 * @brief Allowlist of ONNX operator types permitted by the DNN loader.
 *
 * Any operator in the loaded graph whose type is not in this list causes
 * model_loader.c to reject the file *before* ORT instantiates a session.
 * This is a defence-in-depth measure: ORT already sandboxes execution, but
 * the allowlist lets us reject unexpected custom ops / control-flow ops
 * we never trained against.
 *
 * Keep this list narrow. Adding an op = signing up to audit its runtime
 * complexity and any side-channels it introduces.
 */

#ifndef LIBVMAF_DNN_OP_ALLOWLIST_H_
#define LIBVMAF_DNN_OP_ALLOWLIST_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return true if @p op_type is permitted.
 * @p op_type is a null-terminated ONNX operator name ("Conv", "Gemm", ...).
 */
bool vmaf_dnn_op_allowed(const char *op_type);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_OP_ALLOWLIST_H_ */
