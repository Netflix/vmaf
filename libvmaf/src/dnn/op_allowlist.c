/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <string.h>

#include "op_allowlist.h"

static const char *const ALLOWED_OPS[] = {
    /* structural / shape */
    "Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose",
    "Concat", "Slice", "Gather", "Cast", "Shape", "Expand",
    /* arithmetic */
    "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Pow",
    "Exp", "Log", "Clip", "Min", "Max", "Sum", "Mean",
    /* reductions */
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
    "GlobalAveragePool", "GlobalMaxPool",
    /* dense */
    "Gemm", "MatMul",
    /* convolutional */
    "Conv", "ConvTranspose", "MaxPool", "AveragePool",
    /* normalization */
    "BatchNormalization", "LayerNormalization", "InstanceNormalization",
    /* activations */
    "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softmax",
    "Elu", "Selu", "Softplus", "Softsign",
    "Gelu", "Erf", "HardSigmoid", "HardSwish",
    "PRelu", "Clip",
    /* dropout (no-op in inference) */
    "Dropout",
    /* misc, safe */
    "Constant", "ConstantOfShape",
    NULL,
};

bool vmaf_dnn_op_allowed(const char *op_type)
{
    if (!op_type) {
        return false;
    }
    for (size_t i = 0; ALLOWED_OPS[i] != NULL; ++i) {
        if (strcmp(ALLOWED_OPS[i], op_type) == 0) {
            return true;
        }
    }
    return false;
}
