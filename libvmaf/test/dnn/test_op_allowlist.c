/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include "test.h"

#include "dnn/op_allowlist.h"

static char *test_common_ops_allowed(void)
{
    mu_assert("Conv should be allowed",              vmaf_dnn_op_allowed("Conv"));
    mu_assert("Gemm should be allowed",              vmaf_dnn_op_allowed("Gemm"));
    mu_assert("Relu should be allowed",              vmaf_dnn_op_allowed("Relu"));
    mu_assert("BatchNormalization should be allowed",
              vmaf_dnn_op_allowed("BatchNormalization"));
    mu_assert("GlobalAveragePool should be allowed",
              vmaf_dnn_op_allowed("GlobalAveragePool"));
    return NULL;
}

static char *test_custom_ops_rejected(void)
{
    mu_assert("If should be rejected",        !vmaf_dnn_op_allowed("If"));
    mu_assert("Loop should be rejected",      !vmaf_dnn_op_allowed("Loop"));
    mu_assert("unknown should be rejected",   !vmaf_dnn_op_allowed("custom_op_xyz"));
    mu_assert("NULL should be rejected",      !vmaf_dnn_op_allowed(NULL));
    mu_assert("empty string should be rejected", !vmaf_dnn_op_allowed(""));
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_common_ops_allowed);
    mu_run_test(test_custom_ops_rejected);
    return NULL;
}
