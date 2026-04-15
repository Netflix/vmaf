/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "dnn/tensor_io.h"

static char *test_luma_to_f32_unnormalized(void)
{
    uint8_t src[16] = {  0, 64, 128, 192, 255,  0, 128, 255,
                        10, 20,  30,  40,  50, 60,  70,  80 };
    float dst[16];
    int err = vmaf_tensor_from_luma(src, 16, 16, 1,
                                    VMAF_TENSOR_LAYOUT_NCHW,
                                    VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, dst);
    mu_assert("vmaf_tensor_from_luma failed", err == 0);
    mu_assert("0 did not map to 0",          dst[0] == 0.0f);
    mu_assert("255 did not map to 1",        fabsf(dst[4] - 1.0f) < 1e-6f);
    mu_assert("128 did not round correctly", fabsf(dst[2] - (128.0f / 255.0f)) < 1e-6f);
    return NULL;
}

static char *test_f16_roundtrip(void)
{
    const float inputs[] = { 0.0f, 1.0f, -1.0f, 0.5f, 2.5f, 0.123456f, -7.25f };
    const size_t n = sizeof(inputs) / sizeof(inputs[0]);
    uint16_t h[16];
    float    back[16];
    vmaf_f32_to_f16(inputs, h, n);
    vmaf_f16_to_f32(h, back, n);
    for (size_t i = 0; i < n; ++i) {
        float tol = fabsf(inputs[i]) * 1e-3f + 1e-3f;
        mu_assert("f16 roundtrip exceeded tolerance",
                  fabsf(inputs[i] - back[i]) <= tol);
    }
    return NULL;
}

static char *test_luma_roundtrip(void)
{
    uint8_t src[64], dst[64];
    float   tensor[64];
    for (int i = 0; i < 64; ++i) src[i] = (uint8_t)(i * 4 % 256);
    int err = vmaf_tensor_from_luma(src, 8, 8, 8,
                                    VMAF_TENSOR_LAYOUT_NCHW,
                                    VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, tensor);
    mu_assert("luma→tensor failed", err == 0);
    err = vmaf_tensor_to_luma(tensor, VMAF_TENSOR_LAYOUT_NCHW,
                              VMAF_TENSOR_DTYPE_F32, 8, 8, NULL, NULL,
                              dst, 8);
    mu_assert("tensor→luma failed", err == 0);
    for (int i = 0; i < 64; ++i) {
        mu_assert("luma roundtrip differed", src[i] == dst[i]);
    }
    return NULL;
}

static char *test_rejects_bad_args(void)
{
    uint8_t src[4];
    float   dst[4];
    int err = vmaf_tensor_from_luma(NULL, 2, 2, 2,
                                    VMAF_TENSOR_LAYOUT_NCHW,
                                    VMAF_TENSOR_DTYPE_F32,
                                    NULL, NULL, dst);
    mu_assert("expected -EINVAL on NULL src", err < 0);
    err = vmaf_tensor_from_luma(src, 1, 2, 2,
                                VMAF_TENSOR_LAYOUT_NCHW,
                                VMAF_TENSOR_DTYPE_F32,
                                NULL, NULL, dst);
    mu_assert("expected -EINVAL on stride < width", err < 0);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_luma_to_f32_unnormalized);
    mu_run_test(test_f16_roundtrip);
    mu_run_test(test_luma_roundtrip);
    mu_run_test(test_rejects_bad_args);
    return NULL;
}
