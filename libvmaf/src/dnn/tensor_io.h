/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_DNN_TENSOR_IO_H_
#define LIBVMAF_DNN_TENSOR_IO_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VmafTensorLayout {
    VMAF_TENSOR_LAYOUT_NCHW = 0,
    VMAF_TENSOR_LAYOUT_NHWC = 1,
} VmafTensorLayout;

typedef enum VmafTensorDType {
    VMAF_TENSOR_DTYPE_F32  = 0,
    VMAF_TENSOR_DTYPE_F16  = 1,
} VmafTensorDType;

/**
 * Convert a single-channel 8-bit luma frame to a normalized float32/float16
 * tensor. Values are scaled to [0, 1] (divide by 255) and, if @p mean and
 * @p std are non-NULL, standardized channel-wise.
 *
 * @param src       uint8 luma plane (H rows of W pixels, stride = stride_src)
 * @param stride_src  bytes per source row (>= width)
 * @param width, height  frame dimensions
 * @param layout    NCHW (default for ORT) or NHWC
 * @param dtype     F32 or F16
 * @param mean, std per-channel normalization (length 1 for luma-only); NULL = skip
 * @param dst       destination buffer; required size = width*height*sizeof(dtype)
 *
 * @return 0 on success, -EINVAL on bad args.
 */
int vmaf_tensor_from_luma(const uint8_t *src, size_t stride_src,
                          int width, int height,
                          VmafTensorLayout layout,
                          VmafTensorDType dtype,
                          const float *mean, const float *std,
                          void *dst);

/**
 * Convert a normalized float32/float16 NCHW or NHWC tensor back to an 8-bit
 * luma plane. De-normalizes (if mean/std provided), multiplies by 255 with
 * round-half-to-even, clamps to [0, 255]. Used by learned-filter output.
 */
int vmaf_tensor_to_luma(const void *src,
                        VmafTensorLayout layout,
                        VmafTensorDType dtype,
                        int width, int height,
                        const float *mean, const float *std,
                        uint8_t *dst, size_t stride_dst);

/** Convert float32 ↔ float16 in-place element counts. */
void vmaf_f32_to_f16(const float *src, uint16_t *dst, size_t n);
void vmaf_f16_to_f32(const uint16_t *src, float *dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_TENSOR_IO_H_ */
