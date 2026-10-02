/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef __VMAF_PICTURE_H__
#define __VMAF_PICTURE_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum VmafPixelFormat {
    VMAF_PIX_FMT_UNKNOWN,
    VMAF_PIX_FMT_YUV420P,
    VMAF_PIX_FMT_YUV422P,
    VMAF_PIX_FMT_YUV444P,
    VMAF_PIX_FMT_YUV400P,
};

enum VmafColorRange {
    VMAF_COLOR_RANGE_UNKNOWN,
    VMAF_COLOR_RANGE_LIMITED,
    VMAF_COLOR_RANGE_FULL,
};

enum VmafColorPrimaries {
    VMAF_COLOR_PRIMARIES_UNKNOWN = 0,
    VMAF_COLOR_PRIMARIES_BT709,
    VMAF_COLOR_PRIMARIES_BT2020,
    VMAF_COLOR_PRIMARIES_SMPTE432,
};

enum VmafColorTransferCharacteristic {
    VMAF_COLOR_TRC_UNKNOWN = 0,
    VMAF_COLOR_TRC_BT709,
    VMAF_COLOR_TRC_SMPTE2084,
};

enum VmafColorMatrixCoefficients {
    VMAF_COLOR_MATRIX_UNKNOWN = 0,
    VMAF_COLOR_MATRIX_BT709,
    VMAF_COLOR_MATRIX_BT2020_NCL,
    VMAF_COLOR_MATRIX_ICTCP,
};

typedef struct VmafRef VmafRef;

typedef struct VmafColor {
    enum VmafColorRange range;
    enum VmafColorPrimaries primaries;
    enum VmafColorTransferCharacteristic trc;
    enum VmafColorMatrixCoefficients matrix;
} VmafColor;

typedef struct VmafPicture {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w[3], h[3];
    ptrdiff_t stride[3];
    void *data[3];
    VmafColor color;
    VmafRef *ref;
    void *priv;
} VmafPicture;

int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h);

int vmaf_picture_unref(VmafPicture *pic);

enum VmafResampleFilter {
    VMAF_RESAMPLE_DEFAULT,
    VMAF_RESAMPLE_BILINEAR,
    VMAF_RESAMPLE_BICUBIC,
    VMAF_RESAMPLE_LANCZOS,
};

typedef struct VmafPictureConvertTarget {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w, h;
    VmafColor color;
    enum VmafResampleFilter resample_filter;
} VmafPictureConvertTarget;

typedef struct VmafPictureConvertContext VmafPictureConvertContext;

int vmaf_picture_convert_context_init(VmafPictureConvertContext **ctx,
                                      const VmafPicture *src,
                                      const VmafPictureConvertTarget *target);

int vmaf_picture_convert(VmafPictureConvertContext *ctx, VmafPicture *dst,
                         const VmafPicture *src);

int vmaf_picture_convert_context_close(VmafPictureConvertContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_PICTURE_H__ */
