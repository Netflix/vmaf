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

#include <stdatomic.h>
#include <stddef.h>

enum VmafPixelFormat {
    VMAF_PIX_FMT_UNKNOWN,
    VMAF_PIX_FMT_YUV420P,
    VMAF_PIX_FMT_YUV422P,
    VMAF_PIX_FMT_YUV444P,
};

typedef struct {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w[3], h[3];
    ptrdiff_t stride[3];
    void *data[3];
    atomic_int *ref_cnt;
} VmafPicture;

int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h);

int vmaf_picture_unref(VmafPicture *pic);

#endif /* __VMAF_PICTURE_H__ */
