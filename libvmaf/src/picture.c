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

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mem.h"
#include "picture.h"

#define DATA_ALIGN 32

int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h)
{
    if (!pic) return -EINVAL;
    if (!pix_fmt) return -EINVAL;
    if (bpc < 8 || bpc > 16) return -EINVAL;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pix_fmt;
    pic->bpc = bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = w;
    pic->w[1] = pic->w[2] = w >> ss_hor;
    pic->h[0] = h;
    pic->h[1] = pic->h[2] = h >> ss_ver;
    const int aligned_y = pic->w[0] + DATA_ALIGN - (pic->w[0] % DATA_ALIGN);
    const int aligned_c = pic->w[1] + DATA_ALIGN - (pic->w[1] % DATA_ALIGN);
    const int hbd = pic->bpc > 8;
    pic->stride[0] = aligned_y << hbd;
    pic->stride[1] = pic->stride[2] = aligned_c << hbd;
    const size_t y_sz = pic->stride[0] * pic->h[0];
    const size_t uv_sz = pic->stride[1] * pic->h[1];
    const size_t pic_size = y_sz + 2 * uv_sz;

    uint8_t *data = aligned_malloc(pic_size, DATA_ALIGN);
    if (!data) goto fail;
    memset(data, 0, sizeof(*data));
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;

    //offset data memory allocation
    //the offset data buffer is in int16_t format to handle all bitdepth 8,10,12...
    pic->offset_stride[0] = pic->w[0] * sizeof(int16_t);
    pic->offset_stride[1] = pic->offset_stride[2] = pic->w[1] * sizeof(int16_t);
    const size_t offset_y_sz = pic->offset_stride[0] * pic->h[0];
    const size_t offset_uv_sz = pic->offset_stride[1] * pic->h[1];
    const size_t offset_pic_size = offset_y_sz + 2 * offset_uv_sz;

    uint8_t *offset_data_pointer = aligned_malloc(offset_pic_size, DATA_ALIGN);
    if (!offset_data_pointer) goto fail_offset;
    memset(offset_data_pointer, 0, sizeof(*offset_data_pointer));

    pic->offset_data[0] = offset_data_pointer;
    pic->offset_data[1] = offset_data_pointer + offset_y_sz;
    pic->offset_data[2] = offset_data_pointer + offset_y_sz + offset_uv_sz;

    pic->ref_cnt = malloc(sizeof(*pic->ref_cnt));
    if (!pic->ref_cnt) goto free_data;

    atomic_init(pic->ref_cnt, 1);
    return 0;

free_data:
    free(data);
    data = NULL;
    free(offset_data_pointer);
    offset_data_pointer = NULL;
fail_offset:
    if(data!=NULL)
        free(data);
fail:
    return -ENOMEM;
}

int vmaf_picture_ref(VmafPicture *dst, VmafPicture *src) {
    if (!dst || !src) return -EINVAL;

    memcpy(dst, src, sizeof(*src));
    atomic_fetch_add(src->ref_cnt, 1);
    return 0;
}

int vmaf_picture_unref(VmafPicture *pic) {
    if (!pic) return -EINVAL;
    if (!pic->ref_cnt) return -EINVAL;

    atomic_fetch_sub(pic->ref_cnt, 1);
    if (atomic_load(pic->ref_cnt) == 0) {
        aligned_free(pic->data[0]);
        aligned_free(pic->offset_data[0]);
        free(pic->ref_cnt);
    }
    memset(pic, 0, sizeof(*pic));
    return 0;
}
