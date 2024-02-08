/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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


#include "mem.h"
#include "picture_cuda.h"
#include "common.h"
#include "log.h"

#include <cuda.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int vmaf_cuda_picture_download_async(VmafPicture *cuda_pic, VmafPicture *pic,
                                     uint8_t bitmask)
{
    if (!cuda_pic) return -EINVAL;
    if (!pic) return -EINVAL;

    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;

    VmafPicturePrivate *cuda_priv = cuda_pic->priv;
    for (int i = 0; i < 3; i++) {
        m.srcDevice = cuda_pic->data[i];
        m.srcPitch = cuda_pic->stride[i];
        m.dstHost = pic->data[i];
        m.dstPitch = pic->stride[i];
        m.WidthInBytes = cuda_pic->w[i] * ((pic->bpc + 7) / 8);
        m.Height = cuda_pic->h[i];
        if ((bitmask >> i) & 1)
            CHECK_CUDA(cuMemcpy2DAsync(&m, cuda_priv->cuda.str));
    }

    return CUDA_SUCCESS;
}

int vmaf_cuda_picture_upload_async(VmafPicture *cuda_pic,
                                   VmafPicture *pic, uint8_t bitmask)
{
    if (!cuda_pic) return -EINVAL;
    if (!pic) return -EINVAL;

    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_HOST;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    VmafPicturePrivate *cuda_priv = cuda_pic->priv;
    for (int i = 0; i < 3; i++) {
        m.srcHost = pic->data[i];
        m.srcPitch = pic->stride[i];
        m.dstDevice = cuda_pic->data[i];
        m.dstPitch = cuda_pic->stride[i];
        m.WidthInBytes = cuda_pic->w[i] * ((pic->bpc + 7) / 8);
        m.Height = cuda_pic->h[i];
        if ((bitmask >> i) & 1)
            CHECK_CUDA(cuMemcpy2DAsync(&m, cuda_priv->cuda.str));
    }
    CHECK_CUDA(cuEventRecord(cuda_priv->cuda.ready, cuda_priv->cuda.str));

    return CUDA_SUCCESS;
}

#define DATA_ALIGN_PINNED 32

static int default_release_pinned_picture(VmafPicture *pic, void *cookie)
{
    if (!pic) return -EINVAL;

    VmafPicturePrivate* priv = pic->priv;
    CHECK_CUDA(cuCtxPushCurrent(priv->cuda.ctx));
    CHECK_CUDA(cuMemFreeHost(pic->data[0]));
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    return 0;
}

int vmaf_cuda_picture_alloc_pinned(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                                   unsigned bpc, unsigned w, unsigned h,
                                   VmafCudaState *cuda_state)
{
    if (!pic) return -EINVAL;
    if (!pix_fmt) return -EINVAL;
    if (bpc < 8 || bpc > 16) return -EINVAL;    

    int err = 0;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pix_fmt;
    pic->bpc = bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = w;
    pic->w[1] = pic->w[2] = w >> ss_hor;
    pic->h[0] = h;
    pic->h[1] = pic->h[2] = h >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    const int aligned_y = (pic->w[0] + DATA_ALIGN_PINNED - 1) & ~(DATA_ALIGN_PINNED - 1);
    const int aligned_c = (pic->w[1] + DATA_ALIGN_PINNED - 1) & ~(DATA_ALIGN_PINNED - 1);
    const int hbd = pic->bpc > 8;
    pic->stride[0] = aligned_y << hbd;
    pic->stride[1] = pic->stride[2] = aligned_c << hbd;
    const size_t y_sz = pic->stride[0] * pic->h[0];
    const size_t uv_sz = pic->stride[1] * pic->h[1];
    const size_t pic_size = y_sz + 2 * uv_sz;

    CHECK_CUDA(cuCtxPushCurrent(cuda_state->ctx));
    uint8_t *data;
    CHECK_CUDA(cuMemHostAlloc((void**)&data, pic_size, CU_MEMHOSTALLOC_PORTABLE));
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    if (!data) goto fail;

    memset(data, 0, pic_size);
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->data[1] = pic->data[2] = NULL;

    err |= vmaf_picture_priv_init(pic);
    VmafPicturePrivate* priv = pic->priv;
    priv->cuda.ctx = cuda_state->ctx;
    err |= vmaf_picture_set_release_callback(pic, NULL, default_release_pinned_picture);
    if (err) goto free_data;
    priv->buf_type = VMAF_PICTURE_BUFFER_TYPE_CUDA_HOST_PINNED;

    err |= vmaf_ref_init(&pic->ref);
    if (err) goto free_priv;

    return 0;

free_priv:
    free(pic->priv);
free_data:
    CHECK_CUDA(cuMemFreeHost(data));
fail:
    return -ENOMEM;
}

int vmaf_cuda_picture_alloc(VmafPicture *pic, void *cookie)
{
    if (!pic) return -EINVAL;
    if (!cookie) return -EINVAL;

    VmafCudaCookie *cuda_cookie = cookie;
    if (!cuda_cookie->pix_fmt) return -1;
    if (cuda_cookie->bpc < 8 || cuda_cookie->bpc > 16) return -1;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = cuda_cookie->pix_fmt;
    pic->bpc = cuda_cookie->bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = cuda_cookie->w;
    pic->w[1] = pic->w[2] = cuda_cookie->w >> ss_hor;
    pic->h[0] = cuda_cookie->h;
    pic->h[1] = pic->h[2] = cuda_cookie->h >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    CHECK_CUDA(cuCtxPushCurrent(cuda_cookie->state->ctx));
    
    VmafPicturePrivate* priv = pic->priv = malloc(sizeof(VmafPicturePrivate));
    if (!priv) return -ENOMEM;
    priv->cuda.ctx = cuda_cookie->state->ctx;
    CHECK_CUDA(cuStreamCreate(&priv->cuda.str, CU_STREAM_DEFAULT));
    CHECK_CUDA(cuEventCreate(&priv->cuda.ready, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&priv->cuda.finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventRecord(priv->cuda.finished, priv->cuda.str));
    priv->buf_type = VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE;

    const int hbd = pic->bpc > 8;
    
    for (int i = 0; i < 3; i++) {
        if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P && i > 0) {
            pic->data[1] = pic->data[2] = NULL;
            break;
        }
        CHECK_CUDA(cuMemAllocPitch(&pic->data[i], &pic->stride[i],
                    pic->w[i] * ((pic->bpc + 7) / 8), pic->h[i],
                    8 << hbd));
    }

    CHECK_CUDA(cuCtxPopCurrent(NULL));
    int err = vmaf_ref_init(&pic->ref);
    if (err) return err;

    return 0;
}

int vmaf_cuda_picture_free(VmafPicture *pic, void *cookie)
{
    if (!pic) return -EINVAL;

    int err = vmaf_ref_load(pic->ref);
    if (!err) -EINVAL;
    
    VmafPicturePrivate *priv = pic->priv;
    VmafCudaCookie *cuda_cookie = cookie;
    CHECK_CUDA(cuCtxPushCurrent(cuda_cookie->state->ctx));
    CHECK_CUDA(cuStreamSynchronize(priv->cuda.str));

    for (int i = 0; i < 3; i++) {
	    if (pic->data[i])
		    CHECK_CUDA(cuMemFreeAsync(pic->data[i], priv->cuda.str));
    }

    CHECK_CUDA(cuEventDestroy(priv->cuda.finished));
    CHECK_CUDA(cuEventDestroy(priv->cuda.ready));
    CHECK_CUDA(cuStreamDestroy(priv->cuda.str));
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    free(priv);
    memset(pic, 0, sizeof(*pic));

    return 0;
}

int vmaf_cuda_picture_synchronize(VmafPicture *pic, void *cookie)
{
    if (!pic) return -EINVAL;
    (void)cookie;

    VmafPicturePrivate* priv = pic->priv;
    CHECK_CUDA(cuEventSynchronize(priv->cuda.finished));
    CHECK_CUDA(cuStreamSynchronize(priv->cuda.str));
    CHECK_CUDA(cuCtxPushCurrent(priv->cuda.ctx));
    CHECK_CUDA(cuEventDestroy(priv->cuda.ready));
    CHECK_CUDA(cuEventDestroy(priv->cuda.finished));
    CHECK_CUDA(cuEventCreate(&priv->cuda.finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&priv->cuda.ready, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuCtxPopCurrent(NULL));
    
    return 0;
}

CUstream vmaf_cuda_picture_get_stream(VmafPicture *pic)
{
    VmafPicturePrivate* priv = pic->priv;
    return priv->cuda.str;
}

CUevent vmaf_cuda_picture_get_ready_event(VmafPicture *pic)
{
    VmafPicturePrivate* priv = pic->priv;
    return priv->cuda.ready;
}

CUevent vmaf_cuda_picture_get_finished_event(VmafPicture *pic)
{
    VmafPicturePrivate* priv = pic->priv;
    return priv->cuda.finished;
}
