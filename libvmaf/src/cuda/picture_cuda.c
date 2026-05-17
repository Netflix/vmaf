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
#include "ref.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int vmaf_cuda_picture_download_async(VmafPicture *cuda_pic, VmafPicture *pic, uint8_t bitmask)
{
    if (!cuda_pic)
        return -EINVAL;
    if (!pic)
        return -EINVAL;

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;

    VmafPicturePrivate *cuda_priv = cuda_pic->priv;
    CudaFunctions *cu_f = cuda_priv->cuda.state->f;
    for (int i = 0; i < 3; i++) {
        m.srcDevice = (CUdeviceptr)cuda_pic->data[i];
        m.srcPitch = cuda_pic->stride[i];
        m.dstHost = pic->data[i];
        m.dstPitch = pic->stride[i];
        m.WidthInBytes = (size_t)cuda_pic->w[i] * ((pic->bpc + 7) / 8);
        m.Height = cuda_pic->h[i];
        if ((bitmask >> i) & 1)
            CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m, cuda_priv->cuda.str));
    }

    return 0;
}

int vmaf_cuda_picture_upload_async(VmafPicture *cuda_pic, VmafPicture *pic, uint8_t bitmask)
{
    if (!cuda_pic)
        return -EINVAL;
    if (!pic)
        return -EINVAL;

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_HOST;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    VmafPicturePrivate *cuda_priv = cuda_pic->priv;
    CudaFunctions *cu_f = cuda_priv->cuda.state->f;
    for (int i = 0; i < 3; i++) {
        m.srcHost = pic->data[i];
        m.srcPitch = pic->stride[i];
        m.dstDevice = (CUdeviceptr)cuda_pic->data[i];
        m.dstPitch = cuda_pic->stride[i];
        m.WidthInBytes = (size_t)cuda_pic->w[i] * ((pic->bpc + 7) / 8);
        m.Height = cuda_pic->h[i];
        if ((bitmask >> i) & 1)
            CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m, cuda_priv->cuda.str));
    }
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(cuda_priv->cuda.ready, cuda_priv->cuda.str));

    return 0;
}

#define DATA_ALIGN_PINNED 32

static int default_release_pinned_picture(VmafPicture *pic, void *cookie)
{
    (void)cookie;
    if (!pic)
        return -EINVAL;

    VmafPicturePrivate *priv = pic->priv;
    CudaFunctions *cu_f = priv->cuda.state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(priv->cuda.ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuMemFreeHost(pic->data[0]), fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);
    return 0;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

int vmaf_cuda_picture_alloc_pinned(VmafPicture *pic, enum VmafPixelFormat pix_fmt, unsigned bpc,
                                   unsigned w, unsigned h, VmafCudaState *cuda_state)
{
    if (!pic)
        return -EINVAL;
    if (!pix_fmt)
        return -EINVAL;
    if (bpc < 8 || bpc > 16)
        return -EINVAL;

    int err = 0;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pix_fmt;
    pic->bpc = bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = w;
    /* Ceiling division — mirrors picture.c fix (Research-0094). */
    pic->w[1] = pic->w[2] = (w + ((unsigned)ss_hor)) >> ss_hor;
    pic->h[0] = h;
    pic->h[1] = pic->h[2] = (h + ((unsigned)ss_ver)) >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    const unsigned aligned_y = (pic->w[0] + DATA_ALIGN_PINNED - 1u) & ~(DATA_ALIGN_PINNED - 1u);
    const unsigned aligned_c = (pic->w[1] + DATA_ALIGN_PINNED - 1u) & ~(DATA_ALIGN_PINNED - 1u);
    const int hbd = pic->bpc > 8;
    pic->stride[0] = aligned_y << hbd;
    pic->stride[1] = pic->stride[2] = aligned_c << hbd;
    const size_t y_sz = pic->stride[0] * pic->h[0];
    const size_t uv_sz = pic->stride[1] * pic->h[1];
    const size_t pic_size = y_sz + 2 * uv_sz;
    CudaFunctions *cu_f = cuda_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    uint8_t *data = NULL;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(cuda_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuMemHostAlloc((void **)&data, pic_size, 0x01), fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);
    if (!data)
        goto fail_no_data;

    memset(data, 0, pic_size);
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->data[1] = pic->data[2] = NULL;

    /* vmaf_picture_priv_init allocates pic->priv; check before touching it.
     * Mirrors the fix in picture.c (PR #700, CWE-476): the |= idiom evaluates
     * the right-hand side unconditionally, so a priv-init failure would leave
     * pic->priv == NULL and the subsequent field writes would null-deref. */
    err = vmaf_picture_priv_init(pic);
    if (err)
        goto free_data;

    VmafPicturePrivate *priv = pic->priv;
    priv->cuda.state = cuda_state;
    priv->cuda.ctx = cuda_state->ctx;
    err = vmaf_picture_set_release_callback(pic, NULL, default_release_pinned_picture);
    if (err)
        goto free_priv;
    priv->buf_type = VMAF_PICTURE_BUFFER_TYPE_CUDA_HOST_PINNED;

    err = vmaf_ref_init(&pic->ref);
    if (err)
        goto free_priv;

    return 0;

free_priv:
    free(pic->priv);
free_data:
    (void)cu_f->cuMemFreeHost(data);
fail_no_data:
    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

int vmaf_cuda_picture_alloc(VmafPicture *pic, void *cookie)
{
    if (!pic)
        return -EINVAL;
    if (!cookie)
        return -EINVAL;

    VmafCudaCookie *cuda_cookie = cookie;
    if (!cuda_cookie->pix_fmt)
        return -1;
    if (cuda_cookie->bpc < 8 || cuda_cookie->bpc > 16)
        return -1;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = cuda_cookie->pix_fmt;
    pic->bpc = cuda_cookie->bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = cuda_cookie->w;
    /* Ceiling division — mirrors picture.c fix (Research-0094). */
    pic->w[1] = pic->w[2] = (cuda_cookie->w + ((unsigned)ss_hor)) >> ss_hor;
    pic->h[0] = cuda_cookie->h;
    pic->h[1] = pic->h[2] = (cuda_cookie->h + ((unsigned)ss_ver)) >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    VmafPicturePrivate *priv = pic->priv = malloc(sizeof(VmafPicturePrivate));
    if (!priv)
        return -ENOMEM;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cuda_cookie->state->f, cuCtxPushCurrent(cuda_cookie->state->ctx), fail);
    ctx_pushed = 1;
    priv->cuda.state = cuda_cookie->state;
    priv->cuda.ctx = cuda_cookie->state->ctx;
    CudaFunctions *cu_f = priv->cuda.state->f;
    /* Use CU_STREAM_NON_BLOCKING so this picture-upload stream does not
     * implicitly serialise with the legacy NULL (default) stream.
     * CU_STREAM_DEFAULT causes every operation on this stream to act as if
     * the default stream were involved, meaning all other non-default streams
     * must complete before any work on this stream starts (and vice versa).
     * At sub-4K resolutions that per-frame round-trip serialisation dominates
     * compute time and makes CUDA motion ~0.55× slower than CPU scalar.
     * CU_STREAM_NON_BLOCKING removes the implicit barrier.
     * ADR-0378. */
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&priv->cuda.str, CU_STREAM_NON_BLOCKING, 0),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&priv->cuda.ready, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&priv->cuda.finished, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventRecord(priv->cuda.finished, priv->cuda.str), fail);
    priv->buf_type = VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE;

    const int hbd = pic->bpc > 8;

    for (int i = 0; i < 3; i++) {
        if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P && i > 0) {
            pic->data[1] = pic->data[2] = NULL;
            break;
        }
        CHECK_CUDA_GOTO(cu_f,
                        cuMemAllocPitch((CUdeviceptr *)&pic->data[i], (size_t *)&pic->stride[i],
                                        (size_t)pic->w[i] * ((pic->bpc + 7) / 8), pic->h[i],
                                        8 << hbd),
                        fail);
    }

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);
    int err = vmaf_ref_init(&pic->ref);
    if (err)
        return err;

    return 0;

fail:
    if (ctx_pushed)
        (void)cuda_cookie->state->f->cuCtxPopCurrent(NULL);
fail_after_pop:
    free(priv);
    pic->priv = NULL;
    return _cuda_err;
}

int vmaf_cuda_picture_free(VmafPicture *pic, void *cookie)
{
    if (!pic)
        return -EINVAL;

    long err = vmaf_ref_load(pic->ref);
    if (!err)
        return -EINVAL;

    VmafPicturePrivate *priv = pic->priv;
    VmafCudaCookie *cuda_cookie = cookie;
    CudaFunctions *cu_f = cuda_cookie->state->f;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(cuda_cookie->state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamSynchronize(priv->cuda.str), fail);

    for (int i = 0; i < 3; i++) {
        if (pic->data[i]) {
            CHECK_CUDA_GOTO(cu_f, cuMemFree((CUdeviceptr)pic->data[i]), fail);
        }
    }

    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(priv->cuda.finished), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(priv->cuda.ready), fail);
    CHECK_CUDA_GOTO(cu_f, cuStreamDestroy(priv->cuda.str), fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);
    vmaf_ref_close(pic->ref);
    free(priv);
    memset(pic, 0, sizeof(*pic));

    return 0;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

int vmaf_cuda_picture_synchronize(VmafPicture *pic, void *cookie)
{
    if (!pic)
        return -EINVAL;
    (void)cookie;

    VmafPicturePrivate *priv = pic->priv;
    CudaFunctions *cu_f = priv->cuda.state->f;
    CHECK_CUDA_RETURN(cu_f, cuEventSynchronize(priv->cuda.finished));
    // cuStreamSynchronize after cuEventSynchronize on the same stream is
    // redundant — the event was recorded on this stream, so the sync
    // already guarantees all prior work on the stream is complete.
    // cuCtxPushCurrent/cuCtxPopCurrent with no work between them is a no-op.
    return 0;
}

CUstream vmaf_cuda_picture_get_stream(VmafPicture *pic)
{
    VmafPicturePrivate *priv = pic->priv;
    return priv->cuda.str;
}

CUevent vmaf_cuda_picture_get_ready_event(VmafPicture *pic)
{
    VmafPicturePrivate *priv = pic->priv;
    return priv->cuda.ready;
}

CUevent vmaf_cuda_picture_get_finished_event(VmafPicture *pic)
{
    VmafPicturePrivate *priv = pic->priv;
    return priv->cuda.finished;
}

enum VmafPixelFormat vmaf_cuda_picture_get_pix_fmt(const VmafPicture *pic)
{
    return pic->pix_fmt;
}
