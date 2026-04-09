/**
 *
 *  Copyright 2016-2024 Netflix, Inc.
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

/**
 * SYCL picture management — upload/download Y-plane data between host
 * VmafPicture buffers and SYCL USM device memory.  Also provides the
 * alloc/free callbacks used by VmafPicture pre-allocation pools.
 *
 * NOTE: The primary frame-upload path for SYCL extractors uses the
 * shared-buffer mechanism in common.cpp (vmaf_sycl_shared_frame_upload).
 * The functions here are for:
 *   1. Standalone picture upload/download (testing, future VPL interop).
 *   2. VmafPicture pre-allocation pool callbacks.
 */

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include <sycl/sycl.hpp>

#include "picture_sycl.h"
#include "common.h"

/* ------------------------------------------------------------------ */
/* Upload / download                                                   */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_picture_upload(VmafSyclState *state, void *dst,
                              VmafPicture *pic, unsigned plane)
{
    if (!state || !dst || !pic) return -EINVAL;
    if (plane >= 3) return -EINVAL;

    size_t bpp = (pic->bpc + 7) / 8;
    size_t row_bytes = pic->w[plane] * bpp;
    size_t total = row_bytes * pic->h[plane];

    if (pic->stride[plane] == (ptrdiff_t)row_bytes) {
        /* Contiguous — single memcpy */
        return vmaf_sycl_memcpy_h2d(state, dst, pic->data[plane], total);
    }

    /* Non-contiguous — row-by-row pack into device buffer */
    const uint8_t *src = (const uint8_t *)pic->data[plane];
    uint8_t *d = (uint8_t *)dst;
    for (unsigned y = 0; y < pic->h[plane]; y++) {
        int err = vmaf_sycl_memcpy_h2d(state, d + y * row_bytes,
                                        src + y * pic->stride[plane],
                                        row_bytes);
        if (err) return err;
    }
    return 0;
}

extern "C"
int vmaf_sycl_picture_download(VmafSyclState *state, const void *src,
                                VmafPicture *pic, unsigned plane)
{
    if (!state || !src || !pic) return -EINVAL;
    if (plane >= 3) return -EINVAL;

    size_t bpp = (pic->bpc + 7) / 8;
    size_t row_bytes = pic->w[plane] * bpp;
    size_t total = row_bytes * pic->h[plane];

    if (pic->stride[plane] == (ptrdiff_t)row_bytes) {
        return vmaf_sycl_memcpy_d2h(state, pic->data[plane], src, total);
    }

    /* Non-contiguous — download packed, then scatter rows */
    void *packed = malloc(total);
    if (!packed) return -ENOMEM;

    int err = vmaf_sycl_memcpy_d2h(state, packed, src, total);
    if (!err) {
        uint8_t *dst_ptr = (uint8_t *)pic->data[plane];
        for (unsigned y = 0; y < pic->h[plane]; y++) {
            memcpy(dst_ptr + y * pic->stride[plane],
                   (uint8_t *)packed + y * row_bytes,
                   row_bytes);
        }
    }
    free(packed);
    return err;
}

/* ------------------------------------------------------------------ */
/* VmafPicture pool callbacks                                          */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_picture_alloc(VmafPicture *pic, void *cookie)
{
    if (!pic || !cookie) return -EINVAL;

    VmafSyclCookie *c = (VmafSyclCookie *)cookie;
    size_t bpp = (c->bpc + 7) / 8;
    size_t plane_size = (size_t)c->w * c->h * bpp;

    /* Allocate device memory for Y plane only (VMAF operates on luma) */
    void *dev_buf = vmaf_sycl_malloc_device(c->state, plane_size);
    if (!dev_buf) return -ENOMEM;

    memset(pic, 0, sizeof(*pic));
    pic->data[0] = dev_buf;
    pic->stride[0] = c->w * bpp;
    pic->w[0] = c->w;
    pic->h[0] = c->h;
    pic->bpc = c->bpc;
    pic->pix_fmt = c->pix_fmt;

    return 0;
}

extern "C"
int vmaf_sycl_picture_free(VmafPicture *pic, void *cookie)
{
    if (!pic || !cookie) return -EINVAL;

    VmafSyclCookie *c = (VmafSyclCookie *)cookie;

    if (pic->data[0]) {
        vmaf_sycl_free(c->state, pic->data[0]);
        pic->data[0] = NULL;
    }
    if (pic->data[1]) {
        vmaf_sycl_free(c->state, pic->data[1]);
        pic->data[1] = NULL;
    }
    if (pic->data[2]) {
        vmaf_sycl_free(c->state, pic->data[2]);
        pic->data[2] = NULL;
    }

    return 0;
}
