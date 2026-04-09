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

#ifndef __VMAF_SRC_SYCL_PICTURE_SYCL_H__
#define __VMAF_SRC_SYCL_PICTURE_SYCL_H__

#include "common.h"
#include "libvmaf/picture.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Cookie attached to VmafPicture instances that own SYCL device memory.
 * Used by the pre-allocation and picture management APIs.
 */
typedef struct VmafSyclCookie {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w, h;
    VmafSyclState *state;
} VmafSyclCookie;

/**
 * Upload a single Y-plane from a host VmafPicture to a SYCL USM device
 * buffer.  Handles stride != width (packs rows contiguously).
 *
 * @param state  The SYCL state (provides queue for memcpy).
 * @param dst    Device pointer to receive the packed Y-plane data.
 * @param pic    Source host-side VmafPicture.
 * @param plane  Plane index (normally 0 for Y).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_picture_upload(VmafSyclState *state, void *dst,
                              VmafPicture *pic, unsigned plane);

/**
 * Download a Y-plane from a SYCL USM device buffer back to a host
 * VmafPicture.  Handles stride != width (inserts padding per row).
 *
 * @param state  The SYCL state (provides queue for memcpy).
 * @param src    Device pointer containing packed Y-plane data.
 * @param pic    Destination host-side VmafPicture.
 * @param plane  Plane index (normally 0 for Y).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_picture_download(VmafSyclState *state, const void *src,
                                VmafPicture *pic, unsigned plane);

/**
 * Allocate a SYCL device-backed VmafPicture.
 * Called by the VmafPicture pre-allocation pool.
 *
 * @param pic     The picture to initialise (data pointers set to USM allocs).
 * @param cookie  Pointer to a VmafSyclCookie with dimensions and state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_picture_alloc(VmafPicture *pic, void *cookie);

/**
 * Free a SYCL device-backed VmafPicture.
 * Called by VmafPicture release callback.
 *
 * @param pic     The picture to free.
 * @param cookie  Pointer to a VmafSyclCookie.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_picture_free(VmafPicture *pic, void *cookie);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_SRC_SYCL_PICTURE_SYCL_H__ */
