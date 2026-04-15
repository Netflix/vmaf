/**
 *
 *  Copyright 2024-2026 Lusoris and Claude (Anthropic)
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

#ifndef __VMAF_SRC_SYCL_DMABUF_IMPORT_H__
#define __VMAF_SRC_SYCL_DMABUF_IMPORT_H__

#include "config.h"

#if HAVE_SYCL

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafSyclState VmafSyclState;

/**
 * Import a DMA-BUF file descriptor as a SYCL device pointer via
 * Level Zero external memory import.
 *
 * The caller retains ownership of the fd and must close it when done.
 * The returned pointer must be freed with vmaf_sycl_dmabuf_free().
 *
 * @param state   The SYCL state (must use Level Zero backend).
 * @param fd      DMA-BUF file descriptor.
 * @param size    Size of the DMA-BUF allocation in bytes.
 * @param[out] ptr  Receives the SYCL-accessible device pointer.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_dmabuf_import(VmafSyclState *state, int fd, size_t size,
                             void **ptr);

/**
 * Free a pointer returned by vmaf_sycl_dmabuf_import().
 *
 * @param state  The SYCL state.
 * @param ptr    Pointer to free (previously returned by dmabuf_import).
 */
void vmaf_sycl_dmabuf_free(VmafSyclState *state, void *ptr);

/**
 * Import a VA surface Y-plane into the SYCL shared frame buffers.
 *
 * Primary path (zero-copy): exports the VA surface as a DRM PRIME2
 * DMA-BUF handle, imports it via Level Zero, and uses a SYCL compute
 * kernel to de-tile (Tile4/Y-tiled → linear) directly into the shared
 * buffer. Everything stays on GPU — no CPU copies, no PCIe pixel traffic.
 *
 * Fallback path: if the export or DMA-BUF import fails, uses
 * vaGetImage + vaMapBuffer + H2D memcpy (GPU→CPU→GPU).
 *
 * The VA surface must be in NV12 or P010 format. Only the Y plane is
 * imported (luma). The UV plane is ignored (VMAF only uses luma).
 *
 * @param state       The SYCL state (shared frame buffers must be initialised).
 * @param va_display  The VA display handle (VADisplay).
 * @param va_surface  The VA surface ID (VASurfaceID = unsigned int).
 * @param is_ref      If true, copy to shared_ref_buf; else shared_dis_buf.
 * @param w           Frame width in pixels.
 * @param h           Frame height in pixels.
 * @param bpc         Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_import_va_surface(VmafSyclState *state,
                                 void *va_display,
                                 unsigned int va_surface,
                                 int is_ref,
                                 unsigned w, unsigned h, unsigned bpc);

#ifdef __cplusplus
}
#endif

#endif /* HAVE_SYCL */

#endif /* __VMAF_SRC_SYCL_DMABUF_IMPORT_H__ */
