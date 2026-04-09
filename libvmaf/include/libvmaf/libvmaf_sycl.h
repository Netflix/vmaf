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

#ifndef __VMAF_LIBVMAF_SYCL_H__
#define __VMAF_LIBVMAF_SYCL_H__

#include "libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque SYCL state — holds the sycl::queue, device-memory allocations,
 * shared frame buffers, and profiling state.  Created by
 * vmaf_sycl_state_init() and freed by vmaf_sycl_state_free().
 */
typedef struct VmafSyclState VmafSyclState;

/**
 * Configuration for SYCL device selection and queue creation.
 *
 * device_index — ordinal among Level Zero GPU devices (0 = first Intel GPU).
 *                Use -1 to select the SYCL default device.
 *
 * enable_profiling — if true the queue is created with
 *                    sycl::property::queue::enable_profiling so that
 *                    per-kernel timing can be collected.
 */
typedef struct VmafSyclConfiguration {
    int device_index;
    int enable_profiling;
} VmafSyclConfiguration;

/**
 * Create a new VmafSyclState with an internally-managed SYCL queue.
 *
 * @param[out] sycl_state  Receives the allocated state.
 * @param[in]  cfg         Device index and profiling flag.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_state_init(VmafSyclState **sycl_state,
                          VmafSyclConfiguration cfg);

/**
 * Import a VmafSyclState into a VmafContext.
 * After this call every feature extractor registered with the SYCL flag
 * will receive a pointer to this state during init.
 *
 * @param vmaf        The VMAF context.
 * @param sycl_state  Previously initialised state (ownership is NOT transferred).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_import_state(VmafContext *vmaf, VmafSyclState *sycl_state);

/**
 * Picture pre-allocation method.
 */
enum VmafSyclPicturePreallocationMethod {
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_NONE = 0,
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_DEVICE,
    VMAF_SYCL_PICTURE_PREALLOCATION_METHOD_HOST,
};

/**
 * Configuration for pre-allocating SYCL device pictures.
 */
typedef struct VmafSyclPictureConfiguration {
    struct {
        unsigned w, h;
        unsigned bpc;
        enum VmafPixelFormat pix_fmt;
    } pic_params;
    enum VmafSyclPicturePreallocationMethod pic_prealloc_method;
} VmafSyclPictureConfiguration;

/**
 * Pre-allocate SYCL device pictures for the given dimensions.
 * Must be called before vmaf_read_pictures() / vmaf_read_pictures_sycl().
 *
 * @param vmaf  The VMAF context (must have a SYCL state imported).
 * @param cfg   Picture configuration (width, height, pixel format).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_preallocate_pictures(VmafContext *vmaf,
                                    VmafSyclPictureConfiguration cfg);

/**
 * Fetch a pre-allocated SYCL picture.
 * The returned VmafPicture has device-side USM buffers.
 *
 * @param vmaf  The VMAF context.
 * @param[out] pic  Receives the pre-allocated picture.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_picture_fetch(VmafContext *vmaf, VmafPicture *pic);

/* ------------------------------------------------------------------
 * Zero-copy frame-buffer API
 *
 * The caller writes Y-plane data into shared device buffers (e.g.
 * via Intel VPL → Level Zero memory) and then calls
 * vmaf_read_pictures_sycl() instead of vmaf_read_pictures().
 * This avoids per-frame host→device copies entirely.
 * ------------------------------------------------------------------ */

/**
 * Initialise shared frame buffers for zero-copy operation.
 * Allocates two USM device buffers (ref + dis) sized for the Y plane.
 *
 * @param vmaf  The VMAF context.
 * @param w     Frame width in pixels.
 * @param h     Frame height in pixels.
 * @param bpc   Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_init_frame_buffers(VmafContext *vmaf,
                                  unsigned w, unsigned h, unsigned bpc);

/**
 * Obtain pointers to the shared device buffers.
 * The caller fills these with Y-plane data before calling
 * vmaf_read_pictures_sycl().
 *
 * @param vmaf      The VMAF context.
 * @param[out] ref  Pointer to the reference Y-plane device buffer.
 * @param[out] dis  Pointer to the distorted Y-plane device buffer.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_get_frame_buffers(VmafContext *vmaf,
                                 void **ref, void **dis);

/**
 * Wait for all submitted SYCL compute work to complete.
 * Call after vmaf_read_pictures_sycl() when you need the frame buffers
 * to be safe to overwrite.
 *
 * @param vmaf  The VMAF context.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_wait_compute(VmafContext *vmaf);

/**
 * Process one frame pair via the zero-copy path.
 * Assumes the caller has already written Y-plane data into the buffers
 * returned by vmaf_sycl_get_frame_buffers().
 *
 * @param vmaf   The VMAF context.
 * @param index  Frame index (0-based, sequential).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_read_pictures_sycl(VmafContext *vmaf, unsigned index);

/**
 * Flush all pending SYCL feature extractor work after the last frame.
 * Collects double-buffered results and synchronises the queue.
 *
 * @param vmaf  The VMAF context.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_flush_sycl(VmafContext *vmaf);

/* ------------------------------------------------------------------
 * DMA-BUF / VA surface zero-copy import
 *
 * Import GPU-resident frames (e.g. from VPL decode) directly into
 * SYCL device memory without a host roundtrip.
 *
 * Interop chain:
 *   VPL decode → VA surface → DMA-BUF fd → Level Zero import
 *   → SYCL device pointer → D2D copy into shared frame buffer
 * ------------------------------------------------------------------ */

/**
 * Import a DMA-BUF file descriptor as a SYCL device pointer.
 * Uses Level Zero external memory import.
 *
 * @param sycl_state  The SYCL state (must use Level Zero backend).
 * @param fd          DMA-BUF file descriptor (caller retains ownership).
 * @param size        Size of the DMA-BUF allocation in bytes.
 * @param[out] ptr    Receives the device pointer.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_dmabuf_import(VmafSyclState *sycl_state, int fd, size_t size,
                             void **ptr);

/**
 * Free a pointer returned by vmaf_sycl_dmabuf_import().
 *
 * @param sycl_state  The SYCL state.
 * @param ptr         Pointer to free.
 */
void vmaf_sycl_dmabuf_free(VmafSyclState *sycl_state, void *ptr);

/**
 * Import a VA surface Y-plane into a shared frame buffer.
 * Primary path: exports VA surface as DRM PRIME2 DMA-BUF, imports via
 * Level Zero, and runs a SYCL de-tiling kernel (zero-copy, GPU-only).
 * Fallback: vaGetImage + vaMapBuffer + H2D memcpy (GPU→CPU→GPU).
 * Shared frame buffers must be initialised first via
 * vmaf_sycl_init_frame_buffers().
 *
 * @param sycl_state   The SYCL state.
 * @param va_display   VA display handle (VADisplay, cast to void*).
 * @param va_surface   VA surface ID (VASurfaceID = unsigned int).
 * @param is_ref       If non-zero, import to ref buffer; else dis buffer.
 * @param w            Frame width in pixels.
 * @param h            Frame height in pixels.
 * @param bpc          Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_import_va_surface(VmafSyclState *sycl_state,
                                 void *va_display,
                                 unsigned int va_surface,
                                 int is_ref,
                                 unsigned w, unsigned h, unsigned bpc);

/**
 * Upload a raw Y-plane from host memory into a shared frame buffer.
 * Platform-agnostic path: copies from a host pointer (with pitch) to the
 * SYCL shared ref or dis buffer via H2D memcpy.
 *
 * @param sycl_state   The SYCL state.
 * @param src          Pointer to the Y-plane in host memory.
 * @param pitch        Row stride in bytes (may be > w * bytes_per_pixel).
 * @param is_ref       If non-zero, upload to ref buffer; else dis buffer.
 * @param w            Frame width in pixels.
 * @param h            Frame height in pixels.
 * @param bpc          Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_upload_plane(VmafSyclState *sycl_state,
                            const void *src, unsigned pitch,
                            int is_ref,
                            unsigned w, unsigned h, unsigned bpc);

#ifdef _WIN32
/**
 * Import a D3D11 texture Y-plane into a shared frame buffer (Windows).
 * Creates a staging texture, copies the decoded surface, maps it for
 * CPU read, and uploads via H2D memcpy. Shared frame buffers must be
 * initialised first via vmaf_sycl_init_frame_buffers().
 *
 * @param sycl_state     The SYCL state.
 * @param d3d11_device   ID3D11Device* (cast to void*).
 * @param d3d11_texture  ID3D11Texture2D* (cast to void*).
 * @param subresource    D3D11 subresource index.
 * @param is_ref         If non-zero, import to ref buffer; else dis buffer.
 * @param w              Frame width in pixels.
 * @param h              Frame height in pixels.
 * @param bpc            Bits per component (8 or 10).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_import_d3d11_surface(VmafSyclState *sycl_state,
                                    void *d3d11_device,
                                    void *d3d11_texture,
                                    unsigned subresource,
                                    int is_ref,
                                    unsigned w, unsigned h, unsigned bpc);
#endif /* _WIN32 */

/* ------------------------------------------------------------------
 * Profiling
 * ------------------------------------------------------------------ */

/**
 * Enable per-kernel GPU profiling.
 * Must be called before submitting any frames.
 *
 * @param sycl_state  The SYCL state.
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_profiling_enable(VmafSyclState *sycl_state);

/**
 * Disable profiling and discard accumulated timing data.
 *
 * @param sycl_state  The SYCL state.
 */
void vmaf_sycl_profiling_disable(VmafSyclState *sycl_state);

/**
 * Print profiling results to stdout.
 * Call after vmaf_sycl_wait_compute() or vmaf_flush_sycl().
 *
 * @param sycl_state  The SYCL state.
 */
void vmaf_sycl_profiling_print(VmafSyclState *sycl_state);

/**
 * Get profiling results as a formatted string.
 * Caller must free the returned string with free().
 *
 * @param sycl_state  The SYCL state.
 * @param[out] output Receives the allocated string (NULL on error).
 *
 * @return 0 on success, negative errno on failure.
 */
int vmaf_sycl_profiling_get_string(VmafSyclState *sycl_state,
                                    char **output);

/**
 * Release all resources owned by the SYCL state (queue, buffers, etc.)
 * and reset the pointer. The state must not be imported in any
 * VmafContext when this is called (call vmaf_close() first).
 *
 * @param sycl_state  The SYCL state (freed and set to NULL).
 */
void vmaf_sycl_state_free(VmafSyclState **sycl_state);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_LIBVMAF_SYCL_H__ */
