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
 * SYCL VA-API surface import — import VA-API decoded frames into SYCL
 * shared frame buffers for zero-host-copy VMAF computation.
 *
 * Primary (zero-copy) path:
 *   VA surface → vaExportSurfaceHandle (DRM PRIME2) → DMA-BUF fd
 *   → Level Zero external memory import → SYCL device pointer
 *   → SYCL de-tiling kernel (Tile4/Y-tiled → linear) or D2D memcpy (LINEAR)
 *   → shared frame buffer.
 *   Everything stays on GPU — no CPU copies, no PCIe pixel traffic.
 *
 * Fallback (readback) path:
 *   VA surface → vaGetImage (de-tiles) → vaMapBuffer → H2D memcpy.
 *   Used when vaExportSurfaceHandle or DMA-BUF import fails.
 *
 * DMA-BUF import helpers (vmaf_sycl_dmabuf_import/free):
 *   Raw DMA-BUF fd → Level Zero zeMemAllocDevice → SYCL device pointer.
 */

#include "config.h"

#if HAVE_SYCL

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <level_zero/ze_api.h>

#if HAVE_SYCL_DMABUF
#include <va/va.h>
#include <va/va_drmcommon.h>
#endif

extern "C" {
#include "dmabuf_import.h"
#include "common.h"
#include "log.h"
}

/* ------------------------------------------------------------------ */
/* DMA-BUF → Level Zero → SYCL device pointer                         */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_dmabuf_import(VmafSyclState *state, int fd, size_t size,
                             void **ptr)
{
    if (!state || fd < 0 || !size || !ptr) return -EINVAL;
    *ptr = nullptr;

    try {
        sycl::queue *q = (sycl::queue *)vmaf_sycl_get_queue_ptr(state);
        if (!q) return -EINVAL;

        /* Extract Level Zero native handles from the SYCL queue */
        ze_context_handle_t ze_ctx =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                q->get_context());
        ze_device_handle_t ze_dev =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                q->get_device());

        /* Set up DMA-BUF import descriptor */
        ze_external_memory_import_fd_t import_desc = {};
        import_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
        import_desc.pNext = nullptr;
        import_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
        import_desc.fd = fd;

        /* Chain import descriptor into device allocation descriptor */
        ze_device_mem_alloc_desc_t alloc_desc = {};
        alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        alloc_desc.pNext = &import_desc;
        alloc_desc.flags = 0;
        alloc_desc.ordinal = 0;

        void *ze_ptr = nullptr;
        ze_result_t res = zeMemAllocDevice(ze_ctx, &alloc_desc, size,
                                            0 /* alignment */, ze_dev,
                                            &ze_ptr);
        if (res != ZE_RESULT_SUCCESS) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "Level Zero DMA-BUF import failed: 0x%x\n", res);
            return -EIO;
        }

        *ptr = ze_ptr;
        return 0;

    } catch (const sycl::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SYCL DMA-BUF import exception: %s\n", e.what());
        return -EIO;
    } catch (const std::exception &e) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "DMA-BUF import error: %s\n", e.what());
        return -EIO;
    }
}

extern "C"
void vmaf_sycl_dmabuf_free(VmafSyclState *state, void *ptr)
{
    if (!state || !ptr) return;

    try {
        sycl::queue *q = (sycl::queue *)vmaf_sycl_get_queue_ptr(state);
        if (!q) return;

        ze_context_handle_t ze_ctx =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                q->get_context());

        zeMemFree(ze_ctx, ptr);
    } catch (...) {
        /* Best effort — if SYCL/L0 is already torn down, ignore */
    }
}

#if HAVE_SYCL_DMABUF

/* ------------------------------------------------------------------ */
/* DRM format modifier constants                                       */
/* ------------------------------------------------------------------ */

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0ULL
#endif

/* fourcc_mod_code(INTEL, 9) */
#ifndef I915_FORMAT_MOD_4_TILED
#define I915_FORMAT_MOD_4_TILED 0x0100000000000009ULL
#endif

/* fourcc_mod_code(INTEL, 2) */
#ifndef I915_FORMAT_MOD_Y_TILED
#define I915_FORMAT_MOD_Y_TILED 0x0100000000000002ULL
#endif

/* fourcc_mod_code(INTEL, 1) */
#ifndef I915_FORMAT_MOD_X_TILED
#define I915_FORMAT_MOD_X_TILED 0x0100000000000001ULL
#endif

/* ------------------------------------------------------------------ */
/* Fallback: VA surface → linear readback → H2D copy                   */
/* Used when vaExportSurfaceHandle / DMA-BUF import is not available.  */
/* ------------------------------------------------------------------ */

static int vmaf_sycl_import_va_surface_readback(
    VmafSyclState *state,
    void *va_display_handle,
    unsigned int va_surface_id,
    int is_ref,
    unsigned w, unsigned h, unsigned bpc)
{
    VADisplay va_dpy = (VADisplay)va_display_handle;
    VASurfaceID va_surf = (VASurfaceID)va_surface_id;

    unsigned bytes_per_pixel = (bpc + 7) / 8;
    VAImageFormat y_fmt;
    memset(&y_fmt, 0, sizeof(y_fmt));

    /* Find a suitable image format (NV12 for 8-bit, P010 for 10-bit) */
    {
        int num_fmts = vaMaxNumImageFormats(va_dpy);
        VAImageFormat *fmts = (VAImageFormat *)malloc(
            (size_t)num_fmts * sizeof(VAImageFormat));
        int actual = 0;
        vaQueryImageFormats(va_dpy, fmts, &actual);

        uint32_t target_fourcc = (bpc <= 8) ? VA_FOURCC_NV12 : VA_FOURCC_P010;
        int found = 0;
        for (int i = 0; i < actual; i++) {
            if (fmts[i].fourcc == target_fourcc) {
                y_fmt = fmts[i];
                found = 1;
                break;
            }
        }
        free(fmts);

        if (!found) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "VA image format %s not supported\n",
                     bpc <= 8 ? "NV12" : "P010");
            return -ENOTSUP;
        }
    }

    VAImage va_img;
    memset(&va_img, 0, sizeof(va_img));

    VAStatus va_st = vaCreateImage(va_dpy, &y_fmt, w, h, &va_img);
    if (va_st != VA_STATUS_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vaCreateImage failed: %s\n", vaErrorStr(va_st));
        return -EIO;
    }

    va_st = vaGetImage(va_dpy, va_surf, 0, 0, w, h, va_img.image_id);
    if (va_st != VA_STATUS_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vaGetImage failed: %s\n", vaErrorStr(va_st));
        vaDestroyImage(va_dpy, va_img.image_id);
        return -EIO;
    }

    void *img_data = nullptr;
    va_st = vaMapBuffer(va_dpy, va_img.buf, &img_data);
    if (va_st != VA_STATUS_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vaMapBuffer failed: %s\n", vaErrorStr(va_st));
        vaDestroyImage(va_dpy, va_img.image_id);
        return -EIO;
    }

    uint8_t *y_plane = (uint8_t *)img_data + va_img.offsets[0];
    uint32_t y_pitch = va_img.pitches[0];
    size_t   y_row_bytes = (size_t)w * bytes_per_pixel;

    void *target_buf = is_ref
        ? vmaf_sycl_get_shared_ref(state)
        : vmaf_sycl_get_shared_dis(state);

    if (!target_buf) {
        vaUnmapBuffer(va_dpy, va_img.buf);
        vaDestroyImage(va_dpy, va_img.image_id);
        return -EINVAL;
    }

    sycl::queue *q = (sycl::queue *)vmaf_sycl_get_queue_ptr(state);

    if (y_pitch == y_row_bytes) {
        q->memcpy(target_buf, y_plane, y_row_bytes * h);
    } else {
        uint8_t *src = y_plane;
        uint8_t *dst = (uint8_t *)target_buf;
        for (unsigned row = 0; row < h; row++) {
            q->memcpy(dst, src, y_row_bytes);
            src += y_pitch;
            dst += y_row_bytes;
        }
    }
    q->wait();

    vaUnmapBuffer(va_dpy, va_img.buf);
    vaDestroyImage(va_dpy, va_img.image_id);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Zero-copy: VA surface → DMA-BUF → Level Zero → SYCL de-tile        */
/* ------------------------------------------------------------------ */

extern "C"
int vmaf_sycl_import_va_surface(VmafSyclState *state,
                                 void *va_display_handle,
                                 unsigned int va_surface_id,
                                 int is_ref,
                                 unsigned w, unsigned h, unsigned bpc)
{
    if (!state || !va_display_handle) return -EINVAL;

    VADisplay va_dpy = (VADisplay)va_display_handle;
    VASurfaceID va_surf = (VASurfaceID)va_surface_id;

    /* Sync the VA surface to ensure decode is complete */
    VAStatus va_st = vaSyncSurface(va_dpy, va_surf);
    if (va_st != VA_STATUS_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "vaSyncSurface failed: %s\n", vaErrorStr(va_st));
    }

    /* Export the VA surface as DRM PRIME2 (DMA-BUF fd + layout info) */
    VADRMPRIMESurfaceDescriptor desc;
    memset(&desc, 0, sizeof(desc));

    va_st = vaExportSurfaceHandle(
        va_dpy, va_surf,
        VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
        VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
        &desc);

    if (va_st != VA_STATUS_SUCCESS) {
        vmaf_log(VMAF_LOG_LEVEL_INFO,
                 "vaExportSurfaceHandle failed: %s — using readback path\n",
                 vaErrorStr(va_st));
        return vmaf_sycl_import_va_surface_readback(
            state, va_display_handle, va_surface_id, is_ref, w, h, bpc);
    }

    if (desc.num_layers < 1 || desc.num_objects < 1) {
        for (uint32_t i = 0; i < desc.num_objects; i++)
            close(desc.objects[i].fd);
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "DRM PRIME descriptor has no layers\n");
        return -EIO;
    }

    /* Extract Y plane metadata from the first layer */
    uint32_t y_obj_idx = desc.layers[0].object_index[0];
    int      y_fd      = desc.objects[y_obj_idx].fd;
    uint32_t y_size    = desc.objects[y_obj_idx].size;
    uint64_t modifier  = desc.objects[y_obj_idx].drm_format_modifier;
    uint32_t y_offset  = desc.layers[0].offset[0];
    uint32_t y_pitch   = desc.layers[0].pitch[0];
    unsigned bpp       = (bpc + 7) / 8;

    vmaf_log(VMAF_LOG_LEVEL_DEBUG,
             "[%s] DRM PRIME: fd=%d size=%u modifier=0x%llx "
             "offset=%u pitch=%u (%ux%u @ %u bpp)\n",
             is_ref ? "ref" : "dis",
             y_fd, y_size, (unsigned long long)modifier,
             y_offset, y_pitch, w, h, bpp);

    /* Import the DMA-BUF fd into Level Zero as device memory.
     * This wraps the SAME GPU memory — no copy happens here. */
    void *imported_ptr = nullptr;
    int err = vmaf_sycl_dmabuf_import(state, y_fd, y_size, &imported_ptr);

    /* Close all exported DMA-BUF fds (our API: caller retains ownership) */
    for (uint32_t i = 0; i < desc.num_objects; i++)
        close(desc.objects[i].fd);

    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_INFO,
                 "DMA-BUF import failed (%d) — using readback path\n", err);
        return vmaf_sycl_import_va_surface_readback(
            state, va_display_handle, va_surface_id, is_ref, w, h, bpc);
    }

    /* Get target shared frame buffer */
    void *target_buf = is_ref
        ? vmaf_sycl_get_shared_ref(state)
        : vmaf_sycl_get_shared_dis(state);

    if (!target_buf) {
        vmaf_sycl_dmabuf_free(state, imported_ptr);
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "Shared frame buffer not initialised\n");
        return -EINVAL;
    }

    sycl::queue *q = (sycl::queue *)vmaf_sycl_get_queue_ptr(state);
    size_t row_bytes = (size_t)w * bpp;

    /* Log the zero-copy path once (first successful import) */
    static bool logged_zero_copy = false;
    if (!logged_zero_copy) {
        const char *tiling_name =
            modifier == DRM_FORMAT_MOD_LINEAR ? "LINEAR" :
            modifier == I915_FORMAT_MOD_4_TILED ? "Tile4" :
            modifier == I915_FORMAT_MOD_Y_TILED ? "Y-tiled" : "unknown";
        vmaf_log(VMAF_LOG_LEVEL_INFO,
                 "VA surface zero-copy: DMA-BUF → Level Zero → %s de-tile "
                 "(%ux%u @ %u bpp, pitch=%u)\n",
                 tiling_name, w, h, bpp, y_pitch);
        logged_zero_copy = true;
    }

    if (modifier == DRM_FORMAT_MOD_LINEAR || modifier == 0) {
        /*
         * LINEAR surface → device-to-device memcpy (GPU blitter).
         * No CPU involvement at all.
         */
        vmaf_log(VMAF_LOG_LEVEL_DEBUG, "[%s] zero-copy: LINEAR D2D\n",
                 is_ref ? "ref" : "dis");

        sycl::event ev;
        if (y_pitch == row_bytes && y_offset == 0) {
            ev = q->memcpy(target_buf, imported_ptr, row_bytes * h);
        } else {
            for (unsigned row = 0; row < h; row++) {
                ev = q->memcpy((uint8_t *)target_buf + row * row_bytes,
                          (uint8_t *)imported_ptr + y_offset + row * y_pitch,
                          row_bytes);
            }
        }
        vmaf_sycl_set_detile_event(state, &ev);

    } else if (modifier == I915_FORMAT_MOD_4_TILED) {
        /*
         * Tile4 (Gen12.5+ / DG2): 128 bytes × 32 rows = 4 KB per tile.
         *
         * The intra-tile byte address is a bit-interleave of the byte
         * column x (0-127, 7 bits) and row y (0-31, 5 bits):
         *
         *   offset[3:0]  = x[3:0]   — byte within 16-byte sub-block
         *   offset[5:4]  = y[1:0]   — row within 4-row group
         *   offset[7:6]  = x[5:4]   — 16-byte column (0-3)
         *   offset[8]    = y[2]     — 4-row group
         *   offset[9]    = x[6]     — 64-byte half
         *   offset[10]   = y[3]     — 8-row super-group
         *   offset[11]   = y[4]     — 16-row half
         *
         * Bits 0-3 are purely x, so 4-byte aligned reads are contiguous.
         */
        vmaf_log(VMAF_LOG_LEVEL_DEBUG, "[%s] zero-copy: Tile4 de-tile kernel\n",
                 is_ref ? "ref" : "dis");

        const uint8_t *src = (const uint8_t *)imported_ptr + y_offset;
        uint8_t *dst = (uint8_t *)target_buf;
        unsigned tiles_per_row = y_pitch / 128;

        unsigned words_per_tile_row = 128 / 4; /* = 32 */
        unsigned words_per_row = tiles_per_row * words_per_tile_row;

        sycl::event ev = q->parallel_for(
            sycl::range<2>(h, words_per_row),
            [=](sycl::id<2> id) {
                unsigned py = id[0];
                unsigned word_x = id[1];

                /* Tile address */
                unsigned tc = word_x / words_per_tile_row;
                unsigned wt = word_x % words_per_tile_row;
                unsigned tr = py / 32;
                unsigned ity = py % 32;

                /* Tile4 intra-tile swizzle */
                unsigned x_byte = wt * 4;
                unsigned swizzled =
                    (x_byte & 0x0F)                /* [3:0]  = x[3:0] */
                    | ((ity & 3) << 4)             /* [5:4]  = y[1:0] */
                    | (((x_byte >> 4) & 3) << 6)   /* [7:6]  = x[5:4] */
                    | (((ity >> 2) & 1) << 8)      /* [8]    = y[2]   */
                    | (((x_byte >> 6) & 1) << 9)   /* [9]    = x[6]   */
                    | (((ity >> 3) & 1) << 10)     /* [10]   = y[3]   */
                    | (((ity >> 4) & 1) << 11);    /* [11]   = y[4]   */

                size_t src_off = (size_t)(tr * tiles_per_row + tc) * 4096
                               + swizzled;

                /* Linear destination */
                size_t dst_off = (size_t)py * row_bytes + tc * 128 + wt * 4;

                /* Bounds check — last tile column may exceed frame width */
                if (dst_off + 4 <= (size_t)(py + 1) * row_bytes) {
                    *(uint32_t *)(dst + dst_off) =
                        *(const uint32_t *)(src + src_off);
                } else if (dst_off < (size_t)(py + 1) * row_bytes) {
                    size_t remain = (size_t)(py + 1) * row_bytes - dst_off;
                    for (size_t b = 0; b < remain; b++)
                        dst[dst_off + b] = src[src_off + b];
                }
            });
        vmaf_sycl_set_detile_event(state, &ev);

    } else if (modifier == I915_FORMAT_MOD_Y_TILED) {
        /*
         * Y-tiled: 128 bytes × 32 rows per tile (4 KB).
         * Within each tile, data is in column-major 16-byte OWords:
         *   OWord column 0 (bytes 0-15): rows 0-31 (512 B),
         *   OWord column 1 (bytes 16-31): rows 0-31 (512 B),
         *   ...
         *   OWord column 7 (bytes 112-127): rows 0-31 (512 B).
         */
        vmaf_log(VMAF_LOG_LEVEL_DEBUG, "[%s] zero-copy: Y-tiled de-tile kernel\n",
                 is_ref ? "ref" : "dis");

        const uint8_t *src = (const uint8_t *)imported_ptr + y_offset;
        uint8_t *dst = (uint8_t *)target_buf;
        unsigned tiles_per_row = y_pitch / 128;

        unsigned words_per_tile_row = 128 / 4;
        unsigned words_per_row = tiles_per_row * words_per_tile_row;

        sycl::event ev = q->parallel_for(
            sycl::range<2>(h, words_per_row),
            [=](sycl::id<2> id) {
                unsigned py = id[0];
                unsigned word_x = id[1];

                unsigned tc = word_x / words_per_tile_row;
                unsigned wt = word_x % words_per_tile_row;
                unsigned tr = py / 32;
                unsigned ity = py % 32;

                /* Y-tiled address: OWord column-major */
                unsigned in_tile_byte_x = wt * 4;
                unsigned oword_col = in_tile_byte_x / 16;
                unsigned oword_byte = in_tile_byte_x % 16;

                size_t src_off = (size_t)(tr * tiles_per_row + tc) * 4096
                               + (size_t)oword_col * 512
                               + (size_t)ity * 16 + oword_byte;

                size_t dst_off = (size_t)py * row_bytes + tc * 128 + wt * 4;

                if (dst_off + 4 <= (size_t)(py + 1) * row_bytes) {
                    *(uint32_t *)(dst + dst_off) =
                        *(const uint32_t *)(src + src_off);
                } else if (dst_off < (size_t)(py + 1) * row_bytes) {
                    size_t remain = (size_t)(py + 1) * row_bytes - dst_off;
                    for (size_t b = 0; b < remain; b++)
                        dst[dst_off + b] = src[src_off + b];
                }
            });
        vmaf_sycl_set_detile_event(state, &ev);

    } else {
        /* Unknown tiling — fall back to VA readback path */
        vmaf_sycl_dmabuf_free(state, imported_ptr);
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "Unknown DRM modifier 0x%llx — using readback path\n",
                 (unsigned long long)modifier);
        return vmaf_sycl_import_va_surface_readback(
            state, va_display_handle, va_surface_id, is_ref, w, h, bpc);
    }

    /* Defer the free: the de-tile kernel still needs to read from
     * imported_ptr.  The pointer will be freed by
     * vmaf_sycl_flush_pending_imports() after the next queue wait. */
    vmaf_sycl_defer_import_free(state, imported_ptr);
    return 0;
}

#else /* !HAVE_SYCL_DMABUF */

extern "C"
int vmaf_sycl_import_va_surface(VmafSyclState *state,
                                 void *va_display_handle,
                                 unsigned int va_surface_id,
                                 int is_ref,
                                 unsigned w, unsigned h, unsigned bpc)
{
    (void)state; (void)va_display_handle; (void)va_surface_id;
    (void)is_ref; (void)w; (void)h; (void)bpc;
    return -ENOTSUP;
}

#endif /* HAVE_SYCL_DMABUF */

#endif /* HAVE_SYCL */
