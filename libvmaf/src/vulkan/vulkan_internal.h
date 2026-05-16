/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Internal-only Vulkan context layout. Kernel TUs in
 *  libvmaf/src/feature/vulkan/ include this header (NOT
 *  libvmaf/src/vulkan/vulkan_common.h alone) so they can read the
 *  device / queue / allocator handles. The public surface
 *  (vulkan_common.h) stays opaque so callers can't accidentally bind to
 *  Vulkan-specific types.
 */

#ifndef LIBVMAF_VULKAN_INTERNAL_H_
#define LIBVMAF_VULKAN_INTERNAL_H_

/* volk loads every Vulkan entry point at runtime; defining
 * VK_NO_PROTOTYPES is required so the system Vulkan headers do not
 * declare the static-link prototypes that would conflict with volk's
 * function pointers. The wrap's compile_args already injects
 * `-DVK_NO_PROTOTYPES` for users of `volk_dep`, so this `#define` is
 * defensive in case someone forgets. */
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vk_mem_alloc.h>

/* picture_vulkan.h declares the opaque VmafVulkanBuffer used by the
 * import slots below. Pulled in from the directory's local includes via
 * the meson include_directories list. */
#include "picture_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum frames in flight allowed by the v2 async pending-fence
 * ring (T7-29 part 4, ADR-0251). The default is 4 — the canonical
 * Vulkan game-engine "frames in flight" depth. The hard cap is 8
 * to bound the staging-buffer footprint; callers asking for more
 * via VmafVulkanConfiguration get clamped silently with an
 * internal warning. The ring depth is fixed at state init and
 * never reallocates afterwards (Power of 10 §3 / §6). */
#define VMAF_VULKAN_RING_DEFAULT 4u
#define VMAF_VULKAN_RING_MAX 8u

/* Map a caller-supplied max_outstanding_frames request to a final
 * ring depth: 0 -> DEFAULT, >MAX -> MAX, else passthrough. Callers
 * (state init + lazy_alloc_ring) share this so the value stored in
 * `requested_ring_size` is identical to the one the ring is built
 * with. ADR-0251 follow-up #3. */
static inline unsigned vmaf_vulkan_clamp_ring_size(unsigned requested)
{
    if (requested == 0u)
        return VMAF_VULKAN_RING_DEFAULT;
    if (requested > VMAF_VULKAN_RING_MAX)
        return VMAF_VULKAN_RING_MAX;
    return requested;
}

/* One entry of the v2 async pending-fence ring. Each slot owns a
 * dedicated ref + dis staging buffer pair, a transfer command
 * buffer, and a fence; they are pre-allocated by lazy_alloc_ring
 * on the first import_image call and never reallocated. */
struct VmafVulkanImportSlot {
    VmafVulkanBuffer *ref_buf;
    VmafVulkanBuffer *dis_buf;
    VkCommandBuffer cmd;
    VkFence fence;

    /* Per-direction "data staged for `*_index`" flags. Cleared by
     * vmaf_vulkan_state_build_pictures() once the host has consumed
     * the slot. */
    int ref_pending;
    int dis_pending;
    unsigned ref_index;
    unsigned dis_index;

    /* Non-zero between vkQueueSubmit and the matching
     * vkWaitForFences. submit_to_slot waits + resets if a prior
     * submission is still in flight when the ring wraps; the same
     * field tells vmaf_vulkan_state_free which fences need a final
     * drain. */
    int fence_in_flight;
};

/* Per-state import-ring for the VkImage zero-copy path. Originally
 * one fence + one staging-pair (ADR-0186 v1, synchronous); the v2
 * async pending-fence design (ADR-0251) promotes the single slot
 * to a ring of N slots keyed by `frame_index % ring_size`. */
struct VmafVulkanImportSlots {
    /* Frame geometry pinned by the first import_image call. Subsequent
     * calls must match (or return -EINVAL) — same contract as the
     * SYCL `init_frame_buffers` model. Zero == not yet allocated. */
    unsigned w;
    unsigned h;
    unsigned bpc;
    /* Aligned row stride (bytes) of the staging buffers, matching the
     * DATA_ALIGN-rounded stride that vmaf_picture_alloc would produce.
     * Stored so vmaf_vulkan_read_imported_pictures can hand the buffers
     * to vmaf_read_pictures with correct geometry. */
    size_t stride_bytes;

    /* Ring depth. Set once at state init, never changes. */
    unsigned ring_size;

    /* `ring_size` slots; the unused tail (when ring_size <
     * VMAF_VULKAN_RING_MAX) keeps zero-initialised handles so
     * the teardown path is uniform. */
    struct VmafVulkanImportSlot ring[VMAF_VULKAN_RING_MAX];
};

struct VmafVulkanState {
    VmafVulkanContext *ctx;
    struct VmafVulkanImportSlots import;

    /* Requested ring depth (max_outstanding_frames). Captured from
     * VmafVulkanConfiguration at init time, clamped to
     * [1, VMAF_VULKAN_RING_MAX]. Ring is materialised lazily on the
     * first import_image call (geometry isn't known until then). */
    unsigned requested_ring_size;
};

/* import.c — release any lazily-allocated import slot resources.
 * Safe to call on a state that never imported anything; the slot
 * fields are zero-initialised by calloc in state_init. */
void vmaf_vulkan_import_slots_free(struct VmafVulkanState *state);

/* common.c — fork-internal accessor used by libvmaf.c's
 * vmaf_vulkan_preallocate_pictures (ADR-0238) so the picture pool
 * can borrow the same VkInstance/VkDevice as the imported state.
 * Returns NULL on a NULL state. NOT part of the public ABI. */
struct VmafVulkanContext *vmaf_vulkan_state_context(struct VmafVulkanState *state);

struct VmafVulkanContext {
    /* Set to true after `volkInitialize()` succeeds the first time
     * any context is created. Subsequent contexts skip the global
     * init. */
    int volk_loaded;
    /* Non-zero when libvmaf created the VkInstance + VkDevice
     * itself (vmaf_vulkan_context_new). Zero when the caller
     * supplied them via vmaf_vulkan_state_init_external — in that
     * case context_destroy must NOT call vkDestroyDevice /
     * vkDestroyInstance. The VMA allocator and command pool are
     * always libvmaf-owned regardless. ADR-0186. */
    int owns_handles;

    int device_index; /* Resolved >=0 device ordinal (post auto-pick). */
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family_index;
    VkQueue queue;

    VmaAllocator allocator;
    VkCommandPool command_pool;

    /* Properties of the selected physical device — feature kernels
     * read these to pick group size, sub-group ops, etc. */
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
};

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_VULKAN_INTERNAL_H_ */
