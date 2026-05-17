/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  T5-1b runtime — replaces the T5-1 scaffold's `-ENOSYS` stubs with
 *  a real volk + Vulkan 1.3 + VMA bring-up. Picks a compute-capable
 *  physical device (auto: discrete > integrated > virtual > cpu;
 *  override via the `device_index` argument), creates a dedicated
 *  compute queue family, attaches a VMA allocator, and exposes a
 *  command pool that per-feature dispatch wrappers under
 *  libvmaf/src/feature/vulkan/ reuse.
 */

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf_vulkan.h"
#include "vulkan_common.h"
#include "vulkan_internal.h"

// VK_API_VERSION_1_4 was added in Vulkan Headers 1.3.280 (Jan 2024).
// Ubuntu 22.04 ships libvulkan-dev 1.3.204, which predates this constant.
// Fall back to VK_API_VERSION_1_3 on older SDK headers; the runtime device
// selection path queries the actual device capabilities separately.
// Per ADR-0264 the 1.4 bump is gated on NVIDIA driver regression resolution;
// 1.3 is the active fallback path anyway.
#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_API_VERSION_1_3
#endif

#define VK_OR_FAIL(call_, errno_)                                                                  \
    do {                                                                                           \
        VkResult _vkr = (call_);                                                                   \
        if (_vkr != VK_SUCCESS) {                                                                  \
            err = (errno_);                                                                        \
            goto fail;                                                                             \
        }                                                                                          \
    } while (0)

static int g_volk_loaded = 0;

static int load_volk_once(void)
{
    if (g_volk_loaded)
        return 0;
    VkResult vkr = volkInitialize();
    if (vkr != VK_SUCCESS)
        return -ENOSYS;
    g_volk_loaded = 1;
    return 0;
}

static int create_instance(VkInstance *out_instance)
{
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "libvmaf",
        .applicationVersion = VK_MAKE_API_VERSION(0, 3, 0, 0),
        .pEngineName = "libvmaf-vulkan",
        .engineVersion = VK_MAKE_API_VERSION(0, 3, 0, 0),
        .apiVersion = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VkResult vkr = vkCreateInstance(&create_info, NULL, out_instance);
    if (vkr != VK_SUCCESS)
        return -ENODEV;
    volkLoadInstanceOnly(*out_instance);
    return 0;
}

static int devtype_score(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return 1;
    default:
        return 0;
    }
}

static uint32_t pick_compute_queue_family(VkPhysicalDevice phys)
{
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, NULL);
    if (family_count == 0)
        return UINT32_MAX;

    VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
    if (!families)
        return UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families);

    uint32_t async_compute = UINT32_MAX;
    uint32_t any_compute = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; i++) {
        VkQueueFlags flags = families[i].queueFlags;
        if (!(flags & VK_QUEUE_COMPUTE_BIT))
            continue;
        if (any_compute == UINT32_MAX)
            any_compute = i;
        if (!(flags & VK_QUEUE_GRAPHICS_BIT) && async_compute == UINT32_MAX)
            async_compute = i;
    }
    free(families);

    return (async_compute != UINT32_MAX) ? async_compute : any_compute;
}

static int enumerate_compute_devices(VkInstance instance, VkPhysicalDevice *out, uint32_t *count_io)
{
    uint32_t total = 0;
    VkResult vkr = vkEnumeratePhysicalDevices(instance, &total, NULL);
    if (vkr != VK_SUCCESS)
        return -EIO;
    if (total == 0) {
        *count_io = 0;
        return 0;
    }

    VkPhysicalDevice *all = calloc(total, sizeof(*all));
    if (!all)
        return -ENOMEM;
    vkr = vkEnumeratePhysicalDevices(instance, &total, all);
    if (vkr != VK_SUCCESS) {
        free(all);
        return -EIO;
    }

    int *score = calloc(total, sizeof(*score));
    if (!score) {
        free(all);
        return -ENOMEM;
    }
    uint32_t kept = 0;
    for (uint32_t i = 0; i < total; i++) {
        if (pick_compute_queue_family(all[i]) == UINT32_MAX)
            continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(all[i], &props);
        all[kept] = all[i];
        score[kept] = devtype_score(props.deviceType);
        kept++;
    }

    /* Insertion sort (kept is small in practice). */
    for (uint32_t i = 1; i < kept; i++) {
        VkPhysicalDevice ph = all[i];
        int sc = score[i];
        uint32_t j = i;
        while (j > 0 && score[j - 1] < sc) {
            all[j] = all[j - 1];
            score[j] = score[j - 1];
            j--;
        }
        all[j] = ph;
        score[j] = sc;
    }

    uint32_t to_copy = (kept < *count_io) ? kept : *count_io;
    for (uint32_t i = 0; i < to_copy; i++)
        out[i] = all[i];
    *count_io = kept;
    free(score);
    free(all);
    return 0;
}

int vmaf_vulkan_device_count(void)
{
    int err = load_volk_once();
    if (err)
        return 0;

    VkInstance instance = VK_NULL_HANDLE;
    err = create_instance(&instance);
    if (err)
        return 0;

    VkPhysicalDevice tmp[8];
    uint32_t count = 8;
    err = enumerate_compute_devices(instance, tmp, &count);
    vkDestroyInstance(instance, NULL);
    if (err)
        return err;
    return (int)count;
}

int vmaf_vulkan_context_new(VmafVulkanContext **out, int device_index)
{
    if (!out)
        return -EINVAL;

    int err = load_volk_once();
    if (err)
        return err;
    assert(g_volk_loaded == 1);

    VmafVulkanContext *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return -ENOMEM;
    ctx->volk_loaded = 1;
    ctx->owns_handles = 1;
    ctx->device_index = device_index;
    assert(ctx->instance == VK_NULL_HANDLE);
    assert(ctx->device == VK_NULL_HANDLE);

    err = create_instance(&ctx->instance);
    if (err)
        goto fail_alloc;

    VkPhysicalDevice phys[16];
    uint32_t total = 16;
    err = enumerate_compute_devices(ctx->instance, phys, &total);
    if (err)
        goto fail;
    if (total == 0) {
        err = -ENODEV;
        goto fail;
    }
    if (device_index < 0) {
        ctx->device_index = 0;
    } else if ((uint32_t)device_index >= total) {
        err = -EINVAL;
        goto fail;
    } else {
        ctx->device_index = device_index;
    }
    ctx->physical_device = phys[ctx->device_index];

    vkGetPhysicalDeviceProperties(ctx->physical_device, &ctx->props);
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &ctx->mem_props);

    ctx->queue_family_index = pick_compute_queue_family(ctx->physical_device);
    if (ctx->queue_family_index == UINT32_MAX) {
        err = -ENODEV;
        goto fail;
    }

    /* ADR-0350 follow-up (2026-05-09 review-fix): the two-level GPU
     * reduction shaders (vif_reduce.comp / adm_reduce.comp /
     * motion_reduce.comp) compile to SPIR-V that uses
     * `OpAtomicIAdd` on `int64_t` SSBO members.  Vulkan 1.2 hoisted
     * this capability into core via
     * `VkPhysicalDeviceShaderAtomicInt64Features::shaderBufferInt64Atomics`
     * (rolled up in `VkPhysicalDeviceVulkan12Features`).  If the
     * device does not advertise the capability, `vkCreateShaderModule`
     * /  `vkCreateComputePipelines` will reject our SPIR-V at init
     * time — typical on MoltenVK 1.2.x atop Apple Silicon, where
     * Metal does not expose 64-bit buffer atomics.
     *
     * We probe the feature here, enable it on the
     * `vkCreateDevice` `pNext` chain when present, and reject the
     * Vulkan backend cleanly with `-ENOTSUP` and a stderr log when
     * absent so callers fall back to CPU instead of seeing an
     * opaque pipeline-creation failure later.  See PR #561 review
     * comment 3 / ADR-0350 §Consequences. */
    VkPhysicalDeviceShaderAtomicInt64Features atomic64_feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    };
    VkPhysicalDeviceFeatures2 query_feat2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &atomic64_feat,
    };
    vkGetPhysicalDeviceFeatures2(ctx->physical_device, &query_feat2);
    if (!atomic64_feat.shaderBufferInt64Atomics) {
        fprintf(stderr,
                "libvmaf: Vulkan backend disabled on this device "
                "(\"%s\") — no shaderBufferInt64Atomics support, "
                "required for the two-level reduction shaders "
                "(ADR-0350). Falling back to CPU.\n",
                ctx->props.deviceName);
        err = -ENOTSUP;
        goto fail;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    /* Enable shaderBufferInt64Atomics on the device so the SPIR-V
     * loader accepts our reduction shaders.  We re-use a fresh
     * VkPhysicalDeviceShaderAtomicInt64Features struct (the one
     * populated above is a query-only result; spec disallows
     * passing it back through pNext on `vkCreateDevice`). */
    VkPhysicalDeviceShaderAtomicInt64Features atomic64_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
        .shaderBufferInt64Atomics = VK_TRUE,
    };
    VkPhysicalDeviceFeatures features = {0};
    VkDeviceCreateInfo device_create = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &atomic64_enable,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create,
        .pEnabledFeatures = &features,
    };
    VK_OR_FAIL(vkCreateDevice(ctx->physical_device, &device_create, NULL, &ctx->device), -ENODEV);
    volkLoadDevice(ctx->device);
    vkGetDeviceQueue(ctx->device, ctx->queue_family_index, 0, &ctx->queue);

    VmaVulkanFunctions vma_fns = {
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
    };
    VmaAllocatorCreateInfo alloc_info = {
        .vulkanApiVersion = VK_API_VERSION_1_4,
        .physicalDevice = ctx->physical_device,
        .device = ctx->device,
        .instance = ctx->instance,
        .pVulkanFunctions = &vma_fns,
    };
    VK_OR_FAIL(vmaCreateAllocator(&alloc_info, &ctx->allocator), -ENOMEM);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family_index,
    };
    VK_OR_FAIL(vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool), -ENOMEM);

    /* Power-of-10 §5: pin the post-condition that every handle the
     * caller may dereference is now non-null. Catches a future
     * mistake where one of the create calls is moved out of order. */
    assert(ctx->instance != VK_NULL_HANDLE);
    assert(ctx->physical_device != VK_NULL_HANDLE);
    assert(ctx->device != VK_NULL_HANDLE);
    assert(ctx->queue != VK_NULL_HANDLE);
    assert(ctx->allocator != VK_NULL_HANDLE);
    assert(ctx->command_pool != VK_NULL_HANDLE);

    *out = ctx;
    return 0;

fail:
    if (ctx->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    if (ctx->allocator != VK_NULL_HANDLE)
        vmaDestroyAllocator(ctx->allocator);
    if (ctx->device != VK_NULL_HANDLE)
        vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance != VK_NULL_HANDLE)
        vkDestroyInstance(ctx->instance, NULL);
fail_alloc:
    free(ctx);
    *out = NULL;
    return err;
}

void vmaf_vulkan_context_destroy(VmafVulkanContext *ctx)
{
    if (!ctx)
        return;
    /* command_pool + allocator are always libvmaf-owned, even
     * for externally-supplied instance/device handles. */
    if (ctx->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    if (ctx->allocator != VK_NULL_HANDLE)
        vmaDestroyAllocator(ctx->allocator);
    if (ctx->owns_handles) {
        if (ctx->device != VK_NULL_HANDLE)
            vkDestroyDevice(ctx->device, NULL);
        if (ctx->instance != VK_NULL_HANDLE)
            vkDestroyInstance(ctx->instance, NULL);
    }
    free(ctx);
}

/* External-handle adoption path (T7-29 part 2 / ADR-0186). The
 * caller — typically FFmpeg's AVVulkanDeviceContext — supplies a
 * pre-created VkInstance + VkPhysicalDevice + VkDevice + queue
 * so the source VkImage handles passed via
 * vmaf_vulkan_import_image are valid on libvmaf's compute device.
 *
 * We still create our own VMA allocator + command pool on the
 * supplied device. The volk function-pointer table is global,
 * so we (re-)bind it to the supplied instance/device — see the
 * "mutually exclusive with vmaf_vulkan_state_init" caveat in
 * libvmaf_vulkan.h. */
static int vmaf_vulkan_context_new_external(VmafVulkanContext **out,
                                            const VmafVulkanExternalHandles *h)
{
    if (!out || !h)
        return -EINVAL;
    if (h->instance == 0u || h->physical_device == 0u || h->device == 0u || h->queue == 0u)
        return -EINVAL;

    int err = load_volk_once();
    if (err)
        return err;

    VmafVulkanContext *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return -ENOMEM;
    ctx->volk_loaded = 1;
    ctx->owns_handles = 0;
    ctx->instance = (VkInstance)h->instance;
    ctx->physical_device = (VkPhysicalDevice)h->physical_device;
    ctx->device = (VkDevice)h->device;
    ctx->queue = (VkQueue)h->queue;
    ctx->queue_family_index = h->queue_family_index;

    /* Re-bind volk to the supplied instance/device so all later
     * vk* calls hit the caller's handles, not whatever was loaded
     * by a prior vmaf_vulkan_context_new(). */
    volkLoadInstance(ctx->instance);
    volkLoadDevice(ctx->device);

    vkGetPhysicalDeviceProperties(ctx->physical_device, &ctx->props);
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &ctx->mem_props);

    /* ADR-0350 follow-up: also probe shaderBufferInt64Atomics on the
     * external-handle path.  We can't change the caller's
     * pre-created VkDevice, so if the capability isn't there we
     * just refuse to attach (the caller falls back to CPU). */
    {
        VkPhysicalDeviceShaderAtomicInt64Features ext_atomic64 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
        };
        VkPhysicalDeviceFeatures2 ext_feat2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &ext_atomic64,
        };
        vkGetPhysicalDeviceFeatures2(ctx->physical_device, &ext_feat2);
        if (!ext_atomic64.shaderBufferInt64Atomics) {
            fprintf(stderr,
                    "libvmaf: Vulkan backend disabled on imported "
                    "device (\"%s\") — no shaderBufferInt64Atomics "
                    "support, required for the two-level reduction "
                    "shaders (ADR-0350). Falling back to CPU.\n",
                    ctx->props.deviceName);
            err = -ENOTSUP;
            goto fail;
        }
    }

    VmaVulkanFunctions vma_fns = {
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
    };
    VmaAllocatorCreateInfo alloc_info = {
        .vulkanApiVersion = h->api_version ? h->api_version : VK_API_VERSION_1_4,
        .physicalDevice = ctx->physical_device,
        .device = ctx->device,
        .instance = ctx->instance,
        .pVulkanFunctions = &vma_fns,
    };
    if (vmaCreateAllocator(&alloc_info, &ctx->allocator) != VK_SUCCESS) {
        err = -ENOMEM;
        goto fail;
    }

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family_index,
    };
    if (vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool) != VK_SUCCESS) {
        err = -ENOMEM;
        goto fail;
    }

    *out = ctx;
    return 0;

fail:
    if (ctx->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    if (ctx->allocator != VK_NULL_HANDLE)
        vmaDestroyAllocator(ctx->allocator);
    /* Do NOT destroy the externally-supplied device/instance. */
    free(ctx);
    *out = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  Public state-level API (libvmaf/include/libvmaf/libvmaf_vulkan.h) */
/*                                                                    */
/*  VmafVulkanState is the public opaque handle; it wraps the         */
/*  internal VmafVulkanContext so the public ABI doesn't expose       */
/*  volk/VMA types.                                                   */
/* ------------------------------------------------------------------ */

/* `struct VmafVulkanState` was promoted to vulkan_internal.h in T7-29
 * part 2 (ADR-0186) so the import slots are visible to import.c without
 * a public ABI change. */

int vmaf_vulkan_available(void)
{
    return 1;
}

int vmaf_vulkan_state_init(VmafVulkanState **out, VmafVulkanConfiguration cfg)
{
    if (!out)
        return -EINVAL;
    (void)cfg.enable_validation; /* validation layer toggle: T5-1c follow-up */

    VmafVulkanState *s = calloc(1, sizeof(*s));
    if (!s)
        return -ENOMEM;

    int err = vmaf_vulkan_context_new(&s->ctx, cfg.device_index);
    if (err) {
        free(s);
        return err;
    }

    /* T7-29 part 4 (ADR-0251) + follow-up #3: pick the async
     * pending-fence ring depth from the public config. A
     * default-initialised C struct (cfg.max_outstanding_frames == 0)
     * maps to VMAF_VULKAN_RING_DEFAULT; out-of-range values clamp to
     * [1, VMAF_VULKAN_RING_MAX]. The clamp happens here so
     * `requested_ring_size` always reflects the real depth the ring
     * will be built with — easier debugging than a pre-clamp value
     * that mismatches the runtime. */
    s->requested_ring_size = vmaf_vulkan_clamp_ring_size(cfg.max_outstanding_frames);
    assert(s->requested_ring_size > 0);
    assert(s->requested_ring_size <= VMAF_VULKAN_RING_MAX);
    assert(s->ctx != NULL);

    *out = s;
    return 0;
}

int vmaf_vulkan_state_init_external(VmafVulkanState **out, VmafVulkanExternalHandles handles)
{
    if (!out)
        return -EINVAL;

    VmafVulkanState *s = calloc(1, sizeof(*s));
    if (!s)
        return -ENOMEM;

    int err = vmaf_vulkan_context_new_external(&s->ctx, &handles);
    if (err) {
        free(s);
        return err;
    }

    /* External callers (FFmpeg's vf_libvmaf_vulkan) currently get the
     * canonical default. ADR-0251 follow-up #3 only plumbs the tunable
     * through VmafVulkanConfiguration; extending VmafVulkanExternalHandles
     * is deferred to a separate ABI bump. */
    s->requested_ring_size = VMAF_VULKAN_RING_DEFAULT;
    assert(s->requested_ring_size > 0);
    assert(s->ctx != NULL);

    *out = s;
    return 0;
}

unsigned vmaf_vulkan_state_max_outstanding_frames(const VmafVulkanState *state)
{
    if (!state)
        return 0u;
    /* requested_ring_size was clamped at state init (or set to
     * VMAF_VULKAN_RING_DEFAULT for the external-handles path), so it is
     * already a usable [1, VMAF_VULKAN_RING_MAX] value here. */
    return state->requested_ring_size;
}

VmafVulkanContext *vmaf_vulkan_state_context(VmafVulkanState *state)
{
    /* Fork-internal accessor (declared in vulkan_internal.h, not the
     * public libvmaf_vulkan.h). ADR-0238: the picture pool borrows the
     * VkInstance/VkDevice from the already-imported state so the
     * preallocated buffers live on the same device the kernels run on. */
    if (!state)
        return NULL;
    return state->ctx;
}

void vmaf_vulkan_state_free(VmafVulkanState **state)
{
    if (!state || !*state)
        return;
    /* Release any import-slot resources that survived past the last
     * read_imported_pictures call (T7-29 part 2 / ADR-0186). */
    vmaf_vulkan_import_slots_free(*state);
    vmaf_vulkan_context_destroy((*state)->ctx);
    free(*state);
    *state = NULL;
}

int vmaf_vulkan_list_devices(void)
{
    return vmaf_vulkan_device_count();
}

VmafVulkanContext *vmaf_vulkan_state_get_context(VmafVulkanState *state)
{
    return state ? state->ctx : NULL;
}

/* T7-29 part 2 (ADR-0186): the real implementations for
 * vmaf_vulkan_import_image / vmaf_vulkan_wait_compute live in
 * libvmaf/src/vulkan/import.c. vmaf_vulkan_read_imported_pictures
 * lives in libvmaf/src/libvmaf.c so it can reach into VmafContext. */
