vulkan: enable `VK_KHR_portability_enumeration` on macOS so MoltenVK
devices are visible to `vkEnumeratePhysicalDevices` instead of silently
returning zero devices and falling back to CPU (VK-8,
perf-audit-vulkan-sycl-2026-05-16).
