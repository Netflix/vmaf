**fix(test):** Restore three meson test registrations dropped by PR #1067.
`test_vmaf_use_tiny_model` (suite `dnn`+`fast`), `test_vulkan_pipeline_cache`,
and `test_psnr_vulkan_chroma_geom` (suite `fast`) were silently unregistered
when PR #1067 merged a stale meson.build base; the source `.c` files were
never removed. Restores full coverage of the vmaf_use_tiny_model public API,
VkPipelineCache persistence contract, and Vulkan PSNR chroma geometry guard.
