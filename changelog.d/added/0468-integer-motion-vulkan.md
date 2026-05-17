Add `integer_motion_vulkan` Vulkan feature extractor — a direct port of
`integer_motion_cuda.c` using a new GLSL compute shader
(`shaders/integer_motion.comp`). Provides the explicitly-named
`"integer_motion_vulkan"` extractor alongside the legacy `"motion_vulkan"`
alias, matching the CUDA/SYCL naming convention. Emits
`VMAF_integer_feature_motion_score`, `motion2_score`, and `motion3_score`
with the same host-side blend/moving-average post-processing as the CUDA
twin.
