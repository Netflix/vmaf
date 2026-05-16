**vulkan**: Remove ghost duplicate `moment_vulkan.c` left by PR #1067 clobbering PR #1046's
rename. `libvmaf/src/vulkan/meson.build` now references only `float_moment_vulkan.c`.
