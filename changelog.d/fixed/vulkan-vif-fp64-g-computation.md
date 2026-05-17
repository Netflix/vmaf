**Vulkan VIF shader: promote g/sv_sq to double precision (ADR-0492)**

The Vulkan VIF compute shader (`vif.comp`) now computes the gain factor `g`
and residual variance `sv_sq` in double precision
(`GL_EXT_shader_explicit_arithmetic_types_float64`) to match the CPU
reference path in `integer_vif.c`. The previous `precise float` path
(introduced in PR #1201 to block NVIDIA FMA contraction) eliminated the
FMA-fold bias but left a residual fp32-vs-double divergence of ~7 ULP/pixel,
accumulating to ~2×10⁻⁴ per frame at 576×324 and failing the ADR-0214
places=4 CPU-parity gate.

Device capability is probed at `vmaf_vulkan_context_new` time;
devices without `VkPhysicalDeviceFeatures::shaderFloat64`
(e.g. Apple Silicon via MoltenVK) fall back to CPU with a clear
diagnostic message.
