- **CUDA/SYCL/Vulkan `integer_adm` `adm_min_val` option** (`integer_adm_cuda.c`,
  `integer_adm_sycl.cpp`, `adm_vulkan.c`): the three GPU backends silently
  dropped the `adm_min_val` option, never applying the minimum score floor
  that the CPU reference enforces. Added the option (alias `min`, range
  0.0–1.0, default 0.0) to each backend's `options[]` table and applied a
  host-side clamp `score = MAX(score, adm_min_val)` after the `num / den`
  reduction. Default is 0.0 (no-op), so existing runs are bit-for-bit
  unchanged (ADR-0487).
