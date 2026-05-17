- Fixed `integer_vif_metal`: register `vif_skip_scale0` option and enforce
  scale-0 suppression in `collect_fex_metal`, matching `integer_vif.c` and
  the CUDA/SYCL/Vulkan twins. Previously the option was absent; callers
  setting `vif_skip_scale0=true` had no effect on the Metal backend.
