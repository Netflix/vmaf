- **`integer_vif_metal` `vif_skip_scale0` option gap** (`integer_vif_metal.mm`):
  `integer_vif_metal` had no `vif_skip_scale0` option registration and no
  suppression logic in `collect_fex_metal`, so scale-0 was always included in
  the Metal VIF score regardless of the caller setting. Added the option (with
  `ssclz` alias, matching `integer_vif.c`) and the identical suppression logic
  as the CUDA/SYCL/Vulkan backends: scale-0 emits `0.0`, is excluded from the
  combined `score_num`/`score_den` accumulation, and debug fields
  `integer_vif_num_scale0`/`integer_vif_den_scale0` emit `0.0` when set.
