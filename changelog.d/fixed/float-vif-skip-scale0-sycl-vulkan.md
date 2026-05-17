Wire `vif_skip_scale0` option to `float_vif_sycl` and `float_vif_vulkan` backends.
Previously the option was accepted but silently dropped; scale 0 was always included
in the score aggregation. The suppression logic now matches `float_vif_cuda`.
