- Added `vif_skip_scale0` option to `integer_vif_hip` extractor, closing the
  feature-option parity gap with the CPU `integer_vif` and the CUDA/SYCL/Vulkan
  GPU twins (PR #966). When set, scale-0 primary score is forced to 0.0, scale-0
  is excluded from the aggregate num/den in debug mode, and debug paths emit
  `num_scale0=0.0` / `den_scale0=-1.0` — matching the CPU design exactly.
