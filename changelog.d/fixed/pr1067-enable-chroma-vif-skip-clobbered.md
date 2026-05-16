PR #1067 (bootstrap name-builder refactor) clobbered four GPU feature options:
restore `enable_chroma` in `float_psnr_metal` and `integer_psnr_metal`;
restore `vif_skip_scale0` in `vif_vulkan`; restore ceiling-division chroma
geometry in `psnr_vulkan` (odd-dimension YUV420 parity with CPU and CUDA).
