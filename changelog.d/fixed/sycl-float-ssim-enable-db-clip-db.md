- **SYCL float_ssim `enable_db` / `clip_db` option parity** (`integer_ssim_sycl.cpp`):
  the SYCL backend silently dropped the `enable_db` and `clip_db` options
  present on the CPU extractor and the HIP/Vulkan twins. Added both options
  plus the host-side dB conversion (`-10*log10(1-ssim)`) and dB clamp to
  `collect_fex_sycl`, matching the behaviour of `float_ssim.c`,
  `float_ssim_hip.c` (PR #1087), and the planned Vulkan parity PR (#1040).
