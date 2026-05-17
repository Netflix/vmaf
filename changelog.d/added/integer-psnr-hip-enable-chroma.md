`integer_psnr_hip`: add `enable_chroma` option (default `true`); `psnr_cb` and
`psnr_cr` are now dispatched and emitted on HIP, matching the CUDA/SYCL/Vulkan
twins. YUV400P and `enable_chroma=false` sources still emit luma-only. (ADR-0471)
