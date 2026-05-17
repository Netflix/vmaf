`float_psnr_hip` now exposes the `enable_chroma` option (default `true`),
matching the CUDA/SYCL/Vulkan PSNR twins (ADR-0469). Passing
`enable_chroma=false` is no longer silently dropped.
