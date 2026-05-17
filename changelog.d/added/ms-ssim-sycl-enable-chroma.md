Add `enable_chroma` option to the SYCL MS-SSIM extractor (`integer_ms_ssim_sycl.cpp`),
mirroring ms_ssim_vulkan PR #957. Default `false` (luma-only, behaviour unchanged).
When `true`, `n_planes` follows pix_fmt (1 for YUV400P, 3 otherwise);
multi-plane kernel dispatch is deferred to v2.
