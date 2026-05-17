Add `enable_chroma` option to the Vulkan MS-SSIM extractor (`ms_ssim_vulkan.c`),
mirroring ssim_vulkan PR #956. Default `false` (luma-only, behaviour unchanged).
When `true`, `n_planes` follows pix_fmt (1 for YUV400P, 3 otherwise);
multi-plane kernel dispatch is deferred to v2.
