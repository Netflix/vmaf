Add `enable_chroma` option to the CUDA SSIM extractor (`integer_ssim_cuda.c`),
mirroring CPU PR #939. Default `false` (luma-only, behaviour unchanged).
When `true`, `n_planes` follows pix_fmt (1 for YUV400P, 3 otherwise);
multi-plane kernel dispatch is deferred to v2.
