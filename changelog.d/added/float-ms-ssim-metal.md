Metal backend: add `float_ms_ssim_metal` — float-precision 5-scale MS-SSIM on Apple
Silicon (ADR-0490). Three MSL kernels (decimate / horiz / vert_lcs) match the CUDA
twin layout; host reduction applies Wang (2003) weights in double precision. Wired into
`dispatch_strategy.c` and the feature-extractor registry.
