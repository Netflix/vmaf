fix(sycl): add `enable_lcs` and `enable_chroma` to `float_ms_ssim` SYCL twin (ADR-0486)

The SYCL twin was silently dropping both options; it now emits per-scale
`float_ms_ssim_{l,c,s}_scale{0..4}` intermediates when `enable_lcs=true`,
matching the CUDA twin's behaviour. `enable_chroma` is accepted for symmetry
(MS-SSIM is luma-only by construction).
