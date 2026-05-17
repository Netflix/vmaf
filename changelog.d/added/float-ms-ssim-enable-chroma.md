The `float_ms_ssim` extractor now accepts an `enable_chroma=true` option that
computes and emits per-plane MS-SSIM for the Cb (`float_ms_ssim_cb`) and Cr
(`float_ms_ssim_cr`) chroma channels in addition to the existing luma
`float_ms_ssim` score. The default (`enable_chroma=false`) is
backward-compatible — only `float_ms_ssim` is emitted.
