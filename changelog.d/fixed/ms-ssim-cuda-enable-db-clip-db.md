**integer_ms_ssim_cuda: add missing `enable_db` / `clip_db` options** — the CUDA
twin of `float_ms_ssim` accepted neither `enable_db` nor `clip_db` in its options
table, so callers could not request dB-domain MS-SSIM output on the CUDA path.
Both options are now registered and `collect()` applies the same
`-10*log10(1 - ms_ssim)` transform and peak-SNR ceiling as `float_ms_ssim.c`,
restoring CPU/CUDA parity for log-domain output.
