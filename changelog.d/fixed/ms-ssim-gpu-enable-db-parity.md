**float_ms_ssim CUDA/SYCL: add missing `enable_db` / `clip_db` options (ADR-0485)** — both GPU
twins of `float_ms_ssim` accepted neither `enable_db` nor `clip_db`, so callers could not
request dB-domain MS-SSIM output on the CUDA or SYCL paths.  Both options are now registered
and `collect()` applies the same `-10·log₁₀(1 − MSSSIM)` transform and peak-SNR ceiling as
`float_ms_ssim.c`, restoring CPU/GPU parity for log-domain output.  Default-path output
(`enable_db=false`) is bit-for-bit unchanged.
