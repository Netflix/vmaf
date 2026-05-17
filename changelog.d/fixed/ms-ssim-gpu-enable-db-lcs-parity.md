**fix(cuda,sycl): add `enable_db`, `clip_db`, and `enable_lcs` options to CUDA and SYCL MS-SSIM extractors (ADR-0460)**

The `float_ms_ssim` CUDA extractor was missing `enable_db` and `clip_db`;
the SYCL extractor was missing all three options (`enable_lcs`, `enable_db`,
`clip_db`). Passing any of these to a GPU session silently had no effect,
causing output divergence from the CPU reference. The GPU partial-sum
reduction already computed the per-scale L/C/S means; this fix adds
host-side option dispatch only — no kernel changes. Default-path output
is bit-identical to the pre-fix binary.
