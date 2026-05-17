- **SYCL PSNR `min_sse` option parity** (`integer_psnr_sycl.cpp`): the SYCL
  PSNR extractor was missing the `min_sse` feature option present in the CPU
  reference (`integer_psnr.c`). Without it, callers could not constrain the
  minimum possible SSE, and the per-plane `psnr_max` cap was always the fixed
  `(6*bpc)+12` formula regardless of any `min_sse` argument. Added the option
  and the matching `ceil(10*log10(peak^2 / mse_floor))` init formula, matching
  the CPU reference exactly.
