- `integer_psnr_hip`: add missing `min_sse` option; when set, the PSNR
  ceiling (`psnr_max_y`) is now computed from the SSE floor rather than
  hardcoded to `(6*bpc)+12`, matching `integer_psnr.c` init lines 128-138.
  Previously any `--feature 'psnr=min_sse=X'` on the HIP backend was
  silently dropped, yielding wrong PSNR ceilings at non-default values.
