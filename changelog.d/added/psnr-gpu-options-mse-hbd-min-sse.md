Wire `enable_mse`, `reduced_hbd_peak`, and `min_sse` options into `psnr_cuda` and
`psnr_vulkan`. Previously these three CPU knobs were silently ignored on both GPU
backends; GPU PSNR now honours them identically to `integer_psnr.c`.
