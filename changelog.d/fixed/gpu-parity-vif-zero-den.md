- **CUDA VIF zero-denominator guard** (`integer_vif_cuda.c`): on
  degenerate frames (solid-colour input where all pixels fall below
  `SIGMA_NSQ`) the per-scale VIF denominator is zero; CUDA was returning
  NaN/inf while SYCL and Vulkan already returned 1.0. Added the same
  zero-guard to CUDA's `write_scores` to achieve GPU-parity across all
  three backends.
