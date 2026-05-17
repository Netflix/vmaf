## Changed

- **VIF CUDA filter passes now stage data in shared memory** (`libvmaf/src/feature/cuda/integer_vif/filter1d.cu`):
  all four filter template functions (`filter1d_8_vertical_kernel`,
  `filter1d_8_horizontal_kernel`, `filter1d_16_vertical_kernel`,
  `filter1d_16_horizontal_kernel`) load a per-block tile into `__shared__`
  memory before the convolution loop, eliminating 7–8× redundant L2/DRAM reads
  per output pixel in the horizontal pass and 16 of 17 redundant row loads in
  the vertical pass. Boundary mirroring is handled in the smem load phase.
  Results are bit-identical to the pre-patch implementation (integer fixed-point
  arithmetic unchanged). Estimated speedup: 15–35% on VIF at 1080p
  (ADR-0452, Research-0135).
