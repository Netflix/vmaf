- **perf(cuda/cambi)**: `cambi_spatial_mask_kernel` now stages a 22x22
  zero_deriv tile into `__shared__` memory before the 7x7 box-sum pass.
  Global memory reads per block fall from 37,632 (256 threads x 147 reads) to
  1,452 (484 elements x 3 reads each), a 26x reduction.  No change to
  numerical output; `places=4` parity gate continues to pass.
  (ADR-0464 / perf-audit 2026-05-16 win 3)
