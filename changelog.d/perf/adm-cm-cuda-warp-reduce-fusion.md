- **perf(cuda)**: fuse ADM CM warp-reduce for scales 1-3 — `i4_adm_cm_line_kernel_fused`
  replaces the two-kernel `i4_adm_cm_line_kernel` (compute → global scratch) +
  `adm_cm_reduce_line_kernel_4` (reduce) split.  Eliminates 3 separate reduce-kernel
  launches and 3 `accum_per_thread` global-memory scratch round-trips per frame,
  matching the pattern already used by scale 0 (`adm_cm_line_kernel_8`).
  Estimated 8–12 % speedup on the ADM CM pass; INT64 warp-reduce result is
  bit-identical to the two-kernel path.
