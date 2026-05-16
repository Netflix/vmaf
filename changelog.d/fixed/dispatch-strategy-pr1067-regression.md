**HIP/Metal dispatch regression (PR #1067 clobber):** Restored the
`g_hip_features[]` routing table in `libvmaf/src/hip/dispatch_strategy.c`
(8 entries, ADR-0241 through ADR-0274) that PR #864 introduced but PR #1067
reverted to the unconditional-return-0 stub.  Removed the spurious
`"float_ms_ssim"` entry from `libvmaf/src/metal/dispatch_strategy.c` that
PR #864 had correctly deleted but PR #1067 re-introduced.
