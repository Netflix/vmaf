### Added

- **`float_moment` SVE2 SIMD path** (`libvmaf/src/feature/arm64/moment_sve2.c`):
  the 1st- and 2nd-moment reductions now use the VLA (vector-length-agnostic)
  SVE2 f32→f64 widening pattern on Linux aarch64 hosts that expose
  `HWCAP2_SVE2`.  The path steps by `svcntd()` f32 elements per iteration and
  accumulates into `svfloat64_t`, matching the f64 precision contract of the
  NEON sibling.  NEON remains the fallback when SVE2 is absent; Apple Silicon
  always uses the NEON path (Darwin SVE2 opt-out, ADR-0419).
  See [ADR-0461](../docs/adr/0461-moment-sve2-port.md).
