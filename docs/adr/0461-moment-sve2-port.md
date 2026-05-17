# ADR-0461 — `float_moment` SVE2 port

| Field        | Value                                          |
|--------------|------------------------------------------------|
| Status       | Accepted                                       |
| Date         | 2026-05-16                                     |
| Deciders     | lusoris                                        |
| Tags         | arm64, sve2, simd, float_moment, bit-exactness |

## Context

`float_moment` (the 1st-moment / 2nd-moment feature extractor) has a NEON
kernel (`moment_neon.c`) but no SVE2 sibling.  The NEON path processes four
f32 lanes per iteration using `float32x4_t` and accumulates into `float64x2_t`
pairs — a fixed-width design that cannot exploit SVE2 registers wider than
128 bits on Neoverse V2 / Cortex-X4 and later hardware.

`ssimulacra2_sve2.c` (ADR-0213) established the fork's SVE2 pattern.
`float_moment` is the simplest candidate for a second SVE2 port: two
functions, pure reduction, no libm or branch-heavy code, and an existing
f64-accumulator contract (ADR-0179) that maps cleanly onto `svfloat64_t`.

## Decision

Add `libvmaf/src/feature/arm64/moment_sve2.c` + `moment_sve2.h` implementing
`compute_1st_moment_sve2` and `compute_2nd_moment_sve2` using the SVE2 VLA
f32→f64 widening pattern:

1. Inner loop steps by `svcntd()` f32 elements per iteration (= number of
   f64 lanes per register).
2. Each iteration loads up to `svcntd()` f32 elements under a
   `svwhilelt_b32` predicate, squares (for 2nd moment) in f32, then widens
   with `svcvt_f64_f32_x` under a matching `svwhilelt_b64` predicate.
3. Accumulates into `svfloat64_t dsum`; per-row `svaddv_f64` collapses to
   scalar double.
4. No scalar tail loop: the predicated load handles row tails natively.

The step is `svcntd()` and NOT `svcntw()` because `svcvt_f64_f32_x` widens
only the lower `svcntd()` f32 lanes.  Using `svcntw()` would silently skip
the upper half of the f32 register on SVE2 hardware with registers wider than
128 bits.

### Dispatch

`float_moment.c` selects SVE2 over NEON when
`VMAF_ARM_CPU_FLAG_SVE2` is set:

```c
#if HAVE_SVE2
if (cpu_flags & VMAF_ARM_CPU_FLAG_SVE2) {
    s->moment1 = compute_1st_moment_sve2;
    s->moment2 = compute_2nd_moment_sve2;
}
#endif
```

NEON remains the fallback.  The SVE2 path is purely additive.

### Build gate

The `moment_sve2.c` TU is compiled in its own static library with
`-march=armv9-a+sve2 -ffp-contract=off`, registered under the existing
`if is_sve2_supported` block in `libvmaf/src/meson.build`.  Darwin forces
`is_sve2_supported = false` per ADR-0419; the TU is therefore never linked on
Apple Silicon.

### Parity test

`libvmaf/test/test_moment_simd.c` gains four SVE2 test cases
(`test_sve2_seed_{a,b}`, `test_sve2_aligned_w`, `test_sve2_tiny`) using the
existing `SIMD_BITEXACT_ASSERT_RELATIVE` macro at `MOMENT_REL_TOL = 1e-7`.
The test cases are guarded by `#if ARCH_AARCH64 && HAVE_SVE2` at compile time
and by a `vmaf_get_cpu_flags()` check at runtime so a NEON-only host passes
without executing the SVE2 path.

## Alternatives considered

No alternatives: the implementation is a mechanical port of the NEON pattern
to the canonical SVE2 VLA f32→f64 accumulation pattern.  The only non-obvious
choice (step size `svcntd()` vs `svcntw()`) is dictated by the semantics of
`svcvt_f64_f32_x` and is documented inline in the source.

## Consequences

- ARM Graviton (Neoverse V1/V2) and Cortex-X4 guests running the CI ARM lane
  will exercise the SVE2 path when `HWCAP2_SVE2` is set.
- NEON host CI (e.g. `ubuntu-24.04-arm` on M1-era hardware) skips the SVE2
  test cases at runtime with a log message; the NEON tests still run.
- x86-64 host builds ignore both NEON and SVE2 paths entirely (compile-time
  `#if ARCH_AARCH64` guards).

## References

- req: "Add ARM SVE2 SIMD path for ONE feature extractor that has NEON but not SVE2 yet"
- [ADR-0213](0213-ssimulacra2-sve2.md) — first SVE2 port, established the pattern
- [ADR-0419](0419-sve2-darwin-probe-gate.md) — Darwin SVE2 opt-out
- [ADR-0179](0179-float-moment-neon-bitexact.md) — float_moment NEON bit-exactness contract
- [ADR-0138](0138-simd-bitexact-contract.md) — SIMD bit-exactness general rule
- [ADR-0245](0245-simd-bitexact-test-harness.md) — `simd_bitexact_test.h` harness
