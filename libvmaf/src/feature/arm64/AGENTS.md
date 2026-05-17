# AGENTS.md — libvmaf/src/feature/arm64

Orientation for agents working on the NEON / SVE2 feature SIMD paths.
Parent: [../AGENTS.md](../AGENTS.md). Sister directory:
[`../x86/`](../x86/AGENTS.md).

## Scope

Per-feature aarch64 NEON + SVE2 SIMD implementations. Every TU here
mirrors a scalar reference one level up and is dispatched at runtime
from a feature's `*_dispatch.c` based on `vmaf_get_cpu_flags_arm()`
(see [`../../arm/cpu.c`](../../arm/cpu.c)).

```text
feature/arm64/
  <feature>_neon.{c,h}      # NEON path (aarch64 baseline; always available on ARCH_AARCH64)
  ssimulacra2_sve2.{c,h}    # SVE2 path (T7-38 / ADR-0213) — runtime-gated via HWCAP2_SVE2
  ms_ssim_decimate_neon.*   # 9-tap LPF SIMD (one of four byte-identical TUs — see parent AGENTS.md)
```

## Ground rules

- **Every SIMD `.h` file MUST be self-contained.** Include every
  standard header that names a type used in the file's own declarations
  — do not rely on transitive includes from consumer `.c` files. In
  particular, any header that declares a `ptrdiff_t` parameter MUST
  include `<stddef.h>` directly. Standalone-include failures on Apple
  Clang and Ubuntu ARM Clang are CI regressions (see PR #914 for the
  cambi family; fixed for the motion family in the accompanying PR).
- **Parent rules** apply in full (see [../AGENTS.md](../AGENTS.md) +
  [../../AGENTS.md](../../AGENTS.md)).
- **Bit-exactness with the scalar reference is non-negotiable.** Same
  rule as the AVX2 / AVX-512 sibling — every NEON kernel mirrors the
  scalar TU byte-for-byte under `FLT_EVAL_METHOD == 0`. The bit-exact
  regression tests in [`../../../test/`](../../test/) (`test_*_simd.c`,
  migrated through the [`simd_bitexact_test.h`](../../test/simd_bitexact_test.h)
  harness per ADR-0245) catch ULP drift.
- **`#pragma STDC FP_CONTRACT OFF` is kept at TU level** even though
  aarch64 GCC ignores it with a non-fatal `-Wunknown-pragmas`. The
  pragma is portable and aarch64 GCC does not contract `a + b * c`
  across statements at default optimisation anyway. Removing it on
  rebase loses the cross-architecture documentation.
- **`accumulate_error()` and similar reductions thread accumulators
  by pointer** — do NOT introduce a local-float accumulator inside a
  helper. ADR-0159 burned this lesson into `psnr_hvs_neon.c`: a
  local accumulator drifts the Netflix golden by ~5.5e-5.

## Twin-update rules

These TUs come in twin-bundles. A change to one half **must** ship
with the matching change to the other halves in the **same PR**:

| Group | TUs that move in lockstep |
| --- | --- |
| **SSIM accumulate** (ADR-0139) | `ssim_neon.c` + `../x86/ssim_avx2.c` + `../x86/ssim_avx512.c` + scalar `../iqa/ssim_tools.c` + shared helper `../iqa/ssim_accumulate_lane.h` |
| **IQA convolve** (ADR-0138 + ADR-0143) | `convolve_neon.c` + `../x86/convolve_avx2.c` + `../x86/convolve_avx512.c` + scalar `../iqa/convolve.c` |
| **MS-SSIM decimate LPF** (ADR-0125) | `ms_ssim_decimate_neon.c` + `../x86/ms_ssim_decimate_avx2.c` + `../x86/ms_ssim_decimate_avx512.c` + scalar `../ms_ssim_decimate.c`. The 9-tap filter table appears verbatim in all four. |
| **PSNR-HVS DCT** (ADR-0160) | `psnr_hvs_neon.c` + `../x86/psnr_hvs_avx2.c` + scalar `../third_party/xiph/psnr_hvs.c`. Butterfly block byte-identical across the three; threading `ret` by pointer is load-bearing. |
| **SSIMULACRA 2 SIMD** (ADR-0161 / 0162 / 0163 / 0213 / 0252) | `ssimulacra2_neon.c` + `ssimulacra2_sve2.c` + `../x86/ssimulacra2_avx2.c` + `../x86/ssimulacra2_avx512.c` + `ssimulacra2_host_neon.c` + `../x86/ssimulacra2_host_avx2.c` + scalar `../ssimulacra2.c` + Vulkan host-path `../vulkan/ssimulacra2_vulkan.c` |
| **CAMBI calculate_c_values_row** (ADR-0452) | `cambi_neon.c` (`calculate_c_values_row_neon`) + `../x86/cambi_avx2.c` (`calculate_c_values_row_avx2`) + `../x86/cambi_avx512.c` (`calculate_c_values_row_avx512`) + scalar in `../cambi.c` (`calculate_c_values_row`). Every cambi inner-loop function ported to AVX2 **must** have AVX-512 + NEON siblings in the **same PR**. NEON uses vectorised mask-zero detection (vmaxvq_u16) + scalar per-pixel inner loop for bit-exact output (no gather instruction on NEON). Tested in `../../test/test_cambi_simd.c`. |
| **Motion v2 NEON** (ADR-0145) | `motion_v2_neon.c` uses **arithmetic** right-shift (`vshrq_n_s64(v, 16)` / `vshlq_s64(v, -(int64_t)bpc)`); matches scalar. Sister `../x86/motion_v2_avx2.c` uses `_mm256_srlv_epi64` (logical) — knowingly out-of-spec until the AVX2 audit. **Do NOT port the AVX2 logical pattern here.** 4-lane stride + scalar tails on both sides of the row are load-bearing for the x_conv edge-mirror contract. |

The complete invariants live in [../AGENTS.md
§"Rebase-sensitive invariants"](../AGENTS.md).

## SVE2 invariants (T7-38, ADR-0213)

`ssimulacra2_sve2.c` is the first SVE2 consumer in the fork. It is
**not** a free perf knob:

- The kernel is locked to a fixed 4-lane predicate
  (`svwhilelt_b32(0, 4)`) so its arithmetic order matches the NEON
  sibling regardless of runtime vector length. Widening to
  `svptrue_b32()` exposes lane-count drift across SVE2 hardware
  generations and breaks ADR-0161 bit-identity.
- Build develops against `qemu-aarch64-static`; CI runs the SVE2
  smoke under qemu. Real-hardware verification is opportunistic
  (no SVE2 self-hosted runner yet).
- Runtime gate: `vmaf_get_cpu_flags_arm()` sets the
  `VMAF_ARM_CPU_FLAG_SVE2` bit only when the kernel `AT_HWCAP2`
  reports `HWCAP2_SVE2` (see [`../../arm/cpu.c`](../../arm/cpu.c) +
  [`../../arm/AGENTS.md`](../../arm/AGENTS.md)). On aarch64 hosts
  without SVE2, the dispatcher falls back to NEON automatically.

**On rebase**: do not widen the predicate, do not re-order the
matmul / downsample chain, do not introduce vector `cbrtf` /
`powf` polynomials. The SSIMULACRA 2 invariants apply identically
to NEON and SVE2.

## Adding a new NEON / SVE2 TU

Use [`/add-simd-path`](../../../../.claude/skills/add-simd-path/SKILL.md).
The skill scaffolds the TU + header + dispatch entry + bit-exact
regression test using the shared
[`simd_bitexact_test.h`](../../test/simd_bitexact_test.h) harness
(ADR-0245).

## Upstream-sync notes

Same rules as [`../x86/AGENTS.md`](../x86/AGENTS.md): every TU
carries a Netflix copyright header at the structural level. On
`/sync-upstream` walk the AVX twin + the scalar reference + the
shared SIMD-tail reduction helper before merging. The cross-backend
parity gate at `places=4` catches drift but only after a full run.

## Governing ADRs

See [../AGENTS.md §Governing ADRs](../AGENTS.md) for the full list.
The ones that carve invariants on this directory specifically:

- [ADR-0125](../../../../docs/adr/0125-ms-ssim-decimate-simd.md) —
  MS-SSIM decimate separable SIMD.
- [ADR-0139](../../../../docs/adr/0139-ssim-simd-bitexact-double.md) —
  SSIM accumulate per-lane scalar-double reduction.
- [ADR-0140](../../../../docs/adr/0140-simd-dx-framework.md) —
  `simd_dx.h` framework.
- [ADR-0145](../../../../docs/adr/0145-motion-v2-neon-bitexact.md) —
  `motion_v2` NEON arithmetic-shift contract.
- [ADR-0160](../../../../docs/adr/0160-psnr-hvs-neon-bitexact.md) —
  `psnr_hvs` NEON DCT.
- [ADR-0161](../../../../docs/adr/0161-ssimulacra2-simd-bitexact.md) +
  [ADR-0162](../../../../docs/adr/0162-ssimulacra2-iir-blur-simd.md) +
  [ADR-0163](../../../../docs/adr/0163-ssimulacra2-ptlr-simd.md) +
  [ADR-0213](../../../../docs/adr/0213-ssimulacra2-sve2.md) +
  [ADR-0252](../../../../docs/adr/0252-ssimulacra2-host-xyb-simd.md) —
  SSIMULACRA 2 SIMD ports (NEON + SVE2 + host-path).
- [ADR-0245](../../../../docs/adr/0245-simd-bitexact-test-harness.md) —
  shared bit-exact test harness.
