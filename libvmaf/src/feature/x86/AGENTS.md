# AGENTS.md — libvmaf/src/feature/x86

Orientation for agents working on the AVX2 / AVX-512 feature SIMD
paths. Parent: [../AGENTS.md](../AGENTS.md).

## Scope

Per-feature AVX2 + AVX-512 SIMD implementations. Every TU here mirrors
a scalar reference one level up (e.g. `ssim_avx2.c` ↔ `../iqa/ssim_tools.c`,
`adm_avx2.c` ↔ `../adm.c`) and is dispatched at runtime from a feature's
`*_dispatch.c` based on `vmaf_get_cpu_flags_x86()` (see
[`../../x86/cpu.c`](../../x86/cpu.c)).

```text
feature/x86/
  <feature>_avx2.{c,h}      # AVX2 path (Haswell+ baseline)
  <feature>_avx512.{c,h}    # AVX-512 path (Skylake-X+ baseline; ICL flag for AVX512BW/VBMI2)
  ms_ssim_decimate_*.{c,h}  # 9-tap LPF SIMD (one of four byte-identical TUs — see parent AGENTS.md)
```

The cross-feature plumbing (dispatch tables, the `simd_dx.h` macro
header, the runtime CPUID gate) lives in `../` — this directory
contains only kernel TUs.

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
- **Bit-exactness with the scalar reference is non-negotiable.** Every
  AVX2 / AVX-512 kernel here mirrors a scalar TU byte-for-byte under
  `FLT_EVAL_METHOD == 0`. The bit-exact regression tests in
  [`../../../test/`](../../test/) (`test_*_simd.c`, migrated through
  the [`simd_bitexact_test.h`](../../test/simd_bitexact_test.h) harness
  per ADR-0245) catch ULP drift; pushing through them without a
  paired scalar update is a regression.
- **No FMA on the load-bearing reductions.** `#pragma STDC FP_CONTRACT
  OFF` is set at TU level on every kernel that participates in
  ADR-0138 (`iqa_convolve` widen-then-add) or ADR-0139 (SSIM
  per-lane scalar-double accumulate). The compiler's default
  `-ffp-contract=fast` would silently fuse `a + b * c` and break
  bit-identity vs scalar.
- **Reserved-identifier hygiene** (ADR-0148): no leading-underscore
  names. The IQA tree underwent a sweeping `_iqa_*` →
  `iqa_*` / `_kernel` → `iqa_kernel` / `_ssim_int` →
  `ssim_int` rename; do not reintroduce the old spellings on
  rebase.

## Twin-update rules

These TUs come in twin-bundles. A change to one half **must** ship
with the matching change to the other halves in the **same PR**:

| Group | TUs that move in lockstep |
| --- | --- |
| **SSIM accumulate** (ADR-0139) | `ssim_avx2.c` + `ssim_avx512.c` + `../arm64/ssim_neon.c` + scalar `../iqa/ssim_tools.c` (`ssim_accumulate_default_scalar`) + shared helper `../iqa/ssim_accumulate_lane.h` |
| **IQA convolve** (ADR-0138 + ADR-0143) | `convolve_avx2.c` + `convolve_avx512.c` + `../arm64/convolve_neon.c` + scalar `../iqa/convolve.c` + shared scanline helpers `../common/convolution_avx.c` |
| **MS-SSIM decimate LPF** (ADR-0125) | `ms_ssim_decimate_avx2.c` + `ms_ssim_decimate_avx512.c` + `../arm64/ms_ssim_decimate_neon.c` + scalar `../ms_ssim_decimate.c`. The 9-tap filter table appears verbatim in all four — diff all four when any one moves. |
| **PSNR-HVS DCT** (ADR-0159 + ADR-0350) | `psnr_hvs_avx2.c` + `../arm64/psnr_hvs_neon.c` + scalar `../third_party/xiph/psnr_hvs.c`. Butterfly block is byte-identical across the three. **No `psnr_hvs_avx512.c` — AVX-512 closed as ceiling under T3-9 (a) per [ADR-0350](../../../../docs/adr/0350-psnr-hvs-avx512-ceiling.md): `perf record` cycle share is 78.42 % scalar tail (locked by ADR-0138/0139 bit-exactness) vs 14.82 % DCT, capping a 16-lane widening at 1.07–1.08× over AVX2 (Amdahl ceiling 1.17×) — well below T3-9's 1.3× ship gate.** Re-bench gate: any future upstream change to the Xiph/Daala scalar that shifts the per-block summation tree requires re-running [Research-0091 §7](../../../../docs/research/0091-psnr-hvs-avx512-bench-2026-05-09.md) before claiming the ceiling still holds. |
| **VIF SIMD8** (ADR-0146) | `vif_statistic_avx2.c` (`vif_stat_simd8_compute` + `vif_stat_simd8_reduce` halves around `struct vif_simd8_lane`) + scalar `../vif.c`. Per-lane scalar-float reduction via 32-byte aligned `tmp_n[8]` / `tmp_d[8]` is load-bearing for ADR-0139. |
| **CAMBI calculate_c_values_row** (ADR-0452) | `cambi_avx2.c` (`calculate_c_values_row_avx2`) + `cambi_avx512.c` (`calculate_c_values_row_avx512`) + `../arm64/cambi_neon.c` (`calculate_c_values_row_neon`) + scalar in `../cambi.c` (`calculate_c_values_row`). Every cambi inner-loop function ported to AVX2 **must** have AVX-512 + NEON siblings in the **same PR**. Bit-exact (integer pipeline, no float reduction tree). Tested in `../../test/test_cambi_simd.c`. |
| **SSIMULACRA 2 SIMD** (ADR-0161 / 0162 / 0163 / 0252) | `ssimulacra2_avx2.c` + `ssimulacra2_avx512.c` + `../arm64/ssimulacra2_neon.c` + `../arm64/ssimulacra2_sve2.c` + `ssimulacra2_host_avx2.c` + `../arm64/ssimulacra2_host_neon.c` + scalar `../ssimulacra2.c` + Vulkan host-path call site `../vulkan/ssimulacra2_vulkan.c` |
| **Motion v2 NEON / AVX2 divergence** (ADR-0145) | `motion_v2_avx2.c` (currently uses `_mm256_srlv_epi64` *logical*) is **knowingly out-of-spec** vs scalar; `../arm64/motion_v2_neon.c` matches scalar via arithmetic shift. Do NOT port the AVX2 logical pattern to NEON. The AVX2 audit is a separate batch. |

The complete invariants live in [../AGENTS.md
§"Rebase-sensitive invariants"](../AGENTS.md); this table is the
**index** of which file groups move together.

## simd_dx macros (ADR-0140)

[`../simd_dx.h`](../simd_dx.h) is fork-internal. AVX2 / AVX-512 paths
in this directory consume `SIMD_WIDEN_ADD_F32_F64_AVX2` /
`SIMD_WIDEN_ADD_F32_F64_AVX512`, `SIMD_ALIGNED_F32_BUF_*`,
`SIMD_LANES_*` to encode the ADR-0138 / 0139 patterns by
construction. Macro names are ISA-suffixed on purpose; do not
collapse them into cross-ISA aliases — the fork's SIMD policy
rules out Highway / simde / xsimd (user memory
`feedback_simd_dx_scope.md`).

## Adding a new AVX2 / AVX-512 TU

Use [`/add-simd-path`](../../../../.claude/skills/add-simd-path/SKILL.md).
The skill scaffolds:

1. The TU + header, with `#pragma STDC FP_CONTRACT OFF` at the
   top and the appropriate `#include "../simd_dx.h"`.
2. The dispatch entry in the feature's `*_dispatch.c` so
   `vmaf_get_cpu_flags_x86()` selects the new path.
3. A bit-exact regression test under `../../test/test_<feature>_simd.c`
   using the [`simd_bitexact_test.h`](../../test/simd_bitexact_test.h)
   harness (ADR-0245).

## Upstream-sync notes

- Every TU in this directory carries a Netflix copyright header
  (`Copyright 2016-202x Netflix, Inc.`) — these files are
  upstream-mirror at the structural level even though several
  carry fork-only refactors (ADR-0146 helper splits in
  `vif_statistic_avx2.c`; ADR-0143 `static` + `ptrdiff_t` in
  `convolve_avx2.c`).
- On `/sync-upstream` or `/port-upstream-commit`: if a Netflix
  patch touches any TU in this directory, walk the corresponding
  twin in `../arm64/` + the scalar reference + the SIMD-tail
  reduction helper (`../iqa/ssim_accumulate_lane.h` for SSIM,
  `../iqa/convolve.c` for convolve) before merging. The cross-
  backend parity gate at `places=4`
  ([`scripts/ci/cross_backend_parity_gate.py`](../../../../scripts/ci/cross_backend_parity_gate.py),
  ADR-0214) catches scalar↔SIMD drift but only after a full run.
- VIF kernelscale stays on the precomputed `vif_filter1d_table_s`
  flow ([Research-0024 Strategy E](../../../../docs/research/0024-vif-upstream-divergence.md)).
  Do **not** port Netflix `4ad6e0ea` / `8c645ce3` runtime helpers
  verbatim — they lose the bit-exact contract that ADR-0138 /
  0139 / 0142 / 0143 froze.

## Governing ADRs

See [../AGENTS.md §Governing ADRs](../AGENTS.md) for the full list.
The ones that carve invariants on this directory specifically:

- [ADR-0125](../../../../docs/adr/0125-ms-ssim-decimate-simd.md) —
  MS-SSIM decimate separable SIMD.
- [ADR-0138](../../../../docs/adr/0138-iqa-convolve-avx2-bitexact-double.md) —
  `iqa_convolve` widen-then-add bit-exactness.
- [ADR-0139](../../../../docs/adr/0139-ssim-simd-bitexact-double.md) —
  SSIM accumulate per-lane scalar-double reduction.
- [ADR-0140](../../../../docs/adr/0140-simd-dx-framework.md) —
  `simd_dx.h` framework.
- ADR-0143
  ([`0143-port-netflix-f3a628b4-generalized-avx-convolve.md`](../../../../docs/adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md))
  — generalised AVX convolve scanlines.
- [ADR-0146](../../../../docs/adr/0146-nolint-sweep-function-size.md) —
  IQA / VIF SIMD helper decomposition.
- [ADR-0148](../../../../docs/adr/0148-iqa-rename-and-cleanup.md) —
  reserved-identifier rename.
- [ADR-0159](../../../../docs/adr/0159-psnr-hvs-avx2-bitexact.md) —
  `psnr_hvs` AVX2 DCT.
- [ADR-0161](../../../../docs/adr/0161-ssimulacra2-simd-bitexact.md) +
  [ADR-0162](../../../../docs/adr/0162-ssimulacra2-iir-blur-simd.md) +
  [ADR-0163](../../../../docs/adr/0163-ssimulacra2-ptlr-simd.md) +
  [ADR-0252](../../../../docs/adr/0252-ssimulacra2-host-xyb-simd.md) —
  SSIMULACRA 2 SIMD ports.
- [ADR-0245](../../../../docs/adr/0245-simd-bitexact-test-harness.md) —
  shared bit-exact test harness.
