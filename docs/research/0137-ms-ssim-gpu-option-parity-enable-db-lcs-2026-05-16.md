# Research-0137: MS-SSIM GPU option parity — `enable_db`, `clip_db`, `enable_lcs`

**Date**: 2026-05-16
**Branch**: fix/ms-ssim-gpu-enable-db-lcs-sycl-2026-05-16
**ADR**: [ADR-0460](../adr/0460-ms-ssim-enable-db-clip-db-gpu-parity.md)

## Finding

The wiring audit of 2026-05-16 (`.workingdir/wiring-audit-2026-05-16.md` and
`.workingdir/copy-paste-parity-audit-2026-05-16.md`) identified the following
option-table gaps for the `float_ms_ssim` GPU extractors:

| Backend | Missing option | Effect |
|---------|---------------|--------|
| CUDA | `enable_db` | dB-domain score silently not applied |
| CUDA | `clip_db` | clip before dB conversion silently ignored |
| SYCL | `enable_lcs` | 15 per-scale LCS triples not emitted |
| SYCL | `enable_db` | same as CUDA |
| SYCL | `clip_db` | same as CUDA |

The Vulkan backend (`ms_ssim_vulkan.c`) already exposed all three options
and served as the reference for the correct fix shape.

## Fix shape

Both backends already reduce per-scale L/C/S partial sums on the host and
accumulate them into `l_means[]` / `c_means[]` / `s_means[]` before the
Wang product combine. No kernel changes are needed:

1. Add bool fields to the state struct.
2. Add `VmafOption` entries to the options table.
3. In `collect()`, apply `enable_db` / `clip_db` conversion (mirroring
   `float_ms_ssim.c` lines 131, 185) and add the `enable_lcs` emission
   block (already present verbatim in CUDA since ADR-0243; ported to SYCL
   in this PR).

At default values (`all false`) the output is bit-identical to the
pre-patch binary on both backends — the conversion path is not entered and
the LCS block is not executed.

## Cross-backend correctness

The dB conversion formula used is `-10.0 * log10(1.0 - score)`, which
matches the CPU and Vulkan extractors exactly. The clip guard uses
`std::clamp` in SYCL (C++17, already available in the SYCL TU) and
manual min/max in CUDA (C99-compatible).

## Parity status after this fix

| Option | CPU | CUDA | SYCL | Vulkan |
|--------|-----|------|------|--------|
| `enable_lcs` | yes | yes (ADR-0243) | **yes (this PR)** | yes |
| `enable_db` | yes | **yes (this PR)** | **yes (this PR)** | yes |
| `clip_db` | yes | **yes (this PR)** | **yes (this PR)** | yes |
