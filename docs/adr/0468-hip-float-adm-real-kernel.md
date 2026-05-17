# ADR-0468: HIP float_adm real kernel (ninth HIP consumer)

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `hip`, `build`, `feature-extractor`

## Context

The CUDA backend ships a fully implemented `float_adm_cuda.c` / `float_adm_score.cu`
(ADR-0192 / ADR-0202) that covers the four-stage DWT+CSF+CM ADM pipeline across all four
DWT scales. The HIP backend had scaffold stubs for eight other feature extractors but was
missing `float_adm` — the most signal-rich extractor in the VMAF model. Closing this gap
brings the HIP backend to feature parity with CUDA for ADM scoring, which is the dominant
contributor to the VMAF score.

## Decision

Port `float_adm_cuda.c` and `float_adm/float_adm_score.cu` to HIP as
`feature/hip/float_adm_hip.c` and `feature/hip/float_adm/float_adm_score.hip`. Wire the
HSACO compilation target into `src/meson.build`'s `hip_kernel_sources` dict and add the
`.c` host TU to `hip/meson.build`'s `hip_sources` list. Register
`vmaf_fex_float_adm_hip` in the `#if HAVE_HIP` block of `feature_extractor.c`.

Key HIP adaptations from the CUDA twin:
- Warp size 64 (GCN/RDNA): `__shfl_down` without mask; shared-mem warp-partial arrays
  sized at `FADM_WARPS_PER_BLOCK = 4` (vs. 8 on CUDA).
- Buffer alloc: `hipMalloc` / `hipMemsetAsync` / `hipMemcpyAsync` replace
  `vmaf_cuda_buffer_alloc` + `cuMemsetD8Async`.
- Module API: `hipModuleLoadData` / `hipModuleGetFunction` / `hipModuleLaunchKernel`.
- HSACO binary embedded via `xxd -i` in the `hip_hsaco_sources` custom_target pipeline
  (ADR-0372 pattern).

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep scaffold-only (ENOSYS) | No work | HIP ADM gap persists indefinitely | Gap is the stated motivation |
| Share CUDA PTX via HIP's CUDA compat layer | Single kernel source | Compat layer is experimental on ROCm 6+; warp-size mismatch (32 vs 64) would silently corrupt reductions | Correctness risk outweighs convenience |

## Consequences

- **Positive**: HIP backend now supports `float_adm` and all VMAF model components that
  depend on it (adm2 score + 4 scale subscores). `cross-backend-diff` can be run against
  HIP ADM.
- **Negative**: Ninth HSACO compilation target; adds ~2 s to `enable_hipcc=true` builds.
- **Neutral**: Same scaffold posture as all prior HIP consumers — `init()` returns
  -ENOSYS without `enable_hipcc=true`; no user-visible change on CPU-only builds.

## References

- ADR-0192 / ADR-0202: CUDA float_adm implementation.
- ADR-0372: HIP HSACO kernel build pipeline.
- ADR-0468 closes the last major HIP feature-parity gap for standard VMAF.
- req: "Port CUDA float_adm_cuda.c to HIP float_adm_hip.c + float_adm/*.hip kernels."
