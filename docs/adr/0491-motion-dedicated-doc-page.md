# ADR-0491: Add dedicated `docs/metrics/motion.md` reference page

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `docs`, `metrics`, `motion`, `fork-local`

## Context

`docs/metrics/` has dedicated pages for ANSNR, CAMBI, MS-SSIM, PSNR-HVS,
SSIM, SSIMULACRA2, and VIF, but no standalone page for the motion family of
feature extractors (`motion`, `motion_v2`, `float_motion`). The motion
extractors are covered inline in `features.md`, but that page covers every
extractor and cannot provide the options-table depth, per-variant backend
matrix, or usage examples that a dedicated page affords.

Motion has the widest backend coverage in the fork (all five GPU backends:
CUDA, SYCL, Vulkan, HIP, Metal; plus AVX2, AVX-512, and NEON SIMD) and
three distinct extractor variants with overlapping but different option sets.
Per ADR-0100 (project-wide doc-substance rule), every user-discoverable
surface ships a dedicated doc page.

## Decision

We add `docs/metrics/motion.md` covering all three registered extractors
(`motion`, `motion_v2`, `float_motion`) with an options table, output
feature table, output range, input format constraints, and a complete
backend coverage table for each variant.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Expand the inline `features.md` section | No new file, already partially exists | The inline section would grow unwieldy; no cross-linking possible from per-backend overview pages | Rejected |
| One section per variant in separate files | Maximum granularity | Three tiny files vs one coherent page; the variants share many options | Rejected |

## Consequences

- **Positive**: Users and contributors have a single-page reference for all
  three motion variants including per-backend status and option ranges.
- **Positive**: Backend overview pages can link directly to the per-variant
  tables.
- **Neutral**: `features.md` retains its inline summary; the new page is
  additive and cross-linked from `## See also`.
- **Neutral / follow-up**: When a new motion extractor or backend is added,
  `docs/metrics/motion.md` must be updated in the same PR (same rule as all
  other per-surface doc pages).

## References

- `docs/metrics/features.md` §Motion2 — existing inline documentation
- [ADR-0100](0100-project-wide-doc-substance-rule.md) — per-surface doc
  substance rule
- [ADR-0108](0108-deep-dive-deliverables-rule.md) — six deep-dive deliverables
- [ADR-0193](0193-motion-v2-vulkan.md) — `motion_v2` Vulkan kernel
- [ADR-0196](0196-float-motion-gpu.md) — `float_motion` GPU kernels
- [ADR-0219](0219-motion3-gpu-coverage.md) — motion3 GPU coverage
- [ADR-0273](0273-float-motion-hip.md) — `float_motion` HIP backend
