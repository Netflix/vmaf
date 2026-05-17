# ADR-0460: MS-SSIM `enable_db` and `clip_db` option parity on CUDA and SYCL backends

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: cuda, sycl, ms_ssim, option-parity, bug

## Context

The wiring audit of 2026-05-16 found that the CUDA and SYCL
`float_ms_ssim` extractors did not expose the `enable_db` and `clip_db`
options present in the CPU reference (`float_ms_ssim.c`). In addition,
the SYCL extractor was missing `enable_lcs` entirely (the CUDA extractor
had it from T7-35 / ADR-0243).

With `enable_db=true` the CPU extractor converts the linear MS-SSIM score
to the dB domain via `-10 * log10(1 - ms_ssim)`. With `clip_db=true` it
first clips the linear score to `[0, 1]` to avoid a negative argument to
`log10`. Both are part of the public `VmafOption` surface documented in
`docs/metrics/ms_ssim.md`.

Passing either flag to the CUDA or SYCL extractor silently dropped the
option: the unknown-option path returned without error, and the GPU
extractor continued emitting linear scores while the CPU emitted dB
values. This constitutes a silent output divergence whenever a caller
uses GPU MS-SSIM with `enable_db=true`.

The GPU partials reduction already computes per-scale L/C/S means in both
backends; `enable_lcs` for SYCL and `enable_db`/`clip_db` for both
backends are therefore purely host-side option additions — no kernel
changes are required.

## Decision

For each affected GPU twin:

**CUDA (`integer_ms_ssim_cuda.c`)**:
1. Add `bool enable_db` and `bool clip_db` to `MsSsimStateCuda`.
2. Add matching `VmafOption` entries (type `VMAF_OPT_TYPE_BOOL`,
   `default_val.b = false`) to the existing `options[]` table.
3. In `collect_fex_cuda`, apply the conversion to `score` before the
   `vmaf_feature_collector_append_with_dict` call, mirroring
   `float_ms_ssim.c` exactly.

**SYCL (`integer_ms_ssim_sycl.cpp`)**:
1. Add `bool enable_lcs`, `bool enable_db`, and `bool clip_db` to
   `MsSsimStateSycl`.
2. Replace the empty `options_ms_ssim_sycl[] = {{0}}` sentinel with a
   three-entry table for `enable_lcs`, `enable_db`, and `clip_db`.
3. In `collect_fex_sycl`, apply the dB conversion and the LCS emission
   block, mirroring the CUDA extractor.

At the default values (all `false`) the output is bit-identical to the
pre-patch binary on both backends.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Return `-ENOTSUP` when `enable_db=true` on GPU | Simple | Caller gets an opaque error with no guidance; breaks pipelines that use CPU/GPU interchangeably | Masks the bug rather than fixing it |
| Add all remaining MS-SSIM CPU options in one sweep | Closes more parity gaps | Vulkan already exposes `enable_lcs`/`enable_db`/`clip_db`; a sweep would touch more files without closing the core user-visible bug | Fix the verified gaps; defer future audits |

## Consequences

- **Positive**: `enable_db=true` and `clip_db=true` now produce identical
  dB-domain MS-SSIM scores on CPU, CUDA, and SYCL. `enable_lcs` on SYCL
  now emits the 15 per-scale LCS triples that were already available on
  CPU and CUDA. Cross-backend parity gate is unaffected at default values.
- **Negative**: None. Default-path output is bit-for-bit unchanged.
- **Neutral / follow-ups**: HIP and Metal do not have `float_ms_ssim`
  twins; those backends are not affected. The remaining CUDA/SYCL MS-SSIM
  option gaps (none identified beyond this PR) are considered resolved.

## References

- CPU reference: `libvmaf/src/feature/float_ms_ssim.c` lines 52–76
  (options table) and lines 131, 185–221 (enable_db / enable_lcs paths).
- [ADR-0243](0243-ms-ssim-enable-lcs.md) — enable_lcs on CUDA (T7-35)
- [ADR-0453](0453-psnr-enable-chroma-gpu-parity.md) — precedent for this
  class of option-parity fix
- [ADR-0214](0214-gpu-parity-ci-gate.md) — GPU-parity CI gate
- `.workingdir/wiring-audit-2026-05-16.md` — source finding
- `.workingdir/copy-paste-parity-audit-2026-05-16.md` — severity rating
- PR that implements this: fix/ms-ssim-gpu-enable-db-lcs-sycl-2026-05-16
