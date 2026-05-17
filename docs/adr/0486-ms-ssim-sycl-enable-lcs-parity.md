# ADR-0486: Add enable_lcs and enable_chroma to float_ms_ssim SYCL twin

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris
- **Tags**: `sycl`, `parity`, `options`

## Context

The option-parity audit (.workingdir/feature-option-parity-audit-2026-05-16.md)
identified that the SYCL twin of `float_ms_ssim` was missing `enable_lcs` and
`enable_chroma`, both of which the CUDA twin already exposed (added in PR #933 and
via the enable_chroma cross-backend pass). The SYCL `collect_fex_sycl()` already
computed per-scale L, C, S means internally for the Wang product; the only gap was
that the values were never forwarded to the feature collector when `enable_lcs=true`.
The `enable_chroma` option is structural (MS-SSIM is luma-only by construction) and
is accepted for option-table symmetry without semantic effect.

## Decision

Add `enable_lcs` and `enable_chroma` fields to `MsSsimStateSycl`, expose them in
`options_ms_ssim_sycl[]`, and in `collect_fex_sycl()` emit the 15 per-scale
intermediates (`float_ms_ssim_{l,c,s}_scale{0..4}`) when `enable_lcs=true`.
The default-path output (both flags false) is bit-for-bit unchanged.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Defer until SYCL MS-SSIM is fully refactored | No churn now | Leaves a silent option-drop bug longer | The fix is three lines; deferral adds no value |
| Add only `enable_lcs`, skip `enable_chroma` | Slightly smaller diff | Options table stays asymmetric with CUDA | Symmetry is the whole point of the parity pass |

## Consequences

- **Positive**: SYCL `float_ms_ssim` now matches CUDA's option table; callers using
  `enable_lcs=true` get the same 15 per-scale outputs on both GPU backends.
- **Negative**: None — default-path is unchanged; `enable_chroma` is a no-op
  accepted for future compatibility.
- **Neutral / follow-ups**: The audit row for SYCL `enable_lcs` can be marked DONE.

## References

- Option-parity audit: `.workingdir/feature-option-parity-audit-2026-05-16.md`
- ADR-0485: `enable_db`/`clip_db` parity (this PR extends the same SYCL file)
- CUDA reference implementation: `libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c`
