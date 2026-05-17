# ADR-0487: Wire adm_min_val option into integer_adm GPU backends

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris
- **Tags**: `cuda`, `sycl`, `vulkan`, `adm`, `parity`

## Context

The CPU `integer_adm` extractor exposes an `adm_min_val` option (default 0.0)
that clamps the final ADM score to a caller-specified floor. The CUDA, SYCL,
and Vulkan backends did not register this option, so any non-default value
passed by the caller was silently ignored. This was identified as a
MISSING-bug in the 2026-05-16 feature option-parity audit.

The fix is host-side only: after computing `score = num / den`, apply
`score = MAX(score, adm_min_val)`. No GPU kernel changes are required.
Default value is 0.0, so existing runs are bit-for-bit unchanged.

## Decision

Add `adm_min_val` (alias `min`) to `AdmStateCuda`, `AdmStateSycl`, and
`AdmVulkanState`, register it in each backend's `options[]` table with the
same bounds (0.0–1.0) as the CPU, and apply the floor clamp immediately
after the `score = num / den` computation in each backend's collect path.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| GPU kernel clamp | Keeps all logic on device | Requires shader changes, recompile, extra push constant; overkill for a single scalar compare | Not chosen — host-side clamp is cheaper and equivalent |
| Document as intentional gap | Zero code change | Caller still silently drops the option; parity gap persists | Not chosen — gap is a genuine bug, not a design limitation |

## Consequences

- **Positive**: CUDA/SYCL/Vulkan `integer_adm` now honours `adm_min_val`
  identically to the CPU reference.
- **Negative**: None — default 0.0 is a no-op, so no regression risk.
- **Neutral**: `adm_csf_mode`, `adm_skip_aim`, `adm_dlm_weight`, and
  `adm_skip_scale0` remain open parity gaps on the same three backends;
  those require more substantial changes and are tracked separately.

## References

- CPU reference: `libvmaf/src/feature/integer_adm.c` lines 206–215.
- Parity audit: `.workingdir/feature-option-parity-audit-2026-05-16.md`.
- Related: ADR-0485 (MS-SSIM enable_db parity), ADR-0486 (aiutils dedup).
- req: "pick smallest unaddressed gap" (user session instruction 2026-05-17).
