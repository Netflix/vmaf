# ADR-0484: Metal `float_ssim` option parity — `enable_lcs`, `enable_db`, `clip_db`, `scale`

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `metal`, `ssim`, `option-parity`, `apple-silicon`, `kernel`, `fork-local`

## Context

The Metal `float_ssim_metal` extractor shipped in ADR-0421 with a hardwired
two-pass Gaussian+SSIM kernel that produces a single per-frame SSIM score.
The CPU reference (`float_ssim.c`) exposes four user-facing options:

- `enable_lcs` — emit per-frame luminance (`float_ssim_l`), contrast
  (`float_ssim_c`), and structure (`float_ssim_s`) sub-scores alongside the
  composite SSIM.
- `enable_db` — convert the SSIM score to decibels: `-10·log10(1−SSIM)`.
- `clip_db` — clamp the dB output to a finite maximum derived from frame
  dimensions and bit depth.
- `scale` — decimation scale factor (0=auto, 1=no downscaling).

HIP (`float_ssim_hip.c`, ADR-0374) added `scale` only.  Neither HIP nor Metal
had `enable_lcs`, `enable_db`, or `clip_db`.  An audit of the Metal extractor
against the CPU reference identified this as a gap that makes the Metal path
unsuitable as a drop-in for downstream consumers that rely on the `float_ssim_l`
/ `float_ssim_c` / `float_ssim_s` sub-scores or the dB output form.

## Decision

Extend `float_ssim_metal.mm` and the companion `float_ssim.metal` kernel to
reach full CPU option parity:

1. Add `enable_lcs`, `enable_db`, `clip_db`, and `scale` to the `VmafOption`
   array in the host `.mm`.
2. Extend `float_ssim_vert_combine` in the Metal shader to accept a `lcs_flags`
   uniform and an `lcs_parts` output buffer (3 × partials\_count floats for
   L, C, S partial sums).  When `lcs_flags == 0` the kernel falls through the
   existing single-sum path with no extra ALU cost.
3. Apply `enable_db` / `clip_db` host-side in `collect_fex_metal` after the
   partial-sum reduction — identical to the CPU `convert_to_db()` helper.
4. Validate `scale`: v1 rejects `scale > 1` at `init` time with `-EINVAL` and
   a log message, matching the HIP twin's posture.
5. Extend `provided_features[]` to declare `float_ssim_l`, `float_ssim_c`,
   `float_ssim_s` (conditionally emitted when `enable_lcs == true`).

The kernel change adds one `[[buffer(5)]]` (`lcs_parts`) and one
`[[buffer(6)]]` (`lcs_flags`) argument to `float_ssim_vert_combine`.  When
`enable_lcs == false` the host binds the dummy `par_buf` to slot 5 and passes
`lcs_flags = 0`; the kernel short-circuits the LCS path with a branch on the
uniform, adding zero output writes and negligible ALU overhead.

## Alternatives considered

- **Host-only LCS decomposition** — decompose l/c/s from the aggregated SSIM
  partial sums post-collection.  Not feasible: the reduction discards the
  intermediate per-pixel mu and sigma values needed for the sub-score
  formulae; the decomposition would be numerically different from the
  pixel-level computation.
- **Separate LCS kernel pass** — third dispatch writing per-pixel l/c/s, then
  a fourth reduction pass.  Two extra dispatches per frame; the current
  approach adds zero dispatches (the LCS sums are accumulated in the same
  `float_ssim_vert_combine` threadgroup reduction that already runs).
- **Match HIP — skip LCS, add scale only** — keeps parity with HIP but leaves
  the Metal path below CPU parity.  Rejected: the user request is full CPU
  parity, and the kernel change is contained.

## References

- req: "audit float_ssim_metal.mm; if missing features (enable_db/enable_lcs/clip_db etc per PR #969 pattern), add them"
- ADR-0421 — Metal first kernel batch (T8-1c through T8-1j)
- ADR-0374 — HIP float_ssim (scale-only parity precedent)
- ADR-0453 — PSNR enable_chroma GPU parity (same option-parity pattern)
