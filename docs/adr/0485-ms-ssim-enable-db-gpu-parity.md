# ADR-0485: Wire `enable_db` / `clip_db` into the CUDA and SYCL MS-SSIM twins

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: cuda, sycl, ms-ssim, option-parity, bug, fork-local

## Context

The CHUG-extraction audit of 2026-05-16
(`.workingdir/audit-chug-extraction-cuda-2026-05-16.md`) found that the
CPU `float_ms_ssim` extractor exposes two output-mode options — `enable_db`
(emit the score in dB via `-10 log₁₀(1 − MSSSIM)`) and `clip_db` (clamp
the dB value to a finite per-frame ceiling) — that were absent from both
GPU twins (`float_ms_ssim_cuda` / `integer_ms_ssim_cuda.c` and
`float_ms_ssim_sycl` / `integer_ms_ssim_sycl.cpp`).

Without `enable_db`, a caller that sets the option on a GPU backend
receives no warning and gets the raw linear score, silently diverging from
the CPU path. The audit classified the gap as LOW severity (the default
value is `false` so the default-path output is unaffected), but it is a
straightforward parity fix that follows the established ADR-0453 pattern
(PSNR `enable_chroma` GPU parity, PR #880).

## Decision

For both `integer_ms_ssim_cuda.c` and `integer_ms_ssim_sycl.cpp`:

1. Add `bool enable_db`, `bool clip_db`, and `double max_db` fields to the
   state struct.
2. Add matching `VmafOption` entries to the options table (`enable_db`,
   `clip_db`; `default_val.b = false`).
3. In `init()`, compute `max_db` from the BPC and frame geometry using the
   same formula as the CPU path
   (`ceil(10 log₁₀(peak² / (0.5 / (w × h))))`), or set `max_db = ∞` when
   `clip_db = false`.
4. In `collect()`, after the per-scale Wang product combine, apply the dB
   conversion when `enable_db = true`:
   `min(-10 log₁₀(1 − msssim), max_db)`.

No kernel code is changed. Bit-exactness at the default `enable_db=false`
is guaranteed by construction.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Stub returning `-ENOTSUP` for `enable_db=true` | Simple, no logic needed | Silent divergence replaced with an error — still not parity | Caller cannot use GPU path in dB mode |
| Defer to a follow-up PR | Zero risk in this PR | Gap remains open; ADR-0108 deliverables rule requires closing found bugs | No justification to defer a two-field wiring |

## Consequences

- **Positive**: `enable_db=true` / `clip_db=true` produce identical output
  on CPU, CUDA, and SYCL. GPU-parity gate (ADR-0214) passes for both
  option settings.
- **Negative**: None. Default-path output is bit-for-bit unchanged.
- **Neutral / follow-ups**: Vulkan MS-SSIM twin (`ms_ssim_vulkan.c`) should
  receive the same wiring in a follow-up; not included here as the Vulkan
  extractor is not part of the CHUG GPU pass.

## References

- CPU reference: `libvmaf/src/feature/float_ms_ssim.c` lines 67–79
  (options), 131–136 (init), 185–186 (collect).
- [ADR-0453](0453-psnr-enable-chroma-gpu-parity.md) — PSNR `enable_chroma`
  option-parity precedent.
- [ADR-0214](0214-gpu-parity-ci-gate.md) — GPU-parity CI gate.
- `.workingdir/audit-chug-extraction-cuda-2026-05-16.md` — audit finding
  (table row: `float_ms_ssim_cuda` / `enable_db` / LOW).
- req: user brief 2026-05-17 ("Pick smallest unaddressed CHUG blocker NOT
  done — #1077 ms_ssim enable_db.")
