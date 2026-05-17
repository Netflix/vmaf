# ADR-0469: `float_psnr` HIP twin — wire `enable_chroma` option

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: hip, psnr, option-parity

## Context

ADR-0453 added `enable_chroma` to the three CUDA/SYCL/Vulkan PSNR GPU
twins. The HIP twin (`float_psnr_hip.c`) was not in scope for that PR
and retained an empty `VmafOption options[]` table.

When a caller passes `enable_chroma=false` to the HIP extractor, the
unknown-option path silently drops the flag. Although `float_psnr` is
luma-only today (no chroma dispatch), the silent drop violates the
option-parity contract: callers must be able to set `enable_chroma`
on any PSNR extractor and receive consistent behavior across backends.

## Decision

Add `bool enable_chroma` to `FloatPsnrStateHip` and a matching
`VmafOption` entry (name, help, offset, type, `default_val.b = true`)
to the HIP extractor's options table. The `init()` function documents
that `float_psnr` is luma-only, so the field is wired but does not
gate a chroma dispatch. This mirrors the ADR-0453 pattern and ensures
the option is not silently dropped.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| No change | Zero effort | Silent option drop; violates option-parity | Incorrect |
| Add option + full chroma dispatch | Full parity with integer_psnr | float_psnr has no chroma kernel; out of scope | Follow-up item |

## Consequences

- **Positive**: callers setting `enable_chroma=false` on the HIP extractor
  no longer silently lose the flag; option-parity with CUDA/SYCL/Vulkan twins.
- **Negative**: none.
- **Neutral / follow-ups**: if a future PR adds chroma planes to
  `float_psnr_hip`, the guard in `init()` is already present.

## References

- ADR-0453: PSNR `enable_chroma` option parity across CUDA/SYCL/Vulkan.
- `integer_psnr_cuda.c` as the pattern source.
- PR user direction: paraphrased — add `enable_chroma` to the first
  applicable HIP twin, mirroring the CUDA SSIM PR #950 pattern.
