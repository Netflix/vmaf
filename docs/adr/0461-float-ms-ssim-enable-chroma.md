# ADR-0461: Add `enable_chroma` option to the `float_ms_ssim` extractor

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: ms-ssim, float-ms-ssim, option-parity, metrics, correctness, fork-local

## Context

The feature-option parity audit of 2026-05-16 identified that the CPU
floating-point MS-SSIM extractor (`libvmaf/src/feature/float_ms_ssim.c`,
registered as `float_ms_ssim`) always computed MS-SSIM only on the luma
plane and emitted a single `float_ms_ssim` score per frame. No option
existed to request per-plane chroma scores.

By contrast, the CPU PSNR extractor (`integer_psnr.c`) had already gained
`enable_chroma` support (PR #895, ADR-0453) with a three-plane loop emitting
`psnr_y`, `psnr_cb`, and `psnr_cr`. The MS-SSIM extractor had no equivalent,
making per-channel MS-SSIM for YUV content unavailable from the fast CPU path.

The `float_ms_ssim` extractor uses the IQA library's Gaussian-window
floating-point implementation via the `compute_ms_ssim()` helper in
`ms_ssim.c`. The function takes `(ref, dist, w, h, ref_stride, cmp_stride, …)`
and returns a single scalar score. Because chroma planes have distinct
width/height from luma in 4:2:0 and 4:2:2 content, running the function per
plane requires passing per-plane dimensions; the luma-sized float buffers
allocated in `MsSsimState` are large enough to hold any chroma plane.

## Decision

Add `bool enable_chroma` to `MsSsimState` with a `VmafOption` entry
(default `false` — backward-compatible luma-only output). In `init()`,
clamp `enable_chroma` to `false` when `pix_fmt == VMAF_PIX_FMT_YUV400P`.
In `extract()`, replace the hard-coded single-plane call with a loop over
`n_planes = enable_chroma ? 3 : 1` planes, emitting `float_ms_ssim` (Y),
`float_ms_ssim_cb` (Cb), and `float_ms_ssim_cr` (Cr) using
`ms_ssim_feature_names[]`. Compute the per-plane float stride inline
with `ALIGN_CEIL(ref_pic->w[p] * sizeof(float))` so chroma planes use
the correct stride. Restrict `enable_lcs` output to the luma pass only
(`p == 0`) because the LCS component names do not carry a plane suffix.
Update `provided_features` to advertise all three names.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Default `enable_chroma=true` (parity with PSNR default) | Consistent with `integer_psnr.c` default | Breaking change: callers that parse exactly one `float_ms_ssim` score per frame would receive three. Model JSON files referencing `float_ms_ssim` would need updates. | Backward-compatibility risk outweighs parity benefit; SSIM defaults to `false` per ADR-0460 |
| Add LCS chroma outputs too | More complete chroma statistics | LCS names would need `_cb`/`_cr` suffixes and matching `provided_features` entries; significant scope expansion; low demand | Deferred to follow-up if user demand arises |
| Allocate separate chroma float buffers in `init()` | Avoid per-plane stride recalculation | Adds complexity, separate alloc/free, extra init-time cost | Unnecessary: luma buffers are always large enough to hold chroma planes; chroma stride is cheap to compute inline |
| No change; document the limitation | Zero risk | User-discoverable gap persists | Rejected; the fix is small and contained |

## Consequences

- **Positive**: Per-plane MS-SSIM (`float_ms_ssim_cb`, `float_ms_ssim_cr`) is
  now available from the CPU floating-point path when `enable_chroma=true`.
- **Negative**: None. The luma-only default path is bit-for-bit unchanged.
- **Neutral / follow-ups**: GPU twins (`float_ms_ssim_cuda`, `float_ms_ssim_sycl`,
  `float_ms_ssim_vulkan`) still lack `enable_chroma`. Tracked as follow-up
  in the option-parity backlog.

## References

- `libvmaf/src/feature/float_ms_ssim.c` — implementation
- `libvmaf/src/feature/integer_psnr.c` — `enable_chroma` pattern precedent (ADR-0453)
- [ADR-0460](0460-ssim-enable-chroma.md) — `enable_chroma` for integer SSIM (PR #939)
- req: user brief 2026-05-16 ("Apply enable_chroma pattern to float_ms_ssim and
  integer_ms_ssim extractors; same pattern as psnr enable_chroma PR#895 and PR#939")
