# ADR-0461: Add `enable_chroma` option to `psnr_hvs_vulkan`

- **Status**: Accepted
- **Date**: 2026-05-16
- **Tags**: psnr-hvs, vulkan, option-parity, metrics, fork-local

## Context

The `psnr_hvs_vulkan` extractor always dispatched all three planes and
always emitted four features: `psnr_hvs_y`, `psnr_hvs_cb`, `psnr_hvs_cr`,
and the combined `psnr_hvs` (0.8*Y + 0.1*(Cb+Cr)).  It had no way to
restrict computation to luma only, unlike `psnr_vulkan` which gained
`enable_chroma` in ADR-0453.

Use cases that require only `psnr_hvs_y` (e.g., luma-only quality
monitoring pipelines, HDR workflows where chroma is measured separately)
paid the cost of two extra GPU dispatches and two extra feature
collector appends per frame with no way to opt out.

## Decision

Add a `bool enable_chroma` field to `PsnrHvsVulkanState` with
`default_val.b = true`, matching the ADR-0453 / `psnr_vulkan` precedent.

When `enable_chroma=false`:
- `n_planes` is set to 1 in `init()`.
- The chroma pipeline variants (`pipeline_chroma_u`, `pipeline_chroma_v`)
  are not created in `create_pipeline()`.
- Only the luma buffers are allocated in `alloc_buffers()`.
- Only one descriptor set is allocated in `init()`.
- `extract()` uploads, dispatches, and reads back only plane 0.
- Only `psnr_hvs_y` is appended to the feature collector.
- The combined `psnr_hvs` score is suppressed (it is undefined without
  chroma inputs).

The default (`enable_chroma=true`) is backward-compatible: all existing
callers that do not pass the option see identical behaviour.

## Alternatives considered

| Option | Reason rejected |
|--------|----------------|
| Default `enable_chroma=false` | Would be a breaking change for all existing callers that rely on `psnr_hvs_cb`, `psnr_hvs_cr`, and `psnr_hvs` being emitted |
| Add to CPU `psnr_hvs` extractor instead | The CPU extractor always needs all three planes for the DCT-based HVS metric; the GPU path's per-plane dispatch design makes per-plane opt-out natural and low-cost |
| Suppress combined score but still dispatch chroma | Inconsistent; wastes GPU cycles for data that is not consumed |

## Consequences

- **Positive**: Callers can pass `--feature psnr_hvs_vulkan:enable_chroma=0`
  to reduce per-frame GPU work to one dispatch and one feature emit.
- **Neutral**: The combined `psnr_hvs` score is silently omitted when
  `enable_chroma=false`; callers that require it must keep chroma enabled.
- **No change** to default behaviour, golden-gate assertions, or snapshot
  JSONs.

## Files changed

- `libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c` — implementation

## References

- ADR-0453: `psnr_vulkan` `enable_chroma` pattern precedent
- req: user direction 2026-05-16 ("Mirror PR #956 (SSIM Vulkan) —
  add enable_chroma to psnr_hvs Vulkan twin")
