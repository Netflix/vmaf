# ADR-0490: float_ms_ssim Metal port

- **Status**: Accepted
- **Date**: 2026-05-17
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `metal`, `ms-ssim`, `float`, `apple-silicon`, `fork-local`

## Context

The Metal backend (ADR-0421) shipped eight feature extractors across T8-1b through
T8-1k. The `float_ms_ssim` metric — the float-precision 5-scale MS-SSIM used by the
VMAF model — had Metal twins on CUDA (ADR-0243), SYCL (ADR-0486), and Vulkan but not
Metal, leaving a coverage gap in the Metal backend's dispatch table.

The CPU source (`float_ms_ssim.c`) uses a separable 11-tap Gaussian on each of 5
pyramid levels built from a 9-tap biorthogonal 9/7 LPF decimation, accumulates per-pixel
L/C/S SSIM components, and reduces to a single weighted product using Wang (2003)
weights. The CUDA twin (`integer_ms_ssim_cuda.c` / `ms_ssim_score.cu`) provides a
proven GPU algorithm layout that translates directly to MSL.

## Decision

Port `float_ms_ssim` to Metal using three MSL kernels (`ms_ssim_decimate`,
`ms_ssim_horiz`, `ms_ssim_vert_lcs`) in `float_ms_ssim.metal` and a host wrapper in
`float_ms_ssim_metal.mm`. The implementation follows the same Shared-storage
unified-memory pattern as every other Metal extractor and re-uses the existing
`VmafMetalKernelLifecycle` / `VmafMetalContext` infrastructure. Wang weights and
final product reduction are computed host-side in double precision, matching the CUDA
twin.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Re-use `integer_ms_ssim.metal` kernel names from another branch | Avoids duplicate MSL source | Naming conflict with integer variant if merged; ADR-0436 integer path uses different input format | Fresh float-named kernels are cleaner and avoid merge conflicts |
| Integer-precision Metal port instead | Matches CUDA integer twin naming | CPU `float_ms_ssim` is float-precision; a float Metal twin gives closer CPU parity | Float precision is the correct target for this CPU metric |

## Consequences

- **Positive**: `float_ms_ssim` can now dispatch to Metal on Apple Silicon, completing
  the float-metric column of the Metal backend. The `dispatch_strategy.c` table and
  feature-extractor registry both include `float_ms_ssim_metal`.
- **Negative**: Adds one `.mm` + one `.metal` TU to the Metal build; `xcrun metal`
  compile step grows by one `.air` file.
- **Neutral / follow-ups**: GPU parity validation via `/cross-backend-diff
  float_ms_ssim` recommended before any production promotion claim. The
  `enable_lcs` option is supported; `enable_chroma` and `enable_db`/`clip_db` are
  not wired (luma-only, matching the initial CUDA twin scope).

## References

- CPU source: `libvmaf/src/feature/float_ms_ssim.c`
- CUDA twin: `libvmaf/src/feature/cuda/integer_ms_ssim/`
- Existing Metal twin pattern: `feature/metal/float_ssim_metal.mm` (ADR-0421)
- ADR-0153: min-dim guard (176 × 176)
- ADR-0421: Metal backend feature extractor infrastructure
- req: "Implement `float_ms_ssim_metal` backend — port of the CPU float-precision
  MS-SSIM to Metal."
