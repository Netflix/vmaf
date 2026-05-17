# ADR-0471: Add `enable_chroma` to `integer_psnr_hip` (chroma parity with CUDA/SYCL/Vulkan twins)

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: hip, psnr, option-parity, chroma, fork-local

## Context

ADR-0453 added `enable_chroma` (and full chroma dispatch) to the three GPU PSNR
twins that existed at the time: `integer_psnr_cuda.c`, `integer_psnr_sycl.cpp`, and
`psnr_vulkan.c`. The HIP twin (`integer_psnr_hip.c`) was not included in that scope
because it was still listed as luma-only with a follow-up note in ADR-0372
("Chroma extension is a follow-up").

As a result the HIP extractor:
- only dispatched and emitted `psnr_y`;
- did not advertise `psnr_cb` / `psnr_cr` in `provided_features`;
- silently dropped any `enable_chroma=false` caller intent;
- diverged from all other GPU PSNR twins.

The HIP kernel (`psnr_score.hip`) is plane-agnostic and accepts arbitrary
(pointer, stride, width, height) arguments, so chroma can be dispatched by
invoking the same kernel three times — identical to the CUDA pattern.

## Decision

Mirror the ADR-0453 fix pattern in `integer_psnr_hip.c`:

1. Add `bool enable_chroma` to `PsnrStateHip` with a matching `VmafOption`
   (default `true`).
2. Compute per-plane geometry in `init()` from `pix_fmt` (ceiling division for
   4:2:0 / 4:2:2; same as CUDA twin). Apply `enable_chroma` guard: clamp
   `n_planes` to 1 when false or when `pix_fmt == YUV400P`.
3. Allocate per-plane readback pairs `rb[PSNR_NUM_PLANES]` and staging buffers
   `ref_in[3]` / `dis_in[3]` (luma + up to two chroma planes).
4. In `submit()`, loop over `n_planes` — one HtoD copy + one kernel dispatch
   per plane.
5. In `collect()`, loop over `n_planes` — read each plane's SSE from its
   pinned host slot and emit the score under `psnr_name[p]`.
6. Update `provided_features` to `{"psnr_y", "psnr_cb", "psnr_cr", NULL}`.
7. Update `n_dispatches_per_frame` from 1 to 3.

No changes to the `.hip` kernel source — it is already plane-agnostic.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep HIP luma-only; document as known gap | No code change | Permanent backend divergence; callers using `psnr_cb`/`psnr_cr` on HIP silently get no output | Masks the gap rather than closing it |
| Add a new chroma-specific HIP kernel | Could be optimised for chroma subsampling | Extra kernel maintenance; the existing kernel is already generic | Unnecessary complexity |

## Consequences

- **Positive**: `integer_psnr_hip` now provides full PSNR parity with the
  CUDA/SYCL/Vulkan twins: `psnr_y`, `psnr_cb`, `psnr_cr` all emitted at
  `enable_chroma=true`; luma-only at `enable_chroma=false` or YUV400P.
- **Negative**: None. The default path (`enable_chroma=true`) dispatches three
  kernels per frame instead of one, but each is a small reduction — consistent
  with the CUDA twin's profile.
- **Neutral**: The `.hip` kernel source is unchanged; no HSACO rebuild required
  beyond the normal hipcc pipeline.

## References

- [ADR-0453](0453-psnr-enable-chroma-gpu-parity.md) — original CUDA/SYCL/Vulkan fix
- [ADR-0372](../adr/0372-integer-psnr-hip-real-kernel.md) — HIP PSNR kernel promotion (noted chroma as follow-up)
- CPU reference: `libvmaf/src/feature/integer_psnr.c` (init geometry + enable_chroma guard)
- CUDA twin: `libvmaf/src/feature/cuda/integer_psnr_cuda.c`
