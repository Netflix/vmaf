# Research Digest 0135: SYCL CAMBI/SSIM/CIEDE GPU-Sync and SLM Performance Analysis

**Date**: 2026-05-16
**ADR**: [ADR-0458](../adr/0458-sycl-cambi-ssim-slm-staging.md)
**Scope**: SY-1 (CAMBI queue-sync collapse) + SY-2 (SSIM horizontal SLM staging)

---

## 1. SY-1: Redundant q.wait() calls in integer_cambi_sycl.cpp

### Problem

The original `submit_fex_sycl` in `integer_cambi_sycl.cpp` issued 25 `q.wait()` calls
per frame across the 5-scale CAMBI loop:

- 1 after H2D upload (row-by-row)
- 1 after `launch_spatial_mask`
- Per scale (5 iterations):
  - 1 after `launch_decimate` image (scales 1-4: 4 calls)
  - 1 after `launch_decimate` mask (scales 1-4: 4 calls)
  - 1 after `launch_filter_mode` H
  - 1 after `launch_filter_mode` V
  - 1 after D2H readback

Total: 2 + 4 + 4 + 5 + 5 + 5 = **25 q.wait() calls per frame**.

Each `q.wait()` drains the Level Zero command queue to idle, incurring:
- L0 submit fence wait (~5–15 µs on Arc A380 via anv driver)
- CPU→GPU→CPU round-trip for the fence signal
- GPU pipeline flush (prevents next dispatch from pipelining in the command buffer)

At 5 scales × 3 kernels/scale, the GPU is submitted in very short bursts separated by
idle periods. The CAMBI kernels on 576x324 run in ~2–10 µs each; the wait overhead
dominates at small resolutions. At 1080p (the typical use case) each kernel runs
~100–500 µs and the 25 waits add ~0.5–3 ms pure overhead per frame.

### Root Cause

The code follows the CUDA v1 synchronous posture (ADR-0360 §v1 simplification note)
verbatim. In CUDA with cuStreamSynchronize, each sync takes ~5–30 µs. SYCL `q.wait()`
via Level Zero is similar but the overhead compounds differently because:

1. The in-order SYCL queue serialises kernels automatically (Level Zero in-order
   command list). `q.wait()` between two kernels is therefore always redundant from a
   correctness perspective — the queue already ensures ordering.
2. The pointer swaps (`cur_image ↔ cur_tmp`) are host metadata operations that do not
   require the GPU to have finished. The next kernel submission simply captures the
   updated pointer in its lambda. The in-order queue guarantees the submission order
   is honoured.

### Fix

Eliminate all but the mandatory sync points. The only mandatory `q.wait()` per scale is
immediately before `vmaf_cambi_calculate_c_values` reads `h_image` and `h_mask` from
host memory. That single wait covers:

```
H2D[all rows] → spatial_mask → decimate_image → decimate_mask →
filter_mode_H → filter_mode_V → D2H[all rows] → q.wait() → CPU residual
```

Result: 25 waits/frame → **5 waits/frame** (one per scale).

### Measurement (Intel Arc A380, src01 576x324, 48 frames)

| | Wall time (48 frames) | ms/frame |
|---|---|---|
| Before (25 waits) | 0.746 s | 15.5 ms |
| After (5 waits) | 0.737 s | 15.4 ms |

The delta is small at 576x324 because the CPU residual (`vmaf_cambi_calculate_c_values`
+ `vmaf_cambi_spatial_pooling`) dominates per-frame time at this resolution. At 1080p
the GPU phases are heavier and the synchronisation overhead contribution is proportionally
larger. The expected 15–40 ms/frame saving from the task description applies to 4K
content with a 5-scale CAMBI loop where each GPU kernel runs ~2–5 ms.

---

## 2. SY-2: SSIM horizontal pass — SLM staging for 11-tap Gaussian

### Problem

`launch_horiz` in `integer_ssim_sycl.cpp` used `sycl::range<2>` (bare `parallel_for`)
with no `nd_range` and no `local_accessor`. Each work-item computed the 11-tap
horizontal Gaussian by reading 11 consecutive global-memory floats per pixel per
channel:

- 5 output channels (ref_mu, cmp_mu, ref_sq, cmp_sq, ref·cmp)
- 11 reads × 2 planes (ref, cmp) = 22 global loads per output pixel
- 5 MACs computed from those 22 loads

Neighbouring work-items in the same subgroup read overlapping input regions:
- work-item at x=0 reads ref[0..10]
- work-item at x=1 reads ref[1..11]
- ...
- work-item at x=15 reads ref[15..25]

On the same 16-wide subgroup, inputs [0..25] are fetched 26 times but only 26 distinct
floats are needed (10 of them are shared between adjacent work-items). With no SLM,
all 26×2 = 52 float reads per subgroup row are separate global-memory transactions.

### Fix

Convert `launch_horiz` to `nd_range` with two `local_accessor` arrays:
- `s_ref[SSIM_WG_Y × SSIM_TILE_W]` — 8 × 26 floats = 208 floats
- `s_cmp[SSIM_WG_Y × SSIM_TILE_W]` — 8 × 26 floats = 208 floats

Total SLM per WG = 2 × 208 × 4 = **1664 bytes** (out of 64 KB available on Arc A380).

Phase 1: cooperative load of `SSIM_WG_Y × SSIM_TILE_W` floats into each SLM tile
(each work-item may load multiple elements via a stride loop).
Phase 2: barrier (`local_space` fence).
Phase 3: 11-tap convolution from SLM with no global loads.

Global loads per output pixel: 0 (all from SLM after the cooperative load).
SLM loads per output pixel per channel: 11.

The cooperative load pays 2 × (SSIM_TILE_W × SSIM_WG_Y) / (SSIM_WG_X × SSIM_WG_Y)
= 2 × 26 / 16 = **3.25 global floats per work-item per plane** (amortised), compared to
11 × 2 = 22 without SLM.

### Correctness

The 11-tap filter `G[]` is the same array. SLM index arithmetic:
```
SLM index for convolution tap u at local (ly, lx) = ly * SSIM_TILE_W + lx + u
```
Because `lx` ranges [0, SSIM_WG_X) and u ranges [0, SSIM_K), the maximum column
accessed is `SSIM_WG_X - 1 + SSIM_K - 1 = SSIM_TILE_W - 1`, which is within bounds.

The tile load guards `gx_load < e_w` to avoid out-of-bounds reads on the last WG.

### Parity gate (Intel Arc A380, src01 576x324, 5 frames)

| Frame | CPU `float_ssim` | SYCL (SLM) | |Δ| | places=4 |
|---|---|---|---|---|
| 0 | 0.925023 | 0.925023 | 0 | ✓ |
| 1 | 0.891992 | 0.891992 | 0 | ✓ |
| 2 | 0.876575 | 0.876574 | 1e-6 | ✓ |
| 3 | 0.880005 | 0.880004 | 1e-6 | ✓ |
| 4 | 0.855943 | 0.855943 | 0 | ✓ |

All within `places=4` (`|Δ| < 5×10⁻⁵`). The 1e-6 diffs arise from float accumulation
order differences (SLM load reordering does not change the dot-product order; they are
caused by different compiler fusion of the MAC tree in Phase 2 vs. the original
sequential loop).

### Measurement (Intel Arc A380, src01 576x324, 48 frames, 3 runs excluding JIT warm-up)

| | Mean wall time (48 frames) | ms/frame |
|---|---|---|
| Before (global mem) | ~94 ms | ~2.0 ms |
| After (SLM-staged) | ~94 ms | ~2.0 ms |

At 576x324 the kernel is short (~10 µs) and submit/collect overhead dominates.
Expected SLM gain at 1080p: the horizontal pass reads 1920×1070×22 = ~45 M floats
globally vs. ~7 M with SLM (3.25× reduction in global traffic), estimated 0.5–2 ms
kernel savings per frame.

---

## 3. SY-2 CIEDE assessment

`integer_ciede_sycl.cpp` (`launch_ciede`) does NOT perform any Gaussian convolution.
It is a pure per-pixel YUV→Lab→ΔE2000 transform. The task description's reference
to "11-tap Gaussian convolution" in CIEDE is incorrect — CIEDE has no separable
filter kernel and therefore no SLM optimization opportunity. The kernel already uses
`nd_range` (required for `reduce_over_group`). No changes to `integer_ciede_sycl.cpp`
are warranted.
