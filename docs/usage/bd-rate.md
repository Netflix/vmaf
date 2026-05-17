# BD-rate with VMAF

BD-rate (Bjøntegaard Delta rate) measures the average bitrate
difference at equivalent quality between two encoders. Negative
BD-rate means the test encoder is more efficient — it spends fewer
bits to reach the same VMAF/PSNR/MS-SSIM as the reference. The
metric was introduced in [Bjøntegaard 2001 (VCEG-M33)] and is the
standard codec-comparison reporting axis at JVET, AOM, and most
recent neural-video-compression (NVC) papers (e.g. Gao et al.,
*Advances in Neural Video Compression: A Review and Benchmarking*,
Bristol VI-Lab 2026, where VMAF is one of three reporting metrics
alongside PSNR and MS-SSIM).

This page documents how to run a BD-rate study with VMAF using the
fork's existing tooling.

## What you have

[`python/vmaf/tools/bd_rate.py`](../../python/vmaf/tools/bd_rate.py)
implements the Bjøntegaard piecewise-cubic-Hermite interpolation
(`pchip_interpolate`) with the integration window clamped to the
overlap region of the two RD curves, plus a convex-hull mode for
near-monotonic data. The implementation is validated against
JCTVC-E137 (the JCTVC reference numbers).

[`python/vmaf/tools/bd_rate_calculator.py`](../../python/vmaf/tools/bd_rate_calculator.py)
is the higher-level calculator class.

[`python/test/bd_rate_calculator_test.py`](../../python/test/bd_rate_calculator_test.py)
gates the math against published reference numbers; it runs in CI
under the `Netflix CPU Golden Tests (D24)` lane.

## What this fork ships beyond Netflix upstream

Same calculator. The fork-local additions are around it: a
`--precision` CLI flag (`%.17g` default, IEEE-754 round-trip
lossless — see [`docs/usage/precision.md`](precision.md)) so the
RD-point CSV your encoder run produces survives float-text
round-trip, plus the GPU + SIMD speedups that make sweeping a
BD-rate study tractable on a single workstation.

## Recipe — NVC-style BD-rate against a reference encoder

The Bristol VI-Lab 2026 review reports BD-rate against VTM-20 LD/RA
on UVG, MCL-JCV, HEVC B-E, and AOM A2-A5 with VMAF, PSNR, and
MS-SSIM as the three quality metrics (Tables 3-4 of that paper).
The protocol below mirrors that, scoped to a single sequence pair
to keep the documentation compact.

### 1 — pick the sequences

The standard reference sets:

| Set      | Source                                                                 | Notes                                                |
|----------|------------------------------------------------------------------------|------------------------------------------------------|
| UVG      | [ultravideo.fi](https://ultravideo.fi/dataset.html) (Mercat 2020)      | 1080p / 4K, 7-12 sequences, 120 frames each         |
| MCL-JCV  | [mcl.usc.edu](https://mcl.usc.edu/mcl-jcv-dataset/) (Wang 2016)        | 1080p, 30 sequences, 150 frames each                |
| HEVC B-E | [JCTVC CTC](https://www.itu.int/rec/T-REC-H.265-201906-S) (Bossen)     | 416×240 to 1080p, 5 classes                         |
| AOM A1-A5| [AOM CTC](https://aomedia.googlesource.com/aom/+/refs/heads/main/CONTRIBUTING.md) (Zhao) | A1-A5 resolutions                                   |

The fork's tiny-AI training already uses BVI-DVC + KoNViD-1k +
BVI-AOM (see [`docs/research/0019-tiny-ai-netflix-training.md`](../research/0019-tiny-ai-netflix-training.md)
and [`docs/research/0033-bristol-nvc-review-2026.md`](../research/0033-bristol-nvc-review-2026.md));
those are training corpora and don't substitute for the
codec-comparison sets above.

### 2 — encode at multiple QP/CRF rungs

Standard ladders:

- HEVC/AVC: QP `{22, 27, 32, 37}` (JVET CTC) or 6-rung `{22, 27, 32, 37, 42, 47}` for low-bitrate analysis.
- AV1: CRF `{23, 31, 39, 47, 55, 63}` (AOM CTC).
- VVC: same as HEVC plus `{42, 47}` for modern reporting.

For each rung, log:

- `bitrate_kbps` — actual coded bitrate from the bitstream (`ffprobe -show_streams` or the encoder's stat file).
- `vmaf` — score from libvmaf (this fork's CLI: `vmaf` or via FFmpeg's `libvmaf` filter).
- `psnr` — `psnr_y` is the standard reporting axis; the fork additionally emits `psnr_cb` / `psnr_cr` on the Vulkan backend with `enable_chroma=true` (see [`docs/backends/vulkan/overview.md`](../backends/vulkan/overview.md)).
- `ms_ssim` — `float_ms_ssim` (CPU AVX2/AVX-512/NEON, CUDA, SYCL, Vulkan; the fork also exposes the 15 per-scale L/C/S sub-metrics behind `enable_lcs=true`, see [`docs/metrics/features.md`](../metrics/features.md) §SSIM/MS-SSIM).

### 3 — feed the calculator

```python
from vmaf.tools.bd_rate import calculate_bd_rate

# (rate, quality) tuples — same metric on both sides.
reference = [(8650, 91.2), (4112, 86.5), (1956, 79.8), (920, 70.1)]   # x265 slow
candidate = [(7980, 91.2), (3801, 86.5), (1810, 79.8), (852, 70.1)]   # SVT-AV1

bd_rate_pct = calculate_bd_rate(reference, candidate)
print(f"BD-rate (VMAF): {bd_rate_pct:+.2f}%")
# Negative means `candidate` is more efficient than `reference`.
```

For the convex-hull variant (recommended on noisy or non-monotonic
RD points):

```python
calculate_bd_rate(reference, candidate, use_convex_hull=True)
```

The function clamps the integration window to the metric overlap
between the two curves and raises `BdRateNoOverlapException` if
they don't intersect. Use `min_metric` / `max_metric` to clamp
explicitly.

### 4 — report

For an NVC-style table you want one BD-rate column per quality
metric × test set combination. Example layout (mirroring Tables 3-4
of the Bristol paper):

| Encoder       | UVG (VMAF) | UVG (PSNR-Y) | UVG (MS-SSIM) | MCL-JCV (VMAF) | … |
|---------------|------------|--------------|----------------|-----------------|---|
| x265 slow     | 0.0        | 0.0          | 0.0            | 0.0             |   |
| SVT-AV1 P2    | -8.4%      | -3.1%        | -7.6%          | -10.1%          |   |
| VVenC slower  | -22.3%     | -16.5%       | -21.0%         | -24.8%          |   |

(Values illustrative.)

## Common pitfalls

- **Different VMAF model versions across the two columns.** Always
  pin `vmaf_v0.6.1` or whatever model your study uses; the fork
  auto-loads `vmaf_v0.6.1.json` by default. Verify with
  `vmaf --version --verbose | grep model`.
- **VBV / 1-pass vs 2-pass mismatch.** BD-rate compares like-for-like
  rate-control modes. Two-pass encodes have a different RD shape
  than 1-pass `crf=` encodes; mixing them inflates the apparent
  delta. The fork doesn't enforce this — your harness must.
- **GPU vs CPU score divergence.** CUDA / SYCL / Vulkan backends
  produce ≤ ~1e-4 ULP-different VMAF scores from the CPU reference
  (within `places=4` tolerance, but observable). For competitive
  BD-rate reporting use the same backend on both columns. See
  [`docs/research/0034-ci-pipeline-audit-2026-05.md`](../research/0034-ci-pipeline-audit-2026-05.md)
  and [ADR-0234 (proposed)](../adr/0234-gpu-gen-ulp-calibration.md) on the
  forthcoming `--gpu-calibrated` flag for cross-backend reconciliation.
- **Too few rungs.** The PCHIP interpolant degenerates with fewer
  than 4 RD points; `calculate_bd_rate` raises
  `BdRateNotEnoughPointsException` below the threshold. For a
  publication-grade NVC report, 6 rungs is the AOM CTC norm.

## References

- Bjøntegaard, G., *Calculation of Average PSNR Differences between
  RD-curves*, VCEG-M33, ITU-T VCEG, 2001.
- Bossen, F., *Common Test Conditions and Software Reference
  Configurations*, JCTVC, 2010.
- Gao et al., *Advances in Neural Video Compression: A Review and
  Benchmarking*, Bristol VI-Lab preprint 2026 (audited in
  [`docs/research/0033-bristol-nvc-review-2026.md`](../research/0033-bristol-nvc-review-2026.md)).
- Mercat et al., *UVG dataset: 50/120fps 4K sequences for video
  codec analysis and development*, MMSys 2020.
- Wang et al., *MCL-JCV: a JND-based H.264/AVC video quality
  assessment dataset*, ICIP 2016.
- Zhao et al., *AOM Common Test Conditions v3*, AOM, 2022.

[Bjøntegaard 2001 (VCEG-M33)]: https://www.itu.int/wftp3/av-arch/video-site/0104_Aus/VCEG-M33.doc
