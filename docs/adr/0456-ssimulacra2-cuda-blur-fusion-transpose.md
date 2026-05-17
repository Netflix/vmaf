# ADR-0456: SSIMULACRA2 CUDA Blur: 3-Channel Kernel Fusion and V-Pass Transpose for Coalesced Access

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `cuda`, `perf`, `ssimulacra2`

## Context

The SSIMULACRA2 CUDA extractor (`ssimulacra2_cuda.c` + `ssimulacra2_blur.cu`) processes a
6-scale IIR Gaussian pyramid with 5 separable blur operations per scale. Before this ADR,
`ss2c_blur_3plane` looped over 3 XYB channels and issued one H-pass kernel + one V-pass kernel
per channel per blur call:

- 3 channels × 2 passes = 6 kernel launches per `ss2c_blur_3plane` call
- 5 blurs per scale × up to 6 scales = 30 blur calls per frame
- Total: 6 × 30 = **180 kernel launches per frame**

On RTX 4090, CUDA kernel launch overhead is approximately 2 µs per launch (driver dispatch +
command stream submission). 180 launches contributes ~360 µs of pure driver overhead per frame,
before any GPU compute.

Additionally, the V-pass kernel (`ssimulacra2_blur_v`) reads `in_buf[offset + row * width + col]`
where `row` increments by 1 per IIR step. For a 1920-wide frame, successive reads within a
single thread are 7680 bytes apart. While warp-level coalescing is maintained (32 adjacent
threads for a given IIR step read 32 consecutive floats), the per-thread access stride means
that a single column IIR scan of height=1080 steps touches 1080 distinct cache lines. The
entire scratch buffer (1920 × 1080 × 4 bytes = 7.9 MB) exceeds L1 (256 KB/SM on Ada), causing
repeated L2 traffic.

## Decision

We implement two complementary optimisations in `ssimulacra2_blur.cu` and `ss2c_blur_3plane`:

**Change 1 — 3-channel kernel fusion via `gridDim.z`:** Replace the per-channel loop with
two fused kernels (`ssimulacra2_blur_h3`, `ssimulacra2_blur_v3_transposed`) that use
`blockIdx.z ∈ {0, 1, 2}` to select the XYB channel. A single kernel launch handles all three
planes concurrently. Combined with the transpose launch (see Change 2), this reduces per-blur
launches from 6 to 3, and per-frame launches from 180 to 90.

**Change 2 — V-pass coalescing via in-place transpose:** Before each V-pass, run a transpose
kernel (`ssimulacra2_transpose`) that converts the H-pass output from row-major to column-major
layout in a separate scratch buffer (`d_transpose_buf`). The V-pass then reads
`transposed[c × plane_stride + col × height + row]`, so successive rows for a fixed `col` are
consecutive addresses. The V-pass writes back to row-major directly.

The transpose kernel uses a `float tile[32][33]` shared-memory tile with a +1 column pad to
avoid 32-way bank conflicts on the column-wise store phase. Block shape is 32×32 (1024 threads),
and `gridDim.z = 3` fuses all three channels in a single dispatch.

Bit-exactness is maintained: the IIR recurrence is identical to the original code — same scan
order, same operator sequence, same `--fmad=false` constraint in the meson build. The transpose
changes memory layout without changing values delivered to the IIR.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Channel fusion only (no transpose) | Simple; cuts launches 3× to 60 | V-pass still stride-width per thread; cache behavior unchanged | Leaves the larger V-pass L2 pressure unaddressed |
| Transpose only (no channel fusion) | Reduces per-thread cache lines in V-pass | Adds 1 launch per V-pass; net is +3 launches vs old 6 (worse) without fusion | Must combine with fusion for net improvement |
| Rewrite V-pass with warp-shuffle to scan multiple columns per block | Eliminates transpose overhead entirely | Warp-shuffle IIR requires complex state-sharing; diverges from CPU scalar path making parity audit harder | Fusion + transpose delivers the target speedup with simpler correctness argument |
| CUB / cooperative groups block-scan for V-pass | Library primitives for parallel prefix | IIR is not a standard prefix operation (three independent recurrences per step); CUB block_scan does not directly express this | Not applicable without restructuring the IIR algorithm |

## Consequences

- **Positive**: Per-frame kernel launch count for ssimulacra2 drops from 180 to 90 (launch
  overhead: 360 µs → 180 µs). V-pass reads convert from stride-width per-thread to stride-1
  sequential within a column. One additional device buffer (`d_transpose_buf`, same size as
  `d_blur_scratch`) is allocated per extractor instance.
- **Negative**: Memory allocation increases by one full-resolution 3-plane float buffer
  (~8 MB at 1080p, ~32 MB at 4K). The transpose launch cost partially offsets the launch
  savings from channel fusion; the net is still a reduction.
- **Neutral / follow-ups**: The original single-channel `ssimulacra2_blur_h` and
  `ssimulacra2_blur_v` symbols are retained in the fatbin for future use or debugging but are
  no longer invoked by the dispatch path. The cross-backend parity gate (`places=4`) must pass
  after this change; verified at places=6 (zero diff) on the 576x324 test pair.

## References

- ADR-0192 — GPU long-tail batch 3 (ssimulacra2 CUDA port)
- ADR-0201 — ssimulacra2 Vulkan port precision contract (same `--fmad=false` requirement)
- ADR-0214 — GPU parity CI gate
- [Research digest](../research/0135-ssimulacra2-cuda-perf-2026-05-16.md) — measured before/after frame times
- req: "implement the SSIMULACRA2 blur dispatch consolidation + V-pass transpose"
