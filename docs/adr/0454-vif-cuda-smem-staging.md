# ADR-0454: VIF CUDA shared-memory staging for horizontal and vertical filter passes

- **Status**: Proposed
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: `cuda`, `gpu`, `vif`, `performance`, `smem`, `fork-local`

## Context

The VIF feature extractor is the dominant cost in a full VMAF CUDA run
(~60% of frame time at 1080p). Its two-pass separable filter
(`filter1d_8/16_vertical_kernel` → `filter1d_8/16_horizontal_kernel`) reads
input data entirely from L2/DRAM on every filter tap with no explicit data
reuse:

- **Horizontal pass**: 7 tmp channels × 17 taps = 119 global 32-bit reads per
  output pixel, even though all threads in a block share the same 256-wide
  row slice. Each warp issues up to 17 cache-line misses into L2 per tap group.

- **Vertical pass**: 17 global row reads × 2 planes per output row, with
  adjacent blocks in Y sharing 16 of those 17 rows — all loaded redundantly.

The perf-audit (`perf-audit-cuda-2026-05-16.md`) identified these as wins #1
and #4 with estimated speedups of 20–35% (horizontal) and 15–25% (vertical)
on a VIF-only run.

## Decision

Stage input data into `__shared__` memory before the filter accumulation loop
in all four filter template functions:

- `filter1d_8_vertical_kernel`: tile `(BLOCKY + fwidth_0 - 1) × 128` per plane
  (uint8_t). Block-cooperative load, single `__syncthreads()`.
- `filter1d_8_horizontal_kernel`: tile 273 uint32_t per tmp channel × 7
  channels. Block-cooperative load into 7 `__shared__` arrays.
- `filter1d_16_vertical_kernel`: tile `(BLOCKY + fwidth - 1) × 128` per plane
  (uint16_t).
- `filter1d_16_horizontal_kernel`: same layout as 8-bit horizontal.

The boundary mirror (reflect at `y=0`, `y=h-1`, `x=0`, `x=w-1`) is applied in
the smem load phase; the compute phase reads smem unconditionally.
All arithmetic is unchanged — results are bit-identical to the pre-patch kernel.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Texture cache (read-only path via `__ldg`) | Zero code change to tile logic | Only improves L2 hit-rate; does not eliminate redundant loads across threads | Does not address inter-thread reuse within the block |
| Fused vertical + horizontal single kernel | Eliminates 7 tmp buffer round-trips | 17-tap × 7-channel smem would exceed 48 KB at full block occupancy; requires 2D block restructuring | Deferred to follow-up; the tmp buffers allow pipelined execution today |
| Half-width tile to reduce smem | Halves smem per block, doubles occupancy | Halo threads still issue global loads; the saving is marginal for the dominant interior path | Not worth the indexing complexity |
| `cudaFuncSetCacheConfig(cudaFuncCachePreferL1)` | Zero code change | Moves L1/L2 boundary but does not change the access pattern | Insufficient for 17-tap reuse distance |

## Consequences

- **Positive**: Horizontal kernel reduces unique L2 reads by 7–8× for the
  interior path (7 smem loads replace 7 × 17 global loads per output pixel).
  Vertical kernel eliminates 16 of 17 redundant row loads per warp across
  adjacent blocks in Y. Estimated combined speedup: 15–35% on VIF.
- **Negative**: Smem budget per block increases by ~7.6 KB (horizontal) and
  ~5–10 KB (vertical). At BLOCKX=128, BLOCKY=1 for horizontal and
  BLOCKX=32, BLOCKY=4 for vertical, the SM can still host 6+ active blocks
  per SM (48 KB / 7.6 KB ≈ 6). No occupancy regression expected.
- **Neutral**: The `__syncthreads()` adds one barrier per kernel invocation.
  For 1080p this is ~135 barriers per pass (1920/256 × 1080 blocks) — a
  negligible overhead vs. the eliminated L2 traffic.

## References

- Research digest: [Research-0135](../research/0135-vif-cuda-smem-staging-2026-05-16.md)
- Perf audit: `.workingdir/perf-audit-cuda-2026-05-16.md` wins #1, #4.
- ADR-0214 (GPU parity CI gate — `places=4` must pass).
- NVIDIA CUDA C Best Practices Guide §"Shared Memory".
