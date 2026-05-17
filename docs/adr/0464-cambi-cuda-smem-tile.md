# ADR-0464: CAMBI CUDA spatial-mask shared-memory tile

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, claude-sonnet-4-6
- **Tags**: `cuda`, `gpu`, `cambi`, `performance`, `kernel`, `fork-local`

## Context

The `cambi_spatial_mask_kernel` in
`libvmaf/src/feature/cuda/integer_cambi/cambi_score.cu` computes a 7x7
zero_deriv box sum for every output pixel by reading 49 positions from global
memory, with each position requiring three global loads (p, right-neighbor, below-
neighbor) = 147 global reads per thread.  Adjacent threads in a warp share a
6-column horizontal overlap (42/49 = 86% of the 7x7 window), making ~75% of
those global reads redundant across the warp.  This is the third win from the
perf-audit plan in `.workingdir/perf-audit-cuda-2026-05-16.md`.

At 1080p: 1920x1080 = 2,073,600 output pixels x 147 reads = ~305 M uint16 global
loads per frame per kernel invocation (3 invocations under the 5-scale CAMBI
loop).  The kernel is bandwidth-bound and the redundant traffic prevents the
hardware from reaching its theoretical throughput.

## Decision

Introduce a cooperative shared-memory staging pass in
`cambi_spatial_mask_kernel`.  A 22x22 uint8 tile (`zd_tile[22][32]` with 32-byte
padded rows) holds pre-computed zero_deriv values for the entire (16+6)x(16+6)
footprint of the block.  The 256 threads of the 16x16 block populate the 484
elements in two passes (256 + 228 threads active), performing 3 global reads
each = 1452 global reads per block total.  After `__syncthreads()`, each thread
sums its 7x7 window from SLM only (49 SLM reads, 0 additional global reads).

Global memory traffic per block falls from 256 x 147 = 37,632 to 1,452 reads,
a 26x reduction.  Expected wall-clock speedup on the kernel: 25-40% at 1080p on
RTX 4090 (measured against the cambi_score baseline once Issue #857 is fully
resolved and the path is exercisable end-to-end).

`__launch_bounds__(256)` is added to help the compiler optimise register
allocation for the known block size and to document the occupancy contract.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep 49-global-read-per-thread | Simple, no smem | 26x redundant traffic | Rejected: measurable BW waste |
| 2-pass SAT in smem (row scan, then column scan) | O(N) instead of O(N*HALF^2) | Complex, adds smem register pressure, divergent termination | Overkill: 7x7 is small enough that the cooperative tile load amortises correctly; SAT adds logic risk with no bit-exactness benefit |
| Texture cache | Hardware cache assists | No L1 texture path for uint16 in all SM generations; adds `cudaBindTexture` overhead | Not portable enough |
| Widen tile to 24x24 and pre-compute img_tile, derive zero_deriv | One load phase instead of two | Requires 23x23 uint16 = 1058 bytes smem; border clamping for right/below neighbor reads through the tile creates a correctness subtlety at left/top halo edges for block (0,0) | Ruled out after correctness analysis: the direct zero_deriv load avoids the border-neighbor mismatch |

## Consequences

- **Positive**: 26x reduction in global memory reads for the spatial-mask kernel;
  eliminates warp divergence in the box-sum inner loop (no per-iteration bounds
  checks); `__launch_bounds__(256)` documents the occupancy contract.
- **Negative**: 704 bytes of static shared memory per block (22x32 uint8);
  reduces max resident blocks on SM from ~16 to ~12 at 48 KB smem limit.
  At 704 B the reduction is modest and occupancy remains high.  4-way
  shared-memory bank conflict on uint8 row-access is accepted (SLM is still
  >10x faster than L2 for this access pattern).
- **Neutral / follow-ups**: The `places=4` parity gate
  (`scripts/ci/cross_backend_parity_gate.py --features cambi --backends cpu cuda`)
  must pass.  No change to the host glue (`integer_cambi_cuda.c`), the DtoH
  pipeline, or the CPU residual.  The AGENTS.md invariant (kernels with >=50%
  cross-thread read overlap must use smem tiling) is added in the same PR.

## References

- `.workingdir/perf-audit-cuda-2026-05-16.md` win 3.
- [ADR-0360](0360-cambi-cuda.md) — original CAMBI CUDA port (T3-15a).
- [Research-0135](../research/0135-cambi-cuda-smem-tile-2026-05-16.md) — design
  analysis and tile-size decision matrix.
- `ncu --section MemoryWorkloadAnalysis` on `cambi_spatial_mask_kernel` to
  verify L2 traffic drop post-change.
- `ncu --section LaunchStats` to confirm occupancy remains >=50%.
