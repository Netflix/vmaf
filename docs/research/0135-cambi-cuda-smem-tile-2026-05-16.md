# Research-0135: CAMBI CUDA spatial-mask shared-memory tile design

- **Status**: Active
- **Workstream**: [ADR-0464](../adr/0464-cambi-cuda-smem-tile.md)
- **Last updated**: 2026-05-16

## Question

Can the `cambi_spatial_mask_kernel` global-memory bandwidth be meaningfully
reduced via shared-memory staging without introducing correctness risk or
excessive smem pressure?

## Sources

- Original kernel: `libvmaf/src/feature/cuda/integer_cambi/cambi_score.cu`
  lines 68-125 (pre-change), accessed 2026-05-16.
- Research-0091 — CAMBI CUDA integration trade-offs; noted "49 global reads per
  thread" as the known cost accepted during the initial T3-15a port.
- `.workingdir/perf-audit-cuda-2026-05-16.md` — perf-audit plan; win 3 names
  this exact kernel.
- CUDA Programming Guide, section 5.3.2 (Shared Memory); section 5.3.2.3 (Bank
  Conflicts).
- NVIDIA Nsight Compute documentation: `MemoryWorkloadAnalysis` section for L2
  traffic; `LaunchStats` section for occupancy.

## Findings

### Traffic analysis

Each output pixel in the original kernel performs a 7x7 box sum of zero_deriv
values.  Each zero_deriv value requires reading three global uint16 words (p,
right-neighbor, below-neighbor).  Total per thread: 7x7x3 = 147 global reads.

Adjacent threads in the same warp row share a (7-1)=6-column horizontal overlap
in their 7-wide windows, giving (6/7) = 86% horizontal redundancy.  Across both
dimensions (with a 16x16 block): the 22x22 = 484 unique zero_deriv positions
needed by the entire block are fetched 49 times each (once per output pixel that
uses them on average), yielding 256x49 = 12,544 logical reads that collapse to
484 unique positions — 96% logical redundancy.

With L1 cache (typically 32 KB on Ampere/Ada) some of this is covered, but the
7x7 window at 1080p generates enough unique-row accesses per warp that L1 hit
rate is far below 96% in practice.

### Shared-memory sizing

The 484 zero_deriv values for a 16x16 block need a (16+6)x(16+6) = 22x22 tile.
At uint8 (zero_deriv is 0 or 1), that is 484 bytes without padding.

Padding the row to 32 bytes (22x32 = 704 bytes) was chosen because:
- It places each row on a 32-byte boundary, reducing false sharing between rows
  in the 4-byte-bank shared-memory system.
- A 22-byte natural row stride causes rows 0 and 1 to start at banks 0 and
  5 respectively (22 bytes = 5 full 4-byte banks + 2 bytes), making multi-row
  access patterns less predictable.
- 704 bytes is well within the 48 KB smem limit; at 32x256-thread blocks there
  would be ~68 blocks resident — effectively unlimited from a smem perspective.

4-way bank conflicts remain for uint8 within a row (16 consecutive threads
reading bytes n..n+15 span 4 four-byte banks, 4 threads per bank).  This is
accepted because SLM throughput at 4-way conflict (~4 cycles per access) is
still more than 10x faster than L2 latency (~40-50 cycles), and the bandwidth
benefit of eliminating 37,632 - 1,452 = 36,180 global reads per block
overwhelmingly dominates.

### Correctness analysis: img_tile approach ruled out

A candidate approach was to load an image tile into smem and derive zero_deriv
from it in-register.  Analysis showed this approach produces incorrect
zero_deriv for the left-halo positions of block (0,0):

- For tile position j corresponding to raw_gx = -1: the original kernel
  computes `rx = clamp(-1) = 0`, `r = image[gy][1]` (rx != w-1 so use rx+1).
- The tile approach stores `img_tile[i][j+1] = image[gy][clamp(raw_gx+1)] =
  image[gy][clamp(0)] = image[gy][0]` — same as p — giving a spurious
  eq_right=true when image[gy][0] != image[gy][1].

This discrepancy propagates into the box_sum for output pixel x=0 (which reads
tile position j=0).  The img_tile approach was rejected.

The correct approach: compute each zero_deriv element directly using the same
clamped-coordinate formula as the original kernel, loading from global into the
zd_tile.  Each element requires 3 global reads, for 484x3 = 1,452 per block —
still 26x fewer than the original 37,632.

### Tile-size trade-offs

| Tile type | smem bytes | Global reads | Correctness | Notes |
|---|---|---|---|---|
| No tile (original) | 0 | 37,632 / block | Correct | Baseline |
| img_tile 23x23 uint16 + zd_tile 22x22 uint8 | 1058+484=1542 | ~484x3+484x3=2904 | Bug at left/top halo | Ruled out |
| zd_tile only 22x32 uint8 | 704 | 1452 | Correct | Chosen |
| zd_tile 22x64 uint8 (bank-free) | 1408 | 1452 | Correct | Over-engineered; 2x smem for no meaningful benefit given tiny overall smem budget |

### Load distribution

With 256 threads and 484 elements, two passes suffice:
- Pass 0: all 256 threads load elements 0..255.
- Pass 1: threads 0..227 (228 threads) load elements 256..483.

Thread linearisation: `tid = ty*BLOCK_X + tx`.  Element `k` maps to tile row
`k / ZD_TILE_W` and column `k % ZD_TILE_W`.  The global coordinate for element
k is `(by*BLOCK_Y - SMEM_HALF + row, bx*BLOCK_X - SMEM_HALF + col)`.

In pass 0, 16 consecutive threads (same warp row) load tile positions
(row, col..col+15), which map to 16 consecutive global-memory columns in the
same row.  These 16 uint16 loads form a coalesced 32-byte transaction.
Similarly for pass 1.  The right-neighbor reads (`image[gy][gx+1]`) access the
adjacent column which falls within the same or the immediately following cache
line — effectively coalesced.

## Alternatives explored

The 4 alternatives in the ADR decision matrix.  Additionally:

- **Warp-level shuffle reduction**: could reduce 49-element sums within a warp
  without smem.  Rejected: requires each thread to already hold all 49 values,
  not just one; does not reduce global reads.
- **Texture memory cache**: no explicit uint16 1D texture in CUDA for arbitrary
  widths without pitch setup; adds binding overhead per kernel launch.

## Open questions

- Actual measured speedup on RTX 4090 (cannot benchmark during the cambi_cuda
  segfault window, Issue #857 — path will be exercisable once PR #870 lands).
  Estimated 25-40% from arithmetic; exact number via `ncu
  --section MemoryWorkloadAnalysis` once end-to-end run is possible.
- Occupancy impact of 704 B extra smem: expected small (reduces blocks/SM from
  ~68 to ~60 at the 48 KB limit on Ada — still fully occupancy-bound by warps,
  not smem).
- Whether the `cambi_filter_mode_kernel` would benefit from a 1D smem tile for
  its 3-tap stencil.  Preliminary estimate: 3-tap overlap is only 2/3 = 67%
  but at 1 read per thread vs 3 for zero_deriv, the absolute traffic reduction
  is small.  Not recommended for this PR.

## Related

- [ADR-0360](../adr/0360-cambi-cuda.md) — original CAMBI CUDA port.
- [ADR-0464](../adr/0464-cambi-cuda-smem-tile.md) — this optimization.
- [Research-0091](0091-cambi-cuda-integration.md) — CAMBI CUDA integration
  trade-offs (acknowledged the 49-read cost as a known debt).
- Issue #857 — cambi_cuda segfault (blocks end-to-end validation).
- PR #870 — host-preprocessing fix (prerequisite for end-to-end run).
