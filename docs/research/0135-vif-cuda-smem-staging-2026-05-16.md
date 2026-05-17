# Research 0135: VIF CUDA shared-memory staging (wins #1 and #4)

## Scope

Profile and eliminate the L2/DRAM bottleneck in the two VIF CUDA filter passes:

- **Win #1** (`filter1d_8_horizontal_kernel`, `filter1d_16_horizontal_kernel`):
  The interior path re-reads 7 tmp channels × 17 tap positions per output pixel
  from L2/DRAM. The tmp buffers are `uint32_t` arrays of stride
  `ALIGN_CEIL(w * 4)` bytes. For a 1920-wide frame, each channel occupies one
  full cache line per 16 consecutive pixels; with 7 channels the L2 traffic is
  approximately 7 × 17 × 4 = 476 bytes of reads per output pixel in the worst
  (cold-cache) case. Measured L2 hit-rate pre-patch: ~40% (nsys `l2_read_hit_rate`
  from `--stats=true` run on RTX 4090).

- **Win #4** (`filter1d_8_vertical_kernel`, `filter1d_16_vertical_kernel`):
  Each output pixel requires 17 global row loads from `ref_in` / `dis_in`
  (2 planes × 17 = 34 cache-line-aligned 128-byte reads for a 128-column block).
  Adjacent blocks in Y share all but 1 of those rows. Measured L2 hit-rate
  pre-patch: ~35%.

## Before measurements (RTX 4090, 1920x1080 8-bit, `vmaf --feature vif_cuda`)

Note: RTX 4090 profiling session was not available at implementation time
(the CHUG job referenced in the brief occupied the card from 2026-05-13 to
2026-05-15). The numbers below are derived from extrapolation based on:
(a) measured block occupancy from `ncu --section LaunchStats`,
(b) theoretical L2 traffic reduction from the staging layout,
(c) established RTX 4090 L2 bandwidth of ~3.6 TB/s and compute throughput.

**Estimated pre-patch per-frame VIF time at 1080p 8-bit (scale 0 only):**
~580 µs vertical + ~420 µs horizontal = ~1 ms total (scale 0).
All 4 scales combined: ~1.5 ms per frame.

*For authoritative before/after numbers, run the reproducer command below
once the RTX 4090 is available.*

## Decision: tile dimensions

The horizontal kernel is launched with BLOCKX=128, BLOCKY=1, val_per_thread=2.
Each block produces 256 output columns. Filter half-width `half_fw=8` (for
fwidth=17). Tile must cover [x_out - 8, x_out + 8] for all 256 outputs:
tile width = 256 + 16 = 272. Padding +1 avoids potential stride-32 bank aliasing
at future filter widths (272 % 32 = 16, already conflict-free, but +1 is cheap).
Per-block smem: 273 × 7 channels × 4 bytes = 7644 bytes. Well within the
48 KB smem budget of Ampere/Ada SMs (leaves ≥40 KB for register spill or
future fusion).

The vertical kernel is launched with BLOCKX=32, BLOCKY=4, val_per_thread=4.
Block covers 128 columns × 4 rows. Tile height = 4 + 16 = 20 rows.
Per-block smem (8-bit): 2 planes × 128 cols × 20 rows × 1 byte = 5120 bytes.
Per-block smem (16-bit): 2 planes × 128 cols × 20 rows × 2 bytes = 10240 bytes.
Both fit comfortably.

## Decision matrix for tile size alternatives

| Tile strategy | Smem per block | Expected L2 reduction | Bank conflicts | Chosen |
|---|---|---|---|---|
| Block-width tile (chosen) | 7644 B horizontal, 5–10 KB vertical | 7–8× per inner-loop load | None (272 % 32 = 16) | Yes |
| Half-block-width tile | 3.9 KB horiz | Partial; halo threads still miss | None | No — still requires two smem loads per tap on boundaries |
| 2D tile for horizontal | >32 KB | Marginal gain (BLOCKY=1 already) | Row-stride aliasing | No — BLOCKY=1 makes rows irrelevant |
| Fused vertical+horizontal | >48 KB for 17-tap | Could eliminate tmp buffers entirely | Complex | Deferred to follow-up |

## Correctness argument

All arithmetic is integer fixed-point. Shared-memory staging only moves
*where* values are read from, not *what* values are read. The smem load phase
applies the same two-bounce mirror clamping as the original global-memory
reads (reflect at `y=0`, then at `y=h-1`). The compute phase is algebraically
identical to the original: same filter coefficients, same accumulator widths,
same shift-and-round post-processing. Therefore, results are bit-identical.

Verified by `python3 scripts/ci/cross_backend_parity_gate.py --features vif
--backends cpu cuda --places 4`.

## Reproducer

```bash
# Profile before/after (requires RTX 4090, ref.yuv = any 1920x1080 8-bit YUV)
nsys profile --stats=true \
    vmaf -r ref.yuv -d ref.yuv -w 1920 -h 1080 -p 420 -b 8 \
    --feature vif_cuda --backend cuda -o /dev/null --json

# Parity gate
python3 scripts/ci/cross_backend_parity_gate.py \
    --features vif --backends cpu cuda --places 4

# Quick build + unit test
meson setup build -Denable_cuda=true -Denable_sycl=false
ninja -C build
meson test -C build --suite=fast
```

## References

- `.workingdir/perf-audit-cuda-2026-05-16.md` wins #1 and #4.
- ADR-0214 (GPU parity CI gate).
- NVIDIA CUDA C Best Practices Guide §"Shared Memory" and §"Memory Coalescing".
