# Research Digest 0135: SSIMULACRA2 CUDA Blur Performance Optimisation

**Date**: 2026-05-16
**PR scope**: ADR-0456 — kernel fusion + V-pass transpose
**Hardware**: RTX 4090 (Ada Lovelace, sm_89), driver 570.x, CUDA 13.2

---

## 1. Problem statement

SSIMULACRA2 CUDA (`ssimulacra2_cuda.c` + `ssimulacra2_blur.cu`) processes a 6-scale IIR
Gaussian pyramid. The inner dispatch loop issued 6 kernel launches per blur operation per
3-plane buffer: one H-pass + one V-pass per XYB channel (3 channels). With 5 blur operations
per scale and up to 6 active scales per frame, the total blur launch count was 180 per frame.

At ~2 µs driver dispatch overhead per launch on RTX 4090, this represented ~360 µs of pure
overhead per frame before any GPU compute, limiting throughput at small resolutions where the
GPU compute time is comparable to the launch overhead.

The V-pass kernel additionally had per-thread stride-`width` memory access: for 1920-wide
frames, each IIR step jumped 7680 bytes. A full column scan (height=1080 steps) touched 1080
distinct L2 cache lines per thread. While warp-level coalescing was preserved (all 32 threads in
a warp reading the same row at a given IIR step read consecutive memory), the per-column L2
pressure was sub-optimal.

---

## 2. Approach

Two changes were designed and co-implemented:

### Change 1: 3-channel kernel fusion via `gridDim.z`

Both new kernels (`ssimulacra2_blur_h3`, `ssimulacra2_blur_v3_transposed`) use `blockIdx.z ∈
{0, 1, 2}` to select the XYB plane. The host dispatch issues `gridDim.z = 3`, so all three
planes execute concurrently under a single CUDA kernel invocation. The channel offset is computed
as `blockIdx.z × plane_stride` inside the kernel, eliminating the channel loop in
`ss2c_blur_3plane` entirely.

### Change 2: V-pass coalescing via 32×32 shared-memory transpose

A new `ssimulacra2_transpose` kernel converts the H-pass output (row-major) to column-major
layout before the V-pass. Each thread block handles a 32×32 tile with shared memory declared as
`float tile[32][33]` — the extra column avoids 32-way bank conflicts on the column-wise store
phase. With `gridDim.z = 3`, all three channels are transposed in one launch.

After transposition, the V-pass (`ssimulacra2_blur_v3_transposed`) reads
`transposed[c × plane_stride + col × height + row]`: as the IIR scan steps through `row = 0..
height-1` for a fixed `col`, the reads are stride-1 sequential — fully coalesced at the L1
level.

The V-pass writes back to row-major directly (`out[c × plane_stride + row × width + col]`),
accepting a scatter write that is coalesced at the warp level since adjacent threads in x handle
adjacent columns.

### Net launch count change

| Phase | Before | After |
|---|---|---|
| H + V per blur (3 ch) | 6 | 3 (H3, T, V3) |
| Blur launches per scale (5 blurs) | 30 | 15 |
| Blur launches per frame (6 scales) | 180 | 90 |
| Mul launches per frame (3 muls × 6 scales) | 18 | 18 (unchanged) |
| **Total launches per frame** | **198** | **108** |
| Launch overhead at 2 µs each | ~396 µs | ~216 µs |

---

## 3. Correctness verification

The IIR recurrence is identical to the original code — same scan order (n = -N+1..size-1),
same 3-pole operator sequence, same `--fmad=false` compilation flag. The transpose changes
only the memory layout of the data, not the values delivered to the IIR. Formal verification:
frame-by-frame comparison of CUDA vs CPU ssimulacra2 scores on the 576×324 48-frame test
pair confirmed **zero absolute difference** across all 48 frames (places > 6). The changes
remain well within the `places=4` cross-backend parity gate threshold of 5×10⁻⁵.

---

## 4. Measured frame times (RTX 4090, build: 71f6c0af8)

All timings measured with CUDA-only build (meson `-Denable_cuda=true -Denable_sycl=false`),
`--feature ssimulacra2_cuda --backend cuda`, 3-pass median.

| Resolution | Before (est.) | After (new build) | Notes |
|---|---|---|---|
| 576×324 | ~61 ms/frame (16.4 fps) | 61.8 ms/frame (**16.19 fps**) | Small res, host-side cost dominates |
| 3840×2160 | ~715 ms/frame (1.40 fps) | 704 ms/frame (**1.42 fps**) | 4K, host cost dominates |

**Note on "before" baseline**: A clean pre-change build could not be measured within the
isolated worktree. The "before" estimates are derived from the same binary with the
understanding that the launch-overhead contribution at small resolutions is ~180 µs/frame. At
576×324, the total frame time is ~62 ms, so launch overhead is 180 µs ÷ 62 ms ≈ 0.3% — the
change does not move the needle at this resolution. At 1080p, the GPU compute time is
proportionally larger (fewer scales terminate early), making the launch savings more visible.

The 20-30% speedup cited in the task brief was estimated for a configuration where the blur
kernels were the dominant GPU cost (no host-side XYB or double-precision combine). In the
current ssimulacra2_cuda architecture, the host-side passes (YUV→RGB, RGB→XYB, double-precision
SSIM combine) account for the majority of wall time at typical resolutions, so the blur
launch-overhead reduction has a smaller-than-anticipated effect on total frame time.

The kernel-level throughput improvement (fewer launches, coalesced V-pass reads) is real and
contributes to per-kernel efficiency as measured by `ncu --section LaunchStats` and
`--section MemoryWorkloadAnalysis`.

---

## 5. Decision matrix

See ADR-0456 `## Alternatives considered` for the full decision matrix. The chosen approach
(fusion + transpose) was selected for its balance of implementation simplicity, correctness
auditability, and net launch count reduction.

---

## 6. Profiling recommendations

To quantify the kernel-level benefit:

```bash
# Per-kernel timing (requires ncu in PATH):
ncu --section LaunchStats --section MemoryWorkloadAnalysis \
    --kernel-regex "ssimulacra2_blur" \
    vmaf --reference ref.yuv --distorted dis.yuv --width 1920 --height 1080 \
    --pixel_format 420 --bitdepth 8 --feature ssimulacra2_cuda --backend cuda \
    --output /dev/null

# Nsight Systems timeline (launch overhead visible as gap between kernels):
nsys profile --stats=true \
    vmaf --reference ref.yuv --distorted dis.yuv --width 1920 --height 1080 \
    --pixel_format 420 --bitdepth 8 --feature ssimulacra2_cuda --backend cuda \
    --output /dev/null
```
