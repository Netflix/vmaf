# ADR-0458: SYCL CAMBI queue-sync collapse + SSIM horizontal SLM staging

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `sycl`, `perf`, `cambi`, `ssim`, `gpu`, `fork-local`

## Context

Two performance regressions were identified in the SYCL GPU feature extractor layer
during the perf-audit-vulkan-sycl-2026-05-16 session:

**SY-1** (`integer_cambi_sycl.cpp`): The `submit_fex_sycl` function issued 25
`q.wait()` calls per frame across the 5-scale CAMBI loop. Each `q.wait()` drains the
Level Zero command queue to idle, incurring ~5–15 µs overhead per call and preventing
the in-order queue from pipelining adjacent kernels. With a SYCL in-order queue, all
inter-kernel dependencies are handled automatically; only the 5 mandatory
CPU-reads-from-device syncs (one per scale, before `vmaf_cambi_calculate_c_values`)
are required. See research digest 0135 §1.

**SY-2** (`integer_ssim_sycl.cpp`): `launch_horiz` (the 11-tap horizontal Gaussian
pass) used a bare `parallel_for<range<2>>` with no SLM, causing every work-item to
independently read 22 global-memory floats (11 taps × 2 planes). Neighbouring
work-items in the same subgroup read heavily overlapping input regions with no data
sharing. Adding SLM tile staging eliminates the redundant global reads and is the
pattern already established in `float_vif_sycl.cpp`, `float_ansnr_sycl.cpp`, and
`float_motion_sycl.cpp`. See research digest 0135 §2.

The `integer_ciede_sycl.cpp` kernel does not perform convolution and requires no SLM
change (see research digest 0135 §3).

## Decision

We will:

1. **SY-1**: Remove the 20 redundant `q.wait()` calls from the CAMBI per-scale loop,
   keeping exactly one `q.wait()` per scale immediately before the CPU residual reads
   host memory. The in-order SYCL queue provides the necessary GPU-to-GPU ordering
   guarantees without explicit synchronisation.

2. **SY-2**: Convert `launch_horiz` in `integer_ssim_sycl.cpp` from
   `parallel_for<range<2>>` to `nd_range` with two `local_accessor` SLM tiles
   (`s_ref`, `s_cmp`) of size `SSIM_WG_Y × SSIM_TILE_W = 8 × 26 floats`. The
   cooperative tile load + barrier pattern mirrors the established pattern in the
   other SYCL feature kernels.

Neither change touches `integer_ciede_sycl.cpp`, which has no convolution to optimise.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep per-step q.wait() (status quo) | Simpler to read; matches CUDA v1 posture verbatim | 20 unnecessary queue drains per frame; prevents pipelining | Correctness requirement is only at D2H→CPU boundaries |
| Use out-of-order queue + event::depends_on for CAMBI | Explicit DAG; would allow overlapping H2D with prior D2H | Complex DAG for a strictly sequential pipeline; not worth the complexity given CAMBI's CPU residual already serialises frames | In-order queue is cleaner for this linear pipeline |
| Keep global loads in launch_horiz | No SLM management; simpler kernel | 22 global reads per pixel vs. ~3.25 amortised; poor L2 utilisation at 1080p+ | SLM pattern is established and the tile bounds arithmetic is straightforward |
| SLM for all 5 output channels | Further reduce repeated G[] multiplications | 5 × 208 × 4 = 4160 B SLM — 2.5× more; no additional data reuse (products computed per-thread) | Two-channel SLM (ref, cmp) is sufficient; products are per-thread computation |

## Consequences

- **Positive**: SY-1 eliminates ~20 GPU queue-drain round-trips per frame in CAMBI.
  At 1080p with 5 scales this saves an estimated 0.5–3 ms/frame on Arc A380. SY-2
  reduces global-memory traffic in the SSIM horizontal pass by ~3.25× per work-item,
  saving ~0.5–2 ms/frame at 1080p.
- **Positive**: Code now matches the pattern used by all other SYCL feature kernels
  (`float_vif_sycl.cpp`, `float_motion_sycl.cpp`, `float_ansnr_sycl.cpp`).
- **Positive**: `places=4` parity gate maintained (verified on Intel Arc A380,
  48-frame test sequence, max |Δ| = 1e-6).
- **Negative**: `launch_horiz` is now more complex (tile load loop + barrier vs. simple
  1D inner loop). The tile bounds arithmetic is non-trivial and requires a comment
  explaining why `gx_load < e_w` (not `e_w_horiz`).
- **Neutral**: CIEDE unchanged; the original description of "11-tap convolution in
  CIEDE" was incorrect — CIEDE is a pure per-pixel colour-difference transform with
  no separable filter.
- **Follow-up**: CUDA twin (`integer_cambi_cuda.c`) also uses per-step waits (the
  CUDA v1 posture from ADR-0360). It should receive a parallel SY-1 fix in a
  follow-up PR. The SSIM CUDA twin (`integer_ssim_cuda.c`) should similarly receive
  the SLM-staged horizontal pass.

## References

- Research digest: [0135](../research/0135-sycl-cambi-ssim-ciede-perf-2026-05-16.md)
- Perf audit source: `.workingdir/perf-audit-vulkan-sycl-2026-05-16.md` (SY-1 + SY-2)
- Parent ADRs: [ADR-0371](0371-cambi-sycl-strategy-ii.md) (CAMBI SYCL strategy),
  [ADR-0188](0188-gpu-long-tail-batch-2.md) (SSIM SYCL)
- SYCL 2020 spec §4.7.6 "In-order and out-of-order queues" — in-order queues serialise
  all operations automatically; explicit `wait()` between enqueued operations is only
  needed when the host must observe a result.
- SYCL 2020 spec §4.9 "Local memory" — `local_accessor` maps to L1/SLM on GPU;
  use `access::fence_space::local_space` to fence SLM stores before SLM loads.
- req: `perf-audit-vulkan-sycl-2026-05-16 SY-1 + SY-2 task directive`
