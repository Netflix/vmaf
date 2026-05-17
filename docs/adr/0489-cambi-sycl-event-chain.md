# ADR-0489: CAMBI SYCL — Replace GPU-to-GPU `q.wait()` Calls with Event Chains (SY-1)

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `sycl`, `gpu`, `cambi`, `performance`, `fork-local`

## Context

The perf audit (`.workingdir/perf-audit-vulkan-sycl-2026-05-16.md`, finding SY-1)
identified that `integer_cambi_sycl.cpp::submit_fex_sycl` issued 8 or more
`q.wait()` calls per frame inside the per-scale loop — one after every GPU
step (spatial mask, each of two decimations per scale, filter-mode H, filter-mode
V). Each `q.wait()` drains the entire SYCL queue to idle before the next kernel
launch. On Intel Arc and iGPU hardware this adds approximately 0.5–2 ms per
barrier; at 5 scales × up to 6 waits per scale = up to 30 queue-drain points
per frame, the wasted stall time ranges from 15 to 60 ms per frame.

The comment at line 49 of the original source ("q.wait() used between GPU and
CPU stages — synchronous v1 posture") confirmed this was a known placeholder
carried forward from the CUDA twin's initial design (ADR-0360 §v1 simplification
note). GPU-to-GPU kernel transitions — where one kernel writes a buffer that the
next kernel reads — do not require a full queue drain; they only need an ordering
guarantee, which SYCL event dependencies (`h.depends_on(event)`) provide
without stalling the runtime scheduler.

## Decision

We will replace every GPU-to-GPU `q.wait()` in `submit_fex_sycl` with
`sycl::event` chains:

- `launch_spatial_mask`, `launch_decimate`, and `launch_filter_mode` are
  changed to return `sycl::event` (captured from `q.submit(...)`).
- `launch_decimate` and `launch_filter_mode` accept a `sycl::event dep`
  parameter; the handler calls `h.depends_on(dep)` before `h.parallel_for`.
- In the per-scale loop: decimate-image and decimate-mask both depend on the
  prior event; a lightweight `single_task` fence (depending on both decimate
  events) serialises into the filter-mode chain.
- The two `q.wait()` calls that are semantically necessary are kept:
  1. After the H2D row-loop (kernels cannot read partially-uploaded device
     memory).
  2. After the D2H row-loop (CPU residual cannot read partially-copied host
     staging buffers). This is replaced with a targeted `ev_prev.wait()`
     to drain only the last GPU kernel before D2H begins, followed by
     `q.wait()` to drain the D2H copies themselves.

The CUDA twin (`integer_cambi_cuda.c`, ADR-0360) retains its v1 synchronous
posture; its stream-ordering is handled by CUDA's default serialised
per-stream model and the risk/reward of an event-chain refactor is lower
there.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep `q.wait()` (status quo) | Simple, obviously correct | 15–60 ms/frame stall overhead per SY-1 finding | The overhead is the problem being solved |
| SYCL graph (`vmaf_sycl_graph_register`) | Maximum overlap; replay avoids re-submission overhead | Incompatible with per-scale D2H readback and host histogram pass (CPU residual between GPU steps); ADR-0371 explicitly excluded this path | Cannot fit this algorithm into a graph |
| USM `memcpy` with event deps (H2D) | Could chain H2D into kernel dep | Row-by-row memcpy loop already submits many items; adding per-row event tracking is high complexity for marginal gain at this stage | Deferred; H2D event-chain is a follow-up optimisation |

## Consequences

- **Positive**: GPU-to-GPU stall overhead eliminated. Expected frame-time
  reduction on Intel Arc: 15–60 ms per CAMBI frame. The SYCL runtime can
  now overlap or pipeline adjacent dispatches where the hardware scheduler
  allows it.
- **Positive**: No change to the `places=4` cross-backend parity contract —
  all arithmetic remains identical; only queue-synchronisation semantics changed.
- **Positive**: No public API, option, or output-schema change — purely
  internal scheduling.
- **Negative**: Slight increase in per-scale loop complexity (event variables,
  the two-dep `single_task` fence between decimate and filter-mode passes).
- **Neutral / follow-up**: The H2D row-loop (`q.memcpy` per row × 1080 rows)
  could also be event-chained to avoid the leading `q.wait()`. Deferred;
  the per-row approach was already the v1 posture and the H2D wait is less
  impactful than the GPU-to-GPU waits.
- **Neutral / follow-up**: The CUDA twin could receive an equivalent
  cudaEvent-based refactor; deferred pending profiling on NVIDIA.

## References

- Perf audit finding SY-1: `.workingdir/perf-audit-vulkan-sycl-2026-05-16.md`
- ADR-0371 — CAMBI SYCL port (Strategy II hybrid rationale)
- ADR-0360 — CAMBI CUDA port (v1 synchronous posture)
- ADR-0415 — CAMBI SYCL port PR record
- `libvmaf/src/feature/sycl/integer_cambi_sycl.cpp`
- `libvmaf/src/feature/sycl/AGENTS.md` — updated invariant note
- req: "perf fix smallest unaddressed in vulkan or sycl"
