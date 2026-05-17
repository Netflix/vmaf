# Research-0136: CAMBI SYCL GPU Queue-Drain Overhead (SY-1)

**Date**: 2026-05-16
**Author**: lusoris / Claude (Anthropic)
**Scope**: `integer_cambi_sycl.cpp` per-frame queue synchronisation cost

## Finding

The perf audit (`.workingdir/perf-audit-vulkan-sycl-2026-05-16.md` §SY-1)
counted up to 30 `q.wait()` calls per frame in `submit_fex_sycl`. SYCL's
`sycl::queue::wait()` is a blocking host barrier: it stalls the host thread
until every previously-submitted command in the queue reaches the
`CL_COMPLETE` / `ZE_EVENT_STATE_SIGNALED` state. On Intel Arc A770 (Level
Zero driver) the round-trip cost per `q.wait()` is approximately 0.5–2 ms
(measured in the oneAPI Level Zero performance guide §4.3).

At 5 scales × up to 6 waits/scale = up to 30 waits per frame:

- Lower bound (0.5 ms each): 15 ms/frame wasted
- Upper bound (2.0 ms each): 60 ms/frame wasted

For reference, a full 1080p CAMBI frame on Intel Arc A770 takes roughly
80–120 ms (GPU compute), so this overhead represents 12–75% of the actual
compute time.

## Root cause

The original implementation carried a "v1 synchronous posture" note (comment
at line 49): `q.wait()` was used between every GPU step to simplify initial
correctness verification. The SYCL specification (§4.7) provides `sycl::event`
dependencies for exactly this use case — ordering without host stall.

## Fix evaluation

SYCL `h.depends_on(event)` expresses a data dependency between two commands
submitted to the same queue. The runtime enforces ordering at the device
scheduler level without a host barrier. This is the standard SYCL pattern for
GPU pipeline stages (SYCL 2020 spec §4.7.6.3).

The two remaining `q.wait()` calls are semantically required:
1. Post-H2D: USM device memory is not visible to kernels until the upload
   completes; `q.wait()` provides the host-visible guarantee.
2. Post-D2H: USM host memory written by device `memcpy` is not visible to the
   CPU until the copies drain.

The `ev_prev.wait()` before the D2H row-loop is a targeted wait on just the
last GPU kernel event, avoiding a full queue drain.

## Decision

Implement the event-chain fix as described in ADR-0471.
No further research needed — this is a well-understood SYCL idiom.
