# ADR-0452: Hoist VIF scratch buffer from per-frame allocation to VifState

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris
- **Tags**: perf, vif, cpu, build

## Context

`compute_vif` in `libvmaf/src/feature/vif.c` allocated a contiguous
`10 × ALIGN_CEIL(w × sizeof(float)) × h` byte block on every frame
call and freed it on return. At 1080p (stride = 7712, h = 1080) this is
approximately 79 MB of allocator traffic per frame, consisting of ten
full-plane scratch buffers (`ref_scale`, `dis_scale`, `mu1`, `mu2`,
`ref_sq_filt`, `dis_sq_filt`, `ref_dis_filt`, `num_array`, `den_array`,
`tmpbuf`). The geometry is known at `init_fex` time (from the
`VmafFeatureExtractor` init callback), so the allocation can be lifted
to the init/close lifecycle.

This matches the pattern already used by `float_adm`, `integer_vif`,
`ssimulacra2`, and `integer_adm`. The CPU perf audit (`.workingdir/
perf-audit-cpu-2026-05-16.md`) identified this as Win 1, the top
CPU-side performance opportunity for the float VIF path.

Win 2 (enabling `VIF_OPT_FAST_LOG2`) is a no-op: `vif_options.h` line 28
already defines `VIF_OPT_FAST_LOG2` unconditionally. The macro substitutes
`log2f_approx` (a 9-term Horner polynomial) for libm `log2f`. This was
confirmed present before any code changes; documentation in the research
digest records the discovery for future maintainers.

## Decision

1. Add a `float *vif_buf` field to `VifState` in `float_vif.c`.
2. Change `compute_vif`'s signature to accept `float *data_buf` — the
   caller owns the allocation. `compute_vif` slices sub-pointers from it
   exactly as before.
3. In `float_vif.c::init`, allocate
   `VIF_SCRATCH_BUF_CNT (10) × scaled_float_stride × scaled_h` bytes
   once and assign to `s->vif_buf`.
4. Pass `s->vif_buf` to `compute_vif` from `extract`.
5. Free `s->vif_buf` in `close`.
6. In `vifdiff` (the standalone utility also in `vif.c`), allocate the
   scratch buffer once before the frame loop and free it after — still
   one allocation per video, not one per frame.

Bit-exactness: the scratch buffer is pure write-before-read within each
`compute_vif` call; no cross-frame state is stored in it. The hoist
changes lifetime, not contents.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| Keep per-frame malloc | Zero code change | ~79 MB/frame allocator traffic at 1080p; system allocator lock contention in threaded mode | Performance regression is the status quo; ruled out |
| Thread-local storage for scratch | Zero signature change | Complicates sanitiser / valgrind runs; not idiomatic for this codebase | Not chosen — VifState is already the right place |
| Opaque scratch pointer in VifState (this ADR's decision) | Matches existing `float_adm`, `ssimulacra2` patterns; zero arithmetic change | Requires signature update to `compute_vif` and `vif.h` | Chosen |

## References

- req: perf audit `.workingdir/perf-audit-cpu-2026-05-16.md` Win 1 directive
- [Research digest 0135](../research/0135-vif-cpu-perf-workspace-hoist-2026-05-16.md)
