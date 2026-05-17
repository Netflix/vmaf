### float_motion_vulkan: fix extract_force_zero debug flag and flush idempotency

Two correctness gaps vs `float_motion_cuda.c` (and the Metal parity fix in
commit 982fabacc) were present in `float_motion_vulkan.c`:

**extract_force_zero debug flag not respected.** When `motion_force_zero=true`,
the fast-path emitted `VMAF_feature_motion_score=0` unconditionally for every
frame after the first, ignoring the `debug` option. The correct behaviour
(matching CUDA and Metal) is to emit `motion2_score=0` always and
`motion_score=0` only when `debug=true`. The erroneous frame-index guard
(`if (s->frame_index > 0)`) was also removed — frame 0 correctly receives
`motion2_score=0` in the force-zero path, and conditionally `motion_score=0`
when debug is enabled.

**flush() missing idempotency probe.** `flush()` unconditionally appended the
trailing `motion2_score[frame_index-1]` without first checking whether a
pending collect had already written that slot. This could trip the feature
collector's "cannot be overwritten" warning and surface as a context
synchronisation error on multi-frame sequences where flush ran after a
delayed collect. Added `vmaf_feature_collector_get_score` probe before the
append, matching `float_motion_cuda.c` lines 352-362.
