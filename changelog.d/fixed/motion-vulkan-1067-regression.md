### motion_vulkan: restore PR #1041 extract_force_zero and flush idempotency fixes clobbered by PR #1067

PR #1067 (bootstrap name-builder refactor) accidentally reverted two host-side
correctness fixes that landed in PR #1041:

- **`extract_force_zero`**: re-introduces the `frame_index > 0` guard that
  suppressed `motion2_score` on frame 0, breaking the collector index anchor and
  causing score-0 frames to be silently dropped in force-zero mode.

- **`flush`**: removes the `vmaf_feature_collector_get_score` idempotency probes
  for `motion2_score` and `motion3_score`, causing "cannot be overwritten"
  warnings and context synchronisation errors when flush races a pending-collect.

Both fixes are restored verbatim from PR #1041.
