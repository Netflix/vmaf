fix(metal): restore correct dispatch_strategy keys clobbered by PR #1088

PR #1088's squash merge reverted the five key-string corrections that
PR #1104 applied to `libvmaf/src/metal/dispatch_strategy.c`:

- `motion2_v2_score` → `VMAF_integer_feature_motion2_v2_score`
- `float_motion` → `VMAF_feature_motion_score` + `VMAF_feature_motion2_score`
- `motion2_score` → `VMAF_integer_feature_motion2_score`
- Removed spurious `motion3_score` (Metal never implemented motion3)
- Removed spurious `float_ms_ssim` (no Metal extractor emits this key)

Also adds missing entries for `float_adm_metal` and `integer_vif_metal`,
which were wired as extractors after PR #1104 but never added to the table.
Without correct entries, `vmaf_metal_dispatch_supports()` returns 0 for
valid feature keys, silently falling back to CPU on Metal-capable hardware.
