### Fixed: `test_context` and `test_log` built but never registered as meson test cases

`libvmaf/test/meson.build` defined `test_context` and `test_log` as
`executable()` targets but never called `test()` for either. Both
executables were compiled by `ninja -C build` yet were invisible to
`meson test` (and therefore to CI and the `--suite=fast` pre-push gate).

- `test_context` exercises `vmaf_init`/`vmaf_close`, `vmaf_import_feature_score`,
  `vmaf_feature_score_at_index`, and `vmaf_feature_score_pooled`.
- `test_log` exercises the internal log-callback API.

Both are now registered with `suite : ['fast']` alongside the rest of the
CPU unit-test block.

Identified by the build-matrix and symbol-visibility audit
(`.workingdir/audit-build-matrix-symbols-2026-05-16.md`, finding §5c residual).
