### Fixed

- **CUDA / SYCL / Vulkan CAMBI**: restore `src_width` (alias `srcw`) and
  `src_height` (alias `srch`) option-table entries and conditional-assign
  init pattern that were lost when PR #1067 landed on top of PR #1068.
  Passing `--feature cambi=src_width=N` to any GPU extractor now works
  correctly instead of being silently dropped.
