- Added three unit tests to `libvmaf/test/test_output.c` covering
  previously untested public API entry points identified in the
  2026-05-16 test-coverage audit (§2):
  `vmaf_version()` (non-NULL + digit presence check),
  `vmaf_write_output()` (JSON path-based dispatcher),
  and `vmaf_write_output_with_format()` (custom printf format string
  honoured). Total test count in `test_output` raised from 8 to 11.
