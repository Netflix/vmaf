Add C unit tests for the public `vmaf_write_output` and `vmaf_write_output_with_format`
dispatcher entry points. Both symbols were previously exercised only via the CLI; the
new `test_write_output_dispatcher` and `test_write_output_with_format` cases in
`libvmaf/test/test_output.c` drive them through CSV output with a temp-file path,
closing the last untested surface from the 2026-05-16 coverage audit.
