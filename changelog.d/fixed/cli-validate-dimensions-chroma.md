### Fixed

- **CLI**: `vmaf` now rejects non-positive frame dimensions (zero width or height)
  and odd luma dimensions for chroma-subsampled formats (4:2:0 width/height,
  4:2:2 width) before the main loop, printing a descriptive error to stderr and
  exiting non-zero instead of crashing or producing undefined output.
  The distorted-stream bitdepth range (8–16) is now also checked independently
  of the reference stream. (`libvmaf/tools/vmaf.c`, ADR-0461)
