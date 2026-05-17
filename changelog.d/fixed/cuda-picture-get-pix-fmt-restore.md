### Fixed

- Restore `vmaf_cuda_picture_get_pix_fmt()` accessor dropped by PR #1067 regression.
  The function and its declaration in `picture_cuda.h` were absent from master; callers
  in extractor code that used the accessor would fail to link against a CUDA build.
  Restored with the original `return pic->pix_fmt;` implementation and Doxygen doc
  comment, matching the pattern of `vmaf_cuda_picture_get_stream` and sibling accessors.
