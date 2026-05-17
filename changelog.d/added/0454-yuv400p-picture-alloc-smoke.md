Add `test_picture_alloc_yuv400p_luma_only` smoke test pinning the `VMAF_PIX_FMT_YUV400P`
(monochrome / luma-only) alloc contract: chroma `w`/`h` must be zero, chroma `data[]`
pointers must be `NULL` for both standard and odd-dimension inputs.
