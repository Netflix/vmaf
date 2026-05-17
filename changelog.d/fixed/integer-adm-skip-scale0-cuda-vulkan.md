Wire `adm_skip_scale0` option into the CUDA and Vulkan `integer_adm` backends.
The option was present on the CPU extractor but silently absent from both GPU
paths; scale-0 was always accumulated, diverging from the CPU reference when
callers set `adm_skip_scale0=true`. Default is `false` (no-op), so existing
GPU runs are bit-for-bit unchanged.
