Wire `adm_skip_scale0` and `adm_min_val` into `integer_adm_sycl`,
`integer_adm_hip`, and `integer_adm_metal` backends. The options were
accepted by the CPU, CUDA, and Vulkan extractors but silently ignored
by the three remaining GPU paths; scale-0 was always accumulated and
no score floor was applied.
