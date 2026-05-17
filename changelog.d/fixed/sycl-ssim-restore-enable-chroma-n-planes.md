**fix(sycl):** restore `enable_chroma` option, `n_planes` field, and suppression comment
in `integer_ssim_sycl.cpp` that were accidentally dropped by PR #1095 when it added the
`enable_db`/`clip_db` dB-domain options. The v2 multi-plane dispatch path depends on
`enable_chroma` being present in the options table and `n_planes` being clamped to 1 in v1.
