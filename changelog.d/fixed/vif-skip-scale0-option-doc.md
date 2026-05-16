**docs(vif): add missing `vif_skip_scale0` option to features.md** — the
`vif_skip_scale0` / `ssclz` option was silently absent from the VIF options
table in `docs/metrics/features.md`. All CPU and GPU twins (CUDA, SYCL,
Vulkan) honour the option; callers querying the reference docs had no way
to discover it or its alias.
