Document two undocumented ADM skip options (`adm_skip_scale0` and
`adm_skip_aim` / `adm_skip_aim_scale`) in `docs/metrics/features.md`.
Both options have been in the CPU path since before the fork; `adm_skip_scale0`
was also wired into CUDA by PR #1082 (`integer_adm_cuda`). Neither appeared in
the public documentation.
