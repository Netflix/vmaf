Expose `adm_csf_scale`, `adm_csf_diag_scale`, and `adm_noise_weight` as
proper VmafOption entries in `integer_adm_metal`. The fields existed in the
extractor struct and were used in the kernel dispatch, but were hardcoded to
their defaults in `init_fex_metal` rather than being user-configurable via
`vmaf --feature adm_metal:adm_csf_scale=...`.
