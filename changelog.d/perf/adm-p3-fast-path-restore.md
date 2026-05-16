Restore `adm_sum_cube_s_p3`, `adm_csf_den_scale_s_p3`, and `adm_cm_s_p3` fast-path
functions that were accidentally dropped by PR #1067. These eliminate all per-pixel
`powf()` calls on the default `adm_p_norm == 3.0` path, which is the hot path for
every standard VMAF evaluation.
