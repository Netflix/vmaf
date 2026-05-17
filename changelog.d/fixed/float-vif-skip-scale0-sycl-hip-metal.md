Wire `vif_skip_scale0` into `float_vif_sycl`, `float_vif_hip`, and
`float_vif_metal` backends. The option was registered on the CPU
`float_vif` extractor but absent from all three remaining GPU paths;
scale-0 was always accumulated and emitted regardless of the caller
setting. Host-side suppression only — GPU kernels are unchanged.
