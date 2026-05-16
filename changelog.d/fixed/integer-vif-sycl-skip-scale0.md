- Fixed `integer_vif_sycl`: register `vif_skip_scale0` option and enforce
  scale-0 suppression in `collect_fex_sycl`, matching `integer_vif.c` and
  `integer_vif_cuda` behavior. Previously the option was unregistered;
  callers setting `vif_skip_scale0=true` had no effect on SYCL.
