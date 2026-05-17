Add C unit test for integer_vif `vif_skip_scale0` option (audit gap §1).
Closes the C-level test gap identified in the 2026-05-16 coverage audit:
`test_vif_skip_scale0` verifies that `vif_skip_scale0=true` produces
`scale0_score == 0.0` and that the default path yields a finite positive
value, establishing the CPU ground truth GPU backends must match once the
option is ported to CUDA/SYCL/Vulkan.
