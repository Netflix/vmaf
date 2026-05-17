Add `test_vulkan_motion3_parity` cross-backend gate (T3-15c / ADR-0219): asserts
`VMAF_integer_feature_motion3_score` from the CPU `motion` extractor and the Vulkan
`motion_vulkan` extractor agree within 1e-4 (places=4) on a 256×144 synthetic
fixture. Skips cleanly when no Vulkan compute device is present. Closes the Vulkan
half of the T3-15(c) parity gap; companion tests cover CUDA (PR #922) and SYCL
(PR #927).
