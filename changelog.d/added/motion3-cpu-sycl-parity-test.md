Added `libvmaf/test/test_sycl_motion3_parity.c`: cross-backend parity test
asserting that `VMAF_integer_feature_motion3_score` from the CPU `motion`
extractor and the SYCL `motion_sycl` extractor agree to within 1e-4
(places=4, ADR-0214) at frame index 1. Closes the SYCL half of the
T3-15(c) / ADR-0219 audit gap: the host-side moving-average post-process
is implemented independently in `integer_motion.c` and
`integer_motion_sycl.cpp`; without this test, boundary-condition drift
could silently pollute CHUG-extracted `motion3_mean`/`std` columns.
The test skips cleanly on hosts without a SYCL-capable GPU.
