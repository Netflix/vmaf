Added `libvmaf/test/test_cuda_motion3_parity.c`: cross-backend parity test
asserting that `VMAF_integer_feature_motion3_score` from the CPU `motion`
extractor and the CUDA `motion_cuda` extractor agree to within 1e-4
(places=4, ADR-0214) at frame index 1. Closes a CHUG audit coverage gap
where boundary-condition drift in the host-side moving-average
post-process could silently pollute CHUG-extracted `motion3_mean`/`std`
columns without any cross-backend assertion. The test skips cleanly on
hosts without a CUDA device.
