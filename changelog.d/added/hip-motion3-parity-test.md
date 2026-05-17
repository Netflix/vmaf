- Add motion3 CPU vs. HIP parity gate (`test_hip_motion3_parity`) to close
  the T3-15(c) HIP audit gap; asserts places=4 agreement at frame index 1
  on a 256x144 synthetic YUV420P fixture, skips cleanly when no HIP device
  is present (ADR-0219 / ADR-0214).
