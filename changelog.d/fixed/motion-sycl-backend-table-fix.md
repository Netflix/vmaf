**docs(metrics): correct `motion` GPU backend row — add missing SYCL** — the overview
table in `docs/metrics/features.md` listed `motion` (fixed-point) backends as
`CUDA, Vulkan`, omitting SYCL. `integer_motion_sycl.cpp` (786 lines, registered in
`feature_extractor.c`) has been shipped since it was included in the enable_chroma wave;
the detailed section body already said `CUDA, SYCL, and Vulkan` correctly. Table now
agrees with the body text.
