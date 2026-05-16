Remove 61 duplicate entries (6 SYCL, 55 Vulkan) from `feature_extractor_list[]`
in `libvmaf/src/feature/feature_extractor.c`. The duplicates caused unnecessary
linear-scan overhead on every extractor lookup; the first matching entry was
always returned so feature availability was never affected.
