- **perf(dnn)**: Cache `OrtMemoryInfo` in `VmafOrtSession` at open time; release at
  close. Removes one ORT allocation round-trip per frame in `vmaf_ort_infer` and
  `vmaf_ort_run` (perf audit F2-A / F3-A, 2026-05-16).
- **perf(vulkan/psnr-hvs)**: Move `vkUpdateDescriptorSets` from `extract()` to
  `init()` in `psnr_hvs_vulkan`. Buffer bindings are invariant after init, so
  the per-frame update was redundant (perf audit VK-6, 2026-05-16).
