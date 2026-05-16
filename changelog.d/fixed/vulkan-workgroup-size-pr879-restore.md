**vulkan**: Restore 32-wide workgroup sizes clobbered by PR #1067. `ssimulacra2_blur.comp`
and `cambi_mask_dp.comp` PASS 0/1 revert to `local_size_x = 32` (from the erroneous 1×1×1),
and the matching host dispatch in `ssimulacra2_vulkan.c` / `cambi_vulkan.c` reverts to
`ceil(lines/32)` groups along X (from the erroneous `(h,1,1)` / `(1,lines,1)` forms).
Restores the VK-1 + VK-2 perf fix originally landed in PR #879.
