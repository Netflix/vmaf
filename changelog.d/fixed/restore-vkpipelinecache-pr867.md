fix(vulkan): restore VkPipelineCache persistence clobbered by PR #1067

PR #1067 (refactor/bootstrap-name-builder) inadvertently reverted all
pipeline-cache code originally landed by PR #867 (ADR-0445):
- Dropped `VkPipelineCache pipeline_cache` field from `VmafVulkanContext`
- Dropped the full `pipeline_cache_init` / `pipeline_cache_save` helper
  block (~300 LOC) from `common.c`
- Dropped the three `pipeline_cache_init` / destroy call-sites in
  `vmaf_vulkan_context_new`, `vmaf_vulkan_context_new_external`, and
  `vmaf_vulkan_context_destroy`

This restores both files byte-for-byte to their state immediately
before PR #1067 merged (commit 03bcc8fc0).
