Deduplicate GPU dispatch env-variable tokenizer: extract shared
`vmaf_gpu_dispatch_parse_env` into `libvmaf/src/gpu_dispatch_parse.h`,
eliminating the verbatim `parse_per_feature_override` triplicate across
CUDA, SYCL, and Vulkan dispatch_strategy TUs (ADR-0483).
