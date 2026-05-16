**float_ssim Metal: restore `scale` option clobbered by PR #1067.**
PR #1067 (refactor/ADR-0480) accidentally reverted the `scale_override` struct
field, the `VmafOption` entry, and the `ssim_metal_validate_scale()` path that
PR #1058 (commit `e639d4741`) had added. This restores all three exactly as
originally authored, re-establishing GPU-parity with the HIP and Vulkan twins.
