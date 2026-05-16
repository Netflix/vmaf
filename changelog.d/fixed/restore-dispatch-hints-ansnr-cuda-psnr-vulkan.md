**fix(cuda/ansnr, vulkan/psnr):** Restore `.options` and `.chars` dispatch hints to
`float_ansnr_cuda` and `float_psnr_vulkan` that were dropped when PR #1067 touched those
files. The scheduler was falling back to defaults instead of the correct
single-dispatch / reduction-only profile (ADR-0182 / ADR-0195).
