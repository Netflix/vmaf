Restore `enable_db`, `clip_db`, `max_db` fields and dB-domain clamping logic
to `integer_ssim_cuda.c` that were accidentally stripped by PR #1067.
Also restores `CUmodule` to the state struct (enabling `cuModuleUnload` in
close) and adds the missing `<math.h>` / `<stdbool.h>` includes.
