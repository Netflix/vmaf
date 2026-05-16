Wire `enable_db` and `clip_db` options into `float_ssim_hip`. Previously
passing `--feature float_ssim_hip:enable_db=true` was silently dropped,
producing linear-domain SSIM instead of the requested dB output.
Matches the parity fix already shipped for `float_ssim_cuda` (PR #969).
