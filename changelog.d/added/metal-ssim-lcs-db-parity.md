Add `enable_lcs`, `enable_db`, `clip_db`, and `scale` option parity to
`float_ssim_metal` (ADR-0484). The Metal extractor now matches the CPU
`float_ssim.c` feature surface: luminance/contrast/structure sub-scores,
dB conversion with optional clamping, and scale=1 enforcement at init time.
The `float_ssim.metal` kernel gains a threadgroup-coalesced LCS partial-sum
path that activates only when `enable_lcs=true`, adding zero dispatch overhead
for the default scalar path.
