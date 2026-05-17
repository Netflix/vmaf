**float_vif CUDA: add missing `vif_skip_scale0` option parity.**
`float_vif_cuda` accepted the option but silently included scale-0 in every
score aggregation. The host-side collect path now suppresses scale-0 emission
(`0.0` score, `0.0`/`-1.0` debug num/den) when `vif_skip_scale0=true`,
matching `float_vif.c`, SYCL, and Vulkan.
