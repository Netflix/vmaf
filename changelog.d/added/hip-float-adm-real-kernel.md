## HIP float_adm real kernel (ADR-0468)

Port the CUDA `float_adm_cuda.c` / `float_adm_score.cu` four-stage DWT+CSF+CM pipeline
to HIP as the ninth kernel-template consumer. Adds `float_adm_hip.c`,
`float_adm_hip.h`, and `float_adm/float_adm_score.hip`. Wired into
`hip/meson.build` (host TU), `src/meson.build` (`hip_kernel_sources` HSACO entry),
and `feature_extractor.c` (`vmaf_fex_float_adm_hip` registration under `#if HAVE_HIP`).

With `enable_hipcc=true` and a ROCm 6+ toolchain the kernel runs on device and emits
`VMAF_feature_adm2_score` + per-scale ADM subscores. Without `enable_hipcc` the
scaffold posture is preserved: `init()` returns `-ENOSYS`.

Key HIP adaptations: warp size 64 (GCN/RDNA, 4 shared-mem partial slots vs. 8 on
CUDA), `__shfl_down` without mask, `hipMalloc`/`hipMemsetAsync`/`hipMemcpyAsync`
buffer management, `hipModuleLaunchKernel` dispatch.
