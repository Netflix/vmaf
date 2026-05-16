# Included from libvmaf/src/meson.build via `subdir('hip')`. Paths
# below are relative to libvmaf/src/hip/. The feature stubs live
# under libvmaf/src/feature/hip/, so we use a `..` walk to reach
# them — same shape ADR-0175 uses for the Vulkan scaffold.
#
# Scaffold-only (ADR-0209 / T7-10): every entry point returns -ENOSYS
# until the kernels land, so the build has no HIP / ROCm SDK
# requirement yet. The dependency() probe below stays optional and
# is only consulted if the toolchain happens to expose pkg-config
# metadata for `hip-lang`; the scaffold compiles cleanly on a stock
# Ubuntu runner without any AMD packages installed.
# Kernel-template helpers (T7-10 first-consumer / ADR-0241):
# `kernel_template.c` carries the per-frame async lifecycle helpers
# (stream + event pair + readback alloc/free) that mirror
# `cuda/kernel_template.h`. Bodies return -ENOSYS while the runtime
# is absent; the runtime PR (T7-10b) replaces them with real HIP
# calls without touching consumer TUs.
#
# `feature/hip/integer_psnr_hip.c` is the first kernel-template
# consumer — registers `vmaf_fex_psnr_hip` and exercises every
# template helper in init/submit/collect/close. -ENOSYS at the
# template layer surfaces verbatim through the consumer.
#
# `feature/hip/ciede_hip.c` is the third kernel-template consumer
# (ADR-0257) — registers `vmaf_fex_ciede_hip` and mirrors the CUDA
# twin (`feature/cuda/integer_ciede_cuda.c`). Same scaffold posture:
# the consumer compiles host-only and `init()` surfaces -ENOSYS
# through the kernel-template helpers until T7-10b lands real
# `hipStreamCreate` / `hipMemcpyAsync` bodies.
#
# `feature/hip/float_moment_hip.c` is the fourth kernel-template
# consumer (ADR-0258) — registers `vmaf_fex_float_moment_hip` and
# mirrors the CUDA twin (`feature/cuda/integer_moment_cuda.c`).
# Same scaffold posture as the third consumer; emits four features
# (`float_moment_ref{1st,2nd}`, `float_moment_dis{1st,2nd}`) once
# the runtime kernel arrives.
# `feature/hip/float_ansnr_hip.c` is the fifth kernel-template
# consumer (ADR-0266) — registers `vmaf_fex_float_ansnr_hip` and
# mirrors the CUDA twin (`feature/cuda/float_ansnr_cuda.c`). Same
# scaffold posture as the prior consumers; the submit path
# intentionally bypasses `vmaf_hip_kernel_submit_pre_launch`
# because the kernel writes per-block (sig, noise) float partials
# directly (no atomic, no memset).
#
# `feature/hip/integer_motion_v2_hip.c` is the sixth kernel-template
# consumer (ADR-0267) — registers `vmaf_fex_integer_motion_v2_hip`
# and mirrors the CUDA twin
# (`feature/cuda/integer_motion_v2_cuda.c`). Pins the **temporal**
# extractor shape (`VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag,
# `flush()` callback) and the ping-pong buffer carry-over needed
# for motion-class metrics. Same scaffold posture: `init()`
# returns -ENOSYS until T7-10b.
#
# `feature/hip/float_motion_hip.c` is the seventh kernel-template
# consumer (ADR-0270) — registers `vmaf_fex_float_motion_hip` and
# mirrors the CUDA twin (`feature/cuda/float_motion_cuda.c`). Pins
# the temporal-extractor shape with a raw-pixel cache + blurred-frame
# ping-pong (`uintptr_t ref_in` + `uintptr_t blur[2]` slots tracked
# outside the kernel-template's readback bundle); emits two features
# (`VMAF_feature_motion_score`, `VMAF_feature_motion2_score`) once
# the runtime kernel arrives.
#
# `feature/hip/float_ssim_hip.c` is the eighth kernel-template
# consumer (ADR-0271) — registers `vmaf_fex_float_ssim_hip` and
# mirrors the CUDA twin (`feature/cuda/integer_ssim_cuda.c`, despite
# its filename emits `float_ssim`). Pins the two-dispatch + five
# intermediate float buffers shape; emits one feature
# (`float_ssim`) once the runtime kernel arrives. v1: scale=1 only,
# auto-decimation rejected at init with -EINVAL (mirrors the CUDA
# twin's validation).
hip_sources += files(
    'common.c',
    'picture_hip.c',
    'dispatch_strategy.c',
    'kernel_template.c',
    '../feature/hip/adm_hip.c',
    '../feature/hip/vif_hip.c',
    '../feature/hip/motion_hip.c',
    '../feature/hip/integer_psnr_hip.c',
    '../feature/hip/ciede_hip.c',
    '../feature/hip/float_moment_hip.c',
    '../feature/hip/float_ansnr_hip.c',
    '../feature/hip/integer_motion_v2_hip.c',
    '../feature/hip/float_motion_hip.c',
    '../feature/hip/float_ssim_hip.c',
    # T7-10b first real kernel: float_psnr_hip (ADR-0254). Registered under
    # `#if HAVE_HIP` in feature_extractor.c. With `enable_hipcc=true` the
    # HSACO blob is embedded and the kernel runs on device; without it
    # init() returns -ENOSYS (same scaffold-posture as the other consumers).
    '../feature/hip/float_psnr_hip.c',
)

# T7-10b runtime linkage. ROCm ships `hip_runtime_api.h` under
# `/opt/rocm/include` and the host runtime under
# `/opt/rocm/lib/libamdhip64.so`. The ROCm 7.x package does NOT
# publish a `hip-lang.pc`; the cmake config is the supported
# discovery path. Try meson's pkg-config / cmake probe first; on
# miss, fall back to a manual `find_library` rooted at the usual
# ROCm install prefixes plus an `HIP_PATH` env override.
hip_search_paths = []
hip_path_env = run_command(
    'sh', '-c', 'printf %s "${HIP_PATH:-}"', check: true).stdout()
if hip_path_env != ''
    hip_search_paths += [hip_path_env / 'lib', hip_path_env / 'lib64']
endif
hip_search_paths += [
    '/opt/rocm/lib',
    '/opt/rocm/lib64',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib64',
]

hip_runtime_dep = dependency('hip-lang', required: false)
if not hip_runtime_dep.found()
    hip_lib = cc.find_library('amdhip64',
        dirs: hip_search_paths,
        required: true)
    hip_include_candidates = []
    if hip_path_env != ''
        hip_include_candidates += hip_path_env / 'include'
    endif
    hip_include_candidates += ['/opt/rocm/include', '/usr/include']
    hip_include_dir = ''
    foreach candidate : hip_include_candidates
        if hip_include_dir == '' and run_command(
            'sh', '-c', 'test -f "@0@/hip/hip_runtime_api.h"'.format(candidate),
            check: false).returncode() == 0
            hip_include_dir = candidate
        endif
    endforeach
    if hip_include_dir == ''
        error('HIP runtime header `<hip/hip_runtime_api.h>` not found. ' +
              'Set HIP_PATH or install rocm-hip-runtime-dev.')
    endif
    hip_runtime_dep = declare_dependency(
        dependencies: [hip_lib],
        compile_args: ['-D__HIP_PLATFORM_AMD__=1'],
        include_directories: include_directories(hip_include_dir, is_system: true),
    )
endif
hip_deps += hip_runtime_dep
