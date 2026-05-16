# Included from libvmaf/src/meson.build via `subdir('vulkan')`. Paths
# below are relative to libvmaf/src/vulkan/. The feature kernels live
# under libvmaf/src/feature/vulkan/.
#
# T5-1b runtime wiring (post-T5-1 scaffold, ADR-0175 follow-up):
#  - dependency('vulkan'): system Vulkan loader (libvulkan.so / vulkan-1.dll).
#  - volk: thin Vulkan function loader so we can dlopen at runtime
#    instead of statically linking every entry point. Bundled via
#    libvmaf/subprojects/volk.wrap (vulkan-sdk-1.4.341.0 sha256-pinned).
#  - vk-mem-alloc (VMA): Vulkan Memory Allocator, single-header.
#    Bundled via libvmaf/subprojects/vk-mem-alloc.wrap (v3.3.0 pinned).
#  - glslc: GLSL → SPIR-V compiler from the Vulkan SDK. Used in a
#    custom_target per .comp shader; the .spv blobs are then embedded
#    via xxd-style header generation so the .so carries them inline.

vulkan_sdk_dep = dependency('vulkan', required : true)
volk_dep       = dependency('volk',        fallback : ['volk', 'volk_dep'])
vma_dep        = dependency('vk-mem-alloc', fallback : ['vk-mem-alloc', 'vk_mem_alloc_dep'])

# ADR-0200: pull the volk private-remap header path from the wrap and
# apply `-include <path>` privately on libvmaf's targets via
# `vmaf_cflags_common`. ADR-0198's first iteration applied this through
# `volk_dep.compile_args`, which leaked the build-dir absolute path into
# the generated `libvmaf.pc` Cflags on **static** builds (lawrence
# repro 2026-04-27: BtbN-style fully-static FFmpeg build saw
# `-include /<libvmaf-build-dir>/subprojects/.../volk_priv_remap.h`
# through `pkg-config --cflags libvmaf` after install — path no longer
# existed → fatal "No such file or directory" in FFmpeg's `configure`).
# `c_args` on `library()` calls do NOT propagate to pkg-config; only
# dependency compile_args do. So we apply the `-include` here, as
# libvmaf-private c_args, instead of through the dependency.
volk_subproj           = subproject('volk')
volk_priv_remap_h_path = volk_subproj.get_variable('volk_priv_remap_h_path')
# Note: the actual `+=` to vmaf_cflags_common is done in the parent
# meson.build directly after `subdir('vulkan')` for scope-propagation
# safety. See the block there.

glslc = find_program('glslc', required : true)

# Compute shaders → SPIR-V → embedded C header. Each `.comp` becomes a
# pair: a `<name>.spv` binary + a `<name>_spv.h` header that declares
# `static const uint32_t <name>_spv[]` for direct embedding into the
# host-side dispatch wrapper.
vulkan_shader_sources = [
    'shaders/vif.comp',
    'shaders/motion.comp',
    'shaders/motion_v2.comp',
    'shaders/adm.comp',
    'shaders/psnr.comp',
    'shaders/moment.comp',
    'shaders/ciede.comp',
    'shaders/ssim.comp',
    'shaders/ms_ssim.comp',
    'shaders/ms_ssim_decimate.comp',
    'shaders/psnr_hvs.comp',
    'shaders/float_ansnr.comp',
    'shaders/float_psnr.comp',
    'shaders/float_motion.comp',
    'shaders/float_vif.comp',
    'shaders/float_adm.comp',
    'shaders/ssimulacra2_xyb.comp',
    'shaders/ssimulacra2_blur.comp',
    'shaders/ssimulacra2_mul.comp',
    'shaders/ssimulacra2_ssim.comp',
    # T-GPU-PERF-VK-3 / ADR-0350 — two-level GPU reduction shaders.
    'shaders/vif_reduce.comp',
    'shaders/adm_reduce.comp',
    'shaders/motion_reduce.comp',
    # T7-36 / ADR-0205 — cambi Vulkan integration.
    'shaders/cambi_preprocess.comp',
    'shaders/cambi_derivative.comp',
    'shaders/cambi_filter_mode.comp',
    'shaders/cambi_decimate.comp',
    'shaders/cambi_mask_dp.comp',
]

vulkan_shader_headers = []
# Shaders that need bit-exact float reproduction of CPU semantics
# build with `-O0` to disable SPIR-V optimizer reassociation /
# FMA contraction. psnr_hvs is sensitive at places=4 because the
# per-coefficient mask threshold ((err - thresh) clamp) flips
# discretely on tiny float drift.
psnr_hvs_strict_shaders = [
    'shaders/psnr_hvs.comp',
    # ssimulacra2 blur is a 3-pole recursive IIR — even one FMA
    # reassociation per pixel compounds across the blur radius and
    # causes the per-scale SSIM stats to drift at the 1e-2 level.
    # ADR-0162 sets places=2 for the CPU SIMD twin; we mirror the
    # `-O0` strict-mode here so the GPU twin doesn't lose another
    # decimal place to the optimizer.
    'shaders/ssimulacra2_blur.comp',
    # XYB cube root + opsin matmul: the SPIR-V optimizer's FMA
    # contraction on `(2*y + x/(y*y)) / 3.0` and on the per-pixel
    # 3x3 mat-mul reorders adds vs the CPU's deterministic
    # vmaf_ss2_cbrtf path, which compounds at the per-scale SSIM
    # stats stage. Keep at `-O0` for places=2 contract parity.
    'shaders/ssimulacra2_xyb.comp',
    # The SSIM/EdgeDiff per-WG reductions sum 128 lanes' contributions
    # via shared-memory halving fold — independent of pass=H/V order
    # but the per-pixel stats expression
    # `(num_m * num_s) / denom_s` is FMA-reorderable. Keep strict.
    'shaders/ssimulacra2_ssim.comp',
    # float_vif scale 2/3 numerical saturation bug (PR #718 / ADR-0352):
    # the SPIR-V optimizer's reassociation of the per-pixel vertical-pass
    # inner loop (`a_xx += c_k * ref_v * ref_v`) and the horizontal-pass
    # sigma computation (`sigma1_sq = xx - mu1*mu1`) causes catastrophic
    # cancellation at scales 2+3 where the smoothed signal has very low
    # local variance.  The resulting underestimated sigma1_sq pushes
    # nearly all pixels into the low-sigma branch (`sigma1_sq < 2.0`),
    # inflating the per-scale score to ~1.0.  -O0 disables the
    # reassociation / FMA-contraction that triggers the regression.
    # The precise qualifiers in the shader provide defence-in-depth
    # against driver-side contraction (Vulkan 1.4 NVIDIA and newer MoltenVK
    # allow the shader compiler to contract even after SPIR-V emission).
    'shaders/float_vif.comp',
]
foreach shader : vulkan_shader_sources
    shader_name = shader.split('/')[-1].replace('.comp', '')
    opt_flag = (shader in psnr_hvs_strict_shaders) ? '-O0' : '-O'
    spv = custom_target(shader_name + '.spv',
        input : '../feature/vulkan/' + shader,
        output : shader_name + '.spv',
        command : [glslc, '--target-env=vulkan1.3', opt_flag, '@INPUT@', '-o', '@OUTPUT@'],
    )
    embed = custom_target(shader_name + '_spv.h',
        input : spv,
        output : shader_name + '_spv.h',
        command : ['python3',
                   meson.current_source_dir() / 'spv_embed.py',
                   '@INPUT@', '@OUTPUT@', shader_name],
    )
    vulkan_shader_headers += embed
endforeach

# VMA requires C++17 (aligned_alloc); libvmaf's project default is
# C++11. Compile the VMA implementation TU as its own static lib so
# the cpp_std override doesn't leak into libvmaf's other C++ TUs.
# vma_impl.cpp also needs the volk priv-remap so its `vk*` calls bind
# to volk's `vmaf_priv_vk*` PFN dispatchers (some extension functions
# like `vkGetPhysicalDeviceMemoryProperties2KHR` aren't exported by
# libvulkan.so directly — only the loader's dlsym path resolves them).
# Apply via cpp_args here so it's local to this target.
vma_impl_lib = static_library(
    'vma_impl',
    'vma_impl.cpp',
    override_options : ['cpp_std=c++17', 'warning_level=0'],
    cpp_args : ['-include', volk_priv_remap_h_path],
    dependencies : [vulkan_sdk_dep, volk_dep, vma_dep],
    pic : true,
    install : false,
)

vulkan_sources += files(
    'common.c',
    'dispatch_strategy.c',
    'import.c',
    'picture_vulkan.c',
    'picture_vulkan_pool.c',
    '../feature/vulkan/adm_vulkan.c',
    '../feature/vulkan/vif_vulkan.c',
    '../feature/vulkan/motion_vulkan.c',
    '../feature/vulkan/motion_v2_vulkan.c',
    '../feature/vulkan/psnr_vulkan.c',
    '../feature/vulkan/moment_vulkan.c',
    '../feature/vulkan/ciede_vulkan.c',
    '../feature/vulkan/ssim_vulkan.c',
    '../feature/vulkan/ms_ssim_vulkan.c',
    '../feature/vulkan/psnr_hvs_vulkan.c',
    '../feature/vulkan/float_ansnr_vulkan.c',
    '../feature/vulkan/float_psnr_vulkan.c',
    '../feature/vulkan/float_motion_vulkan.c',
    '../feature/vulkan/float_vif_vulkan.c',
    '../feature/vulkan/float_adm_vulkan.c',
    '../feature/vulkan/ssimulacra2_vulkan.c',
    '../feature/vulkan/cambi_vulkan.c',
)

vulkan_sources += vulkan_shader_headers

vulkan_deps += vulkan_sdk_dep
vulkan_deps += volk_dep
vulkan_deps += vma_dep
vulkan_deps += declare_dependency(link_with : vma_impl_lib)
