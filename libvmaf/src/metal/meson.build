# Included from libvmaf/src/meson.build via `subdir('metal')`. Paths
# below are relative to libvmaf/src/metal/.
#
# T8-1b runtime (ADR-0420): three Objective-C++ TUs drive
# Metal.framework directly. The per-feature consumer scaffolds under
# `feature/metal/` stay pure-C unless they need Metal API dispatch
# (T8-1c flips integer_motion_v2_metal to .mm; the other seven stay
# .c with -ENOSYS init() until each kernel arrives).

# `add_languages` MUST come before any target — meson errors with
# "Tried to use 'add_project_arguments' after a build target has been
# declared" otherwise (CI run 25686136776 / job 75410354618).
add_languages('objcpp', native: false, required: true)

# Apple framework linkage.
foundation_dep = dependency('Foundation', required: true)
metal_dep      = dependency('Metal', required: true)
metalkit_dep   = dependency('MetalKit', required: false)
iosurface_dep  = dependency('IOSurface', required: true)
metal_deps += foundation_dep
metal_deps += metal_dep
metal_deps += iosurface_dep
if metalkit_dep.found()
    metal_deps += metalkit_dep
endif

# Obj-C++ compile arguments for the .mm TUs. ARC, no Obj-C exception
# unwinding, fobjc-weak so we don't pull all of Foundation into every TU.
metal_objcpp_args = [
    '-fobjc-arc',
    '-fno-objc-arc-exceptions',
    '-fobjc-weak',
]

# Separate static_library for the Obj-C++ TUs — `objcpp_args` is set
# per-target, avoiding the `add_project_arguments` ordering trap.
# Mirrors the x86 AVX2 / NEON sibling pattern: build a dedicated
# static lib for sources that need non-default compile args, then
# extract_all_objects() into the main feature library via
# `metal_sources` (consumed by libvmaf_feature_static_lib in the
# parent meson.build).
#
# The base include directory is the project root (libvmaf/src) so
# `#include "../../metal/common.h"` from the feature/metal TUs and
# `#include "dict.h"` (in libvmaf/src/) both resolve.
# Note: `vmaf_include` (which covers feature/ + feature/common) is
# declared after subdir('metal') in the parent meson.build, so we
# build the include set locally from the same paths.
metal_objcpp_includes = include_directories(
    '.', '..',                 # libvmaf/src/metal/, libvmaf/src/
    '../feature', '../feature/common',
    '../../include',           # libvmaf/include
)

metal_objcpp_lib = static_library(
    'metal_objcpp',
    [
        files('common.mm',
              'picture_metal.mm',
              'picture_import.mm',
              'kernel_template.mm',
              '../feature/metal/integer_motion_v2_metal.mm',
              '../feature/metal/float_psnr_metal.mm',
              '../feature/metal/float_moment_metal.mm',
              '../feature/metal/float_ansnr_metal.mm',
              '../feature/metal/integer_psnr_metal.mm',
              '../feature/metal/float_motion_metal.mm',
              '../feature/metal/integer_motion_metal.mm',
              '../feature/metal/integer_vif_metal.mm',
              '../feature/metal/float_ssim_metal.mm'),
    ],
    include_directories: metal_objcpp_includes,
    objcpp_args: metal_objcpp_args,
    dependencies: [foundation_dep, metal_dep, iosurface_dep],
)

# Pure-C TUs that still go through the main feature library's normal
# C compile path — appended to metal_sources directly.
# NOTE: The seven batch-1/batch-2 scaffolds (ADR-0361) that previously
# lived here as .c -ENOSYS stubs have been replaced by real .mm
# dispatch files (T8-1d–T8-1j / ADR-0421) in metal_objcpp_lib above.
metal_sources += files(
    'dispatch_strategy.c',
)

# The Obj-C++ static lib's objects fold into the same feature
# aggregation the parent meson.build consumes (same shape as
# `x86_avx2_static_lib.extract_all_objects()`).
metal_deps += declare_dependency(
    link_with: metal_objcpp_lib,
)

# ---------------------------------------------------------------------
# Metallib compilation (T8-1c / ADR-0421).
#
# Pipeline: `.metal` → `.air` → `.metallib` via xcrun. The compiled
# metallib is embedded into the libvmaf binary's __TEXT,__metallib
# section via the linker flag `-Wl,-sectcreate,__TEXT,__metallib,…`.
# Consumer .mm reads the byte range via linker-defined symbols.
xcrun = find_program('xcrun', required: true)

mv2_air = custom_target('integer_motion_v2_air',
    input: '../feature/metal/integer_motion_v2.metal',
    output: 'integer_motion_v2.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

float_psnr_air = custom_target('float_psnr_air',
    input: '../feature/metal/float_psnr.metal',
    output: 'float_psnr.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

float_moment_air = custom_target('float_moment_air',
    input: '../feature/metal/float_moment.metal',
    output: 'float_moment.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

float_ansnr_air = custom_target('float_ansnr_air',
    input: '../feature/metal/float_ansnr.metal',
    output: 'float_ansnr.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

integer_psnr_air = custom_target('integer_psnr_air',
    input: '../feature/metal/integer_psnr.metal',
    output: 'integer_psnr.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

float_motion_air = custom_target('float_motion_air',
    input: '../feature/metal/float_motion.metal',
    output: 'float_motion.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

integer_motion_air = custom_target('integer_motion_air',
    input: '../feature/metal/integer_motion.metal',
    output: 'integer_motion.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

float_ssim_air = custom_target('float_ssim_air',
    input: '../feature/metal/float_ssim.metal',
    output: 'float_ssim.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

integer_vif_air = custom_target('integer_vif_air',
    input: '../feature/metal/integer_vif.metal',
    output: 'integer_vif.air',
    command: [xcrun, '-sdk', 'macosx', 'metal',
              '-c', '@INPUT@',
              '-o', '@OUTPUT@'],
)

metal_air_files = [
    mv2_air,
    float_psnr_air,
    float_moment_air,
    float_ansnr_air,
    integer_psnr_air,
    float_motion_air,
    integer_motion_air,
    float_ssim_air,
    integer_vif_air,
]

metallib_target = custom_target('libvmaf-metallib',
    input: metal_air_files,
    output: 'default.metallib',
    command: [xcrun, '-sdk', 'macosx', 'metallib',
              '@INPUT@', '-o', '@OUTPUT@'],
)

metal_deps += declare_dependency(
    sources: metallib_target,
    link_args: ['-Wl,-sectcreate,__TEXT,__metallib,' +
                metallib_target.full_path()],
)
