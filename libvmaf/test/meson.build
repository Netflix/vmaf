if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

# Aggregate of all GPU/compute backend dependency objects so test
# binaries that must link against ANY enabled backend can reference
# one variable instead of repeating the per-backend list.
gpu_all_deps = [
  cuda_dependency,
  sycl_dependency,
  vulkan_deps,
  hip_deps,
  metal_deps,
]

test_context = executable('test_context',
    ['test.c', 'test_context.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies:[stdatomic_dependency, pthread_dependency, gpu_all_deps],
)

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c', '../src/ref.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies:[stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps],
)

test_propagate_metadata = executable('test_propagate_metadata',
    ['test.c', 'test_propagate_metadata.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c', '../src/log.c', '../src/predict.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/feature/'), include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, gpu_all_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

# Coverage follow-up to PR #313 (R3 from
# docs/development/coverage-gap-analysis-2026-05-02.md): exercise
# libvmaf/src/output.c writers (XML / JSON / CSV / SUB) end-to-end through
# tmpfile-backed FILE* sinks. test_output.c #include's output.c directly so
# the per-translation-unit .gcno lands in this test's build dir and gcovr
# picks it up. Mirrors the test_feature_collector wiring (link_with libvmaf
# + libsvm objects) for the supporting symbols.
test_output = executable('test_output',
    ['test.c', 'test_output.c', '../src/log.c', '../src/predict.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/feature/'), include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, gpu_all_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_log = executable('test_log',
    ['test.c', 'test_log.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_thread_pool = executable('test_thread_pool',
    ['test.c', 'test_thread_pool.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : [pthread_dependency, thread_lib, sycl_dependency],
)

test_model = executable('test_model',
    ['test.c', 'test_model.c', '../src/dict.c', '../src/pdjson.c', '../src/read_json_model.c', '../src/log.c', json_model_c_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common, '-DJSON_MODEL_PATH="'+join_paths(meson.project_source_root(), '../model/')+'"'],
    dependencies : [pthread_dependency, thread_lib, gpu_all_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/dict.c', '../src/metadata_handler.c',
     '../src/feature/feature_collector.c', '../src/feature/alias.c', '../src/model.c', '../src/log.c',
     '../src/read_json_model.c', '../src/pdjson.c', json_model_c_sources, '../src/feature/feature_name.c', '../src/feature/feature_extractor.c',],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    dependencies : [pthread_dependency, thread_lib, gpu_all_deps],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_feature_extractor = executable('test_feature_extractor',
    # dnn_sources pulled in directly so feature_lpips.c can resolve its
    # libvmaf/dnn.h references (session_open/run/close,
    # vmaf_tensor_from_rgb_imagenet) without linking against libvmaf.so.
    # fex_ctx_vector.c is part of libvmaf_sources (not libvmaf_feature_static_lib)
    # so it must be listed explicitly; the T-CUDA-FEATURE-EXTRACTOR-DOUBLE-WRITE
    # regression test exercises feature_extractor_vector_append() directly.
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c',
     '../src/fex_ctx_vector.c',
     dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test_dict = executable('test_dict',
    ['test.c', 'test_dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_opt = executable('test_opt',
    ['test.c', 'test_opt.c', '../src/opt.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [math_lib, pthread_dependency, sycl_dependency],
)

test_cpu = executable('test_cpu',
    ['test.c', 'test_cpu.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    objects : libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
    dependencies: [pthread_dependency, sycl_dependency],
)

test_ref = executable('test_ref',
    ['test.c', 'test_ref.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_feature = executable('test_feature',
    ['test.c', 'test_feature.c', '../src/feature/alias.c', '../src/dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [pthread_dependency, sycl_dependency],
)

test_ciede = executable('test_ciede',
    ['test.c', 'test_ciede.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, gpu_all_deps],
)

# LPIPS registration tests — verify feature wiring without depending on a
# real ORT session. Links against libvmaf_feature_static_lib for the
# extractor list and pulls in opt.c for the VmafOption typedef.
test_lpips = executable('test_lpips',
    ['test.c', 'test_lpips.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# DISTS-Sq registration tests — mirrors test_lpips and verifies the
# two-input extractor is discoverable without requiring an ORT session.
test_dists = executable('test_dists',
    ['test.c', 'test_dists.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# FastDVDnet temporal pre-filter (T6-7) — same registration-only test
# pattern as test_lpips. Verifies the 5-frame extractor wiring without
# requiring a real ORT session; CLI smoke against
# model/tiny/fastdvdnet_pre.onnx covers the live inference path.
test_fastdvdnet_pre = executable('test_fastdvdnet_pre',
    ['test.c', 'test_fastdvdnet_pre.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# MobileSal saliency registration tests (T6-2a) — same pattern as
# test_lpips above. Verifies the extractor wires up without depending
# on a real ORT session; full inference is exercised by the CLI smoke
# gate against model/tiny/mobilesal.onnx.
test_mobilesal = executable('test_mobilesal',
    ['test.c', 'test_mobilesal.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

# TransNet V2 shot-boundary detector — structural test, no real ORT
# session required (mirrors test_lpips). CLI smoke against
# model/tiny/transnet_v2.onnx covers the live inference path.
test_transnet_v2 = executable('test_transnet_v2',
    ['test.c', 'test_transnet_v2.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test_cambi = executable('test_cambi',
    ['test.c', 'test_cambi.c', '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, gpu_all_deps],
)

cambi_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if cambi_simd_test_archs.contains(host_machine.cpu_family())
test_cambi_simd = executable('test_cambi_simd',
    ['test.c', 'test_cambi_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

# speed_chroma + speed_temporal registration smoke tests — link against
# libvmaf_feature_static_lib for the extractor list (mirrors test_lpips).
# Both extractors live in libvmaf/src/feature/speed.c, which is only
# compiled when -Denable_float=true (see libvmaf/src/meson.build); the
# corresponding entries in feature_extractor.c's extractor list are
# wrapped in `#if VMAF_FLOAT_FEATURES`. Gate the test on the same option
# so a default `meson setup` (enable_float=false) doesn't fail on a
# config that legitimately omits these extractors.
if get_option('enable_float')
test_speed = executable('test_speed',
    ['test.c', 'test_speed.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)
endif
# SpEED-QA extractor tests -- unconditional (speed_qa.c is not float-gated).
# ADR-0253 real-impl: registration, vtable, flat/noise entropy, temporal positivity.
test_speed_qa = executable('test_speed_qa',
    ['test.c', 'test_speed_qa.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib, gpu_all_deps,
                    dnn_deps],
    objects : [
        common_cuda_objects,
        platform_specific_cpu_objects,
        libvmaf_feature_static_lib.extract_all_objects(recursive: true),
        libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
        libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test_luminance_tools = executable('test_luminance_tools',
    ['test.c', 'test_luminance_tools.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
)

test_cli_parse = executable('test_cli_parse',
    ['test.c', 'test_cli_parse.c', '../tools/cli_parse.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), include_directories('../tools/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [compat_cflags],
    dependencies: [pthread_dependency, getopt_dependency, sycl_dependency],
)

# test_y4m_411_oob uses POSIX fmemopen which has no MSVC/MinGW
# equivalent. The reproducer exercises the Y4M parser via an
# in-memory FILE*; on Windows the test isn't built (the bug is
# parser-side, the test wiring isn't platform-specific). Guarded
# off on Windows targets only.
if host_machine.system() != 'windows'
    test_y4m_411_oob = executable('test_y4m_411_oob',
        ['test_y4m_411_oob.c', '../tools/vidinput.c', '../tools/y4m_input.c', '../tools/yuv_input.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), include_directories('../tools/')],
        link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        c_args : [compat_cflags],
        dependencies: [pthread_dependency, sycl_dependency],
    )

    # test_cli_parse_long_only_args forks a child to drive
    # cli_parse() with intentionally-bad argv and asserts the
    # child exits with status 1 + a "usage error" stderr line
    # — instead of dying from the SIGABRT the pre-ADR-0316
    # `error()` assertion would raise. POSIX fork()/waitpid()
    # required; gated off Windows alongside test_y4m_411_oob.
    test_cli_parse_long_only_args = executable('test_cli_parse_long_only_args',
        ['test.c', 'test_cli_parse_long_only_args.c', '../tools/cli_parse.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), include_directories('../tools/')],
        link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        c_args : [compat_cflags],
        dependencies: [pthread_dependency, getopt_dependency, sycl_dependency],
    )
endif

test_psnr = executable('test_psnr',
    ['test.c', 'test_psnr.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
)

test_framesync = executable('test_framesync',
    ['test.c', 'test_framesync.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [pthread_dependency, thread_lib, sycl_dependency],
)

# Ported from upstream Netflix/vmaf c70debb1 ("libvmaf/test: port new
# adm/vif/speed tests"). The vif_tools and speed_chroma test files from that
# commit depend on upstream symbols (NUM_KERNELSCALES, vif_validate_kernelscale,
# speed_get_antialias_filter, the SpEED feature) that have not been ported to
# the fork; only the adm_csf and barten_csf cases are wired here.
test_adm_csf = executable('test_adm_csf',
    ['test.c', 'test_adm_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, pthread_dependency, sycl_dependency],
)

test_barten_csf = executable('test_barten_csf',
    ['test.c', 'test_barten_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, pthread_dependency, sycl_dependency],
)

# Bit-exactness between the scalar-separable reference and the arch-
# specific SIMD paths (x86 AVX2/AVX-512 and arm64 NEON). Links the
# platform SIMD object libraries directly so the kernels can be called
# without CPU-flag gating from inside the test. See
# docs/adr/0125-ms-ssim-decimate-simd.md.
iqa_convolve_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if iqa_convolve_test_archs.contains(host_machine.cpu_family())
test_iqa_convolve = executable('test_iqa_convolve',
    ['test.c', 'test_iqa_convolve.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

psnr_hvs_avx2_test_archs = ['x86_64', 'x86']
if psnr_hvs_avx2_test_archs.contains(host_machine.cpu_family())
test_psnr_hvs_avx2 = executable('test_psnr_hvs_avx2',
    ['test.c', 'test_psnr_hvs_avx2.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

motion_v2_simd_test_archs = ['x86_64', 'x86']
if motion_v2_simd_test_archs.contains(host_machine.cpu_family())
test_motion_v2_simd = executable('test_motion_v2_simd',
    ['test.c', 'test_motion_v2_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

ssimulacra2_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if ssimulacra2_simd_test_archs.contains(host_machine.cpu_family())
# `-ffp-contract=off` is required for bit-exactness on aarch64:
# GCC 10+ and clang default to `-ffp-contract=fast` on aarch64, which
# auto-fuses plain C `a*b+c` into `fmla`. The scalar reference in the
# test TU would fuse; NEON intrinsics do not fuse (compiler treats
# them as opaque). x86 avoids this because `-mfma` is scoped to the
# AVX2/AVX-512 static libs only, so scalar code can't fuse on x86
# regardless of contract setting. See ADR-0161 follow-up notes.
test_ssimulacra2_simd = executable('test_ssimulacra2_simd',
    ['test.c', 'test_ssimulacra2_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines + ['-ffp-contract=off'],
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

psnr_hvs_neon_test_archs = ['aarch64', 'arm64']
if psnr_hvs_neon_test_archs.contains(host_machine.cpu_family())
test_psnr_hvs_neon = executable('test_psnr_hvs_neon',
    ['test.c', 'test_psnr_hvs_neon.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

moment_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if float_enabled and moment_simd_test_archs.contains(host_machine.cpu_family())
test_moment_simd = executable('test_moment_simd',
    ['test.c', 'test_moment_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

ansnr_simd_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if ansnr_simd_test_archs.contains(host_machine.cpu_family())
test_ansnr_simd = executable('test_ansnr_simd',
    ['test.c', 'test_ansnr_simd.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

ms_ssim_decimate_test_archs = ['x86_64', 'x86', 'aarch64', 'arm64']
if ms_ssim_decimate_test_archs.contains(host_machine.cpu_family())
test_ms_ssim_decimate = executable('test_ms_ssim_decimate',
    ['test.c', 'test_ms_ssim_decimate.c',
     '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'),
                           include_directories('../src/feature/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ],
)
endif

test_locale_handling = executable('test_locale_handling',
    ['test.c', 'test_locale_handling.c'],
    include_directories : [
        libvmaf_inc,
        test_inc,
        include_directories('../src/'),
        include_directories('..')
    ],
    c_args : ['-I' + meson.project_build_root() / 'src'] + vmaf_cflags_common,
    dependencies : [pthread_dependency],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

if get_option('enable_cuda')
test_gpu_picture_pool = executable('test_gpu_picture_pool',
    ['test.c', 'test_gpu_picture_pool.c', '../src/gpu_picture_pool.c', '../src/cuda/picture_cuda.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test_cuda_pic_preallocation = executable('test_cuda_pic_preallocation',
    ['test.c', 'test_cuda_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test('test_gpu_picture_pool', test_gpu_picture_pool, suite : ['fast', 'gpu'])
test('test_cuda_pic_preallocation', test_cuda_pic_preallocation, suite : ['fast', 'gpu'])

# Netflix#1420 — graceful-OOM reducer. Exercises the wholesale
# CHECK_CUDA_GOTO/_RETURN rewrite by requesting an impossible
# 1 TiB cuMemAlloc and asserting the returned errno is -ENOMEM
# rather than an aborted process. GPU-gated: skips cleanly if
# no driver/device is visible to the process.
test_cuda_buffer_alloc_oom = executable('test_cuda_buffer_alloc_oom',
    ['test.c', 'test_cuda_buffer_alloc_oom.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)
test('test_cuda_buffer_alloc_oom', test_cuda_buffer_alloc_oom, suite : ['fast', 'gpu'])

# Netflix#1300 / ADR-0157 — preallocation-leak reducer. Runs N
# init/preallocate/fetch/close cycles with FULL cleanup on each cycle
# (vmaf_close + vmaf_cuda_state_free + vmaf_model_destroy). ASan is the
# gate: pre-fix, the framework leaked the VmafCudaState struct, the
# CudaFunctions dlopen table, and the ring-buffer mutex state on every
# cycle. GPU-gated: skips cleanly if no driver/device is visible.
test_cuda_preallocation_leak = executable('test_cuda_preallocation_leak',
    ['test.c', 'test_cuda_preallocation_leak.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, cuda_dependency],
    c_args: ['-DHAVE_CUDA=1']
)
test('test_cuda_preallocation_leak', test_cuda_preallocation_leak, suite : ['fast', 'gpu'])
endif

# T6-2b / ADR-0247 — vmaf-roi sidecar core helpers (per-CTU mean reduce
# + saliency-to-QP-offset). Header-only test that compiles
# libvmaf/tools/vmaf_roi_core.h and exercises clamp / sign / monotonicity
# without needing the full binary.
test_vmaf_roi_libm = cc.find_library('m', required : false)
test_vmaf_roi = executable('test_vmaf_roi',
    ['test.c', 'test_vmaf_roi.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../tools/')],
    dependencies: [pthread_dependency, test_vmaf_roi_libm],
)
test('test_vmaf_roi', test_vmaf_roi, suite : ['fast'])

# picture pool test - exercises vmaf_preallocate_pictures / vmaf_fetch_preallocated_picture.
# These symbols are always linked into libvmaf (no build gate), so this test
# always runs.
test_pic_preallocation = executable('test_pic_preallocation',
    ['test.c', 'test_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, thread_lib],
)
test('test_pic_preallocation', test_pic_preallocation, suite : ['fast'])

# ADR-0152 — monotonic-index guard on vmaf_read_pictures (Netflix#910).
test_read_pictures_monotonic = executable('test_read_pictures_monotonic',
    ['test.c', 'test_read_pictures_monotonic.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, thread_lib],
)
test('test_read_pictures_monotonic', test_read_pictures_monotonic, suite : ['fast'])

# ADR-0154 — score_pooled returns -EAGAIN instead of -EINVAL for
# not-yet-written features (Netflix#755).
test_score_pooled_eagain = executable('test_score_pooled_eagain',
    ['test.c', 'test_score_pooled_eagain.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, pthread_dependency, thread_lib],
)
test('test_score_pooled_eagain', test_score_pooled_eagain, suite : ['fast'])

# ADR-0153 — float_ms_ssim init rejects <176x176 (Netflix#1414).
test_float_ms_ssim_min_dim = executable('test_float_ms_ssim_min_dim',
    ['test.c', 'test_float_ms_ssim_min_dim.c', '../src/mem.c', '../src/picture.c',
     '../src/ref.c', '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)
test('test_float_ms_ssim_min_dim', test_float_ms_ssim_min_dim, suite : ['fast'])

# Research-0094 regression: motion init() must reject frames < 3x3 with -EINVAL
# instead of reading out-of-bounds via the reflect-101 mirror-padding formula.
test_motion_min_dim = executable('test_motion_min_dim',
    ['test.c', 'test_motion_min_dim.c', '../src/mem.c', '../src/picture.c',
     '../src/ref.c', '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c', '../src/thread_locale.c', '../src/gpu_picture_pool.c', dnn_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), dnn_inc],
    c_args : vmaf_cflags_common + dnn_defines,
    dependencies : [math_lib, stdatomic_dependency, pthread_dependency, thread_lib,
                    gpu_all_deps, dnn_deps],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)
test('test_motion_min_dim', test_motion_min_dim, suite : ['fast'])

if get_option('enable_sycl')
test_sycl = executable('test_sycl',
    ['test.c', 'test_sycl.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
    c_args: ['-DHAVE_SYCL=1']
)

test_sycl_pic_preallocation = executable('test_sycl_pic_preallocation',
    ['test.c', 'test_sycl_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
    c_args: ['-DHAVE_SYCL=1']
)

test('test_sycl', test_sycl, suite : ['fast', 'gpu'])
test('test_sycl_pic_preallocation', test_sycl_pic_preallocation, suite : ['fast', 'gpu'])

test_integer_cambi_sycl = executable('test_integer_cambi_sycl',
    ['test.c', 'test_integer_cambi_sycl.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, sycl_dependency],
    c_args: ['-DHAVE_SYCL=1']
)
test('test_integer_cambi_sycl', test_integer_cambi_sycl, suite : ['fast', 'gpu'])
endif

# ADR-0175 / T5-1: Vulkan backend scaffold smoke. Pins the -ENOSYS
# contract so a future runtime PR can't accidentally enable the
# backend without flipping the smoke expectations.
if get_option('enable_vulkan').enabled()
test_vulkan_smoke = executable('test_vulkan_smoke',
    ['test.c', 'test_vulkan_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_smoke', test_vulkan_smoke, suite : ['fast', 'gpu'])

# T7-29 part 4 (ADR-0251): contract smoke for the v2 async
# pending-fence ring. Public-API only — the GPU-end-to-end
# ring-wrap correctness path lives in the FFmpeg filter
# downstream (T7-29 part 3) and the cross-backend gate.
test_vulkan_async_pending_fence = executable('test_vulkan_async_pending_fence',
    ['test.c', 'test_vulkan_async_pending_fence.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_async_pending_fence', test_vulkan_async_pending_fence, suite : ['fast', 'gpu'])

# ADR-0238: contract smoke for the public Vulkan picture-preallocation
# surface (vmaf_vulkan_preallocate_pictures / vmaf_vulkan_picture_fetch).
test_vulkan_pic_preallocation = executable('test_vulkan_pic_preallocation',
    ['test.c', 'test_vulkan_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_VULKAN=1']
)
test('test_vulkan_pic_preallocation', test_vulkan_pic_preallocation, suite : ['fast', 'gpu'])
endif

# ADR-0209 / T5-2: embedded MCP scaffold smoke. Pins the -ENOSYS
# contract on every public entry point. T5-2b runtime
# (ADR-0209 § "Status update 2026-05-08") flipped init/start_stdio/
# close to a working dispatcher; the test now drives a real JSON-RPC
# round-trip over pipes. SSE/UDS expectations remain pinned at -ENOSYS.
if get_option('enable_mcp')
test_mcp_smoke = executable('test_mcp_smoke',
    ['test.c', 'test_mcp_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_MCP=1']
)
test('test_mcp_smoke', test_mcp_smoke, suite : ['slow'], timeout: 60)
endif

# ADR-0212 / T7-10: HIP (AMD ROCm) backend smoke.
# Scaffold-shape sub-tests pin the public-API contract; the T7-10b
# runtime sub-tests probe `hipGetDeviceCount` and skip the device-
# resident assertions when no AMD GPU is visible to the runtime.
if get_option('enable_hip') == true
test_hip_smoke = executable('test_hip_smoke',
    ['test.c', 'test_hip_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency, hip_deps],
    c_args: ['-DHAVE_HIP=1']
)
test('test_hip_smoke', test_hip_smoke, suite : ['fast', 'gpu'])
endif

# ADR-0338 / T8-1: Metal (Apple Silicon) backend scaffold smoke. Same
# pattern as the HIP / Vulkan scaffolds above — pins the -ENOSYS
# contract for every public C-API entry point until the runtime PR
# (T8-1b) lands. Built whenever the parent meson resolved
# is_metal_enabled to true (auto on macOS, or `enabled` explicitly).
metal_test_opt = get_option('enable_metal')
if metal_test_opt.enabled() or (metal_test_opt.auto() and host_machine.system() == 'darwin')
test_metal_smoke = executable('test_metal_smoke',
    ['test.c', 'test_metal_smoke.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_METAL=1']
)
test('test_metal_smoke', test_metal_smoke, suite : ['fast', 'gpu'])

# ADR-0437 — install-gate: verify libvmaf_metal.h is reachable via the
# installed include path and that vmaf_metal_import_state + the IOSurface
# sub-API are declared with the correct prototypes (compile+link check only;
# no Metal runtime calls, so safe on runners without Apple-Family-7 devices).
# Gated to macOS only: the header requires Apple SDK types even in stub mode.
if host_machine.system() == 'darwin'
test_metal_install_header = executable('test_metal_install_header',
    ['test.c', 'test_metal_install_header.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [pthread_dependency],
    c_args: ['-DHAVE_METAL=1']
)
test('test_metal_install_header', test_metal_install_header, suite : ['fast', 'gpu'])
endif
endif

test('test_picture', test_picture, suite : ['fast'])
test('test_feature_collector', test_feature_collector, suite : ['fast'])
test('test_output', test_output, suite : ['fast'])
test('test_thread_pool', test_thread_pool, suite : ['fast'])
test('test_model', test_model, suite : ['fast'])
test('test_predict', test_predict, suite : ['fast'])
test('test_feature_extractor', test_feature_extractor, suite : ['fast'])
test('test_dict', test_dict, suite : ['fast'])
test('test_opt', test_opt, suite : ['fast'])
test('test_cpu', test_cpu, suite : ['fast'])
test('test_ref', test_ref, suite : ['fast'])
test('test_feature', test_feature, suite : ['fast'])
test('test_ciede', test_ciede, suite : ['fast'])
test('test_lpips', test_lpips, suite : ['fast'])
test('test_dists', test_dists, suite : ['fast'])
test('test_fastdvdnet_pre', test_fastdvdnet_pre, suite : ['fast'])
test('test_mobilesal', test_mobilesal, suite : ['fast'])
test('test_transnet_v2', test_transnet_v2, suite : ['fast'])
test('test_cambi', test_cambi, suite : ['fast'])

if cambi_simd_test_archs.contains(host_machine.cpu_family())
test('test_cambi_simd', test_cambi_simd, suite : ['fast', 'simd'])
endif
if get_option('enable_float')
test('test_speed', test_speed, suite : ['fast'])
endif
test('test_speed_qa', test_speed_qa, suite : ['fast'])
test('test_luminance_tools', test_luminance_tools, suite : ['fast'])
test('test_cli_parse', test_cli_parse, suite : ['fast'])
if host_machine.system() != 'windows'
    test('test_y4m_411_oob', test_y4m_411_oob, suite : ['fast'])
    test('test_cli_parse_long_only_args', test_cli_parse_long_only_args, suite : ['fast'])
endif
test('test_psnr', test_psnr, suite : ['fast'])
test('test_framesync', test_framesync, suite : ['fast'])
test('test_propagate_metadata', test_propagate_metadata, suite : ['fast'])
test('test_adm_csf', test_adm_csf, suite : ['fast'])
test('test_barten_csf', test_barten_csf, suite : ['fast'])
if ms_ssim_decimate_test_archs.contains(host_machine.cpu_family())
test('test_ms_ssim_decimate', test_ms_ssim_decimate, suite : ['fast', 'simd'])
endif

if float_enabled and moment_simd_test_archs.contains(host_machine.cpu_family())
test('test_moment_simd', test_moment_simd, suite : ['fast', 'simd'])
endif

if ansnr_simd_test_archs.contains(host_machine.cpu_family())
test('test_ansnr_simd', test_ansnr_simd, suite : ['fast', 'simd'])
endif

if iqa_convolve_test_archs.contains(host_machine.cpu_family())
test('test_iqa_convolve', test_iqa_convolve, suite : ['fast', 'simd'])
endif

if psnr_hvs_avx2_test_archs.contains(host_machine.cpu_family())
test('test_psnr_hvs_avx2', test_psnr_hvs_avx2, suite : ['fast', 'simd'])
endif

if motion_v2_simd_test_archs.contains(host_machine.cpu_family())
test('test_motion_v2_simd', test_motion_v2_simd, suite : ['fast', 'simd'])
endif

if psnr_hvs_neon_test_archs.contains(host_machine.cpu_family())
test('test_psnr_hvs_neon', test_psnr_hvs_neon, suite : ['fast', 'simd'])
endif

if ssimulacra2_simd_test_archs.contains(host_machine.cpu_family())
test('test_ssimulacra2_simd', test_ssimulacra2_simd, suite : ['fast', 'simd'])
endif

test('test_locale_handling', test_locale_handling, suite : ['fast'])

subdir('dnn')
subdir('fuzz')
