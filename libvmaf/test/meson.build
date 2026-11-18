if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

test_context = executable('test_context',
    ['test.c', 'test_context.c'],
    include_directories : [libvmaf_inc, test_inc, pthreadwin32_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies:[stdatomic_dependency, cuda_dependency],
)

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c', '../src/ref.c', '../src/log.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies:[stdatomic_dependency, thread_lib, cuda_dependency],
)

test_propagate_metadata = executable('test_propagate_metadata',
    ['test.c', 'test_propagate_metadata.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c', '../src/log.c', '../src/predict.c', '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/feature/'), include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [ cuda_dependency, stdatomic_dependency, thread_lib, rev_dep ],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_log = executable('test_log',
    ['test.c', 'test_log.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
)

test_thread_pool = executable('test_thread_pool',
    ['test.c', 'test_thread_pool.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : thread_lib,
)

test_model = executable('test_model',
    ['test.c', 'test_model.c', '../src/dict.c', '../src/pdjson.c', '../src/read_json_model.c', '../src/log.c', json_model_c_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common, '-DJSON_MODEL_PATH="'+join_paths(meson.project_source_root(), '../model/')+'"'],
    dependencies : [thread_lib, cuda_dependency, stdatomic_dependency],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/dict.c', '../src/metadata_handler.c',
     '../src/feature/feature_collector.c', '../src/feature/alias.c', '../src/model.c', '../src/log.c',
     '../src/read_json_model.c', '../src/pdjson.c', json_model_c_sources, '../src/feature/feature_name.c', '../src/feature/feature_extractor.c',],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    dependencies : [thread_lib, cuda_dependency, stdatomic_dependency],
    objects : libsvm_static_lib.extract_all_objects(recursive: true),
)

test_feature_extractor = executable('test_feature_extractor',
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : [math_lib, stdatomic_dependency, thread_lib, cuda_dependency],
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
)

test_cpu = executable('test_cpu',
    ['test.c', 'test_cpu.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    objects : libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
)

test_ref = executable('test_ref',
    ['test.c', 'test_ref.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : stdatomic_dependency,
)

test_feature = executable('test_feature',
    ['test.c', 'test_feature.c', '../src/feature/alias.c', '../src/dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
)

test_ciede = executable('test_ciede',
    ['test.c', 'test_ciede.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],

    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [thread_lib, cuda_dependency, stdatomic_dependency]
)

test_cambi = executable('test_cambi',
    ['test.c', 'test_cambi.c', '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [thread_lib, cuda_dependency, stdatomic_dependency]
)

test_luminance_tools = executable('test_luminance_tools',
    ['test.c', 'test_luminance_tools.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

test_cli_parse = executable('test_cli_parse',
    ['test.c', 'test_cli_parse.c', '../tools/cli_parse.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/'), include_directories('../tools/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [compat_cflags],
    dependencies : getopt_dependency
)

test_psnr = executable('test_psnr',
    ['test.c', 'test_psnr.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [thread_lib, stdatomic_dependency],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

test_framesync = executable('test_framesync',
    ['test.c', 'test_framesync.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [thread_lib, stdatomic_dependency],
)

test_adm_csf = executable('test_adm_csf',
    ['test.c', 'test_adm_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : math_lib,
)

test_barten_csf = executable('test_barten_csf',
    ['test.c', 'test_barten_csf.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : math_lib,
)

test_conversion_policy = executable('test_conversion_policy',
    ['test.c', 'test_conversion_policy.c', '../src/conversion_policy.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
)

test_motion_blend = executable('test_motion_blend',
    ['test.c', 'test_motion_blend.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
	dependencies : [stdatomic_dependency]
)

test_speed_filter = executable('test_speed_filter',
    ['test.c', 'test_speed_filter.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : math_lib,
)
test('test_speed_filter', test_speed_filter)

# The vertical pass of adm_dwt2_16() used to add up filter taps in an int32_t
# in an order that overflows for samples of 42456 and above. Signed overflow
# wraps back to the right value on the usual targets, so the output is correct
# either way and only an instrumented build can tell. Build the three
# translation units that hold the code under test (integer_adm.c, x86/adm_avx2.c
# and x86/adm_avx512.c) with signed integer overflow trapping, for this test
# only: the other objects come from the regular build, minus the regular
# objects of those three files so no symbol is defined twice. Without compiler
# support for the check the test still compares against the 64-bit reference.
adm_dwt2_16_trap_args = ['-fsanitize=signed-integer-overflow',
                         '-fno-sanitize-recover=signed-integer-overflow']
adm_dwt2_16_trap = cc.has_multi_arguments(adm_dwt2_16_trap_args) and cc.links(
    'int main(void) { return 0; }',
    args : adm_dwt2_16_trap_args,
    name : 'with signed integer overflow trapping',
)
adm_dwt2_16_link_args = []
adm_dwt2_16_objects = [
    common_cuda_objects,
    libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
    libsvm_static_lib.extract_all_objects(recursive: true),
]

if adm_dwt2_16_trap
    adm_dwt2_16_link_args += adm_dwt2_16_trap_args
    adm_dwt2_16_inc = [libvmaf_include, vmaf_include, include_directories('../src/cuda/')]

    adm_dwt2_16_feature_sources = []
    foreach src : libvmaf_feature_sources
        if src != feature_src_dir + 'integer_adm.c'
            adm_dwt2_16_feature_sources += src
        endif
    endforeach
    adm_dwt2_16_trap_adm = static_library(
        'adm_dwt2_16_trap_adm',
        '../src/feature/integer_adm.c',
        include_directories : adm_dwt2_16_inc,
        dependencies : [thread_lib, stdatomic_dependency, cuda_dependency, builtin_clz_dependency],
        c_args : adm_dwt2_16_trap_args,
    )
    adm_dwt2_16_objects += [
        libvmaf_feature_static_lib.extract_objects(adm_dwt2_16_feature_sources),
        adm_dwt2_16_trap_adm.extract_all_objects(recursive: true),
    ]

    if is_asm_enabled and host_machine.cpu_family().startswith('x86')
        adm_dwt2_16_avx2_sources = []
        foreach src : x86_avx2_sources
            if src != feature_src_dir + 'x86/adm_avx2.c'
                adm_dwt2_16_avx2_sources += src
            endif
        endforeach
        adm_dwt2_16_trap_avx2 = static_library(
            'adm_dwt2_16_trap_avx2',
            '../src/feature/x86/adm_avx2.c',
            include_directories : vmaf_base_include,
            c_args : avx2_cflags + vmaf_cflags_common + adm_dwt2_16_trap_args,
            dependencies : [builtin_clz_dependency],
        )
        adm_dwt2_16_objects += [
            x86_avx2_static_lib.extract_objects(adm_dwt2_16_avx2_sources),
            adm_dwt2_16_trap_avx2.extract_all_objects(recursive: true),
        ]

        if is_avx512_enabled and is_avx512_supported
            adm_dwt2_16_avx512_sources = []
            foreach src : x86_avx512_sources
                if src != feature_src_dir + 'x86/adm_avx512.c'
                    adm_dwt2_16_avx512_sources += src
                endif
            endforeach
            adm_dwt2_16_trap_avx512 = static_library(
                'adm_dwt2_16_trap_avx512',
                '../src/feature/x86/adm_avx512.c',
                include_directories : vmaf_base_include,
                c_args : avx512_cflags + vmaf_cflags_common + adm_dwt2_16_trap_args,
                dependencies : [builtin_clz_dependency],
            )
            adm_dwt2_16_objects += [
                x86_avx512_static_lib.extract_objects(adm_dwt2_16_avx512_sources),
                adm_dwt2_16_trap_avx512.extract_all_objects(recursive: true),
            ]
        endif
    else
        adm_dwt2_16_objects += platform_specific_cpu_objects
    endif
else
    adm_dwt2_16_objects += [
        platform_specific_cpu_objects,
        libvmaf_feature_static_lib.extract_all_objects(recursive: true),
    ]
endif

test_adm_dwt2_16 = executable('test_adm_dwt2_16',
    ['test.c', 'test_adm_dwt2_16.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
     '../src/metadata_handler.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : [math_lib, stdatomic_dependency, thread_lib, cuda_dependency],
    link_args : adm_dwt2_16_link_args,
    objects : adm_dwt2_16_objects,
)

if float_enabled
    test_vif_tools = executable('test_vif_tools',
        ['test.c', 'test_vif_tools.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
        link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        dependencies : math_lib,
    )

    test_speed_chroma = executable('test_speed_chroma',
        ['test.c', 'test_speed_chroma.c', '../src/picture.c', '../src/mem.c', '../src/ref.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
        link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        dependencies : math_lib,
    )

    test('test_vif_tools', test_vif_tools)
    test('test_speed_chroma', test_speed_chroma)
endif

if get_option('enable_cuda')
test_ring_buffer = executable('test_ring_buffer',
    ['test.c', 'test_ring_buffer.c', '../src/cuda/ring_buffer.c', '../src/cuda/picture_cuda.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [cuda_dependency, thread_lib, stdatomic_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test_cuda_pic_preallocation = executable('test_cuda_pic_preallocation',
    ['test.c', 'test_cuda_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [cuda_dependency, thread_lib, stdatomic_dependency],
    c_args: ['-DHAVE_CUDA=1']
)

test('test_ring_buffer', test_ring_buffer)
test('test_cuda_pic_preallocation', test_cuda_pic_preallocation)
endif

test_pic_preallocation = executable('test_pic_preallocation',
    ['test.c', 'test_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc, pthreadwin32_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)
test('test_pic_preallocation', test_pic_preallocation)

test_picture_wrap_integration = executable('test_picture_wrap_integration',
    ['test.c', 'test_picture_wrap_integration.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency,
)
test('test_picture_wrap_integration', test_picture_wrap_integration)

test('test_picture', test_picture)
test('test_feature_collector', test_feature_collector)
test('test_thread_pool', test_thread_pool)
test('test_model', test_model)
test('test_predict', test_predict)
test('test_feature_extractor', test_feature_extractor)
test('test_dict', test_dict)
test('test_cpu', test_cpu)
test('test_ref', test_ref)
test('test_feature', test_feature)
test('test_ciede', test_ciede)
test('test_cambi', test_cambi)
test('test_luminance_tools', test_luminance_tools)
test('test_cli_parse', test_cli_parse)
test('test_psnr', test_psnr)
test('test_framesync', test_framesync)
test('test_adm_dwt2_16', test_adm_dwt2_16)
test('test_propagate_metadata', test_propagate_metadata)
test('test_adm_csf', test_adm_csf)
test('test_barten_csf', test_barten_csf)
test('test_motion_blend', test_motion_blend)
test('test_conversion_policy', test_conversion_policy)

if is_asm_enabled and (host_machine.cpu_family().startswith('arm64') or host_machine.cpu_family().startswith('aarch64'))
    test_motion_neon = executable('test_motion_neon',
        ['test.c', 'test_motion_neon.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
         '../src/dict.c', '../src/opt.c', '../src/log.c', '../src/predict.c',
         '../src/metadata_handler.c'],
        include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
        dependencies : [math_lib, stdatomic_dependency, thread_lib, cuda_dependency],
        objects : [
          common_cuda_objects,
          platform_specific_cpu_objects,
          libvmaf_feature_static_lib.extract_all_objects(recursive: true),
          libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
          libsvm_static_lib.extract_all_objects(recursive: true),
        ]
    )

    test('test_motion_neon', test_motion_neon)
endif

if get_option('enable_checkasm')
    subdir('checkasm')
endif

if get_option('enable_zimg')
test_colorspace = executable('test_colorspace',
    ['test.c', 'test_colorspace.c', '../src/picture.c',
     '../src/mem.c', '../src/ref.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies: [zimg_dependency, thread_lib],
    c_args: ['-DHAVE_ZIMG=1']
)

test('test_colorspace', test_colorspace)
endif

test_read_pictures_convert_args = ['-DJSON_MODEL_PATH="'+join_paths(meson.project_source_root(), '../model/')+'"']
if get_option('enable_zimg')
    test_read_pictures_convert_args += ['-DHAVE_ZIMG=1']
endif
test_read_pictures_convert = executable('test_read_pictures_convert',
    ['test.c', 'test_read_pictures_convert.c'],
    include_directories : [libvmaf_inc, test_inc],
    c_args : test_read_pictures_convert_args,
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)
test('test_read_pictures_convert', test_read_pictures_convert)
