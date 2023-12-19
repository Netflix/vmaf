if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

test_context = executable('test_context',
    ['test.c', 'test_context.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies:[stdatomic_dependency, cuda_dependency],
)

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c', '../src/ref.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies:[stdatomic_dependency, thread_lib, cuda_dependency],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/feature/'), include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency
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
    ['test.c', 'test_model.c', '../src/dict.c', '../src/svm.cpp', '../src/pdjson.c', '../src/read_json_model.c', '../src/log.c', json_model_c_sources],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : [vmaf_cflags_common, '-DJSON_MODEL_PATH="'+join_paths(meson.project_source_root(), '../model/')+'"'],
    cpp_args : vmaf_cflags_common,
    dependencies : [thread_lib, cuda_dependency],
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/dict.c',
     '../src/feature/feature_collector.c', '../src/feature/alias.c', '../src/model.c', '../src/svm.cpp', '../src/log.c',
     '../src/read_json_model.c', '../src/pdjson.c', json_model_c_sources, '../src/feature/feature_name.c', '../src/feature/feature_extractor.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [thread_lib, cuda_dependency],
)

test_feature_extractor = executable('test_feature_extractor',
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c', '../src/ref.c',
     '../src/dict.c', '../src/opt.c', '../src/log.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    dependencies : [math_lib, stdatomic_dependency, cuda_dependency],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
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
)

test_feature = executable('test_feature',
    ['test.c', 'test_feature.c', '../src/feature/alias.c', '../src/dict.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
)

test_ciede = executable('test_ciede',
    ['test.c', 'test_ciede.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency
)

test_cambi = executable('test_cambi',
    ['test.c', 'test_cambi.c', '../src/picture.c', '../src/mem.c', '../src/ref.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency
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
)

test_psnr = executable('test_psnr',
    ['test.c', 'test_psnr.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
)

if get_option('enable_cuda')
test_ring_buffer = executable('test_ring_buffer',
    ['test.c', 'test_ring_buffer.c', '../src/cuda/ring_buffer.c', '../src/cuda/picture_cuda.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../src/')],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency,
    c_args: ['-DHAVE_CUDA=1']
)

test_cuda_pic_preallocation = executable('test_cuda_pic_preallocation',
    ['test.c', 'test_cuda_pic_preallocation.c'],
    include_directories : [libvmaf_inc, test_inc],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies: cuda_dependency,
    c_args: ['-DHAVE_CUDA=1']
)

test('test_ring_buffer', test_ring_buffer)
test('test_cuda_pic_preallocation', test_cuda_pic_preallocation)
endif

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
