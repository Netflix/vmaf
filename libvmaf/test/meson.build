if not get_option('enable_tests')
    subdir_done()
endif

test_inc = include_directories('.')

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
    dependencies:[stdatomic_dependency],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c',],
    include_directories : [libvmaf_inc, test_inc, '../src/feature/'],
)

test_thread_pool = executable('test_thread_pool',
    ['test.c', 'test_thread_pool.c', '../src/thread_pool.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
    dependencies : thread_lib,
)

test_model = executable('test_model',
    ['test.c', 'test_model.c', '../src/svm.cpp', '../src/unpickle.cpp'],
    include_directories : [libvmaf_inc, test_inc, opencontainers_include,
                           '../src/third_party/ptools/', '../src'],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    objects : libptools.extract_all_objects(),
    dependencies : thread_lib,
)

test_predict = executable('test_predict',
    ['test.c', 'test_predict.c', '../src/predict.c',
     '../src/feature/feature_collector.c', '../src/model.c', '../src/svm.cpp',
     '../src/unpickle.cpp'],
    include_directories : [libvmaf_inc, test_inc, opencontainers_include,
                           '../src/third_party/ptools/', '../src'],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    objects : libptools.extract_all_objects(),
    dependencies : thread_lib,
)

test_feature_extractor = executable('test_feature_extractor',
    ['test.c', 'test_feature_extractor.c', '../src/mem.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
    dependencies : [math_lib, stdatomic_dependency],
    objects : [
      convolution_and_psnr_avx_static_lib.extract_all_objects(),
      libvmaf_feature_static_lib.extract_all_objects(),
      libvmaf_rc_feature_static_lib.extract_all_objects(),
    ]
)

test_dict = executable('test_dict',
    ['test.c', 'test_dict.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
)

test('test_picture', test_picture)
test('test_feature_collector', test_feature_collector)
test('test_thread_pool', test_thread_pool)
test('test_model', test_model)
test('test_predict', test_predict)
test('test_feature_extractor', test_feature_extractor)
test('test_dict', test_dict)
