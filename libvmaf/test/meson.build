test_inc = include_directories('.')

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c',],
    include_directories : [libvmaf_inc, test_inc, '../src/feature/'],
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
    ['test.c', 'test_feature_extractor.c',
     '../src/feature/feature_extractor.c',
     '../src/feature/feature_collector.c',
     '../src/feature/integer_ssim.c',
     '../src/mem.c',
     '../src/feature/offset.c',
     '../src/feature/common/alignment.c',
     '../src/feature/common/cpu.c',
     '../src/feature/picture_copy.c',
     '../src/feature/adm.c',
     '../src/feature/adm_tools.c',
     '../src/feature/float_adm.c',
     '../src/feature/float_vif.c',
     '../src/feature/vif.c',
     '../src/feature/vif_tools.c',
     '../src/feature/float_ssim.c',
     '../src/feature/iqa/math_utils.c',
     '../src/feature/iqa/convolve.c',
     '../src/feature/iqa/decimate.c',
     '../src/feature/iqa/ssim_tools.c',
     '../src/feature/ssim.c',
     '../src/feature/integer_psnr.c',
     '../src/feature/float_psnr.c',
     '../src/feature/psnr.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
    dependencies : math_lib,
    objects : [
        convolution_and_psnr_avx_static_lib.extract_all_objects(),
    ],
)

test('test_picture', test_picture)
test('test_feature_collector', test_feature_collector)
test('test_model', test_model)
test('test_predict', test_predict)
test('test_feature_extractor', test_feature_extractor)
