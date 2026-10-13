# checkasm is fetched via the `subprojects/checkasm.wrap` file. It is first
# looked up as an external dependency (e.g. via pkg-config), falling back to
# the vendored subproject otherwise. See:
# https://checkasm.videolan.me/integration.html
checkasm_dependency = dependency('checkasm',
    fallback : ['checkasm', 'checkasm_dep'],
    required : false,
)

if not checkasm_dependency.found()
    message('checkasm dependency not found, skipping checkasm tests')
    subdir_done()
endif

checkasm_sources = [
    'checkasm.c',
    'check_motion.c',
    'check_adm.c',
    'check_speed.c',
    'check_cambi.c',
    'check_vif.c',
    # linked in to satisfy symbols referenced from within the feature static
    # lib objects below (mirrors test_feature_extractor.c's link set)
    '../../src/mem.c',
    '../../src/picture.c',
    '../../src/ref.c',
    '../../src/dict.c',
    '../../src/opt.c',
    '../../src/log.c',
    '../../src/predict.c',
    '../../src/metadata_handler.c',
]

checkasm = executable('checkasm',
    checkasm_sources,
    include_directories : [libvmaf_inc, include_directories('../../src/')],
    dependencies : [checkasm_dependency, math_lib, stdatomic_dependency,
                    thread_lib, cuda_dependency],
    objects : [
      common_cuda_objects,
      platform_specific_cpu_objects,
      libvmaf_feature_static_lib.extract_all_objects(recursive: true),
      libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
      libsvm_static_lib.extract_all_objects(recursive: true),
    ]
)

test('checkasm', checkasm, suite : 'checkasm')
benchmark('checkasm', checkasm, suite : 'checkasm', args : '--bench')
