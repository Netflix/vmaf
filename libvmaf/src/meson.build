# Build libptools
ptools_path = './third_party/ptools/'
opencontainers_path = ptools_path + 'opencontainers_1_8_4'
opencontainers_include = include_directories(opencontainers_path + '/include')

vmaf_soversion = vmaf_api_version_major

if cc.get_id() != 'msvc'
    ptools_cpp_args = [
        '-fno-strict-aliasing',
        '-DLINUX_',
        '-DOC_NEW_STYLE_INCLUDES',
        '-D_REENTRANT'
    ]
else
    ptools_cpp_args = [
      '-wd4028', # parameter different from declaration
      '-wd4996', # use of POSIX functions
      '-DOC_NEW_STYLE_INCLUDES',
      '-D_REENTRANT',
    ]
endif

ptools_sources = [
    ptools_path + 'm2pythontools.cc',
    ptools_path + 'valpython.cc',
    ptools_path + 'midassocket.cc',
    ptools_path + 'valprotocol2.cc',
    ptools_path + 'm2ser.cc',
    ptools_path + 'm2streamdataenc.cc',
    ptools_path + 'm2convertrep.cc',
    ptools_path + 'timeconv.cc'
]

thread_lib = dependency('threads')
math_lib = cc.find_library('m', required : false)

libptools = shared_library(
    'ptools',
    ptools_sources,
    include_directories : [
        opencontainers_include,
    ],
    cpp_args : ptools_cpp_args,
    dependencies : thread_lib,
)

# Build libvmaf
third_party_dir = './third_party/'
feature_src_dir = './feature/'
src_dir = './'

vmaf_base_include = include_directories('./', './feature/common')

convolution_and_psnr_avx_sources = [
    feature_src_dir + 'common/convolution_avx.c',
    feature_src_dir + 'psnr_tools.c'
]

if cc.get_id() != 'msvc'
    vmaf_cflags_common = [
        '-pedantic',
        '-DOC_NEW_STYLE_INCLUDES',
    ]
else
    vmaf_cflags_common = [
      '-wd4028', # parameter different from declaration
      '-wd4996', # use of POSIX functions
      '-DOC_NEW_STYLE_INCLUDES',
    ]
endif

convolution_and_psnr_avx_static_lib = static_library(
    'convolution_and_psnr_avx',
    convolution_and_psnr_avx_sources,
    include_directories : vmaf_base_include,
    c_args : ['-mavx'] + vmaf_cflags_common,
)

vmaf_include = include_directories(
    opencontainers_path + '/include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
    ptools_path,
    opencontainers_path + '/include'
)

libvmaf_feature_sources = [
    feature_src_dir + 'common/alignment.c',
    feature_src_dir + 'common/convolution.c',
    feature_src_dir + 'common/cpu.c',
    feature_src_dir + 'offset.c',
    feature_src_dir + 'adm.c',
    feature_src_dir + 'adm_tools.c',
    feature_src_dir + 'ansnr.c',
    feature_src_dir + 'ansnr_tools.c',
    feature_src_dir + 'vif.c',
    feature_src_dir + 'vif_tools.c',
    feature_src_dir + 'motion.c',
    feature_src_dir + 'psnr.c',
    feature_src_dir + 'ssim.c',
    feature_src_dir + 'ms_ssim.c',
    feature_src_dir + 'moment.c',
    feature_src_dir + 'all.c',
    feature_src_dir + 'common/blur_array.c',
    feature_src_dir + 'iqa/math_utils.c',
    feature_src_dir + 'iqa/convolve.c',
    feature_src_dir + 'iqa/decimate.c',
    feature_src_dir + 'iqa/ssim_tools.c',
]

libvmaf_feature_static_lib = static_library(
    'libvmaf_feature',
    libvmaf_feature_sources,
    include_directories : [vmaf_include, vmaf_base_include],
    dependencies: [stdatomic_dependency],
)

vmaf_sources = [
    third_party_dir + 'pugixml/pugixml.cpp',
    src_dir + 'mem.c',
    src_dir + 'combo.c',
    src_dir + 'cpu_info.c',
    src_dir + 'svm.cpp',
    src_dir + 'darray.c',
    src_dir + 'libvmaf.cpp',
    src_dir + 'vmaf.cpp',
]

libvmaf = both_libraries(
    'vmaf',
    vmaf_sources,
    include_directories : vmaf_include,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [thread_lib, stdatomic_dependency],
    objects : [
        convolution_and_psnr_avx_static_lib.extract_all_objects(),
        libptools.extract_all_objects(),
        libvmaf_feature_static_lib.extract_all_objects(),
    ],
    install: true,
    soversion: vmaf_soversion,
    version: vmaf_soname_version,
)

libvmaf_rc_include = include_directories(
    opencontainers_path + '/include',
    '../include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
    ptools_path,
    opencontainers_path + '/include',
)

libvmaf_rc_feature_sources = [
  feature_src_dir + 'picture_copy.c',
  feature_src_dir + 'integer_psnr.c',
  feature_src_dir + 'feature_extractor.c',
  feature_src_dir + 'alias.c',
  feature_src_dir + 'float_adm.c',
  feature_src_dir + 'feature_collector.c',
  feature_src_dir + 'float_psnr.c',
  feature_src_dir + 'integer_motion.c',
  feature_src_dir + 'float_motion.c',
  feature_src_dir + 'float_ssim.c',
  feature_src_dir + 'float_ms_ssim.c',
  feature_src_dir + 'float_vif.c',
  feature_src_dir + 'integer_vif.c',
  feature_src_dir + 'integer_ssim.c',
  feature_src_dir + 'float_moment.c',
]

libvmaf_rc_feature_static_lib = static_library(
    'libvmaf_rc_feature',
    libvmaf_rc_feature_sources,
    include_directories : [libvmaf_rc_include],
    dependencies: [stdatomic_dependency],
)

libvmaf_rc_sources = [
    src_dir + 'libvmaf.rc.c',
    src_dir + 'predict.c',
    src_dir + 'model.c',
    src_dir + 'unpickle.cpp',
    src_dir + 'svm.cpp',
    src_dir + 'picture.c',
    src_dir + 'mem.c',
    src_dir + 'picture.c',
    src_dir + 'output.c',
    src_dir + 'fex_ctx_vector.c',
    src_dir + 'thread_pool.c',
    src_dir + 'dict.c',
]

libvmaf_rc = both_libraries(
    'vmaf_rc',
    libvmaf_rc_sources,
    include_directories : [vmaf_include, libvmaf_inc],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [
      thread_lib,
      math_lib,
      stdatomic_dependency,
    ],
    objects : [
        convolution_and_psnr_avx_static_lib.extract_all_objects(),
        libptools.extract_all_objects(),
        libvmaf_feature_static_lib.extract_all_objects(),
        libvmaf_rc_feature_static_lib.extract_all_objects(),
    ],
    install: false,
)

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries: libvmaf,
    version: meson.project_version(),
    name: 'libvmaf',
    filebase: 'libvmaf',
    description: 'VMAF, Video Multimethod Assessment Fusion',
    subdirs: 'libvmaf'
)
