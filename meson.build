project('vmaf', 'c', 'cpp')

# Build libptools

ptools_path = './src/ptools/'
opencontainers_path = ptools_path + 'opencontainers_1_8_4'
opencontainers_include = include_directories(opencontainers_path + '/include')

ptools_cpp_args = [
    '-Wall',
    '-Wextra',
    '-fno-strict-aliasing',
    '-DLINUX_',
    '-DOC_NEW_STYLE_INCLUDES',
    '-Wno-deprecated',
    '-D_REENTRANT'
 ]

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

ccp = meson.get_compiler('cpp')
pthread_lib = ccp.find_library('pthread')

ptools_shared_lib = shared_library(
    'ptools',
    ptools_sources,
    include_directories : opencontainers_include,
    cpp_args : ptools_cpp_args,
    dependencies : pthread_lib
)

# Build libvmaf

feature_src_dir = './src/libvmaf/src/'
src_dir = './src/libvmaf/src/'

vmaf_base_include = include_directories('./src/libvmaf/src', './src/libvmaf/src/common')

convolution_and_psnr_avx_sources = [
    feature_src_dir + 'common/convolution_avx.c',
    feature_src_dir + 'psnr_tools.c'
]

vmaf_cflags_common = [
    '-w',
    '-Wextra',
    '-pedantic',
    '-D_FILE_OFFSET_BITS=64',
    '-D MULTI_THREADING',
    '-DOC_NEW_STYLE_INCLUDES'
]

convolution_and_psnr_avx_static_lib = static_library(
    'convolution_and_psnr_avx',
    convolution_and_psnr_avx_sources,
    include_directories : vmaf_base_include,
    c_args : ['-mavx'] + vmaf_cflags_common,
)

vmaf_c_args = ['-std=c99'] + vmaf_cflags_common
vmaf_cpp_args = ['-std=c++11'] + vmaf_cflags_common

vmaf_sources = [
    feature_src_dir + 'common/alloc.c',
    feature_src_dir + 'common/alignment.c',
    feature_src_dir + 'common/file_io.c',
    feature_src_dir + 'common/frame.c',
    feature_src_dir + 'common/convolution.c',
    feature_src_dir + 'common/cpu.c',
    feature_src_dir + 'adm.c',
    feature_src_dir + 'adm_tools.c',
    feature_src_dir + 'ansnr.c',
    feature_src_dir + 'ansnr_tools.c',
    feature_src_dir + 'vif.c',
    feature_src_dir + 'vif_tools.c',
    feature_src_dir + 'motion.c',
    feature_src_dir + 'psnr.c',
    feature_src_dir + 'iqa/math_utils.c',
    feature_src_dir + 'iqa/convolve.c',
    feature_src_dir + 'iqa/decimate.c',
    feature_src_dir + 'iqa/ssim_tools.c',
    feature_src_dir + 'ssim.c',
    feature_src_dir + 'ms_ssim.c',
    feature_src_dir + 'moment.c',
    feature_src_dir + 'all.c',
    feature_src_dir + 'common/blur_array.c',
    src_dir + 'combo.c',
    src_dir + 'cpu_info.c',
    src_dir + 'svm.cpp',
    src_dir + 'darray.c',
    src_dir + 'libvmaf.cpp',
    src_dir + 'vmaf.cpp',
    src_dir + 'pugixml/pugixml.cpp'
]

vmaf_include = include_directories(
    opencontainers_path + '/include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
    ptools_path,
    opencontainers_path + '/include'
)

vmaf_static_lib = static_library(
    'vmaf',
    vmaf_sources,
    include_directories : vmaf_include,
    c_args : vmaf_c_args,
    cpp_args : vmaf_cpp_args,
    dependencies : pthread_lib,
    objects : [
        convolution_and_psnr_avx_static_lib.extract_all_objects(),
        ptools_shared_lib.extract_all_objects()
    ]
)

libvmaf_dep = declare_dependency(
    link_with : vmaf_static_lib,
    include_directories : include_directories(
        src_dir,
        feature_src_dir + 'common'
    )
)

vmafossexec = executable(
    'vmafossexec',
    src_dir + 'main.cpp',
    include_directories : vmaf_include,
    c_args : vmaf_c_args,
    cpp_args : vmaf_cpp_args,
    link_with : vmaf_static_lib,
)
