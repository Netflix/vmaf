if host_machine.system() == 'windows'
    # On Windows, we use a compatibility layer to emulate pthread
    thread_dependency = []
    thread_compat_dep = declare_dependency(sources : files('win32/thread.c'))
else
    thread_dependency = dependency('threads')
    thread_compat_dep = []
endif

# Build libptools
ptools_path = './third_party/ptools/'
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

libptools = shared_library(
    'ptools',
    ptools_sources,
    include_directories : opencontainers_include,
    dependencies : [
        thread_dependency,
        thread_compat_dep,
    ],
    cpp_args : ptools_cpp_args,
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

vmaf_cflags_common = [
    '-w',
    '-Wextra',
    '-pedantic',
    '-DOC_NEW_STYLE_INCLUDES',
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
    feature_src_dir + 'common/convolution.c',
    feature_src_dir + 'common/cpu.c',
    feature_src_dir + 'all_options.c',
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
    third_party_dir + 'pugixml/pugixml.cpp',
    src_dir + 'combo.c',
    src_dir + 'cpu_info.c',
    src_dir + 'svm.cpp',
    src_dir + 'darray.c',
    src_dir + 'libvmaf.cpp',
    src_dir + 'vmaf.cpp',
]

vmaf_include = include_directories(
    opencontainers_path + '/include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
    ptools_path,
    opencontainers_path + '/include'
)

libvmaf = both_libraries(
    'vmaf',
    vmaf_sources,
    include_directories : vmaf_include,
    c_args : vmaf_c_args,
    cpp_args : vmaf_cpp_args,
    dependencies : [
        thread_dependency,
        thread_compat_dep,
    ],
    objects : [
        convolution_and_psnr_avx_static_lib.extract_all_objects(),
        libptools.extract_all_objects()
    ],
    install: true,
)

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries: libvmaf,
    version: meson.project_version(),
    name: 'libvmaf',
    filebase: 'vmaf',
    description: 'VMAF, Video Multimethod Assessment Fusion'
)
