compat_cflags = []
if cc.has_function('strsep')
  compat_cflags += '-DHAVE_STRSEP'
endif

vmaf = executable(
    'vmaf',
    ['vmaf.c', 'cli_parse.c', 'y4m_input.c', 'vidinput.c', 'yuv_input.c'],
    include_directories : [libvmaf_inc, vmaf_include],
    dependencies: [stdatomic_dependency, cuda_dependency],
    c_args : [vmaf_cflags_common, compat_cflags],
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    install : true,
)

subdir('test')
