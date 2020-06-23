compat_cflags = []
if cc.has_function('strsep')
  compat_cflags += '-DHAVE_STRSEP'
endif

vmafossexec = executable(
    'vmafossexec',
    [src_dir + 'main.cpp', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    dependencies: [stdatomic_dependency],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf.get_static_lib(),
    install : true,
)

vmaf_rc = executable(
    'vmaf_rc',
    ['vmaf.c', 'cli_parse.c', 'y4m_input.c', 'vidinput.c', 'yuv_input.c'],
    include_directories : [libvmaf_inc, vmaf_include],
    dependencies: [stdatomic_dependency],
    c_args : [vmaf_cflags_common, compat_cflags],
    link_with : libvmaf_rc.get_static_lib(),
    install : false,
)

vmaf_feature = executable(
    'vmaf_feature',
    [src_dir + 'vmaf_feature_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    dependencies: [stdatomic_dependency],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf.get_static_lib(),
    install : false,
)
