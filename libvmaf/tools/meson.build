libvmaf_dep = declare_dependency(
    link_with : libvmaf,
)

vmafossexec = executable(
    'vmafossexec',
    [src_dir + 'main.cpp', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf.get_static_lib(),
    install : true,
)

psnr = executable(
    'psnr',
    [src_dir + 'psnr_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf,
    install : false,
)

moment = executable(
    'moment',
    [src_dir + 'moment_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf,
    install : false,
)

ssim = executable(
    'ssim',
    [src_dir + 'ssim_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf,
    install : false,
)

ms_ssim = executable(
    'ms_ssim',
    [src_dir + 'ms_ssim_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf,
    install : false,
)

vmaf = executable(
    'vmaf',
    [src_dir + 'vmaf_main.c', src_dir + 'read_frame.c'],
    include_directories : libvmaf_inc,
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    link_with : libvmaf,
    install : false,
)
