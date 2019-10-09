libvmaf_dep = declare_dependency(
    link_with : libvmaf,
    #include_directories : include_directories(
    #    src_dir,
    #    feature_src_dir + 'common'
    #)
)

vmafossexec = executable(
    'vmafossexec',
    src_dir + 'main.cpp',
    include_directories : vmaf_include,
    c_args : vmaf_c_args,
    cpp_args : vmaf_cpp_args,
    link_with : libvmaf,
    install : true,
)
