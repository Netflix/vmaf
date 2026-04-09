compat_cflags = []
if cc.has_function('strsep')
  compat_cflags += '-DHAVE_STRSEP'
endif

if not get_option('enable_tools')
    subdir_done()
endif

vmaf_tool_deps = [stdatomic_dependency, cuda_dependency]
vmaf_tool_cflags = [vmaf_cflags_common, compat_cflags]
vmaf_tool_link_args = []
if get_option('enable_cuda')
    vmaf_tool_cflags += ['-DHAVE_CUDA=1']
endif
if get_option('enable_sycl')
    vmaf_tool_cflags += ['-DHAVE_SYCL=1']
    vmaf_tool_deps += sycl_dependency
endif

vmaf_tool_link_target = libvmaf
if get_option('default_library') == 'both'
    vmaf_tool_link_target = libvmaf.get_shared_lib()
endif

vmaf = executable(
    'vmaf',
    ['vmaf.c', 'cli_parse.c', 'y4m_input.c', 'vidinput.c', 'yuv_input.c'],
    include_directories : [libvmaf_inc, vmaf_include],
    dependencies: vmaf_tool_deps,
    c_args : vmaf_tool_cflags,
    link_args : vmaf_tool_link_args,
    link_with : vmaf_tool_link_target,
    install : true,
)

vmaf_bench_deps = [stdatomic_dependency]
vmaf_bench_cflags = vmaf_cflags_common
vmaf_bench_link_args = []
if get_option('enable_cuda')
    vmaf_bench_deps += cuda_dependency
    vmaf_bench_cflags += ['-DHAVE_CUDA=1']
endif
if get_option('enable_sycl')
    vmaf_bench_deps += sycl_dependency
    vmaf_bench_cflags += ['-DHAVE_SYCL=1']
endif

vmaf_bench_link_target = get_option('default_library') == 'both' ? libvmaf.get_shared_lib() : libvmaf

vmaf_bench = executable(
    'vmaf_bench',
    ['vmaf_bench.c'],
    include_directories : [libvmaf_inc, vmaf_include],
    dependencies: vmaf_bench_deps,
    c_args : vmaf_bench_cflags,
    link_args : vmaf_bench_link_args,
    link_with : vmaf_bench_link_target,
)

# VPL decode → SYCL pipeline tool
# Requires SYCL + VA-API + Level Zero + VPL
if get_option('enable_sycl')
    vpl_dep = dependency('vpl', required : false)

    vmaf_vpl_deps = [stdatomic_dependency, vpl_dep]
    vmaf_vpl_ok = vpl_dep.found()

    # Linux: VA-API + libva-drm
    libva_dep_tools = dependency('libva', required : false)
    libva_drm_dep_tools = dependency('libva-drm', required : false)
    vmaf_vpl_ok = vmaf_vpl_ok and libva_dep_tools.found() and libva_drm_dep_tools.found()
    if vmaf_vpl_ok
        vmaf_vpl_deps += [libva_dep_tools, libva_drm_dep_tools]
    endif

    if vmaf_vpl_ok
        vmaf_vpl = executable(
            'vmaf_vpl',
            ['vmaf_vpl.c'],
            include_directories : [libvmaf_inc, vmaf_include],
            dependencies: vmaf_vpl_deps,
            c_args : vmaf_cflags_common + ['-DHAVE_SYCL=1'],
            link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
        )
        message('vmaf_vpl tool: enabled (VPL + VA-API + SYCL)')
    else
        message('vmaf_vpl tool: disabled (missing VPL or platform surface deps)')
    endif
endif

subdir('test')
