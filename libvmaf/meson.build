project('libvmaf', ['c', 'cpp'],
    version : '3.0.0',
    default_options : ['c_std=c11',
                       'cpp_std=c++11',
                       'warning_level=2',
                       'buildtype=release',
                       'default_library=both',
                      ],
    meson_version: '>= 0.56.1')

vmaf_soname_version       = '3.0.0'
vmaf_api_version_array    = vmaf_soname_version.split('.')
vmaf_api_version_major    = vmaf_api_version_array[0]
vmaf_api_version_minor    = vmaf_api_version_array[1]
vmaf_api_version_revision = vmaf_api_version_array[2]

libvmaf_src_root = meson.current_source_dir()
cc = meson.get_compiler('c')
libvmaf_inc = include_directories(['include'])

# Arguments in test_args will be used even on feature tests
test_args = []
if host_machine.system() == 'linux'
    test_args += '-D_GNU_SOURCE'
    add_project_arguments('-D_GNU_SOURCE', language: 'c')
elif host_machine.system() == 'darwin'
    test_args += '-D_DARWIN_C_SOURCE'
    add_project_arguments('-D_DARWIN_C_SOURCE', language: 'c')
else
    test_args += '-D_XOPEN_SOURCE=600'
    add_project_arguments('-D_XOPEN_SOURCE=600', language: 'c')
endif

# Header checks
stdatomic_dependency = []
if not cc.check_header('stdatomic.h')
    if cc.get_id() == 'msvc'
        # we have a custom replacement for MSVC
        stdatomic_dependency = declare_dependency(
            include_directories : include_directories('src/compat/msvc'),
        )
    elif cc.compiles('''int main() { int v = 0; return __atomic_fetch_add(&v, 1, __ATOMIC_SEQ_CST); }''',
                     name : 'GCC-style atomics', args : test_args)
        stdatomic_dependency = declare_dependency(
            include_directories : include_directories('src/compat/gcc'),
        )
    else
        error('Atomics not supported')
    endif
endif

subdir('include')
subdir('src')
subdir('tools')
subdir('doc')
subdir('test')
