project('libvmaf', ['c', 'cpp'],
    version : '3.2.0',
    default_options : ['c_std=c11',
                       'cpp_std=c++11',
                       'warning_level=2',
                       'buildtype=release',
                       'default_library=both',
                      ],
    meson_version: '>= 0.56.1')

vmaf_soname_version       = '3.2.0'
vmaf_api_version_array    = vmaf_soname_version.split('.')
vmaf_api_version_major    = vmaf_api_version_array[0]
vmaf_api_version_minor    = vmaf_api_version_array[1]
vmaf_api_version_revision = vmaf_api_version_array[2]

libvmaf_src_root = meson.current_source_dir()
cc = meson.get_compiler('c')
libvmaf_inc = include_directories(['include'])

# Arguments in test_args will be used even on feature tests
test_args = []
if host_machine.system() == 'linux' or host_machine.system() == 'windows' or host_machine.system() == 'cygwin'
    test_args += '-D_GNU_SOURCE'
    add_project_arguments('-D_GNU_SOURCE', language: ['c', 'cpp'])
elif host_machine.system() == 'darwin'
    test_args += '-D_DARWIN_C_SOURCE'
    add_project_arguments('-D_DARWIN_C_SOURCE', language: ['c', 'cpp'])
endif

# M_PI, M_E, etc are only defined in math.h if _USE_MATH_DEFINES is defined before including math.h
if cc.get_id() == 'msvc'
	add_project_arguments('-D_USE_MATH_DEFINES', language: ['c', 'cpp'])
endif

# Header checks
if cc.has_header('malloc.h', args: test_args)
    add_project_arguments('-DHAVE_MALLOC_H', language: ['c', 'cpp'])
endif
if cc.has_header('alloca.h', args: test_args)
    add_project_arguments('-DHAVE_ALLOCA_H', language: ['c', 'cpp'])
endif

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

builtin_clz_dependency = []
if cc.get_id() == 'msvc'
    builtin_clz_dependency = declare_dependency(
        include_directories : include_directories('src/compat/msvc'),
        compile_args : ['-DHAVE_BUILTIN_CLZ_COMPAT'],
    )
endif

if cc.check_header('getopt.h', args: test_args)
    getopt_dependency = declare_dependency(compile_args : '-DHAVE_GETOPT_H')
else
    getopt_dependency = declare_dependency(
        compile_args : '-DHAVE_GETOPT_H',
        sources: ['src/compat/getopt/getopt.c'],
        include_directories : include_directories('src/compat/getopt'),
    )
endif

pthreadwin32_inc = []
pthreadwin32_lib = []

if cc.get_id() == 'msvc'
    if get_option('bundled_winpthreads')
        # Use the bundled GerHobbelt/pthread-win32 via CMake subproject
        cmake = import('cmake')

        pthreadwin32_opts = cmake.subproject_options()
        pthreadwin32_opts.add_cmake_defines({
            'BUILD_SHARED_LIBS': false,
        })
        pthreadwin32_proj = cmake.subproject('pthread-win32', options: pthreadwin32_opts)
        pthreadwin32_lib = pthreadwin32_proj.dependency('pthreadVC3')
        pthreadwin32_inc = include_directories('./subprojects/pthread-win32')
    else
        # --- External pthreads detection ---

        # 1. First try to find the library
        external_pthread = dependency('pthread', required: false)

        if not external_pthread.found()
            foreach name : [
                'pthreadVC3', 'pthreadVC2', 'pthreadVC',
                'pthreadVCE3', 'pthreadVCE2', 'pthreadVCE',
                'pthreadVSE3', 'pthreadVSE2', 'pthreadVSE',
                'pthread', 'libpthread', 'pthreads'
            ]
                external_pthread = cc.find_library(name, required: false)
                if external_pthread.found()
                    break
                endif
            endforeach
        endif

        if not external_pthread.found()
            error('MSVC build requested with -Dbundled_winpthreads=false, ' +
                  'but no external pthreads library was found.\n' +
                  'Please install a Windows pthreads implementation and make sure ' +
                  'the library is discoverable, or use the bundled one (default).')
        endif

        # 2. Try to find the header (pthread.h)
        # This helps when the library is found but headers are in a non-standard location
        pthread_header = cc.check_header('pthread.h', required: false)

        if not pthread_header
            # Try common extra include locations (optional, can be extended)
            # Most of the time the user should have the include path already in the environment
            error('External pthreads library was found, but <pthread.h> could not be located.\n' +
                  'Please make sure the include directory is available (e.g. via INCLUDE ' +
                  'environment variable or -I flags).')
        endif

        pthreadwin32_lib = external_pthread
    endif
endif

subdir('include')
subdir('src')
subdir('tools')
subdir('doc')
subdir('test')
