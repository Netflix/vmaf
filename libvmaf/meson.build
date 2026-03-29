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
thread_lib = []

if cc.get_id() == 'msvc'
    if get_option('bundled_winpthreads')
        # Bundled GerHobbelt/pthread-win32 via CMake subproject
        cmake = import('cmake')

        pthreadwin32_opts = cmake.subproject_options()
        pthreadwin32_opts.add_cmake_defines({
            'BUILD_SHARED_LIBS': false,
        })
        pthreadwin32_proj = cmake.subproject('pthread-win32', options: pthreadwin32_opts)

        # Meson usually exposes shared-style CMake targets (pthreadVC3), not libpthreadVC3.
        _pw_targets = pthreadwin32_proj.target_list()
        if 'libpthreadVC3' in _pw_targets
            pthreadwin32_lib = pthreadwin32_proj.dependency('libpthreadVC3')
            _ptw32_static = true
        elif 'pthreadVC3' in _pw_targets
            pthreadwin32_lib = pthreadwin32_proj.dependency('pthreadVC3')
            _ptw32_static = false
        else
            error('pthread-win32 subproject: neither libpthreadVC3 nor pthreadVC3 found.\n' +
                  'Available targets: ' + ', '.join(_pw_targets))
        endif

        pthreadwin32_inc = include_directories('./subprojects/pthread-win32')
    else
        # External pthreads (system / vcpkg / custom prefix)
        external_pthread = dependency('pthread', required: false)
        _ptw32_static = true

        if not external_pthread.found()
            # MSVC DEBUG_POSTFIX "d": Debug installs libpthreadVC3d.lib / pthreadVC3d.lib
            _pthread_names = []
            if get_option('buildtype') == 'debug'
                _pthread_names += [
                    'libpthreadVC3d', 'pthreadVC3d',
                    'libpthreadVCE3d', 'pthreadVCE3d',
                    'libpthreadVSE3d', 'pthreadVSE3d',
                ]
            endif
            _pthread_names += [
                'libpthreadVC3', 'pthreadVC3',
                'libpthreadVC2', 'pthreadVC2',
                'libpthreadVC',  'pthreadVC',
                'libpthreadVCE3', 'pthreadVCE3',
                'libpthreadVSE3', 'pthreadVSE3',
                'pthread', 'libpthread', 'pthreads',
            ]

            foreach name : _pthread_names
                external_pthread = cc.find_library(name, required: false)
                if external_pthread.found()
                    # Heuristic: names with 'libpthread' are static archives
                    if name.startswith('libpthread')
                        _ptw32_static = true
                    else
                        _ptw32_static = false
                    endif
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

        if not cc.check_header('pthread.h', required: false)
            error('External pthreads library was found, but <pthread.h> could not be located.\n' +
                  'Please make sure the include directory is available (e.g. via INCLUDE ' +
                  'environment variable or -I flags).')
        endif

        pthreadwin32_lib = external_pthread
    endif

    # Match pthreads4w VC cleanup; STATIC only when linking a static archive
    _ptw32_args = ['-D__PTW32_CLEANUP_C']
    if _ptw32_static
        _ptw32_args += ['-D__PTW32_STATIC_LIB']
    endif

    thread_lib = declare_dependency(
        dependencies: [pthreadwin32_lib],
        include_directories: pthreadwin32_inc,
        compile_args: _ptw32_args,
    )
else
    thread_lib = dependency('threads')
endif

subdir('include')
subdir('src')
subdir('tools')
subdir('doc')
subdir('test')
