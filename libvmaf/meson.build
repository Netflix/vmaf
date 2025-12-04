project('libvmaf', ['c', 'cpp'],
    version : '3.0.0',
    default_options : ['c_std=c11',
                       'cpp_std=c++11',
                       'warning_level=2',
                       'buildtype=release',
                       'default_library=both',
                      ],
    meson_version: '>= 0.56.1')

warning('Please run meson in top source directory instead of libvmaf/ subdir.')

subdir('include')
subdir('src')
subdir('tools')
subdir('doc')
subdir('test')
