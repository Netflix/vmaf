project('libvmaf', ['c', 'cpp'],
    version : '1.3.16',
    meson_version: '>= 0.47.0')

vmaf_soname_version       = '0.0.0'
vmaf_api_version_array    = vmaf_soname_version.split('.')
vmaf_api_version_major    = vmaf_api_version_array[0]
vmaf_api_version_minor    = vmaf_api_version_array[1]
vmaf_api_version_revision = vmaf_api_version_array[2]

libvmaf_src_root = meson.current_source_dir()
cc = meson.get_compiler('c')
libvmaf_inc = include_directories('include')

subdir('src')
subdir('include')
subdir('tools')
subdir('doc')
