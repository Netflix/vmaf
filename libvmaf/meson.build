project('libvmaf', 'c', 'cpp',
         version : '1.4.0')

libvmaf_src_root = meson.current_source_dir()
libvmaf_inc = include_directories('include')

subdir('doc')
subdir('src')
subdir('tools')

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries: libvmaf,
    version: meson.project_version(),
    name: 'libvmaf',
    filebase: 'vmaf',
    description: 'VMAF, Perceptual video quality assessment based on multi-method fusion'
)
