# installed version.h header generation
version_h_data = configuration_data()
version_h_data.set('VMAF_API_VERSION_MAJOR', vmaf_api_version_major)
version_h_data.set('VMAF_API_VERSION_MINOR', vmaf_api_version_minor)
version_h_data.set('VMAF_API_VERSION_PATCH', vmaf_api_version_revision)
version_h_target = configure_file(input: 'version.h.in',
                                  output: 'version.h',
                                  configuration: version_h_data)

is_cuda_enabled = get_option('enable_cuda') == true

platform_specific_headers = []

if is_cuda_enabled
  platform_specific_headers += 'libvmaf_cuda.h'
endif

# install headers
install_headers(['libvmaf.h',
                'feature.h',
                'model.h',
                'picture.h'],
                version_h_target,
                platform_specific_headers,
                subdir : 'libvmaf')
