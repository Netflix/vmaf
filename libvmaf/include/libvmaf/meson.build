# installed version.h header generation
version_h_data = configuration_data()
version_h_data.set('VMAF_API_VERSION_MAJOR', vmaf_api_version_major)
version_h_data.set('VMAF_API_VERSION_MINOR', vmaf_api_version_minor)
version_h_data.set('VMAF_API_VERSION_PATCH', vmaf_api_version_revision)
version_h_target = configure_file(input: 'version.h.in',
                                  output: 'version.h',
                                  configuration: version_h_data)

# install headers
install_headers(['libvmaf.h',
                'compute_vmaf.h',
                'feature.h',
                'model.h',
                'picture.h'],
                version_h_target,
                subdir : 'libvmaf')
