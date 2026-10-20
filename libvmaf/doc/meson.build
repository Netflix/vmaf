if not get_option('enable_docs')
    subdir_done()
endif

doxygen = find_program('doxygen', required: false)

if doxygen.found()
    conf_data = configuration_data()
    doxygen_input = [source_root / 'libvmaf/include/libvmaf', source_root / 'libvmaf/src/feature']

    conf_data.set('DOXYGEN_INPUT', ' '.join(doxygen_input))
    conf_data.set('DOXYGEN_STRIP', source_root / 'libvmaf/include')
    conf_data.set('DOXYGEN_OUTPUT', meson.current_build_dir())
    doxyfile = configure_file(input: 'Doxyfile.in',
                              output: 'Doxyfile',
                              configuration: conf_data)

    custom_target('doc',
                  build_by_default: false,
                  command: [doxygen, doxyfile],
                  output: ['html']
    )
endif
