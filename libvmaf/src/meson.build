vmaf_soversion = vmaf_api_version_major

# Build libvmaf
feature_src_dir = './feature/'
src_dir = './'
model_dir = '../../model/'

vmaf_base_include = include_directories('./', './feature/common')

if cc.get_id() != 'msvc'
    vmaf_cflags_common = [
        '-pedantic',
        '-DOC_NEW_STYLE_INCLUDES',
    ]
else
    vmaf_cflags_common = [
      '-wd4028', # parameter different from declaration
      '-wd4996', # use of POSIX functions
      '-DOC_NEW_STYLE_INCLUDES',
    ]
endif

cdata = configuration_data()
cdata_asm = configuration_data()

if cc.symbols_have_underscore_prefix()
    cdata.set10('PREFIX', true)
    cdata_asm.set10('PREFIX', true)
endif

is_asm_enabled = get_option('enable_asm') == true
is_avx512_enabled = get_option('enable_avx512') == true

if is_asm_enabled
    if host_machine.cpu_family().startswith('x86')
        cdata.set10('HAVE_ASM', is_asm_enabled)
        cdata.set10('ARCH_X86', host_machine.cpu_family().startswith('x86'))
        cdata.set10('ARCH_X86_64', host_machine.cpu_family() == 'x86_64')
        cdata.set10('ARCH_X86_32', host_machine.cpu_family() == 'x86')

        # check NASM version
        nasm = find_program('nasm')
        if nasm.found()
            nasm_r = run_command(nasm, '-v', check: true)

            if nasm_r.returncode() != 0
                error('failed running nasm to obtain its version')
            endif

            out = nasm_r.stdout().strip().split()
            if out[1].to_lower() == 'version'
                if out[2].version_compare('<2.13.02')
                    error('nasm 2.13.02 or later is required, found nasm @0@'.format(out[2]))
                elif out[2].version_compare('<2.14') and get_option('enable_avx512')
                    error('nasm 2.14 or later is required for AVX-512 asm.\n' +
                           'AVX-512 asm can be disabled with \'-Denable_avx512=false\'')
                endif
                cdata.set10('HAVE_AVX512', get_option('enable_avx512'))
                cdata_asm.set10('HAVE_AVX512', get_option('enable_avx512'))
            else
                error('unexpected nasm version string: @0@'.format(nasm_r.stdout()))
            endif
        endif

        # Generate config.asm
        cdata_asm.set10('ARCH_X86_64', host_machine.cpu_family() == 'x86_64')
        cdata_asm.set10('ARCH_X86_32', host_machine.cpu_family() == 'x86')
        cdata_asm.set10('PIC', true)
        config_asm_target = configure_file(output: 'config.asm', output_format: 'nasm', configuration: cdata_asm)

        if host_machine.system() == 'windows'
            nasm_format = 'win'
        elif host_machine.system() == 'darwin'
            nasm_format = 'macho'
        else
            nasm_format = 'elf'
        endif
        if host_machine.cpu_family() == 'x86_64'
            nasm_format += '64'
        else
            nasm_format += '32'
        endif

        nasm_gen = generator(nasm,
            output: '@BASENAME@.obj',
            depfile: '@BASENAME@.obj.ndep',
            arguments: [
                '-f', nasm_format,
                '-I', '@0@/src/'.format(libvmaf_src_root),
                '-I', '@0@/'.format(meson.current_build_dir()),
                '-MQ', '@OUTPUT@', '-MF', '@DEPFILE@',
                '@EXTRA_ARGS@',
                '@INPUT@',
                '-o', '@OUTPUT@'
            ])
    endif
endif

json_model_c_sources = []
built_in_models_enabled = get_option('built_in_models') == true
float_enabled = get_option('enable_float') == true
cdata.set10('VMAF_FLOAT_FEATURES', float_enabled)

if built_in_models_enabled
    xxd = find_program('xxd', required: false)
    if xxd.found()
        model_files = [
            'vmaf_v0.6.1.json',
            'vmaf_b_v0.6.3.json',
            'vmaf_v0.6.1neg.json',
            'vmaf_4k_v0.6.1.json',
        ]

        if float_enabled
            model_files += [
              'vmaf_float_v0.6.1neg.json',
              'vmaf_float_v0.6.1.json',
              'vmaf_float_b_v0.6.3.json',
              'vmaf_float_4k_v0.6.1.json',
            ]
        endif

        foreach model_file : model_files
            json_model_c_sources += custom_target(
                  model_file,
                  output : '@PLAINNAME@.c',
                  input : configure_file(
                    input : model_dir + model_file,
                    output : model_file,
                    copy: true
                  ),
                  command : [xxd, '--include', '@INPUT@', '@OUTPUT@'],
            )
        endforeach

        cdata.set10('VMAF_BUILT_IN_MODELS', built_in_models_enabled)
    endif
endif

config_h_target = configure_file(output: 'config.h', configuration: cdata)

libvmaf_cpu_sources = [
    src_dir + 'cpu.c',
]

if is_asm_enabled
    if host_machine.cpu_family().startswith('x86')
        # NASM source files
        libvmaf_sources_asm = files(
            'x86/cpuid.asm',
        )
        libvmaf_nasm_objects = nasm_gen.process(libvmaf_sources_asm)
        libvmaf_cpu_sources += libvmaf_nasm_objects

        libvmaf_cpu_sources += files(
            src_dir + 'x86/cpu.c',
        )
    endif
endif

libvmaf_include = include_directories(
    '.',
    '../include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
)

libvmaf_cpu_static_lib = static_library(
    'libvmaf_cpu',
    libvmaf_cpu_sources,
    include_directories : [libvmaf_include],
)

platform_specific_cpu_objects = []

if is_asm_enabled
    if host_machine.cpu_family().startswith('x86')
      x86_avx2_sources = [
          feature_src_dir + 'common/convolution_avx.c',
          feature_src_dir + 'x86/motion_avx2.c',
          feature_src_dir + 'x86/vif_avx2.c',
          feature_src_dir + 'x86/adm_avx2.c',
          feature_src_dir + 'x86/cambi_avx2.c',
      ]

      x86_avx2_static_lib = static_library(
          'x86_avx2',
          x86_avx2_sources,
          include_directories : vmaf_base_include,
          c_args : ['-mavx', '-mavx2'] + vmaf_cflags_common,
      )

      platform_specific_cpu_objects += x86_avx2_static_lib.extract_all_objects(recursive: true)

      if is_avx512_enabled
        x86_avx512_sources = [
            feature_src_dir + 'x86/motion_avx512.c',
            feature_src_dir + 'x86/vif_avx512.c',
        ]

        x86_avx512_static_lib = static_library(
            'x86_avx512',
            x86_avx512_sources,
            include_directories : vmaf_base_include,
            c_args : ['-mavx512f', '-mavx512dq', '-mavx512bw',
                      '-mavx512vbmi', '-mavx512vbmi2', '-mavx512vl'] +
                     vmaf_cflags_common,
        )

        platform_specific_cpu_objects += x86_avx512_static_lib.extract_all_objects(recursive: true)
      endif

    endif
endif

thread_lib = dependency('threads')
math_lib = cc.find_library('m', required : false)

vmaf_include = include_directories(
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
)

libvmaf_feature_sources = [
    feature_src_dir + 'picture_copy.c',
    feature_src_dir + 'integer_psnr.c',
    feature_src_dir + 'third_party/xiph/psnr_hvs.c',
    feature_src_dir + 'feature_extractor.c',
    feature_src_dir + 'feature_name.c',
    feature_src_dir + 'alias.c',
    feature_src_dir + 'integer_adm.c',
    feature_src_dir + 'feature_collector.c',
    feature_src_dir + 'integer_motion.c',
    feature_src_dir + 'integer_vif.c',
    feature_src_dir + 'ciede.c',
    feature_src_dir + 'common/alignment.c',
    feature_src_dir + 'mkdirp.c',

    feature_src_dir + 'float_ssim.c',
    feature_src_dir + 'float_ms_ssim.c',
    feature_src_dir + 'ssim.c',
    feature_src_dir + 'ms_ssim.c',
    feature_src_dir + 'iqa/decimate.c',
    feature_src_dir + 'iqa/ssim_tools.c',
    feature_src_dir + 'iqa/math_utils.c',
    feature_src_dir + 'iqa/convolve.c',
    feature_src_dir + 'cambi.c',
    feature_src_dir + 'luminance_tools.c',
    feature_src_dir + 'null.c',
]

if float_enabled
    libvmaf_feature_sources += [
        feature_src_dir + 'float_adm.c',
        feature_src_dir + 'float_psnr.c',
        feature_src_dir + 'float_ansnr.c',
        feature_src_dir + 'float_motion.c',
        feature_src_dir + 'float_vif.c',
        feature_src_dir + 'float_moment.c',

        feature_src_dir + 'common/convolution.c',
        feature_src_dir + 'offset.c',
        feature_src_dir + 'adm.c',
        feature_src_dir + 'adm_tools.c',
        feature_src_dir + 'ansnr.c',
        feature_src_dir + 'ansnr_tools.c',
        feature_src_dir + 'vif.c',
        feature_src_dir + 'vif_tools.c',
        feature_src_dir + 'motion.c',
        feature_src_dir + 'psnr.c',
        feature_src_dir + 'psnr_tools.c',
        feature_src_dir + 'moment.c',
        feature_src_dir + 'all.c',
        feature_src_dir + 'common/blur_array.c',
    ]
endif

libvmaf_feature_static_lib = static_library(
    'libvmaf_feature',
    libvmaf_feature_sources,
    include_directories : [libvmaf_include, vmaf_include],
    dependencies: [stdatomic_dependency],
)

libvmaf_sources = [
    src_dir + 'libvmaf.c',
    src_dir + 'compute_vmaf.c',
    src_dir + 'predict.c',
    src_dir + 'model.c',
    src_dir + 'svm.cpp',
    src_dir + 'picture.c',
    src_dir + 'mem.c',
    src_dir + 'picture.c',
    src_dir + 'output.c',
    src_dir + 'fex_ctx_vector.c',
    src_dir + 'thread_pool.c',
    src_dir + 'dict.c',
    src_dir + 'opt.c',
    src_dir + 'ref.c',
    src_dir + 'read_json_model.c',
    src_dir + 'pdjson.c',
    src_dir + 'log.c',
]

if host_machine.system() == 'windows'
    vmaf_soversion = ''
else
    vmaf_soversion = vmaf_api_version_major
endif

libvmaf = library(
    'vmaf',
    [libvmaf_sources, rev_target, json_model_c_sources],
    include_directories : [libvmaf_inc, vmaf_include],
    c_args : vmaf_cflags_common,
    cpp_args : vmaf_cflags_common,
    dependencies : [
      thread_lib,
      math_lib,
      stdatomic_dependency,
    ],
    objects : [
        platform_specific_cpu_objects,
        libvmaf_feature_static_lib.extract_all_objects(recursive: true),
        libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
    ],
    version : vmaf_soname_version,
    soversion : vmaf_soversion,
    install: true,
)

pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries: libvmaf,
    version: meson.project_version(),
    name: 'libvmaf',
    filebase: 'libvmaf',
    description: 'VMAF, Video Multimethod Assessment Fusion',
    subdirs: [ '.', 'libvmaf']
)
