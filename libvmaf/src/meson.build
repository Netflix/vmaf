vmaf_soversion = vmaf_api_version_major

# Build libvmaf
feature_src_dir = './feature/'
src_dir = './'
model_dir = '../../model/'
cuda_dir = './cuda/'

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
is_cuda_enabled = get_option('enable_cuda') == true
is_avx512_enabled = get_option('enable_avx512') == true
is_nvtx_enabled = get_option('enable_nvtx') == true

if is_nvtx_enabled    
    cdata.set10('HAVE_NVTX', is_nvtx_enabled)
endif
if is_cuda_enabled
    cdata.set10('HAVE_CUDA', is_cuda_enabled)
endif

if is_asm_enabled
    if host_machine.cpu_family().startswith('arm64') or host_machine.cpu_family().startswith('aarch64')
        cdata.set10('HAVE_ASM', is_asm_enabled)
        cdata.set10('ARCH_AARCH64', host_machine.cpu_family() == 'aarch64' or host_machine.cpu() == 'arm64')
        cdata.set10('ARCH_ARM',     host_machine.cpu_family().startswith('arm') and host_machine.cpu() != 'arm64')

         if host_machine.system() == 'darwin'
            asm_format = 'macho'
        else
            asm_format = 'elf'
        endif
    endif

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
                    warning('nasm 2.14 or later is required for AVX-512 asm. Disabling AVX-512.')
                    is_avx512_supported = false
                else
                    is_avx512_supported = true
                endif
                cdata.set10('HAVE_AVX512', get_option('enable_avx512') and is_avx512_supported)
                cdata_asm.set10('HAVE_AVX512', get_option('enable_avx512') and is_avx512_supported)
            else
                error('unexpected nasm version string: @0@'.format(nasm_r.stdout()))
            endif
        endif

        # Generate config.asm
        cdata_asm.set10('ARCH_X86_64', host_machine.cpu_family() == 'x86_64')
        cdata_asm.set10('ARCH_X86_32', host_machine.cpu_family() == 'x86')
        cdata_asm.set10('PIC', true)
        config_asm_target = configure_file(output: 'config.asm', output_format: 'nasm', configuration: cdata_asm)

        if host_machine.system() == 'windows' or host_machine.system() == 'cygwin'
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
        xdd_supports_n = run_command(xxd, '-n', 'test', check: false).returncode() == 0
        if meson.is_subproject () and not xdd_supports_n
            error('xxd with support for -n option is required building as a subproject. Please install a newer version of xxd or disable built-in models.')
        endif

        model_files = [
            'vmaf_v0.6.1.json',
            'vmaf_b_v0.6.3.json',
            'vmaf_v0.6.1neg.json',
            'vmaf_4k_v0.6.1.json',
            'vmaf_4k_v0.6.1neg.json',
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
            if xdd_supports_n
                command = [xxd, '-i', '-n', 'src_@PLAINNAME@', '@INPUT@', '@OUTPUT@']
            else
                command = [xxd, '-i', '@INPUT@', '@OUTPUT@']
            endif
            json_model_c_sources += custom_target(
                  model_file,
                  output : '@PLAINNAME@.c',
                  input : configure_file(
                    input : model_dir + model_file,
                    output : model_file,
                    copy: true
                  ),
                  command : command,
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

    if host_machine.cpu_family().startswith('arm64') or host_machine.cpu_family().startswith('aarch64')
        libvmaf_cpu_sources += files(
            src_dir + 'arm/cpu.c',
        )
    endif
endif

libvmaf_include = include_directories(
    '.',
    '../include',
    src_dir,
    feature_src_dir,
    feature_src_dir + 'common',
    is_nvtx_enabled ? '/usr/local/cuda/include' : '',
)

libvmaf_cpu_static_lib = static_library(
    'libvmaf_cpu',
    libvmaf_cpu_sources,
    include_directories : [libvmaf_include],
)

platform_specific_cpu_objects = []

if is_asm_enabled
    if host_machine.cpu_family().startswith('arm64') or host_machine.cpu_family().startswith('aarch64')
        arm64_sources = [
          feature_src_dir + 'arm64/vif_neon.c',
          feature_src_dir + 'arm64/adm_neon.c',
        ]

          arm64_static_lib = static_library(
          'arm64_v8',
          arm64_sources,
          include_directories : vmaf_base_include,
          c_args : vmaf_cflags_common + ['-DARCH_AARCH64']
        )

        platform_specific_cpu_objects += arm64_static_lib.extract_all_objects()
    endif

    if host_machine.cpu_family().startswith('x86')
      x86_avx2_sources = [
          feature_src_dir + 'common/convolution_avx.c',
          feature_src_dir + 'x86/motion_avx2.c',
          feature_src_dir + 'x86/vif_avx2.c',
          feature_src_dir + 'x86/adm_avx2.c',
          feature_src_dir + 'x86/cambi_avx2.c',
          feature_src_dir + 'x86/psnr_avx2.c',
      ]

      x86_avx2_static_lib = static_library(
          'x86_avx2',
          x86_avx2_sources,
          include_directories : vmaf_base_include,
          c_args : ['-mavx', '-mavx2'] + vmaf_cflags_common,
      )

      platform_specific_cpu_objects += x86_avx2_static_lib.extract_all_objects(recursive: true)

      if is_avx512_enabled and is_avx512_supported
        x86_avx512_sources = [
            feature_src_dir + 'x86/motion_avx512.c',
            feature_src_dir + 'x86/vif_avx512.c',
        ]

        x86_avx512_static_lib = static_library(
            'x86_avx512',
            x86_avx512_sources,
            include_directories : vmaf_base_include,
            c_args : ['-mavx512f', '-mavx512dq', '-mavx512bw', '-mavx512cd', '-mavx512dq',
                      '-mavx512vbmi', '-mavx512vl'] +
                     vmaf_cflags_common,
        )

        platform_specific_cpu_objects += x86_avx512_static_lib.extract_all_objects(recursive: true)
      endif

    endif
endif

cuda_dependency = []
common_cuda_objects = []
cuda_inc = []

if is_nvtx_enabled
    cuda_dependency += declare_dependency(
        link_with: shared_library('dl'),
        include_directories: include_directories('/usr/local/cuda/include'))
endif

if is_cuda_enabled
    # Check for required CUDA headers
    if not cc.has_header('ffnvcodec/dynlink_cuda.h')
        error('ffnvcodec/dynlink_cuda.h not found. Please install nv-codec-headers from https://code.ffmpeg.org/FFmpeg/nv-codec-headers (commit 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d or greater required).')
    endif
    if not cc.has_header('ffnvcodec/dynlink_loader.h')
        error('ffnvcodec/dynlink_loader.h not found. Please install nv-codec-headers from https://code.ffmpeg.org/FFmpeg/nv-codec-headers (commit 876af32a202d0de83bd1d36fe74ee0f7fcf86b0d or greater required).')
    endif

    cuda_cu_sources = {
        'adm_dwt2' : [feature_src_dir + 'cuda/integer_adm/adm_dwt2.cu'],
        'adm_cm' : [feature_src_dir + 'cuda/integer_adm/adm_cm.cu'],
        'adm_csf' : [feature_src_dir + 'cuda/integer_adm/adm_csf.cu'],
        'adm_csf_den' : [feature_src_dir + 'cuda/integer_adm/adm_csf_den.cu'],
        'adm_decouple' : [feature_src_dir + 'cuda/integer_adm/adm_decouple.cu'],
        'filter1d' : [feature_src_dir + 'cuda/integer_vif/filter1d.cu'],
        'motion_score' : [feature_src_dir + 'cuda/integer_motion/motion_score.cu'],
    }
    message(cuda_cu_sources)
    cuda_sources = [
        cuda_dir + 'common.c',
        cuda_dir + 'picture_cuda.c',
        cuda_dir + 'ring_buffer.c',
    ]
    gencode = []
    if get_option('enable_nvcc')
        cuda_lang = add_languages('cuda', required : true)
        cuda_compiler = meson.get_compiler('cuda')
        nvcc_exe = find_program('nvcc')

        gencode = [
            '--fatbin',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
        ]
        message('Found CUDA version = @0@'.format(cuda_compiler.version()))
        if cuda_compiler.version().version_compare('<13')
            gencode += '-gencode=arch=compute_50,code=compute_50'
        endif
        # always compile device code to enable quick startup on newer GPUs, for the last supported GPU also generate PTX for future compatibility
        if cuda_compiler.version().version_compare('>11.8')
            gencode += '-gencode=arch=compute_90,code=sm_90'
            if cuda_compiler.version().version_compare('>12.8')
                gencode += [
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_120,code=sm_120',
                    '-gencode=arch=compute_120,code=compute_120'
                ]
            else
                gencode += '-gencode=arch=compute_90,code=compute_90'
            endif
        else
            gencode += '-gencode=arch=compute_80,code=compute_80'
        endif

    else
        nvcc_exe = find_program('clang')
        gencode = [
            '--cuda-gpu-arch=sm_75',
            '--cuda-device-only',
            '-S'
            # many complaints about device intrinsics when compiling without CTK and the ffmpeg cuda runtime header
            #            '-nocudainc',
            #            '-nocudalib',
        ]
    endif
    message('CUDA gencode = @0@'.format(gencode))


    cuda_flags = []
    if get_option('buildtype') == 'debug'
        cuda_flags += ['-DCUDA_DEBUG', '-G']
    else
        if is_nvtx_enabled
            cuda_flags += ['-lineinfo']
        endif
    endif

    ptx_files = {}
    foreach name, _cu : cuda_cu_sources
        t = custom_target('cu_ptx_target_' + name,
            build_by_default: true,
            output : ['@0@.fatbin'.format(name)],
            input : _cu,
            command : [nvcc_exe,
                gencode,
                '@INPUT@',
                '-o', '@OUTPUT@' ,
                '-I', './src',
                '-I', '../src',
                '-I', '../include',
                '-I', '../src/feature',
                '-I', '../src/' + cuda_dir,
                '-DDEVICE_CODE',
            ] + cuda_flags
        )
        ptx_files += {name : [t]}
    endforeach

    message('ptx_files = @0@'.format(ptx_files.keys()))

    # bin2c is distributed along with cuda tools. Use '--padd 0x00' to add a NULL-terminator byte
    # to the end of the generated array.
    bin2c_exe = find_program('bin2c')
    ptx_arrays = []
    foreach name, _ptx : ptx_files
        t = custom_target('ptx_bin2c_@0@'.format(name),
            build_by_default: true,
            output : ['@PLAINNAME@.c'],
            input : _ptx,
            capture : true,
            command : [bin2c_exe, '--const', '--padd', '0x00',
                '--name', '@BASENAME@_ptx', '@INPUT@',
            ]
        )
        ptx_arrays += t
    endforeach

    cuda_static_lib = static_library(
        'cuda_common_vmaf_lib',
        [cuda_sources, ptx_arrays,],
        include_directories : [
            libvmaf_include,
            vmaf_base_include,
            cuda_dir,
            feature_src_dir,
            include_directories('../src/cuda/'),
            cuda_inc,
        ],
        c_args : vmaf_cflags_common,
        cuda_args: cuda_flags # + ['-gencode', 'arch=compute_86,code=sm_86' ] #, '--use_fast_math']
    )

    common_cuda_objects += cuda_static_lib.extract_all_objects()
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

if is_cuda_enabled
    libvmaf_feature_sources += [
        feature_src_dir + 'cuda/integer_adm_cuda.c',
        feature_src_dir + 'cuda/integer_vif_cuda.c',
        feature_src_dir + 'cuda/integer_motion_cuda.c',
    ]
endif

libvmaf_feature_static_lib = static_library(
    'libvmaf_feature',
    libvmaf_feature_sources,
    include_directories : [libvmaf_include, vmaf_include, cuda_dir],
    dependencies: [stdatomic_dependency, cuda_dependency],
    objects: common_cuda_objects
)

libsvm_cpp_args = vmaf_cflags_common
if meson.get_compiler('cpp').get_id() == 'clang' or meson.get_compiler('cpp').get_id() == 'apple-clang'
    libsvm_cpp_args += ['-U_LIBCPP_ENABLE_ASSERTIONS']
endif

libsvm_static_lib = static_library(
    'libsvm',
    src_dir + 'svm.cpp',
    include_directories : [libvmaf_include, vmaf_include],
    dependencies : [thread_lib],
    cpp_args : libsvm_cpp_args,
)

libvmaf_sources = [
    src_dir + 'libvmaf.c',
    src_dir + 'predict.c',
    src_dir + 'model.c',
    src_dir + 'picture.c',
    src_dir + 'mem.c',
    src_dir + 'output.c',
    src_dir + 'fex_ctx_vector.c',
    src_dir + 'thread_pool.c',
    src_dir + 'dict.c',
    src_dir + 'opt.c',
    src_dir + 'ref.c',
    src_dir + 'read_json_model.c',
    src_dir + 'pdjson.c',
    src_dir + 'log.c',
    src_dir + 'framesync.c',
    src_dir + 'metadata_handler.c',
]

if is_cuda_enabled
    vmaf_cflags_common += '-DHAVE_CUDA'
endif
if is_nvtx_enabled
    vmaf_cflags_common += '-DHAVE_NVTX'
endif

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
    dependencies : [
      thread_lib,
      math_lib,
      stdatomic_dependency,
      cuda_dependency,
    ],
    objects : [
        platform_specific_cpu_objects,
        libvmaf_feature_static_lib.extract_all_objects(recursive: true),
        libvmaf_cpu_static_lib.extract_all_objects(recursive: true),
        libsvm_static_lib.extract_all_objects(recursive: true),
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
