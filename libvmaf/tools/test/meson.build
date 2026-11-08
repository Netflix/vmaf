# The Python harness launches the target directly, without Meson's exe_wrapper.
if get_option('enable_tests') and not meson.is_cross_build()
    python = import('python').find_installation()
    test('test_vmaf_cli_invalid_unsigned', python,
        args : [files('test_vmaf_cli_invalid_unsigned.py'), vmaf.full_path()],
        depends : vmaf,
    )
endif

if get_option('enable_cuda')
    test_vmaf_cuda_gpumask = find_program('test_vmaf_cuda_gpumask.sh')
    test('test_vmaf_cuda_gpumask', test_vmaf_cuda_gpumask)
endif
