# DNN unit tests.
#
# These exercise the internal helpers in libvmaf/src/dnn/ directly (tensor I/O,
# operator allowlist, model loader). They compile the dnn sources into the
# test binaries instead of linking libvmaf, because the helpers are
# file-private to the dnn/ TU set.

dnn_test_inc = [
    libvmaf_inc,
    test_inc,
    include_directories('../../src'),
    include_directories('../../src/dnn'),
]

test_tensor_io = executable('test_tensor_io',
    ['../test.c', 'test_tensor_io.c', '../../src/dnn/tensor_io.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
    dependencies : [math_lib],
)

test_op_allowlist = executable('test_op_allowlist',
    ['../test.c', 'test_op_allowlist.c', '../../src/dnn/op_allowlist.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
)

test_model_loader = executable('test_model_loader',
    ['../test.c', 'test_model_loader.c', '../../src/dnn/model_loader.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
)

test('test_tensor_io',    test_tensor_io,    suite : 'dnn')
test('test_op_allowlist', test_op_allowlist, suite : 'dnn')
test('test_model_loader', test_model_loader, suite : 'dnn')

bash_prog = find_program('bash', required : false)
if bash_prog.found()
    test('test_cli',
         bash_prog,
         args : [files('test_cli.sh')],
         env  : { 'VMAF_BIN' : vmaf.full_path() },
         depends : vmaf,
         suite : 'dnn')
endif
