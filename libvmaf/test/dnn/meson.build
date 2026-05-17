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
    ['../test.c', 'test_model_loader.c',
     '../../src/dnn/model_loader.c',
     '../../src/dnn/onnx_scan.c',
     '../../src/dnn/op_allowlist.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
)

# T6-9 / ADR-0209 — Sigstore-bundle verification failure modes. Compiles
# model_loader.c directly so the test binary exercises the in-tree
# vmaf_dnn_verify_signature() implementation; does NOT require `cosign`
# on the host because every covered branch fails before posix_spawnp().
test_tiny_model_verify = executable('test_tiny_model_verify',
    ['../test.c', 'test_tiny_model_verify.c',
     '../../src/dnn/model_loader.c',
     '../../src/dnn/onnx_scan.c',
     '../../src/dnn/op_allowlist.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
)

test_onnx_scan = executable('test_onnx_scan',
    ['../test.c', 'test_onnx_scan.c',
     '../../src/dnn/onnx_scan.c',
     '../../src/dnn/op_allowlist.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
)

# Public-surface smoke test for vmaf_dnn_session_run(). Links against
# libvmaf so it exercises the real stub / real-ORT path that downstream
# consumers get, not a compiled-in copy.
test_dnn_session_api = executable('test_dnn_session_api',
    ['../test.c', 'test_dnn_session_api.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [cuda_dependency, sycl_dependency],
)

# Execution-provider selection + fp16_io round-trip. Needs the real
# libvmaf so that the attached-EP accessor is resolved from the same
# TU that constructed the session; running against a compiled-in copy
# of ort_backend.c would defeat the test. The fp16 branch requires the
# real ORT build; on stub builds every path short-circuits to NULL/ENOSYS.
test_ep_fp16 = executable('test_ep_fp16',
    ['../test.c', 'test_ep_fp16.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, cuda_dependency, sycl_dependency],
)

# Direct unit tests for ort_backend.c internal helpers exposed via
# ort_backend_internal.h (fp16 conversion edges, resolve_name,
# NULL-guard branches that the public dnn.h surface never reaches on a
# CPU-only ORT CI build). See ADR-0112.
test_ort_internals = executable('test_ort_internals',
    ['../test.c', 'test_ort_internals.c'],
    include_directories : dnn_test_inc,
    c_args : dnn_defines,
    link_with : get_option('default_library') == 'both' ? libvmaf.get_static_lib() : libvmaf,
    dependencies : [math_lib, cuda_dependency, sycl_dependency],
)

test('test_tensor_io',       test_tensor_io,       suite : 'dnn')
test('test_op_allowlist',    test_op_allowlist,    suite : 'dnn')
test('test_model_loader',    test_model_loader,    suite : 'dnn')
test('test_tiny_model_verify', test_tiny_model_verify, suite : 'dnn')
test('test_onnx_scan',       test_onnx_scan,       suite : 'dnn')
test('test_dnn_session_api', test_dnn_session_api,
     workdir : meson.project_source_root() / '..',
     suite : 'dnn')
test('test_ep_fp16',         test_ep_fp16,
     workdir : meson.project_source_root() / '..',
     suite : 'dnn')
test('test_ort_internals',   test_ort_internals,
     workdir : meson.project_source_root() / '..',
     suite : 'dnn')

bash_prog = find_program('bash', required : false)
if bash_prog.found()
    # test_cli shells out to the built `vmaf` CLI binary, which only
    # exists when -Denable_tools=true. Skip the CLI-level smoke test
    # when tools are disabled (e.g. on the Windows MSVC build-only
    # legs where <unistd.h> / <getopt.h> aren't available) or when
    # cross-compiling (bash invokes VMAF_BIN directly, bypassing meson's
    # exe_wrapper; qemu-user then fails without QEMU_LD_PREFIX).
    if get_option('enable_tools') and not meson.is_cross_build()
        test('test_cli',
             bash_prog,
             args : [files('test_cli.sh')],
             env  : { 'VMAF_BIN' : vmaf.full_path() },
             depends : vmaf,
             suite : 'dnn')
    endif

    # Registry integrity — verifies every model/tiny/*.onnx listed in
    # registry.json exists and matches its pinned sha256. Runs against the
    # source tree, not the build tree. Fails loudly if a tiny-model file
    # is tampered with without a registry bump.
    test('test_registry',
         bash_prog,
         args : [files('test_registry.sh')],
         env  : { 'TINY_DIR' : meson.project_source_root() / '..' / 'model' / 'tiny' },
         suite : 'dnn')
endif

# vmaf_tiny_v2 ONNX smoke (ADR-0244). Pure-Python; verifies the
# registry entry, sidecar, and ONNX inference shape. Skips cleanly when
# python3 / onnxruntime is missing — the registry sha256 is already
# guarded by test_registry above; this test adds the sidecar contract +
# onnxruntime load-path coverage.
python_prog = find_program('python3', required : false)
if python_prog.found()
    test('test_vmaf_tiny_v2',
         python_prog,
         args : [files('test_vmaf_tiny_v2.py')],
         workdir : meson.project_source_root() / '..',
         suite : 'dnn')
endif
