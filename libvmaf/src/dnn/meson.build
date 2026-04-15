# DNN runtime — ONNX Runtime wrapper for tiny-model inference.
#
# `enable_dnn` is a `feature` option (auto / enabled / disabled).
#   - enabled  → ORT must be discoverable via pkg-config; build fails otherwise
#   - auto     → use ORT if found, fall back to stub symbols otherwise
#   - disabled → build stub symbols, never link ORT
#
# All translation units guard their real bodies behind `VMAF_HAVE_DNN`; when
# the macro is unset they compile into -ENOSYS stubs, so libvmaf always
# exports the public `libvmaf/dnn.h` symbols.

enable_dnn_opt = get_option('enable_dnn')

ort_dep = dependency('libonnxruntime', required: false)
if not ort_dep.found()
  ort_dep = dependency('onnxruntime', required: false)
endif

dnn_have_ort = false
dnn_defines  = []
dnn_deps     = []

if enable_dnn_opt.enabled() and not ort_dep.found()
  error('-Denable_dnn=enabled but ONNX Runtime was not found via pkg-config (libonnxruntime / onnxruntime)')
endif

if (enable_dnn_opt.auto() or enable_dnn_opt.enabled()) and ort_dep.found()
  dnn_have_ort = true
  dnn_defines += '-DVMAF_HAVE_DNN=1'
  dnn_deps    += ort_dep
endif

dnn_sources = files(
  'dnn_api.c',
  'model_loader.c',
  'op_allowlist.c',
  'ort_backend.c',
  'tensor_io.c',
)
dnn_inc = include_directories('.', '../../include')
