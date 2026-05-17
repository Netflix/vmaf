## Fixed

- `dev/Containerfile`: add Intel GPU compute-runtime (`intel-opencl-icd`,
  `libze-intel-gpu1`) from Intel's GPU apt repo so the Level Zero loader can
  probe the Arc GPU ICD inside the container. Without these packages
  `sycl-ls` returned "No platforms found" even with the oneAPI compiler and
  Level Zero loader already present.
- `dev/Containerfile`: add `/opt/intel/oneapi/umf/latest/lib` to
  `LD_LIBRARY_PATH` so the Unified Runtime Level Zero adapter
  (`libur_adapter_level_zero.so.0`) can dlopen `libumf.so.1` at startup.
  Root-caused via `LD_DEBUG=libs` on the host Arc A380: the adapter was
  dlopen-failing with "libumf.so.1: cannot open shared object file" because
  the oneAPI compiler lib path only exposes `libumf.so` (no `.1` soname).
- `dev/Containerfile`: document that `rocm-hip-runtime-dev` is sufficient for
  HIP kernel dispatch on AMD hardware (it includes `libhsa-runtime64.so` via
  `hsa-rocr` and `libamdhip64.so`); no additional host-side ROCm install is
  needed beyond the `amdgpu` kernel module and device nodes.
- `dev/Containerfile`: add build-time `sycl-ls` / `clinfo --list` /
  `rocminfo` probe `RUN` step (non-fatal, `|| true`) so GPU runtime install
  gaps surface in the build log rather than silently at container startup.
