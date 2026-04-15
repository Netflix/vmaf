# Bundling libvmaf_sycl for Self-Contained Deployment

## Problem

When deploying FFmpeg with `libvmaf_sycl` on a system without Intel oneAPI installed, SYCL fails with:

```
SYCL exception: No device of requested type available
```

Even though the Intel iGPU is present and VA-API works, the SYCL runtime libraries are missing.

## Required Runtime Libraries

### Intel oneAPI Runtime (from `/opt/intel/oneapi/compiler/latest/lib/`)

| Library | Purpose |
|---------|---------|
| `libsycl.so` | SYCL runtime |
| `libze_loader.so` | Level Zero loader (GPU compute API) |
| `libsvml.so` | Intel short vector math library |
| `libirc.so` | Intel compiler runtime |
| `libpi_level_zero.so` | SYCL plugin for Level Zero backend (runtime-loaded) |

### FFmpeg Integration

| Library | Purpose |
|---------|---------|
| `libvpl.so.2` | Intel VPL dispatcher (QSV interop for zero-copy decode→VMAF) |

### System Libraries (may be missing in minimal/container environments)

| Library | Purpose |
|---------|---------|
| `libdrm.so.2` | DRM access (required by Level Zero) |
| `libva.so` | VA-API (required for DMA-BUF zero-copy path) |
| `libva-drm.so` | VA-API DRM backend |

### Transitive Dependencies

Check the build machine for additional transitive deps:

```bash
ldd /opt/intel/oneapi/compiler/latest/lib/libze_loader.so
ldd /opt/intel/oneapi/compiler/latest/lib/libsycl.so
```

Any non-standard deps (e.g. `libspdlog`, `libfmt`) also need bundling.

## Cannot Be Bundled (must exist on target)

- `i915` or `xe` kernel module (Intel GPU driver)
- `/dev/dri/render*` device node access
- Standard glibc (`libc.so`, `libm.so`, `libpthread.so`)

## Bundling Steps

1. **Copy the `.so` files** into the FFmpeg binary directory (or a `lib/` subdirectory).

2. **Set RPATH at link time** so the binary finds them without `LD_LIBRARY_PATH`:
   ```bash
   # Same directory as binary
   -Wl,-rpath,'$ORIGIN'
   # Or a lib/ subdirectory
   -Wl,-rpath,'$ORIGIN/lib'
   ```

3. **Alternatively**, have users set `LD_LIBRARY_PATH` at runtime:
   ```bash
   export LD_LIBRARY_PATH=/path/to/bundled/libs:$LD_LIBRARY_PATH
   ```

## Verifying

Check which libraries are missing at runtime:

```bash
ldd /path/to/ffmpeg | grep -E 'sycl|ze_loader|svml|irc|pi_level_zero|vpl|drm|libva'
```

Any "not found" entries need to be bundled.

## Notes

- SPIR-V device code is embedded in the binary at link time via `clang-offload-wrapper`, so no extra device code files are needed.
- The kernel driver (`i915` or `xe`) must still be loaded on the target system — it cannot be bundled.
- `libva` and `libva-drm` are only needed if using the DMA-BUF zero-copy path; otherwise the CPU upload path is used.
