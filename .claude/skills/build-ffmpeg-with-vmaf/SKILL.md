---
name: build-ffmpeg-with-vmaf
description: Clone (or update) ffmpeg, apply our fork's patches under ffmpeg-patches/, configure with --enable-libvmaf against our local libvmaf.so, build, and run smoke tests on both the 'libvmaf' and 'vmaf_pre' filters.
---

# /build-ffmpeg-with-vmaf

End-to-end verification that our libvmaf (CLI, C API, tiny-AI surface) integrates
correctly with ffmpeg.

## Invocation

```
/build-ffmpeg-with-vmaf [--ffmpeg-ref=master|n7.0|<sha>] [--ffmpeg-dir=/tmp/ffmpeg]
                        [--libvmaf-build=libvmaf/build] [--jobs=N]
                        [--run-filter-smoketest]
```

Defaults: `master` at HEAD, `/tmp/ffmpeg` checkout, our latest `libvmaf/build`, `$(nproc)`
jobs, smoke-test enabled.

## Steps

1. Ensure `libvmaf/build/src/libvmaf.so.3.0.0` + `libvmaf/build/src/libvmaf.pc` exist.
   If not, call `/build-vmaf --backend=cpu` first.
2. Clone or fetch the ffmpeg repo into `--ffmpeg-dir`. `git clean -fdx` before applying
   patches (to keep idempotent).
3. Checkout `--ffmpeg-ref`.
4. Apply every `*.patch` in `ffmpeg-patches/` in lexicographic order via `git am`.
   On conflict: abort the `am`, report the failing patch + hunks, suggest
   `/refresh-ffmpeg-patches`.
5. Configure ffmpeg:
   ```
   PKG_CONFIG_PATH=$repo/libvmaf/build/src:$PKG_CONFIG_PATH \
   LD_LIBRARY_PATH=$repo/libvmaf/build/src:$LD_LIBRARY_PATH \
   ./configure --prefix=/tmp/ffmpeg-install --enable-libvmaf --enable-gpl \
               --enable-version3
   ```
6. `make -j$jobs && make install`.
7. Smoke test (if `--run-filter-smoketest`):
   - `ffmpeg -i testdata/ref_576x324_48f.yuv -i testdata/dis_576x324_48f.yuv \
      -lavfi "[0:v][1:v]libvmaf=log_path=/tmp/vmaf.xml" -f null -`
   - `ffmpeg ... -lavfi "vmaf_pre=tiny_model=model/tiny/psnr_proxy.onnx"` (only if the
      tiny-AI model file is present — skip silently otherwise).
   - Verify `/tmp/vmaf.xml` contains a `pooled_metrics` block with `vmaf` mean ≥ 0.
8. Report: ffmpeg ref built, patches applied (N), filter test pass/fail,
   resulting ffmpeg binary path.

## Notes

- Uses our local `libvmaf.so`, NOT a system-installed one. `LD_LIBRARY_PATH` scoping
  is critical to avoid false positives against an older installed copy.
- If no patches exist yet (`ffmpeg-patches/` is empty or absent), skip step 4 and
  configure against vanilla ffmpeg — still a valid integration test.
- Does NOT push or install ffmpeg system-wide.
