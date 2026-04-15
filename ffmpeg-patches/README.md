# ffmpeg-patches/

Local patches against FFmpeg **n8.1** for integrating this VMAF fork into
`libavfilter/vf_libvmaf*` plus a new `vf_vmaf_pre` filter.

## Contents

- **`0001-libvmaf-add-tiny-model-option.patch`** — adds `tiny_model` /
  `tiny_device` / `tiny_threads` / `tiny_fp16` options on the existing
  `libvmaf` filter; calls `vmaf_use_tiny_model()` when set.
- **`0002-add-vmaf_pre-filter.patch`** — new `vmaf_pre` filter: luma-in /
  luma-out learned pre-processing via `vmaf_dnn_session_*`. Chroma
  planes pass through.
- **`0003-libvmaf-wire-sycl-backend-selector.patch`** — adds `sycl_device`
  / `sycl_profile` options; invokes `vmaf_sycl_state_init()` +
  `vmaf_sycl_import_state()` when `sycl_device >= 0`.

Every patch is guarded by `check_pkg_config` so it degrades gracefully when
libvmaf was built without the relevant feature (`-Denable_dnn`, `-Denable_sycl`).

## How to apply

```bash
cd /path/to/ffmpeg    # must be at tag n8.1
for p in $(grep -v '^\s*#' /path/to/vmaf/ffmpeg-patches/series.txt); do
    git apply --3way /path/to/vmaf/ffmpeg-patches/$p
done
```

Or via the helper skill: `/ffmpeg-apply-patches /path/to/ffmpeg`.

## How to smoke-test

```bash
bash ffmpeg-patches/test/build-and-run.sh
```

Requires `libvmaf` to be installed (`pkg-config --cflags libvmaf` must
resolve). Set `VMAF_PREFIX` to point at a non-standard install prefix.
Pins `FFMPEG_SHA=n8.1`; override to test against a newer tag.

## How to regenerate

After editing FFmpeg locally, run `/ffmpeg-build-patches` to diff against
the tracked upstream base and rewrite the numbered patches in place.

## License

BSD-3-Clause-Plus-Patent for patches authored in this repo; patches are
applied against FFmpeg (LGPL / GPL) — the resulting linked binary's
distribution terms are governed by FFmpeg's license, not this one.
