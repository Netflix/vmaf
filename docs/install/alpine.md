# Installing on Alpine Linux (3.20+)

Alpine uses **musl libc**, which the CUDA and Intel oneAPI toolchains do
not officially support. The Alpine setup therefore builds a **CPU-only**
VMAF; `ENABLE_CUDA=1` and `ENABLE_SYCL=1` are silently ignored.

```bash
bash scripts/setup/alpine.sh
```

## Manual install

```bash
apk add --no-cache \
    build-base meson ninja pkgconf nasm \
    python3 py3-pip \
    clang cppcheck doxygen
```

## Build

```bash
cd libvmaf
meson setup ../build
ninja -C ../build
```

## Why Alpine?

The Alpine build is a **portability gate**: if libvmaf compiles and passes
its tests under musl, we have some confidence that the codebase is free
of glibc-specific assumptions. The CI runs this via a nightly job.
