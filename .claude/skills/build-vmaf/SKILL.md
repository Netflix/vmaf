---
name: build-vmaf
description: Build the libvmaf library + CLI with the requested backend combination (cpu, cuda, sycl, hip, all) and build type. Wraps meson setup + ninja. Reports wall time + output artifact paths.
---

# /build-vmaf

Thin wrapper over `meson setup` + `ninja` that enforces the canonical command lines for
this repo.

## Invocation

```
/build-vmaf [--backend=cpu|cuda|sycl|hip|all] [--config=debug|release|relwithdebinfo]
            [--sanitizers=asan,ubsan,tsan|none] [--reconfigure] [--clean]
```

Defaults: `--backend=cpu --config=release --sanitizers=none`.

## What it does

1. `cd` to `libvmaf/`.
2. If `--clean` is passed, remove `build/`.
3. `meson setup build [--reconfigure]` with the backend flags:
   - `cpu`:  `-Denable_cuda=false -Denable_sycl=false`
   - `cuda`: `-Denable_cuda=true -Denable_sycl=false`
   - `sycl`: `-Denable_cuda=false -Denable_sycl=true`
   - `hip`:  `-Denable_cuda=false -Denable_sycl=false -Denable_hip=true`  (when the
             backend is scaffolded; otherwise errors)
   - `all`:  all enabled together
4. Build type flags per `--config`:
   - `debug`           → `--buildtype=debug`
   - `release`         → `--buildtype=release`
   - `relwithdebinfo`  → `--buildtype=release -Db_ndebug=true` plus `-g` via env
5. Sanitizers per `--sanitizers`:
   - `asan`  → `-Db_sanitize=address`
   - `ubsan` → `-Db_sanitize=undefined`
   - Combine via `address,undefined`. `tsan` is standalone.
6. `ninja -C build -j$(nproc)`.
7. Reports: wall time, path to `build/tools/vmaf`, path to `build/src/libvmaf.so.3.0.0`.

## Constraints

- Sanitizers force `--buildtype=debug` (ASan+UBSan need it to be useful).
- TSan is mutually exclusive with ASan.
- CUDA + SYCL in one build requires both compilers available; skill errors out early
  if `nvcc` or `icpx` is missing.

## Uses

`.claude/skills/build-vmaf/build.sh` (the agent can invoke it directly). When running
outside Claude, call `./.claude/skills/build-vmaf/build.sh --backend=cpu`.
