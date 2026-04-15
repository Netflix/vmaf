---
name: add-gpu-backend
description: Scaffold a complete new GPU backend (vulkan, hip, rocm, metal, opencl, etc.) with runtime, feature kernel stubs, public header, Meson options, CI workflow, doc stub, and smoke test. Flagship scaffolding skill.
---

# /add-gpu-backend

Creates a fully-wired new GPU backend directory tree + build integration. Generated
code compiles as a stub (empty kernels returning zero scores) so the CI/build pipeline
stays green while the implementation is filled in incrementally.

## Invocation

```
/add-gpu-backend <name>
```

`<name>` is lowercase, e.g. `vulkan`, `hip`, `metal`, `opencl`, `rocm`.

## Files created

| Path                                                         | Purpose                              |
|--------------------------------------------------------------|--------------------------------------|
| `libvmaf/src/<name>/common.{c,h}`                            | Context/device selection             |
| `libvmaf/src/<name>/picture_<name>.{c,h}`                    | Picture allocation/lifecycle         |
| `libvmaf/src/<name>/<name>_types.h`                          | Backend-local types                  |
| `libvmaf/src/feature/<name>/integer_adm_<name>.{c,h}`        | ADM kernel stub                      |
| `libvmaf/src/feature/<name>/integer_vif_<name>.{c,h}`        | VIF kernel stub                      |
| `libvmaf/src/feature/<name>/integer_motion_<name>.{c,h}`     | Motion kernel stub                   |
| `libvmaf/include/libvmaf/libvmaf_<name>.h`                   | Public header (fetch pic, device sel)|
| `libvmaf/test/test_<name>_smoke.c`                           | Build + init + single-frame test     |
| `docs/<name>/README.md`                                      | Backend doc stub (build, caveats)    |
| `.github/workflows/<name>.yml` (or ci.yml matrix row)        | CI job                               |

## Files patched

- `libvmaf/meson_options.txt` — new `enable_<name>` feature flag.
- `libvmaf/meson.build` — conditional subdir + dependency declaration.
- `libvmaf/src/meson.build` — feature sources + link.
- `libvmaf/src/libvmaf.c` — backend dispatch entry.
- `libvmaf/src/feature/feature_extractor.c` — registry add.
- `CLAUDE.md` + `AGENTS.md` — mention new backend in §1 "what this repo is".
- `docs/principles.md` — if the backend requires any rule relaxation, a PR-specific
  note (otherwise untouched).

## Templates

Located under `.claude/skills/add-gpu-backend/templates/` (not created by the skill —
authored by hand once for each candidate backend). Expected templates:

- `templates/vulkan/` — VkInstance, VkDevice, compute queue, SPIR-V shader skeleton.
- `templates/hip/`    — hipInit, HIP kernel skeleton.
- `templates/metal/`  — MTLDevice, MTLCommandQueue, metal shader skeleton.
- `templates/opencl/` — clGetPlatformIDs, kernel skeleton.
- `templates/generic/` — fallback when no template exists: creates minimal no-op
  stubs + the CI job.

## Workflow

1. Validate `<name>` is a known backend (has a template) or fall back to generic.
2. Check no `libvmaf/src/<name>/` exists yet; if it does, refuse.
3. Copy template files, substituting `@NAME@`, `@NAME_UPPER@`, and `@COPYRIGHT@`
   (= `Copyright 2024-2026 Lusoris and Claude (Anthropic)`) placeholders.
4. Apply patches to shared files (search-and-replace-based).
5. Run `/build-vmaf --backend=<name>` to confirm the stub compiles.
6. Run `build/test/test_<name>_smoke` to confirm init path works.
7. Open a PR checklist comment with:
   - Backend implementation checklist (picture_<name>.c, feature kernels, sanitizer
     clean, ncu/vtune/rocprof profile).
   - CODEOWNERS review required.
   - Required CI checks listed.

## Guardrails

- Never overwrites existing files.
- Never activates the backend by default (`enable_<name>` starts as `disabled`).
- Never claims bit-exactness until `/cross-backend-diff` confirms ULP ≤ 2.
