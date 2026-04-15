---
name: add-simd-path
description: Scaffold a new SIMD implementation for an existing feature. Creates intrinsics source + header + a bit-exact-vs-scalar comparison test; wires into runtime dispatch.
---

# /add-simd-path

## Invocation

```
/add-simd-path <isa> <feature>
```

- `<isa>` ∈ `avx2`, `avx512`, `avx512icl`, `neon`.
- `<feature>` ∈ the set of feature names under `libvmaf/src/feature/` (e.g. `vif`,
  `adm`, `ansnr`, `motion`, `ciede`, `ssim`).

## Files created

| Path                                                       | Purpose                         |
|------------------------------------------------------------|---------------------------------|
| `libvmaf/src/feature/x86/<feature>_<isa>.c`                | Intrinsics impl (or arm64/…)    |
| `libvmaf/src/feature/x86/<feature>_<isa>.h`                | Prototype + ISA guard           |
| `libvmaf/test/test_<feature>_<isa>_bitexact.c`             | Bit-exact vs scalar comparison  |

## Files patched

- `libvmaf/src/cpu.c` or `cpu.h` — dispatch table entry if not present.
- `libvmaf/src/feature/<feature>.c` — select SIMD impl when `cpu_supports_<isa>()`.
- `libvmaf/src/meson.build` — conditional compile with `-mavx2` / `-mavx512f` etc.
- `libvmaf/test/meson.build` — register the new bit-exact test.

## Template behavior

The intrinsics stub loads data via `loadu`, does a pass-through computation (scalar
semantics), stores via `storeu`. It compiles and passes the bit-exact test by
definition. The author replaces the stub body with the actual intrinsics.

## Guardrails

- Refuses if `libvmaf/src/feature/x86/<feature>_<isa>.c` already exists.
- The bit-exact test MUST pass before merge. If author changes reduction order, add
  a `_mm*_cvt*_pd` double-accumulate pattern (see `simd-reviewer` guidance).
- Runs `/build-vmaf --backend=cpu` at end to confirm compilation.
