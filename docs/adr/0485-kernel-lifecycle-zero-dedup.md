# ADR-0485: Extract `VMAF_LIFECYCLE_ZERO` macro to eliminate struct-init duplication across HIP and Metal kernel templates

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: `cuda`, `framework`, `lint`, `build`

## Context

The 2026-05-16 GPU-template dedup audit (`dedup-audit-gpu-templates-2026-05-16.md`)
identified opportunity #2: the HIP and Metal `kernel_template` init functions each
contain an identical field-by-field zero-init block for their lifecycle and
readback/buffer structs:

```c
/* HIP kernel_template.c — lifecycle_init */
lc->str      = 0;
lc->submit   = 0;
lc->finished = 0;

/* Metal kernel_template.mm — lifecycle_init */
lc->cmd_queue = 0;
lc->submit    = 0;
lc->finished  = 0;
```

And in the readback/buffer alloc functions:

```c
/* HIP */   rb->device = NULL; rb->host_pinned = NULL; rb->bytes = bytes;
/* Metal */ buf->buffer = 0; buf->host_view = NULL; buf->bytes = bytes;
```

Both patterns exist because each backend's struct was written independently and
the struct definitions cannot be unified (the `uintptr_t` fields store handle
types with different semantics: `hipStream_t` vs `MTLCommandQueue`).  However,
the zero-init action is identical: bring all fields to the all-zeros sentinel that
signals "not yet initialised."

A field-by-field pattern is fragile: adding a new field to either struct silently
skips the zero-out unless the author remembers to update both the HIP and Metal
init functions.

## Decision

Introduce `libvmaf/src/kernel_lifecycle_common.h` with a single
`VMAF_LIFECYCLE_ZERO(lc)` macro backed by `memset(lc, 0, sizeof(*lc))`.  Both
`hip/kernel_template.c` and `metal/kernel_template.mm` include this header and
replace the field-by-field zero blocks with `VMAF_LIFECYCLE_ZERO`.

The struct definitions and all other lifecycle logic remain backend-local.  The
shared header is intentionally minimal — it contains only the zero macro and the
`#include <string.h>` it requires.

## Alternatives considered

| Option | Pros | Cons |
|---|---|---|
| Keep field-by-field zeroing | No new file; zero coupling | Fragile: new fields silently missed; pattern duplicates across backends |
| Full cross-backend lifecycle struct | Maximum sharing | Impossible: handle types differ; header-purity invariant (no `<hip/hip_runtime.h>` in public surface) forbids it |
| `memset` inline at each callsite (no shared header) | No new file | Does not establish a named, searchable pattern; still duplicated |
| `VMAF_LIFECYCLE_ZERO` macro in shared header (chosen) | One place to update; new fields auto-covered; grep-able | Adds one 55-line header |

## Consequences

**Positive**
- New fields added to `VmafHipKernelLifecycle`, `VmafHipKernelReadback`,
  `VmafMetalKernelLifecycle`, or `VmafMetalKernelBuffer` are automatically
  zero-initialised; no manual update to the init functions required.
- The zero-init pattern is named and grep-able (`VMAF_LIFECYCLE_ZERO`).

**Negative**
- Adds `libvmaf/src/kernel_lifecycle_common.h` as a new internal header that
  future GPU backends should also include.

**Neutral**
- No change to struct field types, API signatures, or observable behaviour.
- No bit-exactness impact: the macro produces the same all-zeros result as the
  replaced field-by-field assignments on all POSIX targets.

## References

- Audit file: `.workingdir/dedup-audit-gpu-templates-2026-05-16.md` — opportunity #2
- `libvmaf/src/kernel_lifecycle_common.h` — implementation
- `libvmaf/src/hip/kernel_template.c` — HIP consumer
- `libvmaf/src/metal/kernel_template.mm` — Metal consumer
- [ADR-0241](0241-hip-first-consumer-psnr.md) — HIP kernel template introduction
- [ADR-0361](0361-metal-compute-backend.md) — Metal kernel template introduction
- [ADR-0484](0484-kernel-scaffolding-hip-metal-doc.md) — kernel-scaffolding doc extension
