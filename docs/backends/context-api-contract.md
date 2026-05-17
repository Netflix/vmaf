# GPU Backend Context-API Contract

This document records the three-function API surface that every GPU backend in
libvmaf must expose.  Following a consistent shape prevents per-backend API
drift and lets higher-level integration code (CLI, feature-extractor glue,
ffmpeg filter) stay uniform across backends.

See [ADR-0486](../adr/0486-context-api-contract-doc.md) for the rationale and
decision record.

---

## Required functions

Every backend named `<backend>` (e.g., `hip`, `metal`, `vulkan`) **must** expose
exactly these three functions in its public header
`libvmaf/src/<backend>/common.h`:

```c
/* Allocate and initialise a new context bound to device_index.
 *
 * Returns  0            on success; *ctx is non-NULL.
 * Returns -EINVAL       if ctx is NULL or device_index is out of range.
 * Returns -ENODEV       if no device with that index exists.
 * Returns -ENOMEM       on allocation failure.
 *
 * The caller owns *ctx and must release it with vmaf_<backend>_context_destroy.
 */
int vmaf_<backend>_context_new(Vmaf<Backend>Context **ctx, int device_index);

/* Release all resources owned by ctx.
 *
 * Safe to call with ctx == NULL (no-op).
 */
void vmaf_<backend>_context_destroy(Vmaf<Backend>Context *ctx);

/* Return the number of available devices of this type.
 *
 * Returns >= 0          number of devices (0 means "none found, but
 *                       device discovery succeeded").
 * Returns -ENODEV       if device discovery itself failed.
 */
int vmaf_<backend>_device_count(void);
```

---

## Error-return contract

| Condition                         | Return value |
|-----------------------------------|--------------|
| Success                           | `0`          |
| `ctx` or required pointer is NULL | `-EINVAL`    |
| Device index out of range         | `-EINVAL`    |
| No device found                   | `-ENODEV`    |
| Allocation failure                | `-ENOMEM`    |
| Runtime / driver error            | `-EIO` (fallback; prefer a more specific errno where the underlying runtime maps cleanly) |

`context_destroy` never returns an error — it is `void`.  Any cleanup error
that cannot be handled silently should be logged via `vmaf_log` before
discarding it.

---

## CUDA deviation

The legacy CUDA backend pre-dates this contract (inherited from Netflix
upstream) and uses `vmaf_cuda_state_init` / `vmaf_cuda_release` with a
different signature.  It is exempt from the naming rule; **do not rename it**
without a dedicated ADR and a matching update to `ffmpeg-patches/`.

---

## Opaque-handle accessors (optional)

Where a backend context owns GPU handles that consumer TUs need (e.g., stream,
queue, device handle), expose them through typed accessors:

```c
/* Returns the opaque handle; NULL for a NULL ctx.
 * Lifetime: tied to ctx — callers must NOT release the returned pointer. */
void *vmaf_<backend>_context_<handle_name>(Vmaf<Backend>Context *ctx);
```

This follows the pattern established in HIP (`vmaf_hip_context_stream`,
ADR-0212), Metal (`vmaf_metal_context_device_handle`,
`vmaf_metal_context_queue_handle`, ADR-0361), and CUDA
(`vmaf_cuda_context_stream`, ADR-0246).

---

## Checklist for new backends

When adding a new GPU backend, verify:

- [ ] `vmaf_<backend>_context_new` is declared in `libvmaf/src/<backend>/common.h`.
- [ ] `vmaf_<backend>_context_destroy` accepts NULL without crashing.
- [ ] `vmaf_<backend>_device_count` returns `>= 0` on success, `-ENODEV` on
      discovery failure.
- [ ] Error returns use POSIX errno values (not raw runtime codes).
- [ ] Opaque-handle accessors follow the `void *` + lifetime-tied pattern.
- [ ] Public header is installed via `meson.build` `install_headers`.
