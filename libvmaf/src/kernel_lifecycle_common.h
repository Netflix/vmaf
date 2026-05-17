/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Shared struct-zero helpers for GPU kernel lifecycle structs.
 *
 *  HIP and Metal kernel_template bodies both contain the same
 *  field-by-field zero-init pattern in their lifecycle_init and
 *  lifecycle_close functions (dedup opportunity #2 from the
 *  2026-05-16 GPU-template audit; ADR-0485).
 *
 *  This header provides a single `VMAF_LIFECYCLE_ZERO` macro that
 *  both backends include.  Struct definitions stay backend-local —
 *  this file cannot unify them because the handle types (`uintptr_t`
 *  fields) differ in meaning (hipStream_t vs MTLCommandQueue) even
 *  though they share the same C shape.  The macro is purely about
 *  zeroing; it does not introduce cross-backend coupling.
 *
 *  Usage:
 *
 *      VMAF_LIFECYCLE_ZERO(lc);   // equivalent to memset(lc, 0, sizeof(*lc))
 *
 *  Precondition: `lc` is a non-NULL pointer to a POD struct whose
 *  zero bit-pattern is the canonical "not-yet-initialised" sentinel
 *  (i.e. all `uintptr_t` slots == 0 means "no handle allocated").
 *  All current GPU lifecycle structs on POSIX targets satisfy this.
 */

#ifndef LIBVMAF_KERNEL_LIFECYCLE_COMMON_H_
#define LIBVMAF_KERNEL_LIFECYCLE_COMMON_H_

#include <string.h>

/*
 * Zero-initialise a lifecycle / readback / buffer struct pointer.
 *
 * Using `memset` rather than field-by-field assignment:
 *   1. Adding a new field to a struct cannot silently miss the zero-out.
 *   2. Eliminates the verbatim 3-line block that HIP and Metal
 *      both duplicate in `lifecycle_init` and `lifecycle_close`.
 *   3. `memset` to 0 on a POSIX target produces null pointers and
 *      zero integers — the same result as the explicit assignments.
 *
 * `(void)sizeof(*(lc))` — not a real expression; it just forces a
 * compile-time check that `lc` is a pointer-to-struct so the macro
 * cannot be accidentally applied to a scalar.
 */
#define VMAF_LIFECYCLE_ZERO(lc)                                                                    \
    do {                                                                                           \
        (void)sizeof(*(lc));                                                                       \
        memset((lc), 0, sizeof(*(lc)));                                                            \
    } while (0)

#endif /* LIBVMAF_KERNEL_LIFECYCLE_COMMON_H_ */
