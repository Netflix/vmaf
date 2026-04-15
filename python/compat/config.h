/*
 * Stub config.h for building the Python/Cython extension.
 *
 * The Cython extension directly includes libvmaf .c source files which
 * pull in cpu.h -> config.h.  The meson-generated config.h enables
 * architecture-specific SIMD paths (ARCH_X86, ARCH_AARCH64) whose
 * implementation files are NOT compiled into the extension, causing
 * undefined-symbol errors at import time.
 *
 * This header disables all architecture dispatching so the generic C
 * fallback code is used instead.
 */

#pragma once

#define ARCH_X86 0
#define ARCH_X86_32 0
#define ARCH_X86_64 0
#define ARCH_AARCH64 0
#define HAVE_ASM 0
#define HAVE_AVX512 0
#define HAVE_CUDA 0
#define HAVE_SYCL 0
