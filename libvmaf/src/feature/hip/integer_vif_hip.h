/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the integer VIF feature extractor.
 *  Mirrors libvmaf/src/feature/cuda/integer_vif_cuda.h field-for-field,
 *  replacing CUDA Driver API types with HIP equivalents.
 */

#ifndef FEATURE_INTEGER_VIF_HIP_H_
#define FEATURE_INTEGER_VIF_HIP_H_

#include <stdint.h>

/* Enhancement gain limit default — mirrors the CUDA twin. */
#ifndef DEFAULT_VIF_ENHN_GAIN_LIMIT
#define DEFAULT_VIF_ENHN_GAIN_LIMIT (100.0)
#endif

/**
 * Per-scale accumulator matching vif_accums in integer_vif_cuda.h.
 * All fields are int64_t to accommodate full-frame accumulations at
 * 4K resolution without overflow.
 */
typedef struct vif_accums_hip {
    int64_t x;
    int64_t x2;
    int64_t num_x;
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
} vif_accums_hip;

/**
 * Device-side buffer layout for the VIF HIP kernels.
 * Field order and stride semantics mirror VifBufferCuda.
 * All pointer fields are plain device pointers (uintptr_t casted to typed
 * pointers inside the host TU, matching the CUDA Driver API pattern).
 */
typedef struct VifBufferHip {
    /* Half-resolution downsampled ref/dis (CUdeviceptr equivalent). */
    uintptr_t ref;
    uintptr_t dis;

    /* 16-bit mu planes. */
    uint16_t *mu1;
    uint16_t *mu2;

    /* 32-bit moment planes. */
    uint32_t *mu1_32;
    uint32_t *mu2_32;
    uint32_t *ref_sq;
    uint32_t *dis_sq;
    uint32_t *ref_dis;

    /* Per-scale accumulator (device, one vif_accums_hip per scale). */
    int64_t *accum;

    /* Temporary intermediate planes (vertical-pass output). */
    struct {
        uint32_t *mu1;
        uint32_t *mu2;
        uint32_t *ref;
        uint32_t *dis;
        uint32_t *ref_dis;
        uint32_t *ref_convol;
        uint32_t *dis_convol;
        uint32_t *padding;
    } tmp;

    ptrdiff_t stride;     /* byte stride for raw ref/dis planes       */
    ptrdiff_t rd_stride;  /* byte stride for half-res downsampled buf  */
    ptrdiff_t stride_16;  /* byte stride for 16-bit mu planes          */
    ptrdiff_t stride_32;  /* byte stride for 32-bit moment planes      */
    ptrdiff_t stride_64;  /* byte stride for 64-bit accumulator plane  */
    ptrdiff_t stride_tmp; /* byte stride for 32-bit tmp planes         */
} VifBufferHip;

#ifdef HAVE_HIPCC
/**
 * Symbol produced by:
 *   hipcc --genco --offload-arch=<gfx> -o vif_statistics.hsaco  \
 *             feature/hip/integer_vif/vif_statistics.hip
 *   xxd -i vif_statistics.hsaco > vif_statistics_hsaco.c
 * The meson `hip_hsaco_sources` custom_target pipeline embeds the blob
 * exactly as it does for psnr_score.hsaco (ADR-0372).
 */
extern const unsigned char vif_statistics_hsaco[];
extern const unsigned int vif_statistics_hsaco_len;
#endif /* HAVE_HIPCC */

#endif /* FEATURE_INTEGER_VIF_HIP_H_ */
