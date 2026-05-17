/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the integer ADM feature extractor.
 *  Mirrors libvmaf/src/feature/cuda/integer_adm_cuda.h call-graph-for-call-graph.
 *  The four HSACO blobs (adm_dwt2, adm_csf, adm_csf_den, adm_cm) are declared
 *  here when HAVE_HIPCC is defined; in scaffold builds the header guards the
 *  symbols away so host-only translation units compile without ROCm headers.
 */

#ifndef FEATURE_ADM_HIP_H_
#define FEATURE_ADM_HIP_H_

#include "config.h"
#include "integer_adm.h"
#include "common.h"

/* --------------------------------------------------------------------- */
/* DWT band structs — mirror the CUDA twins in integer_adm_cuda.h         */
/* --------------------------------------------------------------------- */
typedef struct hip_adm_dwt_band_t {
    union {
        struct {
            int16_t *band_a; /* Low-pass V + low-pass H. */
            int16_t *band_h; /* High-pass V + low-pass H. */
            int16_t *band_v; /* Low-pass V + high-pass H. */
            int16_t *band_d; /* High-pass V + high-pass H. */
        };
        int16_t *bands[4];
    };
} hip_adm_dwt_band_t;

typedef struct hip_i4_adm_dwt_band_t {
    union {
        struct {
            int32_t *band_a; /* Low-pass V + low-pass H. */
            int32_t *band_h; /* High-pass V + low-pass H. */
            int32_t *band_v; /* Low-pass V + high-pass H. */
            int32_t *band_d; /* High-pass V + high-pass H. */
        };
        int32_t *bands[4];
    };
} hip_i4_adm_dwt_band_t;

/* --------------------------------------------------------------------- */
/* Fixed per-frame parameters passed to every ADM kernel                  */
/* --------------------------------------------------------------------- */
typedef struct AdmFixedParametersHip {
    float rfactor[3 * 4];
    uint32_t i_rfactor[3 * 4];
    float factor1[4];
    float factor2[4];
    float log2_h;
    float log2_w;

    double adm_norm_view_dist;
    double adm_enhn_gain_limit;
    int32_t adm_ref_display_height;

    int32_t dwt2_db2_coeffs_lo[10];
    int32_t dwt2_db2_coeffs_hi[10];
    int32_t dwt2_db2_coeffs_lo_sum;
    int32_t dwt2_db2_coeffs_hi_sum;
} AdmFixedParametersHip;

/* --------------------------------------------------------------------- */
/* Device-side buffer bundle (device pointers, managed by init/close)     */
/* --------------------------------------------------------------------- */
typedef struct AdmBufferHip {
    size_t ind_size_x, ind_size_y; /* strides for intermediate buffers */

    /* int16 (scale 0) and int32 (scales 1-3) DWT band arrays */
    hip_adm_dwt_band_t ref_dwt2;
    hip_adm_dwt_band_t dis_dwt2;
    hip_adm_dwt_band_t csf_f;

    hip_i4_adm_dwt_band_t i4_ref_dwt2;
    hip_i4_adm_dwt_band_t i4_dis_dwt2;
    hip_i4_adm_dwt_band_t i4_csf_f;

    /* Per-scale accumulator arrays */
    int64_t *adm_cm[4];
    uint64_t *adm_csf_den[4];

    /* Pinned host readback buffer (sizeof(int64_t) * RES_BUFFER_SIZE) */
    void *results_host;

    /* Backing device allocations */
    void *data_buf;    /* packed DWT bands */
    void *tmp_ref;     /* scale 1-3 vertical-pass scratch (ref) */
    void *tmp_dis;     /* scale 1-3 vertical-pass scratch (dis) */
    void *tmp_accum;   /* CM per-pixel accumulator */
    void *tmp_accum_h; /* CM row accumulator */
    void *tmp_res;     /* packed CM + CSF-den output accumulator */
} AdmBufferHip;

/* --------------------------------------------------------------------- */
/* HSACO blobs — embedded by xxd -i during the meson hipcc pipeline       */
/* --------------------------------------------------------------------- */
#ifdef HAVE_HIPCC
extern const unsigned char adm_dwt2_hsaco[];
extern const unsigned int adm_dwt2_hsaco_len;
extern const unsigned char adm_csf_hsaco[];
extern const unsigned int adm_csf_hsaco_len;
extern const unsigned char adm_csf_den_hsaco[];
extern const unsigned int adm_csf_den_hsaco_len;
extern const unsigned char adm_cm_hsaco[];
extern const unsigned int adm_cm_hsaco_len;
#endif /* HAVE_HIPCC */

#endif /* FEATURE_ADM_HIP_H_ */
