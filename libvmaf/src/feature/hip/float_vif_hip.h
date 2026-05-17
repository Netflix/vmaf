/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the float_vif feature extractor — ninth
 *  kernel-template consumer (T7-10b batch-5 / ADR-0379).
 *
 *  Mirrors libvmaf/src/feature/cuda/float_vif_cuda.h. The HSACO
 *  symbol declared here is produced by the meson
 *  `hip_hsaco_c_float_vif_score` custom_target pipeline:
 *  `xxd -i -n float_vif_score_hsaco float_vif_score.hsaco >
 *  float_vif_score_hsaco.c` (same pattern as ADR-0372 / ADR-0374).
 */
#ifndef FEATURE_FLOAT_VIF_HIP_H_
#define FEATURE_FLOAT_VIF_HIP_H_

/* HSACO blob embedded by xxd; consumed by `hipModuleLoadData` in
 * `float_vif_hip.c:fvif_hip_module_load()`. */
extern const unsigned char float_vif_score_hsaco[];
extern const unsigned int float_vif_score_hsaco_len;

#endif /* FEATURE_FLOAT_VIF_HIP_H_ */
