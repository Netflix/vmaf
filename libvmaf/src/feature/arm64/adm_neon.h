
#ifndef ARM_64_ADM_H_
#define ARM_64_ADM_H_

#include "feature/integer_adm.h"

void adm_dwt2_8_neon(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride);

void adm_decouple_neon(AdmBuffer *buf, int w, int h, int stride,
                       double adm_enhn_gain_limit, int32_t *adm_div_lookup);

float adm_cm_neon(AdmBuffer *buf, int w, int h, int src_stride, int csf_a_stride,
                  double adm_norm_view_dist, int adm_ref_display_height,
                  int adm_csf_mode, double adm_csf_scale, double adm_csf_diag_scale,
                  double adm_noise_weight, bool measure_aim);

#endif /* ARM64_ADM_H_ */
