
#ifndef ARM_64_ADM_H_
#define ARM_64_ADM_H_

#include "feature/integer_adm.h"

void adm_dwt2_8_neon(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride);

#endif /* ARM64_ADM_H_ */
