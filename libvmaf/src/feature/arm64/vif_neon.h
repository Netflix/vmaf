
#ifndef ARM64_VIF_H_
#define ARM64_VIF_H_

#include "feature/integer_vif.h"

void vif_subsample_rd_8_neon(VifBuffer buf, unsigned w, unsigned h);

void vif_subsample_rd_16_neon(VifBuffer buf, unsigned w, unsigned h, int scale,
                             int bpc);

void vif_statistic_8_neon(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h);

void vif_statistic_16_neon(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale);

#endif /* ARM64_VIF_H_ */
