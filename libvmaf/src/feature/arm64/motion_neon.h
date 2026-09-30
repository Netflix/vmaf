#ifndef ARM64_MOTION_NEON_H_
#define ARM64_MOTION_NEON_H_

#include <stddef.h>
#include <stdint.h>

uint64_t motion_score_pipeline_8_neon(const uint8_t *prev, ptrdiff_t prev_stride,
                                      const uint8_t *cur, ptrdiff_t cur_stride,
                                      int32_t *y_row, unsigned w, unsigned h,
                                      unsigned bpc);

#endif /* ARM64_MOTION_NEON_H_ */
