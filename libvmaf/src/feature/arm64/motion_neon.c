#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "feature/integer_motion.h"

uint64_t motion_score_pipeline_8_neon(const uint8_t *prev, ptrdiff_t prev_stride,
                                      const uint8_t *cur, ptrdiff_t cur_stride,
                                      int32_t *y_row, unsigned w, unsigned h,
                                      unsigned bpc)
{
    (void)prev; (void)prev_stride; (void)cur; (void)cur_stride;
    (void)y_row; (void)w; (void)h; (void)bpc;
    return 0;
}
