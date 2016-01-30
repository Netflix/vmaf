#pragma once

#ifndef ADM_OPTIONS_H_
#define ADM_OPTIONS_H_

/* Whether to use a trigonometry-free method for comparing angles. */
#define ADM_OPT_AVOID_ATAN

/* Whether to save intermediate results to files. */
/* #define ADM_OPT_DEBUG_DUMP */

/* Whether to perform division by reciprocal-multiplication. */
#define ADM_OPT_RECIP_DIVISION

/* Whether to use single precision for computation. */
#define ADM_OPT_SINGLE_PRECISION
//#define ADM_OPT_DOUBLE_PRECISION

#endif /* ADM_OPTIONS_H_ */
