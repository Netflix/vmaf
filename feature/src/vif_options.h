#pragma once

#ifndef VIF_OPTIONS_H_
#define VIF_OPTIONS_H_

/* Whether to use an alternate arrangement of the core VIF statistic. */
#define VIF_OPT_ALTERNATE_STATISTIC

/* Whether to use an approximate implementation of log2 / log2f. */
#define VIF_OPT_FAST_LOG2

/* Whether to save intermedate results to files. */
/* #define VIF_OPT_DEBUG_DUMP */

/* Whether to keep the borders of the image after filtering. */
/* #define VIF_OPT_HANDLE_BORDERS */

/* Whether to use a 1-D formulation of the Gaussian filter. */
#define VIF_OPT_FILTER_1D

/* Whether to use single precision for computation. */
#define VIF_OPT_SINGLE_PRECISION
//#define VIF_OPT_DOUBLE_PRECISION

#endif /* VIF_OPTIONS_H_ */
