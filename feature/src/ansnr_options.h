#pragma once

#ifndef ANSNR_OPTIONS_H_
#define ANSNR_OPTIONS_H_

/* Whether to use border replication instead of zero extension. */
/* #define ANSNR_OPT_BORDER_REPLICATE */

/* Whether to save intermediate results to files. */
/* #define ANSNR_OPT_DEBUG_DUMP */

/* Whether to use a 1-D approximation of filters. */
/* #define ANSNR_OPT_FILTER_1D */

/* Whether to normalize result by dividing against maximum ANSNR. */
/* #define ANSNR_OPT_NORMALIZE */

/* Whether to use single precision for computation. */
#define ANSNR_OPT_SINGLE_PRECISION
//#define ANSNR_OPT_DOUBLE_PRECISION

#endif /* ANSNR_OPTIONS_H_ */
