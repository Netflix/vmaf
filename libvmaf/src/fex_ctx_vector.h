#ifndef __VMAF_SRC_FEX_CTX_VECTOR_H__
#define __VMAF_SRC_FEX_CTX_VECTOR_H__

#include "feature/feature_extractor.h"

typedef struct {
    VmafFeatureExtractorContext **fex_ctx;
    unsigned cnt, capacity;
} RegisteredFeatureExtractors;

int feature_extractor_vector_init(RegisteredFeatureExtractors *rfe);

int feature_extractor_vector_append(RegisteredFeatureExtractors *rfe,
                                    VmafFeatureExtractorContext *fex_ctx);

void feature_extractor_vector_destroy(RegisteredFeatureExtractors *rfe);

#endif /* __VMAF_SRC_FEX_CTX_VECTOR_H__ */
