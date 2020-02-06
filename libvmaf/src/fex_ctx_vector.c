#include <errno.h>

#include "feature/feature_extractor.h"
#include "fex_ctx_vector.h"

int feature_extractor_vector_init(RegisteredFeatureExtractors *rfe)
{
    rfe->cnt = 0;
    rfe->capacity = 8;
    size_t sz = sizeof(*(rfe->fex_ctx)) * rfe->capacity;
    rfe->fex_ctx = malloc(sz);
    if (!rfe->fex_ctx) return -ENOMEM;
    memset(rfe->fex_ctx, 0, sz);
    return 0;
}

int feature_extractor_vector_append(RegisteredFeatureExtractors *rfe,
                                    VmafFeatureExtractorContext *fex_ctx)
{
    if (!rfe) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    for (unsigned i = 0; i < rfe->cnt; i++) {
        if (!strcmp(rfe->fex_ctx[i]->fex->name, fex_ctx->fex->name))
            return vmaf_feature_extractor_context_destroy(fex_ctx);
    }

    if (rfe->cnt >= rfe->capacity) {
        size_t capacity = rfe->capacity * 2;
        VmafFeatureExtractorContext **fex_ctx =
            realloc(rfe->fex_ctx, sizeof(*(rfe->fex_ctx)) * capacity);
        if (!fex_ctx) return -ENOMEM;
        rfe->fex_ctx = fex_ctx;
        rfe->capacity = capacity;
        for (unsigned i = rfe->cnt; i < rfe->capacity; i++)
            rfe->fex_ctx[i] = NULL;
    }

    rfe->fex_ctx[rfe->cnt++] = fex_ctx;
    return 0;
}

void feature_extractor_vector_destroy(RegisteredFeatureExtractors *rfe)
{
    if (!rfe) return;
    for (unsigned i = 0; i < rfe->cnt; i++) {
        vmaf_feature_extractor_context_close(rfe->fex_ctx[i]);
        vmaf_feature_extractor_context_destroy(rfe->fex_ctx[i]);
    }
    free(rfe->fex_ctx);
    return;
}
