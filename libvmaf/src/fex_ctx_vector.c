/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <string.h>

#include "feature/feature_extractor.h"
#include "feature/feature_name.h"
#include "fex_ctx_vector.h"
#include "log.h"

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
                                    VmafFeatureExtractorContext *fex_ctx,
                                    uint64_t flags)
{
    if (!rfe) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    (void) flags;

    for (unsigned i = 0; i < rfe->cnt; i++) {
        char *feature_a =
            vmaf_feature_name_from_options(rfe->fex_ctx[i]->fex->name,
                    rfe->fex_ctx[i]->fex->options, rfe->fex_ctx[i]->fex->priv);

        char *feature_b =
            vmaf_feature_name_from_options(fex_ctx->fex->name,
                    fex_ctx->fex->options, fex_ctx->fex->priv);

        int ret = 1;
        if (feature_a && feature_b)
            ret = strcmp(feature_a, feature_b);

        free(feature_a);
        free(feature_b);

        if (ret) continue;

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

    const unsigned cnt = fex_ctx->opts_dict ? fex_ctx->opts_dict->cnt : 0;
    vmaf_log(VMAF_LOG_LEVEL_DEBUG,
             "feature extractor \"%s\" registered "
             "with %d opts\n",
             fex_ctx->fex->name, cnt);

    for (unsigned i = 0; i < cnt; i++) {
        vmaf_log(VMAF_LOG_LEVEL_DEBUG,"%s: %s\n",
                 fex_ctx->opts_dict->entry[i].key,
                 fex_ctx->opts_dict->entry[i].val);
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
