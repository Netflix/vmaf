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
                                    VmafFeatureExtractorContext *fex_ctx,
                                    uint64_t flags)
{
    if (!rfe) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    for (unsigned i = 0; i < rfe->cnt; i++) {
        if (!strcmp(rfe->fex_ctx[i]->fex->name, fex_ctx->fex->name)) {
            /* same fex */
            if (flags & VMAF_FEATURE_EXTRACTOR_CONTEXT_DO_NOT_OVERWRITE) {
                /* if do not overwrite, check opts_dict consistency */
                if (!rfe->fex_ctx[i]->opts_dict && !fex_ctx->opts_dict) {
                    /* skip if both opts_dict are NULL */
                    return 0;
                } else if (!rfe->fex_ctx[i]->opts_dict || !fex_ctx->opts_dict) {
                    /* error if one dict is NULL and the other is not.
                     * Note that this does not handle the case the non-NULL dict's value is
                     * equal to the default value of the NULL dict's. But this is not fixable
                     * in the current framework unless the default values of fex's options
                     * are exposed in the current layer. FIXME */
                    return -ENOMEM;
                }
                VmafDictionary *d =
                        vmaf_dictionary_merge(&rfe->fex_ctx[i]->opts_dict,
                                              &fex_ctx->opts_dict, VMAF_DICT_DO_NOT_OVERWRITE);
                if (!d) {
                    return -ENOMEM;
                }
                else {
                    vmaf_dictionary_free(&d);
                    return 0;
                }
            } else {
                /* if allow overwrite, merge opt_dict */
                VmafDictionary *d =
                        vmaf_dictionary_merge(&rfe->fex_ctx[i]->opts_dict,
                                              &fex_ctx->opts_dict, 0);
                vmaf_dictionary_free(&rfe->fex_ctx[i]->opts_dict);
                VmafFeatureExtractorContext *f = rfe->fex_ctx[i];
                f->opts_dict = d;
                if (f->fex->options && f->fex->priv) {
                    int err = parse_options(f);
                    if (err) return err;
                }
                return vmaf_feature_extractor_context_destroy(fex_ctx);
            }
        }
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
