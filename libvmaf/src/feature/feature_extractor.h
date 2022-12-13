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

#ifndef __VMAF_FEATURE_EXTRACTOR_H__
#define __VMAF_FEATURE_EXTRACTOR_H__

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#include "dict.h"
#include "feature_collector.h"
#include "opt.h"

#include "libvmaf/picture.h"

#ifdef HAVE_CUDA
#include "cuda/common.h"
#endif

enum VmafFeatureExtractorFlags {
    VMAF_FEATURE_EXTRACTOR_TEMPORAL = 1 << 0,
    VMAF_FEATURE_EXTRACTOR_CUDA = 1 << 1,
};

typedef struct VmafFeatureExtractor {
    const char *name; ///< Name of feature extractor.
    /**
     * Initialization callback. Optional, preallocate fex->priv buffers here.
     *
     * @param     fex self.
     * @param pix_fmt VmafPixelFormat of all subsequent pictures.
     * @param     bpc Bitdepth of all subsequent pictures.
     * @param       w Width of all subsequent pictures.
     * @param       h Height of all subsequent pictures.
     */
    int (*init)(struct VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h);
    /**
     * Feature extraction callback. Called for every pair of pictures. Unless
     * the VMAF_FEATURE_EXTRACTOR_TEMPORAL flag is set, there is no guarantee
     * that this callback is called in any specific order.
     *
     *
     * @param               fex self.
     * @param           ref_pic Reference VmafPicture.
     * @param        ref_pic_90 Reference VmafPicture, translated 90 degrees.
     * @param          dist_pic Distorted VmafPicture.
     * @param       dist_pic_90 Distorted VmafPicture, translated 90 degrees.
     * @param             index Picture index.
     * @param feature_collector VmafFeatureCollector used to write out scores.
     */
    int (*extract)(struct VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector);
    /**
     * Buffer flush callback. Optional.
     * Called only when the VMAF_FEATURE_EXTRACTOR_TEMPORAL flag is set.
     *
     * @param               fex self.
     * @param feature_collector VmafFeatureCollector used to write out scores.
     */
    int (*flush)(struct VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector);
    /**
     * Close callback. Optional, clean up fex->priv buffers here.
     *
     * @param               fex self.
     */
    int (*close)(struct VmafFeatureExtractor *fex);
    const VmafOption *options; ///< Optional initialization options.
    void *priv; ///< Custom data.
    size_t priv_size; ///< sizeof private data.
    uint64_t flags; ///< Feauture extraction flags, binary or'd.
    const char **provided_features; ///< Provided feature list, NULL terminated.

    #ifdef HAVE_CUDA
    VmafCudaState *cu_state; ///< VmafCudaState, set by framework
    #endif

} VmafFeatureExtractor;

VmafFeatureExtractor *vmaf_get_feature_extractor_by_name(const char *name);
VmafFeatureExtractor *vmaf_get_feature_extractor_by_feature_name(const char *name,
                                                                 unsigned flags);

enum VmafFeatureExtractorContextFlags {
    VMAF_FEATURE_EXTRACTOR_CONTEXT_DO_NOT_OVERWRITE = 1 << 0,
};

typedef struct VmafFeatureExtractorContext {
    bool is_initialized, is_closed;
    VmafDictionary *opts_dict;
    VmafFeatureExtractor *fex;
} VmafFeatureExtractorContext;

int vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **fex_ctx,
                                          VmafFeatureExtractor *fex,
                                          VmafDictionary *opts_dict);

int vmaf_feature_extractor_context_init(VmafFeatureExtractorContext *fex_ctx,
                                        enum VmafPixelFormat pix_fmt,
                                        unsigned bpc, unsigned w, unsigned h);

int vmaf_feature_extractor_context_extract(VmafFeatureExtractorContext *fex_ctx,
                                           VmafPicture *ref, VmafPicture *ref_90,
                                           VmafPicture *dist, VmafPicture *dist_90,
                                           unsigned pic_index,
                                           VmafFeatureCollector *vfc);

int vmaf_feature_extractor_context_flush(VmafFeatureExtractorContext *fex_ctx,
                                         VmafFeatureCollector *vfc);

int vmaf_feature_extractor_context_close(VmafFeatureExtractorContext *fex_ctx);

int vmaf_feature_extractor_context_delete(VmafFeatureExtractorContext *fex_ctx);

int vmaf_feature_extractor_context_destroy(VmafFeatureExtractorContext *fex_ctx);

typedef struct VmafFeatureExtractorContextPool {
    struct fex_list_entry {
        VmafFeatureExtractor *fex;
        VmafDictionary *opts_dict;
        struct {
            VmafFeatureExtractorContext *fex_ctx;
            bool in_use;
        } *ctx_list;
        atomic_int capacity, in_use;
        pthread_cond_t full;
    } *fex_list;
    unsigned cnt, capacity;
    pthread_mutex_t lock;
    unsigned n_threads;
} VmafFeatureExtractorContextPool;

int vmaf_fex_ctx_pool_create(VmafFeatureExtractorContextPool **pool,
                             unsigned n_threads);

int vmaf_fex_ctx_pool_aquire(VmafFeatureExtractorContextPool *pool,
                             VmafFeatureExtractor *fex,
                             VmafDictionary *opts_dict,
                             VmafFeatureExtractorContext **fex_ctx);

int vmaf_fex_ctx_pool_release(VmafFeatureExtractorContextPool *pool,
                             VmafFeatureExtractorContext *fex_ctx);

int vmaf_fex_ctx_pool_flush(VmafFeatureExtractorContextPool *pool,
                            VmafFeatureCollector *feature_collector);

int vmaf_fex_ctx_pool_destroy(VmafFeatureExtractorContextPool *pool);

#endif /* __VMAF_FEATURE_EXTRACTOR_H__ */
