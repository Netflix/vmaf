#ifndef __VMAF_FEATURE_EXTRACTOR_H__
#define __VMAF_FEATURE_EXTRACTOR_H__

#include <stdint.h>
#include <stdlib.h>

#include "feature_collector.h"

#include "libvmaf/picture.h"

enum VmafFeatureExtractorFlags {
    VMAF_FEATURE_EXTRACTOR_TEMPORAL = 1 << 0,
};

typedef struct VmafFeatureExtractor {
    const char *name;
    int (*init)(struct VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h);
    int (*extract)(struct VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector);
    int (*close)(struct VmafFeatureExtractor *fex);
    void *priv;
    size_t priv_size;
    uint64_t flags;
    const char *feature[];
} VmafFeatureExtractor;

VmafFeatureExtractor *vmaf_get_feature_extractor_by_name(char *name);
VmafFeatureExtractor *vmaf_get_feature_extractor_by_feature_name(char *name);

typedef struct VmafFeatureExtractorContext VmafFeatureExtractorContext;

int vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **fex_ctx,
                                          VmafFeatureExtractor *fex);

int vmaf_feature_extractor_context_init(VmafFeatureExtractorContext *fex_ctx,
                                        enum VmafPixelFormat pix_fmt,
                                        unsigned bpc, unsigned w, unsigned h);

int vmaf_feature_extractor_context_extract(VmafFeatureExtractorContext *fex_ctx,
                                           VmafPicture *ref, VmafPicture *dist,
                                           unsigned pic_index,
                                           VmafFeatureCollector *vfc);

int vmaf_feature_extractor_context_close(VmafFeatureExtractorContext *fex_ctx);

int vmaf_feature_extractor_context_delete(VmafFeatureExtractorContext *fex_ctx);

#endif /* __VMAF_FEATURE_EXTRACTOR_H__ */
