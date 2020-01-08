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
    int (*init)(struct VmafFeatureExtractor *fex);
    int (*extract)(struct VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector);
    int (*close)(struct VmafFeatureExtractor *fex);
    void *priv;
    size_t priv_size;
    uint64_t flags;
    const char *feature[];
} VmafFeatureExtractor;

VmafFeatureExtractor *get_feature_extractor_by_name(char *name);
VmafFeatureExtractor *get_feature_extractor_by_feature_name(char *feature_name);

#endif /* __VMAF_FEATURE_EXTRACTOR_H__ */
