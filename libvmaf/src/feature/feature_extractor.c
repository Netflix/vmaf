#include <string.h>

#include "feature_extractor.h"

static VmafFeatureExtractor *feature_extractor_list[] = {
    NULL
};

VmafFeatureExtractor *get_feature_extractor_by_name(char *name)
{
    if (!name) return NULL;

    VmafFeatureExtractor *fex = NULL;
    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        if (!strcmp(name, fex->name))
           return fex;
    }
    return NULL;
}

VmafFeatureExtractor *get_feature_extractor_by_feature_name(char *feature_name)
{
    if (!feature_name) return NULL;

    VmafFeatureExtractor *fex = NULL;
    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        const char *fname = NULL;
        for (unsigned j = 0; (fname = fex->feature[j]); j++) {
            if (!strcmp(feature_name, fname))
                return fex;
        }
    }
    return NULL;
}
