#include <stdint.h>

#include "feature/feature_extractor.h"
#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"

static char *test_get_feature_extractor_by_name_and_feature_name()
{
    VmafFeatureExtractor *fex = NULL;
    fex = vmaf_get_feature_extractor_by_name("");
    mu_assert("problem during vmaf_picture_unref", !fex);
    fex = vmaf_get_feature_extractor_by_feature_name("");
    mu_assert("problem during vmaf_picture_unref", !fex);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_get_feature_extractor_by_name_and_feature_name);
    return NULL;
}
