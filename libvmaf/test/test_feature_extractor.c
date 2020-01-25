#include <stdint.h>

#include "feature/feature_extractor.h"
#include "feature/common/cpu.h"
#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"

enum vmaf_cpu cpu;
// ^ FIXME, this is a global in the old libvmaf
// A few wrapped floating point feature extractors rely on it being a global
// After we clean those up, We'll add this to the VmafContext

static char *test_get_feature_extractor_by_name_and_feature_name()
{
    cpu = cpu_autodetect(); //FIXME, see above

    VmafFeatureExtractor *fex;
    fex = vmaf_get_feature_extractor_by_name("");
    mu_assert("problem during vmaf_get_feature_extractor_by_name", !fex);
    fex = vmaf_get_feature_extractor_by_name("float_vif");
    mu_assert("problem vmaf_get_feature_extractor_by_name",
              !strcmp(fex->name, "float_vif"));

    fex =
        vmaf_get_feature_extractor_by_feature_name("'VMAF_feature_adm2_score'");
    mu_assert("problem during vmaf_get_feature_extractor_by_feature_name",
              fex && !strcmp(fex->name, "float_adm"));

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_get_feature_extractor_by_name_and_feature_name);
    return NULL;
}
