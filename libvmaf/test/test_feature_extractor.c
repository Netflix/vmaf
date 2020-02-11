#include <stdint.h>
#include <string.h>

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

static char *test_get_feature_extractor_context_pool()
{
    int err = 0;

    const unsigned n_threads = 8;
    VmafFeatureExtractorContextPool *pool;
    err = vmaf_fex_ctx_pool_create(&pool, n_threads);
    mu_assert("problem during vmaf_fex_ctx_pool_create", !err);

    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("ssim");
    mu_assert("problem during vmaf_get_feature_extractor_by_name", fex);

    VmafFeatureExtractorContext *fex_ctx[n_threads];
    for (unsigned i = 0; i < n_threads; i++) {
        err = vmaf_fex_ctx_pool_aquire(pool, fex, &fex_ctx[i]);
        mu_assert("problem during vmaf_fex_ctx_pool_aquire", !err);
        mu_assert("fex_ctx[i] should be ssim feature extractor",
                  !strcmp(fex_ctx[i]->fex->name, "ssim"));
    }

    for (unsigned i = 0; i < n_threads; i++) {
        err = vmaf_fex_ctx_pool_release(pool, fex_ctx[i]);
        mu_assert("problem during vmaf_fex_ctx_pool_release", !err);
    }

    err = vmaf_fex_ctx_pool_destroy(pool);
    mu_assert("problem during vmaf_fex_ctx_pool_destroy", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_get_feature_extractor_by_name_and_feature_name);
    mu_run_test(test_get_feature_extractor_context_pool);
    return NULL;
}
