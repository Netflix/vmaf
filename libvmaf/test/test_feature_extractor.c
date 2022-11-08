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

#include <stdint.h>
#include <string.h>

#include "config.h"
#include "dict.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"

static char *test_get_feature_extractor_by_name_and_feature_name()
{
    VmafFeatureExtractor *fex;
    fex = vmaf_get_feature_extractor_by_name("");
    mu_assert("problem during vmaf_get_feature_extractor_by_name", !fex);
    fex = vmaf_get_feature_extractor_by_name("vif");
    mu_assert("problem vmaf_get_feature_extractor_by_name",
              !strcmp(fex->name, "vif"));

    fex = vmaf_get_feature_extractor_by_feature_name(
            "VMAF_integer_feature_adm2_score", 0);
    mu_assert("problem during vmaf_get_feature_extractor_by_feature_name",
              fex && !strcmp(fex->name, "adm"));

#if HAVE_CUDA
    unsigned flags = VMAF_FEATURE_EXTRACTOR_CUDA;
    fex = vmaf_get_feature_extractor_by_feature_name(
            "VMAF_integer_feature_adm2_score", flags);
    mu_assert("problem during vmaf_get_feature_extractor_by_feature_name",
              fex && !strcmp(fex->name, "adm_cuda"));
#endif

    return NULL;
}

static char *test_feature_extractor_context_pool()
{
    int err = 0;

    const unsigned n_threads = 8;
    VmafFeatureExtractorContextPool *pool;
    err = vmaf_fex_ctx_pool_create(&pool, n_threads);
    mu_assert("problem during vmaf_fex_ctx_pool_create", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("float_ssim");
    mu_assert("problem during vmaf_get_feature_extractor_by_name", fex);

    VmafFeatureExtractorContext *fex_ctx[n_threads];
    for (unsigned i = 0; i < n_threads; i++) {
        err = vmaf_fex_ctx_pool_aquire(pool, fex, NULL, &fex_ctx[i]);
        mu_assert("problem during vmaf_fex_ctx_pool_aquire", !err);
        mu_assert("fex_ctx[i] should be float_ssim feature extractor",
                  !strcmp(fex_ctx[i]->fex->name, "float_ssim"));
    }

    for (unsigned i = 0; i < n_threads; i++) {
        err = vmaf_fex_ctx_pool_release(pool, fex_ctx[i]);
        mu_assert("problem during vmaf_fex_ctx_pool_release", !err);
    }

    err = vmaf_fex_ctx_pool_destroy(pool);
    mu_assert("problem during vmaf_fex_ctx_pool_destroy", !err);

    return NULL;
}

static char *test_feature_extractor_flush()
{
    int err = 0;

    VmafFeatureExtractor *fex;
    fex = vmaf_get_feature_extractor_by_name("motion");
    mu_assert("problem vmaf_get_feature_extractor_by_name",
              !strcmp(fex->name, "motion"));
    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("problem during vmaf_feature_extractor_context_create", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);

    VmafFeatureCollector *vfc;
    err = vmaf_feature_collector_init(&vfc);
    mu_assert("vmaf_feature_collector_init", !err);

    double score;
    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL, &dist,
                                                 NULL, 0, vfc);
    mu_assert("problem during vmaf_feature_extractor_context_extract", !err);
    err = vmaf_feature_collector_get_score(vfc, "VMAF_integer_feature_motion2_score",
                                           &score, 0);
    mu_assert("problem during vmaf_feature_collector_get_score", !err);
    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL, &dist,
                                                 NULL, 1, vfc);
    mu_assert("problem during vmaf_feature_extractor_context_extract", !err);
    err = vmaf_feature_collector_get_score(vfc, "VMAF_integer_feature_motion2_score",
                                           &score, 0);
    mu_assert("problem during vmaf_feature_extractor_context_flush", !err);
    err = vmaf_feature_extractor_context_flush(fex_ctx, vfc);
    mu_assert("problem during vmaf_feature_extractor_context_flush", !err);
    err = vmaf_feature_collector_get_score(vfc, "VMAF_integer_feature_motion2_score",
                                           &score, 1);
    mu_assert("problem during vmaf_feature_collector_get_score", !err);

    err = vmaf_feature_extractor_context_close(fex_ctx);
    mu_assert("problem during vmaf_feature_extractor_context_close", !err);
    err = vmaf_feature_extractor_context_destroy(fex_ctx);
    mu_assert("problem during vmaf_feature_extractor_context_destroy", !err);

    vmaf_feature_collector_destroy(vfc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

static char *test_feature_extractor_initialization_options()
{
    int err = 0;

    VmafFeatureExtractor *fex;
    fex = vmaf_get_feature_extractor_by_name("psnr");
    mu_assert("problem vmaf_get_feature_extractor_by_name",
              !strcmp(fex->name, "psnr"));

    VmafDictionary *opts_dict = NULL;
    err = vmaf_dictionary_set(&opts_dict, "enable_chroma", "false", 0);
    mu_assert("problem during vmaf_dictionary_set", !err);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, opts_dict);
    mu_assert("problem during vmaf_feature_extractor_context_create", !err);

    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);

    VmafFeatureCollector *vfc;
    err = vmaf_feature_collector_init(&vfc);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL, &dist,
                                                 NULL, 0, vfc);
    mu_assert("problem during vmaf_feature_extractor_context_extract", !err);

    double score;
    err = vmaf_feature_collector_get_score(vfc, "psnr_cb", &score, 0);
    mu_assert("chroma PSNR was not disabled via option", err);

    err = vmaf_feature_extractor_context_close(fex_ctx);
    mu_assert("problem during vmaf_feature_extractor_context_close", !err);
    err = vmaf_feature_extractor_context_destroy(fex_ctx);
    mu_assert("problem during vmaf_feature_extractor_context_destroy", !err);

    vmaf_feature_collector_destroy(vfc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_get_feature_extractor_by_name_and_feature_name);
    mu_run_test(test_feature_extractor_context_pool);
    mu_run_test(test_feature_extractor_flush);
    mu_run_test(test_feature_extractor_initialization_options);
    return NULL;
}
