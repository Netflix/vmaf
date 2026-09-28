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

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

static char *test_context_init_and_close()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static char *test_get_feature_score()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    err = vmaf_import_feature_score(vmaf, "feature_a", 100., 0);
    err |= vmaf_import_feature_score(vmaf, "feature_a", 200., 1);
    err |= vmaf_import_feature_score(vmaf, "feature_a", 300., 2);
    mu_assert("problem during vmaf_import_feature_score", !err);

    double score;
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 0);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 100.);
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 1);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 200.);
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 2);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 300.);

    err = vmaf_feature_score_pooled(vmaf, "feature_a", VMAF_POOL_METHOD_MEAN,
                                    &score, 0, 2);
    mu_assert("problem during vmaf_feature_score_pooled", !err);
    mu_assert("pooled feature score does not match expected value",
              score == 200.);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static int read_pair(VmafContext *vmaf, unsigned ref_bpc, unsigned dist_bpc,
                     unsigned index)
{
    VmafPicture ref, dist;
    int err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, ref_bpc, 16, 16);
    err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, dist_bpc, 16, 16);
    if (err) return -ENOMEM;

    err = vmaf_read_pictures(vmaf, &ref, &dist, index);
    if (err) {
        // Release whatever the call left with the caller. Who owns a rejected
        // pair is not what this test checks: the unrefs below free the pair if
        // the call kept it, and return -EINVAL without freeing anything if the
        // call already released it (it clears the struct).
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dist);
    }
    return err;
}

static char *test_read_pictures_rejects_bpc_mismatch()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    // First pair: the reference sets the context's bit depth, so only a
    // mismatch between ref and dist can be caught here.
    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = read_pair(vmaf, 10, 8, 0);
    mu_assert("10-bit ref with 8-bit dist accepted on the first pair",
              err == -EINVAL);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    // Later pairs must also match the bit depth the context started with.
    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = read_pair(vmaf, 8, 8, 0);
    mu_assert("matching 8-bit pair rejected", !err);
    err = read_pair(vmaf, 8, 10, 1);
    mu_assert("8-bit ref with 10-bit dist accepted on a later pair",
              err == -EINVAL);
    err = read_pair(vmaf, 10, 10, 1);
    mu_assert("10-bit pair accepted by a context that started at 8 bits",
              err == -EINVAL);
    err = read_pair(vmaf, 8, 8, 1);
    mu_assert("matching 8-bit pair rejected after a rejected pair", !err);
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem flushing context", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_context_init_and_close);
    mu_run_test(test_get_feature_score);
    mu_run_test(test_read_pictures_rejects_bpc_mismatch);
    return NULL;
}
