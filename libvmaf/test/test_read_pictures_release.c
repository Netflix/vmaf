/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "ref.h"

/*
 * vmaf_read_pictures() takes ownership of both pictures on every return once
 * it has a context and two pictures. These tests submit a pair whose shape
 * differs from the stream's, which the call rejects with -EINVAL.
 *
 * test_pic_pool_pair_released() cannot finish on a library that keeps the
 * rejected pair: vmaf_close() waits for every pool picture. The meson test()
 * entry carries a timeout so such a regression fails the test instead of
 * hanging the suite.
 */

static VmafContext *open_context(void)
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    if (vmaf_init(&vmaf, cfg)) return NULL;
    if (vmaf_use_feature(vmaf, "psnr", NULL)) {
        vmaf_close(vmaf);
        return NULL;
    }
    return vmaf;
}

static int alloc_pair(VmafPicture *ref, VmafPicture *dist, unsigned w,
                      unsigned h)
{
    int err = vmaf_picture_alloc(ref, VMAF_PIX_FMT_YUV420P, 8, w, h);
    err |= vmaf_picture_alloc(dist, VMAF_PIX_FMT_YUV420P, 8, w, h);
    return err;
}

// the first pair fixes the stream's shape at 32x32
static int read_first_pair(VmafContext *vmaf)
{
    VmafPicture ref, dist;
    int err = alloc_pair(&ref, &dist, 32, 32);
    if (err) return err;
    return vmaf_read_pictures(vmaf, &ref, &dist, 0);
}

static char *test_rejected_pair_released()
{
    VmafContext *vmaf = open_context();
    mu_assert("problem during open_context", vmaf);
    mu_assert("first pair rejected", !read_first_pair(vmaf));

    VmafPicture ref, dist, keep_ref, keep_dist;
    mu_assert("problem during alloc_pair", !alloc_pair(&ref, &dist, 64, 48));
    // a second reference of the test's own, as vmaf_picture_ref() makes
    keep_ref = ref;
    keep_dist = dist;
    vmaf_ref_fetch_increment(keep_ref.ref);
    vmaf_ref_fetch_increment(keep_dist.ref);
    mu_assert("two references expected", vmaf_ref_load(keep_ref.ref) == 2);

    int err = vmaf_read_pictures(vmaf, &ref, &dist, 1);
    mu_assert("pair of another shape accepted", err == -EINVAL);
    mu_assert("rejected ref still referenced by the context",
              vmaf_ref_load(keep_ref.ref) == 1);
    mu_assert("rejected dist still referenced by the context",
              vmaf_ref_load(keep_dist.ref) == 1);

    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&keep_ref));
    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&keep_dist));
    mu_assert("problem during vmaf_read_pictures flush",
              !vmaf_read_pictures(vmaf, NULL, NULL, 0));
    mu_assert("problem during vmaf_close", !vmaf_close(vmaf));
    return NULL;
}

static char *test_unref_after_error_is_harmless()
{
    VmafContext *vmaf = open_context();
    mu_assert("problem during open_context", vmaf);
    mu_assert("first pair rejected", !read_first_pair(vmaf));

    VmafPicture ref, dist;
    mu_assert("problem during alloc_pair", !alloc_pair(&ref, &dist, 64, 48));
    mu_assert("pair of another shape accepted",
              vmaf_read_pictures(vmaf, &ref, &dist, 1) == -EINVAL);

    // the context cleared the caller's structs: a second unref has nothing
    // to release
    mu_assert("ref not cleared", !ref.ref && !ref.priv);
    mu_assert("dist not cleared", !dist.ref && !dist.priv);
    mu_assert("unref after the error must fail", vmaf_picture_unref(&ref) == -EINVAL);
    mu_assert("unref after the error must fail", vmaf_picture_unref(&dist) == -EINVAL);

    mu_assert("problem during vmaf_read_pictures flush",
              !vmaf_read_pictures(vmaf, NULL, NULL, 0));
    mu_assert("problem during vmaf_close", !vmaf_close(vmaf));
    return NULL;
}

static char *test_calls_that_take_nothing()
{
    VmafContext *vmaf = open_context();
    mu_assert("problem during open_context", vmaf);
    mu_assert("first pair rejected", !read_first_pair(vmaf));

    VmafPicture ref, dist;
    mu_assert("problem during alloc_pair", !alloc_pair(&ref, &dist, 32, 32));

    // one picture only
    mu_assert("one picture accepted",
              vmaf_read_pictures(vmaf, &ref, NULL, 1) == -EINVAL);
    mu_assert("one picture accepted",
              vmaf_read_pictures(vmaf, NULL, &dist, 1) == -EINVAL);
    mu_assert("ref taken", vmaf_ref_load(ref.ref) == 1);
    mu_assert("dist taken", vmaf_ref_load(dist.ref) == 1);

    // no context
    mu_assert("no context accepted",
              vmaf_read_pictures(NULL, &ref, &dist, 1) == -EINVAL);
    mu_assert("ref taken", vmaf_ref_load(ref.ref) == 1);
    mu_assert("dist taken", vmaf_ref_load(dist.ref) == 1);

    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&ref));
    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&dist));

    // flush
    mu_assert("problem during vmaf_read_pictures flush",
              !vmaf_read_pictures(vmaf, NULL, NULL, 0));
    mu_assert("second flush accepted",
              vmaf_read_pictures(vmaf, NULL, NULL, 0) == -EINVAL);
    mu_assert("problem during vmaf_close", !vmaf_close(vmaf));
    return NULL;
}

static char *test_pic_pool_pair_released()
{
    VmafContext *vmaf = open_context();
    mu_assert("problem during open_context", vmaf);
    mu_assert("first pair rejected", !read_first_pair(vmaf));

    VmafPictureConfiguration pic_cfg = {
        .pic_params = { .w = 64, .h = 48, .bpc = 8,
                        .pix_fmt = VMAF_PIX_FMT_YUV420P },
        .pic_cnt = 2,
    };
    mu_assert("problem during vmaf_preallocate_pictures",
              !vmaf_preallocate_pictures(vmaf, pic_cfg));

    VmafPicture ref, dist;
    mu_assert("problem during vmaf_fetch_preallocated_picture",
              !vmaf_fetch_preallocated_picture(vmaf, &ref));
    mu_assert("problem during vmaf_fetch_preallocated_picture",
              !vmaf_fetch_preallocated_picture(vmaf, &dist));

    mu_assert("pair of another shape accepted",
              vmaf_read_pictures(vmaf, &ref, &dist, 1) == -EINVAL);

    // both pool pictures are back: the pool hands them out again
    mu_assert("problem during vmaf_fetch_preallocated_picture",
              !vmaf_fetch_preallocated_picture(vmaf, &ref));
    mu_assert("problem during vmaf_fetch_preallocated_picture",
              !vmaf_fetch_preallocated_picture(vmaf, &dist));
    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&ref));
    mu_assert("problem during vmaf_picture_unref", !vmaf_picture_unref(&dist));

    mu_assert("problem during vmaf_read_pictures flush",
              !vmaf_read_pictures(vmaf, NULL, NULL, 0));
    mu_assert("problem during vmaf_close", !vmaf_close(vmaf));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_rejected_pair_released);
    mu_run_test(test_unref_after_error_is_harmless);
    mu_run_test(test_calls_that_take_nothing);
    mu_run_test(test_pic_pool_pair_released);
    return NULL;
}
