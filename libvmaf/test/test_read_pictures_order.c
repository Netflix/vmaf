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

static int read_blank_pair(VmafContext *vmaf, unsigned index)
{
    VmafPicture ref, dist;
    int err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 16, 16);
    err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 16, 16);
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

static char *test_read_pictures_requires_increasing_index()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = vmaf_use_feature(vmaf, "motion", NULL);
    mu_assert("problem during vmaf_use_feature", !err);

    mu_assert("index 0 rejected", !read_blank_pair(vmaf, 0));
    mu_assert("index 1 rejected", !read_blank_pair(vmaf, 1));
    mu_assert("index 2 rejected", !read_blank_pair(vmaf, 2));
    // A repeated or earlier index that an extractor already scored also fails
    // in the feature collector, after the extractors ran. The check in
    // vmaf_read_pictures() rejects them before any state changes, and it
    // rejects an earlier index that was never submitted (3 after 4) too.
    mu_assert("repeated index 2 accepted",
              read_blank_pair(vmaf, 2) == -EINVAL);
    mu_assert("earlier index 1 accepted",
              read_blank_pair(vmaf, 1) == -EINVAL);
    mu_assert("index 4 rejected", !read_blank_pair(vmaf, 4));
    mu_assert("index 3 accepted after index 4",
              read_blank_pair(vmaf, 3) == -EINVAL);
    mu_assert("index 5 rejected after a rejected index", !read_blank_pair(vmaf, 5));

    // Flushing passes NULL pictures and an index that is not compared.
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("flush rejected after pictures with higher indices", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    // The order is tracked per context.
    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    mu_assert("index 0 rejected by a new context", !read_blank_pair(vmaf, 0));
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem flushing context", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_read_pictures_requires_increasing_index);
    return NULL;
}
