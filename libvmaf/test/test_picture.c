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

#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"
#include "ref.h"

static char *test_picture_alloc_ref_and_unref()
{
    int err;

    VmafPicture pic_a, pic_b;
    err = vmaf_picture_alloc(&pic_a, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("pic_a.ref->cnt should be 1", vmaf_ref_load(pic_a.ref) == 1);
    err = vmaf_picture_ref(&pic_b, &pic_a);
    mu_assert("problem during vmaf_picture_ref", !err);
    mu_assert("pic_a.ref->cnt should be 2", vmaf_ref_load(pic_a.ref) == 2);
    mu_assert("pic_b.ref->cnt should be 2", vmaf_ref_load(pic_b.ref) == 2);
    err = vmaf_picture_unref(&pic_a);
    mu_assert("problem during vmaf_picture_unref", !err);
    mu_assert("pic_b.ref->cnt should be 1", vmaf_ref_load(pic_b.ref) == 1);
    err = vmaf_picture_unref(&pic_b);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}

static char *test_picture_data_alignment()
{
    int err;

    VmafPicture pic;
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 10, 1920+1, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("picture data is not 32-byte alligned",
        !(((uintptr_t) pic.data[0]) % 32) &&
        !(((uintptr_t) pic.data[1]) % 32) &&
        !(((uintptr_t) pic.data[2]) % 32) &&
        !(pic.stride[0] % 32) &&
        !(pic.stride[1] % 32) &&
        !(pic.stride[2] % 32)
    );
    err = vmaf_picture_unref(&pic);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_picture_alloc_ref_and_unref);
    mu_run_test(test_picture_data_alignment);
    return NULL;
}
