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
#include <stdint.h>
#include <string.h>

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

static char *test_picture_color_metadata_defaults()
{
    int err;

    VmafPicture pic;
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("color.range should default to unknown",
              pic.color.range == VMAF_COLOR_RANGE_UNKNOWN);
    mu_assert("color.primaries should default to unknown",
              pic.color.primaries == VMAF_COLOR_PRIMARIES_UNKNOWN);
    mu_assert("color.trc should default to unknown",
              pic.color.trc == VMAF_COLOR_TRC_UNKNOWN);
    mu_assert("color.matrix should default to unknown",
              pic.color.matrix == VMAF_COLOR_MATRIX_UNKNOWN);

    pic.color.range = VMAF_COLOR_RANGE_FULL;
    pic.color.primaries = VMAF_COLOR_PRIMARIES_BT2020;
    pic.color.trc = VMAF_COLOR_TRC_SMPTE2084;
    pic.color.matrix = VMAF_COLOR_MATRIX_ICTCP;

    VmafPicture pic_ref;
    err = vmaf_picture_ref(&pic_ref, &pic);
    mu_assert("problem during vmaf_picture_ref", !err);
    mu_assert("color.range should be preserved by vmaf_picture_ref",
              pic_ref.color.range == VMAF_COLOR_RANGE_FULL);
    mu_assert("color.primaries should be preserved by vmaf_picture_ref",
              pic_ref.color.primaries == VMAF_COLOR_PRIMARIES_BT2020);
    mu_assert("color.trc should be preserved by vmaf_picture_ref",
              pic_ref.color.trc == VMAF_COLOR_TRC_SMPTE2084);
    mu_assert("color.matrix should be preserved by vmaf_picture_ref",
              pic_ref.color.matrix == VMAF_COLOR_MATRIX_ICTCP);

    err = vmaf_picture_unref(&pic);
    mu_assert("problem during vmaf_picture_unref", !err);
    err = vmaf_picture_unref(&pic_ref);
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

#ifndef HAVE_ZIMG
static char *test_picture_convert_without_zimg()
{
    VmafPicture src, dst;
    int err = vmaf_picture_alloc(&src, VMAF_PIX_FMT_YUV420P, 8, 16, 16);
    mu_assert("problem during vmaf_picture_alloc", !err);
    memset(&dst, 0, sizeof(dst));

    VmafPictureConvertTarget target = {
        .pix_fmt = VMAF_PIX_FMT_YUV444P,
        .bpc = 8,
    };
    VmafPictureConvertContext *ctx = NULL;
    err = vmaf_picture_convert_context_init(&ctx, &src, &target);
    mu_assert("init should be unsupported without zimg", err == -ENOTSUP);
    mu_assert("no context should be created without zimg", !ctx);

    err = vmaf_picture_convert(ctx, &dst, &src);
    mu_assert("convert should be unsupported without zimg", err == -ENOTSUP);
    mu_assert("dst should not be allocated without zimg", !dst.ref);

    err = vmaf_picture_convert_context_close(ctx);
    mu_assert("close should be unsupported without zimg", err == -ENOTSUP);

    err = vmaf_picture_unref(&src);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}
#endif

char *run_tests()
{
    mu_run_test(test_picture_alloc_ref_and_unref);
    mu_run_test(test_picture_color_metadata_defaults);
    mu_run_test(test_picture_data_alignment);
#ifndef HAVE_ZIMG
    mu_run_test(test_picture_convert_without_zimg);
#endif
    return NULL;
}
