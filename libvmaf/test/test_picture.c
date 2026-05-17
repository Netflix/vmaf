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
#include <stdint.h>

#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"
#include "ref.h"

// NOLINTNEXTLINE(readability-function-size): test scaffolding — explicitly walks every alloc / fill / ref / unref state to keep failures localised; splitting hides the assertion that fired.
static char *test_picture_alloc_ref_and_unref()
{
    int err;

    VmafPicture pic_a;
    VmafPicture pic_b;
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
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 10, 1920 + 1, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("picture data is not 32-byte alligned",
              !(((uintptr_t)pic.data[0]) % 32) && !(((uintptr_t)pic.data[1]) % 32) &&
                  !(((uintptr_t)pic.data[2]) % 32) && !(pic.stride[0] % 32) &&
                  !(pic.stride[1] % 32) && !(pic.stride[2] % 32));
    err = vmaf_picture_unref(&pic);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}

/*
 * Regression test for Research-0094: odd-height / odd-width YUV 4:2:0 inputs
 * must produce ceil(luma/2) chroma rows/columns, not floor.  Pre-fix,
 * picture_compute_geometry used plain right-shift (floor), which under-
 * allocated chroma planes by one row for any input with an odd luma dimension.
 * Consumers such as ciede::scale_chroma_planes would then walk one row past
 * the allocation, causing an ASan-detected heap OOB.
 *
 * Canonical reproducer: 577x323 YUV 4:2:0 (both dimensions odd).
 *   luma:   w=577, h=323
 *   chroma: w=ceil(577/2)=289, h=ceil(323/2)=162   (correct, post-fix)
 *           w=floor(577/2)=288, h=floor(323/2)=161  (wrong, pre-fix)
 */
static char *test_picture_odd_dim_chroma_ceiling()
{
    int err;
    VmafPicture pic;

    /* 4:2:0, both luma dims odd. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 577, 323);
    mu_assert("vmaf_picture_alloc failed for 577x323 YUV420", !err);
    mu_assert("chroma w must be ceil(577/2)=289 for odd-width 4:2:0", pic.w[1] == 289);
    mu_assert("chroma w[2] must equal w[1]", pic.w[2] == pic.w[1]);
    mu_assert("chroma h must be ceil(323/2)=162 for odd-height 4:2:0", pic.h[1] == 162);
    mu_assert("chroma h[2] must equal h[1]", pic.h[2] == pic.h[1]);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed", !err);

    /* 4:2:0, even dims — ceiling must equal floor (no change). */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 576, 324);
    mu_assert("vmaf_picture_alloc failed for 576x324 YUV420", !err);
    mu_assert("chroma w must be 288 for even-width 4:2:0", pic.w[1] == 288);
    mu_assert("chroma h must be 162 for even-height 4:2:0", pic.h[1] == 162);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed", !err);

    /* 4:2:2, odd width only — height must be full luma height. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV422P, 8, 577, 323);
    mu_assert("vmaf_picture_alloc failed for 577x323 YUV422", !err);
    mu_assert("chroma w must be ceil(577/2)=289 for odd-width 4:2:2", pic.w[1] == 289);
    mu_assert("chroma h must equal luma h for 4:2:2", pic.h[1] == 323);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed", !err);

    /* 4:4:4, odd dims — no subsampling, chroma == luma. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV444P, 8, 577, 323);
    mu_assert("vmaf_picture_alloc failed for 577x323 YUV444", !err);
    mu_assert("chroma w must equal luma w for 4:4:4", pic.w[1] == 577);
    mu_assert("chroma h must equal luma h for 4:4:4", pic.h[1] == 323);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed", !err);

    return NULL;
}

/*
 * Regression test for audit finding #10: integer overflow in
 * picture_compute_geometry when caller passes near-UINT_MAX width.
 * Before the fix, (w + DATA_ALIGN - 1u) wrapped to 0, producing a
 * zero-byte allocation that passed silently and caused OOB on any pixel
 * read.  After the fix, vmaf_picture_alloc rejects w > 32768 or h > 32768
 * with -EINVAL before any arithmetic.  CERT INT30-C.
 */
static char *test_picture_alloc_rejects_overflow_dimensions()
{
    int err;
    VmafPicture pic;

    /* w == 0 must be rejected. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 0, 1080);
    mu_assert("vmaf_picture_alloc must reject w=0 with -EINVAL", err == -EINVAL);

    /* h == 0 must be rejected. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 1920, 0);
    mu_assert("vmaf_picture_alloc must reject h=0 with -EINVAL", err == -EINVAL);

    /* w > 32768 must be rejected. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 32769, 1080);
    mu_assert("vmaf_picture_alloc must reject w=32769 with -EINVAL", err == -EINVAL);

    /* h > 32768 must be rejected. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 1920, 32769);
    mu_assert("vmaf_picture_alloc must reject h=32769 with -EINVAL", err == -EINVAL);

    /* Boundary: w==32768, h==32768 must succeed. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 8, 32768, 32768);
    mu_assert("vmaf_picture_alloc must accept w=32768, h=32768", !err);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed for 32768x32768", !err);

    return NULL;
}

/*
 * Smoke test for VMAF_PIX_FMT_YUV400P (luma-only / monochrome).
 *
 * picture.c sets chroma w/h to 0 and chroma data[] pointers to NULL for
 * YUV400P pictures.  This test pins that contract so regressions in
 * picture_compute_geometry or vmaf_picture_alloc are caught before any
 * consumer (e.g. a luma-only PSNR extractor) walks a NULL chroma pointer.
 */
static char *test_picture_alloc_yuv400p_luma_only(void)
{
    int err;
    VmafPicture pic;

    /* Standard HD luma-only picture. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV400P, 8, 1920, 1080);
    mu_assert("vmaf_picture_alloc failed for 1920x1080 YUV400P", !err);
    mu_assert("YUV400P luma data[0] must be non-NULL", pic.data[0] != NULL);
    mu_assert("YUV400P chroma data[1] must be NULL", pic.data[1] == NULL);
    mu_assert("YUV400P chroma data[2] must be NULL", pic.data[2] == NULL);
    mu_assert("YUV400P luma w[0] must equal requested width", pic.w[0] == 1920);
    mu_assert("YUV400P luma h[0] must equal requested height", pic.h[0] == 1080);
    mu_assert("YUV400P chroma w[1] must be zero", pic.w[1] == 0);
    mu_assert("YUV400P chroma w[2] must be zero", pic.w[2] == 0);
    mu_assert("YUV400P chroma h[1] must be zero", pic.h[1] == 0);
    mu_assert("YUV400P chroma h[2] must be zero", pic.h[2] == 0);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed for YUV400P", !err);

    /* Odd dimensions: geometry zeroing must not depend on even luma dims. */
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV400P, 8, 577, 323);
    mu_assert("vmaf_picture_alloc failed for odd-dim YUV400P", !err);
    mu_assert("YUV400P odd-dim luma data[0] must be non-NULL", pic.data[0] != NULL);
    mu_assert("YUV400P odd-dim chroma data[1] must be NULL", pic.data[1] == NULL);
    mu_assert("YUV400P odd-dim chroma data[2] must be NULL", pic.data[2] == NULL);
    mu_assert("YUV400P odd-dim chroma w[1] must be zero", pic.w[1] == 0);
    mu_assert("YUV400P odd-dim chroma h[1] must be zero", pic.h[1] == 0);
    err = vmaf_picture_unref(&pic);
    mu_assert("vmaf_picture_unref failed for odd-dim YUV400P", !err);

    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_picture_alloc_ref_and_unref);
    mu_run_test(test_picture_data_alignment);
    mu_run_test(test_picture_odd_dim_chroma_ceiling);
    mu_run_test(test_picture_alloc_rejects_overflow_dimensions);
    mu_run_test(test_picture_alloc_yuv400p_luma_only);
    return NULL;
}
