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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

#define W 64
#define H 64
#define N_FRAMES 5

static char *read_file(const char *path)
{
    /* text mode: vmaf_write_output() writes in text mode, so CRLF folds */
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    char *buf = calloc((size_t) size + 1, 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    /* a text-mode read may return fewer bytes than ftell() reports */
    size_t n = fread(buf, 1, (size_t) size, f);
    if (ferror(f)) {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static char *test_rejected_picture_does_not_empty_pooled_output()
{
    const char *output_path = "test_read_pictures_output.json";
    VmafContext *vmaf;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    err = vmaf_use_feature(vmaf, "psnr", NULL);
    mu_assert("problem during vmaf_use_feature", !err);

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, W, H);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, W, H);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    VmafPicture rejected_ref, rejected_dist;
    err = vmaf_picture_alloc(&rejected_ref, VMAF_PIX_FMT_YUV420P, 8, 96, H);
    mu_assert("problem during vmaf_picture_alloc", !err);
    err = vmaf_picture_alloc(&rejected_dist, VMAF_PIX_FMT_YUV420P, 8, 96, H);
    mu_assert("problem during vmaf_picture_alloc", !err);
    err = vmaf_read_pictures(vmaf, &rejected_ref, &rejected_dist, N_FRAMES);
    mu_assert("mismatched picture size should be rejected", err == -EINVAL);
    mu_assert("rejected ref should be released",
              !rejected_ref.ref && !rejected_ref.priv);
    mu_assert("rejected dist should be released",
              !rejected_dist.ref && !rejected_dist.priv);

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem flushing pictures", !err);
    err = vmaf_write_output(vmaf, output_path, VMAF_OUTPUT_FORMAT_JSON);
    mu_assert("problem during vmaf_write_output", !err);
    vmaf_close(vmaf);

    char *output = read_file(output_path);
    remove(output_path);
    mu_assert("could not read JSON output", output);
    mu_assert("pooled psnr metrics should be populated after a rejected pair",
              strstr(output, "\"psnr_y\": {\n      \"min\":") != NULL);
    free(output);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_rejected_picture_does_not_empty_pooled_output);
    return NULL;
}
