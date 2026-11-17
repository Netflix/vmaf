/**
 *
 *  Copyright 2026 luoye-cpu
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "cuda/common.h"
#include "fex_ctx_vector.h"
#include "libvmaf/feature.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"

#define TEST_W 512
#define TEST_H 512
#define TEST_FRAMES 6

static int cuda_device_available(void)
{
    CudaFunctions *f = NULL;
    if (cuda_load_functions(&f, NULL) || !f)
        return 0;

    int n_devices = 0;
    const int ok = f->cuInit(0) == CUDA_SUCCESS &&
                   f->cuDeviceGetCount(&n_devices) == CUDA_SUCCESS &&
                   n_devices > 0;
    cuda_free_functions(&f);
    return ok;
}

static void fill_luma_8b(uint8_t *data, ptrdiff_t stride, unsigned w,
                         unsigned h, unsigned index)
{
    for (unsigned y = 0; y < h; y++) {
        uint8_t *row = data + y * stride;
        for (unsigned x = 0; x < w; x++) {
            const unsigned block = ((x >> 4) ^ (y >> 4)) * 37;
            row[x] = (uint8_t)((x * 7 + y * 13 + index * 29 + block) & 0xff);
        }
    }
}

static void fill_chroma_8b(uint8_t *data, ptrdiff_t stride, unsigned w,
                           unsigned h, unsigned index, unsigned plane)
{
    for (unsigned y = 0; y < h; y++) {
        uint8_t *row = data + y * stride;
        for (unsigned x = 0; x < w; x++) {
            row[x] = (uint8_t)((x * 3 + y * 5 + index * 17 + plane * 61) & 0xff);
        }
    }
}

static void fill_picture_8b(VmafPicture *pic, unsigned index)
{
    fill_luma_8b(pic->data[0], pic->stride[0], pic->w[0], pic->h[0], index);
    for (unsigned p = 1; p < 3; p++)
        fill_chroma_8b(pic->data[p], pic->stride[p], pic->w[p], pic->h[p],
                       index, p);
}

static int run_feature_cpu(const char *fex_name, const char *opt_key,
                           const char *opt_val, unsigned n_frames,
                           double *scores, const char *feature_name)
{
    VmafConfiguration cfg = { 0 };
    VmafContext *vmaf = NULL;
    int err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    VmafFeatureDictionary *opts = NULL;
    if (opt_key)
        err = vmaf_feature_dictionary_set(&opts, opt_key, opt_val);
    if (!err)
        err = vmaf_use_feature(vmaf, fex_name, opts);
    if (err) goto fail;

    for (unsigned i = 0; i < n_frames; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8,
                                 TEST_W, TEST_H);
        if (err) goto fail;
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8,
                                 TEST_W, TEST_H);
        if (err) {
            vmaf_picture_unref(&ref);
            goto fail;
        }
        fill_picture_8b(&ref, i);
        fill_picture_8b(&dist, i);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) goto fail;
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) goto fail;

    for (unsigned i = 0; i < n_frames; i++) {
        err = vmaf_feature_score_at_index(vmaf, feature_name, &scores[i], i);
        if (err) goto fail;
    }

    err = vmaf_close(vmaf);
    return err;

fail:
    vmaf_close(vmaf);
    return err;
}

static int run_feature_cuda(const char *fex_name, const char *opt_key,
                            const char *opt_val, unsigned n_frames,
                            double *scores, const char *feature_name)
{
    VmafConfiguration cfg = { 0 };
    VmafContext *vmaf = NULL;
    VmafCudaState *cu_state = NULL;
    int err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    VmafCudaConfiguration cuda_cfg = { 0 };
    err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    if (err) {
        vmaf_close(vmaf);
        return err;
    }
    err = vmaf_cuda_import_state(vmaf, cu_state);
    if (err) goto fail;

    VmafFeatureDictionary *opts = NULL;
    if (opt_key)
        err = vmaf_feature_dictionary_set(&opts, opt_key, opt_val);
    if (!err)
        err = vmaf_use_feature(vmaf, fex_name, opts);
    if (err) goto fail;

    for (unsigned i = 0; i < n_frames; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8,
                                 TEST_W, TEST_H);
        if (err) goto fail;
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8,
                                 TEST_W, TEST_H);
        if (err) {
            vmaf_picture_unref(&ref);
            goto fail;
        }
        fill_picture_8b(&ref, i);
        fill_picture_8b(&dist, i);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) goto fail;
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) goto fail;

    for (unsigned i = 0; i < n_frames; i++) {
        err = vmaf_feature_score_at_index(vmaf, feature_name, &scores[i], i);
        if (err) goto fail;
    }

    return vmaf_close(vmaf);

fail:
    vmaf_close(vmaf);
    return err;
}

static char *test_motion_default_cuda_matches_cpu(void)
{
    static const char *features[2] = {
        "VMAF_integer_feature_motion2_score",
        "VMAF_integer_feature_motion3_score",
    };
    double cpu[2][TEST_FRAMES], gpu[2][TEST_FRAMES];
    static char msg[128];

    for (unsigned f = 0; f < 2; f++) {
        int err = run_feature_cpu("motion", NULL, NULL, TEST_FRAMES,
                                  cpu[f], features[f]);
        mu_assert("CPU motion failed", !err);
    }

    for (unsigned f = 0; f < 2; f++) {
        int err = run_feature_cuda("motion_cuda", NULL, NULL, TEST_FRAMES,
                                   gpu[f], features[f]);
        mu_assert("CUDA motion failed", !err);
    }

    for (unsigned f = 0; f < 2; f++) {
        for (unsigned i = 0; i < TEST_FRAMES; i++) {
            if (cpu[f][i] != gpu[f][i]) {
                snprintf(msg, sizeof(msg),
                         "%s mismatch at frame %u: cpu=%f cuda=%f",
                         features[f], i, cpu[f][i], gpu[f][i]);
                return msg;
            }
        }
    }
    return NULL;
}

static char *test_motion_five_frame_cuda_matches_cpu(void)
{
    /* Options are part of the feature name: the five-frame window appends
     * the "mffw" option alias. */
    static const char *features[2] = {
        "integer_motion2_mffw",
        "integer_motion3_mffw",
    };
    double cpu[2][TEST_FRAMES], gpu[2][TEST_FRAMES];
    static char msg[128];

    for (unsigned f = 0; f < 2; f++) {
        int err = run_feature_cpu("motion", "motion_five_frame_window", "true",
                                  TEST_FRAMES, cpu[f], features[f]);
        mu_assert("CPU five-frame motion failed", !err);
    }

    for (unsigned f = 0; f < 2; f++) {
        int err = run_feature_cuda("motion_cuda", "motion_five_frame_window",
                                   "true", TEST_FRAMES, gpu[f], features[f]);
        mu_assert("CUDA five-frame motion failed", !err);
    }

    for (unsigned f = 0; f < 2; f++) {
        for (unsigned i = 0; i < TEST_FRAMES; i++) {
            if (cpu[f][i] != gpu[f][i]) {
                snprintf(msg, sizeof(msg),
                         "%s mismatch at frame %u: cpu=%f cuda=%f",
                         features[f], i, cpu[f][i], gpu[f][i]);
                return msg;
            }
        }
    }
    return NULL;
}


char *run_tests()
{
    /* Keep the suite usable on CPU-only builders: meson has no per-test skip
     * protocol for these binaries, so probe the driver once and report a
     * skip as success without registering any test. */
    if (!cuda_device_available()) {
        fprintf(stderr, "skipped: no CUDA device available\n");
        return NULL;
    }

    mu_run_test(test_motion_default_cuda_matches_cpu);
    mu_run_test(test_motion_five_frame_cuda_matches_cpu);
    return NULL;
}
