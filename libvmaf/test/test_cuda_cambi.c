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

static char *test_cambi_cuda_matches_cpu(void)
{
    double cpu[TEST_FRAMES], gpu[TEST_FRAMES];
    static char msg[128];

    int err = run_feature_cpu("cambi", NULL, NULL, TEST_FRAMES, cpu,
                              "Cambi_feature_cambi_score");
    mu_assert("CPU cambi failed", !err);

    err = run_feature_cuda("cambi_cuda", NULL, NULL, TEST_FRAMES, gpu,
                           "Cambi_feature_cambi_score");
    mu_assert("CUDA cambi failed", !err);

    for (unsigned i = 0; i < TEST_FRAMES; i++) {
        if (cpu[i] != gpu[i]) {
            snprintf(msg, sizeof(msg),
                     "cambi mismatch at frame %u: cpu=%f cuda=%f",
                     i, cpu[i], gpu[i]);
            return msg;
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

    mu_run_test(test_cambi_cuda_matches_cpu);
    return NULL;
}
