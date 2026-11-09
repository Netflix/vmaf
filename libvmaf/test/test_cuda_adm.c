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

#define ADM_VIEW_DIST_COUNT 2
#define ADM_FEATURE_COUNT 7

/* Names the scores are registered under. A default option value is not
 * rendered into a feature name, and the name is only aliased when at least
 * one non-default option is present: the primary instance (viewing distance
 * 3.0, the default) keeps the provided feature name, while the secondary
 * instance (5.0) renders the aliased name plus the nvd suffix. */
static const char *adm_feature_names[ADM_VIEW_DIST_COUNT][ADM_FEATURE_COUNT] = {
    {
        "VMAF_integer_feature_adm2_score",
        "VMAF_integer_feature_aim_score",
        "VMAF_integer_feature_adm3_score",
        "integer_adm_scale0",
        "integer_adm_scale1",
        "integer_adm_scale2",
        "integer_adm_scale3",
    },
    {
        "integer_adm2_nvd_5",
        "integer_aim_nvd_5",
        "integer_adm3_nvd_5",
        "integer_adm_scale0_nvd_5",
        "integer_adm_scale1_nvd_5",
        "integer_adm_scale2_nvd_5",
        "integer_adm_scale3_nvd_5",
    },
};

static int run_adm_two_view_dist(int cuda, unsigned n_frames,
                                 double scores[ADM_VIEW_DIST_COUNT]
                                               [ADM_FEATURE_COUNT][TEST_FRAMES])
{
    static const char *nvd[ADM_VIEW_DIST_COUNT] = { "3.0", "5.0" };
    VmafConfiguration cfg = { 0 };
    VmafContext *vmaf = NULL;
    VmafCudaState *cu_state = NULL;
    int err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    if (cuda) {
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            vmaf_close(vmaf);
            return err;
        }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) goto fail;
    }

    /* Two viewing distances in one context: the CPU extractor folds them into
     * a single instance (adm_norm_view_dist_extra), the CUDA extractor keeps
     * two instances. Both must expose the same scores. */
    for (unsigned v = 0; v < ADM_VIEW_DIST_COUNT; v++) {
        VmafFeatureDictionary *opts = NULL;
        err = vmaf_feature_dictionary_set(&opts, "adm_norm_view_dist",
                                          nvd[v]);
        if (!err)
            err = vmaf_use_feature(vmaf, cuda ? "adm_cuda" : "adm", opts);
        if (err) goto fail;
    }

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

    for (unsigned v = 0; v < ADM_VIEW_DIST_COUNT; v++) {
        for (unsigned f = 0; f < ADM_FEATURE_COUNT; f++) {
            for (unsigned i = 0; i < n_frames; i++) {
                err = vmaf_feature_score_at_index(vmaf,
                                                  adm_feature_names[v][f],
                                                  &scores[v][f][i], i);
                if (err) goto fail;
            }
        }
    }

    return vmaf_close(vmaf);

fail:
    vmaf_close(vmaf);
    return err;
}

static char *test_adm_multi_view_dist_cuda_matches_cpu(void)
{
    double cpu[ADM_VIEW_DIST_COUNT][ADM_FEATURE_COUNT][TEST_FRAMES];
    double gpu[ADM_VIEW_DIST_COUNT][ADM_FEATURE_COUNT][TEST_FRAMES];
    static char msg[160];

    int err = run_adm_two_view_dist(0, TEST_FRAMES, cpu);
    mu_assert("CPU multi-view-dist ADM failed", !err);
    err = run_adm_two_view_dist(1, TEST_FRAMES, gpu);
    mu_assert("CUDA multi-view-dist ADM failed", !err);

    for (unsigned v = 0; v < ADM_VIEW_DIST_COUNT; v++) {
        for (unsigned f = 0; f < ADM_FEATURE_COUNT; f++) {
            for (unsigned i = 0; i < TEST_FRAMES; i++) {
                if (cpu[v][f][i] != gpu[v][f][i]) {
                    snprintf(msg, sizeof(msg),
                             "%s (view dist %s) mismatch at frame %u: "
                             "cpu=%f cuda=%f",
                             adm_feature_names[v][f], v ? "5.0" : "3.0", i,
                             cpu[v][f][i], gpu[v][f][i]);
                    return msg;
                }
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

    mu_run_test(test_adm_multi_view_dist_cuda_matches_cpu);
    return NULL;
}
