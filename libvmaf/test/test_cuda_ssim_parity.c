/**
 *
 *  Copyright 2026 Bardie Høgh Joensen
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

/* Compare CPU and CUDA SSIM, including options and odd dimensions.
 * A missing CUDA device is reported as a Meson SKIP (77). */

#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"
#include "cuda/common.h"

#define N_FRAMES 5

static const char *score_keys[] = {
    "float_ssim", "float_ssim_l", "float_ssim_c", "float_ssim_s",
};
#define N_KEYS (sizeof(score_keys) / sizeof(score_keys[0]))

typedef struct ParityCase {
    const char *name;
    enum VmafPixelFormat pix_fmt;
    unsigned bpc, w, h;
    unsigned index_step;
    int scale;
    bool enable_db, clip_db;
} ParityCase;

static uint32_t lcg_state;

static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 16;
}

static void fill_pictures(VmafPicture *ref, VmafPicture *dist, unsigned bpc,
                          unsigned index)
{
    /* Noisy pairs, inverted pixels, identical texture, then identical black. */
    const unsigned peak = (1 << bpc) - 1;
    lcg_state = 12345u + index * 7919u;

    for (unsigned p = 0; p < 3; p++) {
        if (bpc == 8) {
            uint8_t *r = ref->data[p];
            uint8_t *d = dist->data[p];
            for (unsigned i = 0; i < ref->h[p]; i++) {
                for (unsigned j = 0; j < ref->w[p]; j++) {
                    const int v = (i + j + lcg_next()) % (peak + 1);
                    const int noise = (int)(lcg_next() % 15) - 7;
                    int vd = v + noise;
                    if (vd < 0) vd = 0;
                    if (vd > (int)peak) vd = peak;
                    if (index == N_FRAMES - 3) vd = peak - v;
                    if (index == N_FRAMES - 2) vd = v;
                    r[j] = index == N_FRAMES - 1 ? 0 : v;
                    d[j] = index == N_FRAMES - 1 ? 0 : vd;
                }
                r += ref->stride[p];
                d += dist->stride[p];
            }
        } else {
            uint16_t *r = ref->data[p];
            uint16_t *d = dist->data[p];
            for (unsigned i = 0; i < ref->h[p]; i++) {
                for (unsigned j = 0; j < ref->w[p]; j++) {
                    const int v = (i + j + lcg_next()) % (peak + 1);
                    const int noise = (int)(lcg_next() % 61) - 30;
                    int vd = v + noise;
                    if (vd < 0) vd = 0;
                    if (vd > (int)peak) vd = peak;
                    if (index == N_FRAMES - 3) vd = peak - v;
                    if (index == N_FRAMES - 2) vd = v;
                    r[j] = index == N_FRAMES - 1 ? 0 : v;
                    d[j] = index == N_FRAMES - 1 ? 0 : vd;
                }
                r += ref->stride[p] / 2;
                d += dist->stride[p] / 2;
            }
        }
    }
}

/* Return 1 when CUDA is unavailable; report test failures through fail. */
static int run_pass(int use_cuda, const ParityCase *test_case,
                    double scores[N_FRAMES][N_KEYS],
                    char **fail)
{
    int err = 0;
    *fail = NULL;
    const unsigned step = test_case->index_step ? test_case->index_step : 1;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR,
        .n_threads = use_cuda ? 2 : 0, // threads + CUDA: the double-flush path
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) { *fail = "problem during vmaf_init"; return 0; }

    VmafCudaState *cu_state = NULL;
    if (use_cuda) {
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            vmaf_close(vmaf);
            if (cu_state) cuda_free_functions(&cu_state->f);
            free(cu_state);
            return 1; // no CUDA device, skip
        }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { *fail = "problem during vmaf_cuda_import_state"; return 0; }
    }

    VmafFeatureDictionary *ssim_dict = NULL;
    char scale[16];
    snprintf(scale, sizeof(scale), "%d", test_case->scale);
    err = vmaf_feature_dictionary_set(&ssim_dict, "enable_lcs", "true");
    err |= vmaf_feature_dictionary_set(&ssim_dict, "scale", scale);
    if (test_case->enable_db)
        err |= vmaf_feature_dictionary_set(&ssim_dict, "enable_db", "true");
    if (test_case->clip_db)
        err |= vmaf_feature_dictionary_set(&ssim_dict, "clip_db", "true");
    if (err) { *fail = "problem configuring ssim options"; return 0; }

    err = vmaf_use_feature(vmaf, use_cuda ? "ssim_cuda" : "float_ssim",
                           ssim_dict);
    if (err) { *fail = "problem during vmaf_use_feature ssim"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, test_case->pix_fmt, test_case->bpc,
                                 test_case->w, test_case->h);
        err |= vmaf_picture_alloc(&dist, test_case->pix_fmt, test_case->bpc,
                                  test_case->w, test_case->h);
        if (err) { *fail = "problem during vmaf_picture_alloc"; return 0; }
        fill_pictures(&ref, &dist, test_case->bpc, i);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i * step);
        if (err) { *fail = "problem during vmaf_read_pictures"; return 0; }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) { *fail = "problem during vmaf_read_pictures flush"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            err = vmaf_feature_score_at_index(vmaf, score_keys[k],
                                              &scores[i][k], i * step);
            if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
        }
    }

    err = vmaf_close(vmaf);
    if (cu_state) cuda_free_functions(&cu_state->f);
    free(cu_state);
    if (err) { *fail = "problem during vmaf_close"; return 0; }

    return 0;
}

static char *parity(const ParityCase *test_case)
{
    double cpu[N_FRAMES][N_KEYS], gpu[N_FRAMES][N_KEYS];
    char *fail = NULL;

    run_pass(0, test_case, cpu, &fail);
    if (fail) return fail;

    if (run_pass(1, test_case, gpu, &fail)) {
        fprintf(stderr, "no CUDA device available, skipping\n");
        exit(77);
    }
    if (fail) return fail;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            const bool infinite = i >= N_FRAMES - 2 && k == 0 &&
                                  test_case->enable_db && !test_case->clip_db;
            if (isnan(cpu[i][k]) || isnan(gpu[i][k]) ||
                (!infinite && (!isfinite(cpu[i][k]) || !isfinite(gpu[i][k]))) ||
                memcmp(&cpu[i][k], &gpu[i][k], sizeof(cpu[i][k]))) {
                fprintf(stderr, "mismatch %s, frame %u, %s: "
                        "cpu=%a gpu=%a\n", test_case->name, i,
                        score_keys[k], cpu[i][k], gpu[i][k]);
                return "cpu/cuda score mismatch";
            }
        }
    }

    for (unsigned k = 1; k < N_KEYS; k++)
        mu_assert("identical-picture SSIM component must equal one",
                  gpu[N_FRAMES - 1][k] == 1.0);
    if (!test_case->enable_db)
        mu_assert("identical-picture SSIM must equal one", gpu[N_FRAMES - 1][0] == 1.0);
    else if (!test_case->clip_db)
        mu_assert("identical-picture SSIM dB must be positive infinity",
                  isinf(gpu[N_FRAMES - 1][0]) && gpu[N_FRAMES - 1][0] > 0);

    return NULL;
}

static char *test_ssim_cuda_parity_420_8bpc_auto_scale(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 8 bpc 768x432 auto scale",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 8, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_ssim_cuda_parity_420_10bpc_auto_scale(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 10 bpc 768x432 auto scale",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 10, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_ssim_cuda_parity_444_12bpc_odd_scale_3(void)
{
    const ParityCase test_case = {
        .name = "yuv444p 12 bpc 65x49 scale 3",
        .pix_fmt = VMAF_PIX_FMT_YUV444P,
        .bpc = 12, .w = 65, .h = 49,
        .scale = 3,
    };
    return parity(&test_case);
}

static char *test_ssim_cuda_parity_422_16bpc_scale_1_options(void)
{
    const ParityCase test_case = {
        .name = "yuv422p 16 bpc 64x48 scale 1 with options",
        .pix_fmt = VMAF_PIX_FMT_YUV422P,
        .bpc = 16, .w = 64, .h = 48,
        .scale = 1,
        .enable_db = true,
        .clip_db = true,
    };
    return parity(&test_case);
}

static char *test_ssim_cuda_unclipped_db(void)
{
    const ParityCase test_case = {
        .name = "11x11 grayscale with unclipped dB",
        .pix_fmt = VMAF_PIX_FMT_YUV400P,
        .bpc = 8, .w = 11, .h = 11, .enable_db = true,
    };
    return parity(&test_case);
}

static char *test_ssim_cuda_scale_edges(void)
{
    const ParityCase cases[] = {
        { .name = "below auto-scale threshold", .pix_fmt = VMAF_PIX_FMT_YUV400P,
          .bpc = 8, .w = 512, .h = 383 },
        { .name = "at auto-scale threshold", .pix_fmt = VMAF_PIX_FMT_YUV400P,
          .bpc = 8, .w = 512, .h = 384 },
        { .name = "odd 10-bit without downscaling", .pix_fmt = VMAF_PIX_FMT_YUV444P,
          .bpc = 10, .w = 17, .h = 13, .scale = 1 },
        { .name = "maximum scale with skipped indices", .pix_fmt = VMAF_PIX_FMT_YUV444P,
          .bpc = 16, .w = 131, .h = 111, .scale = 10, .index_step = 2 },
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *fail = parity(&cases[i]);
        if (fail) return fail;
    }
    return NULL;
}

#include "test_cuda_score_error.h"

static char *test_ssim_cuda_score_error(void)
{
    return test_cuda_score_error("ssim_cuda", "float_ssim");
}

char *run_tests()
{
    mu_run_test(test_ssim_cuda_parity_420_8bpc_auto_scale);
    mu_run_test(test_ssim_cuda_parity_420_10bpc_auto_scale);
    mu_run_test(test_ssim_cuda_parity_444_12bpc_odd_scale_3);
    mu_run_test(test_ssim_cuda_parity_422_16bpc_scale_1_options);
    mu_run_test(test_ssim_cuda_unclipped_db);
    mu_run_test(test_ssim_cuda_scale_edges);
    mu_run_test(test_ssim_cuda_score_error);
    return NULL;
}
