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

/*
 * CPU vs CUDA parity test for the psnr_cuda and ssim_cuda feature
 * extractors. Runs the same deterministic synthetic frames through the CPU
 * extractors (psnr, float_ssim) and the CUDA extractors (psnr_cuda,
 * ssim_cuda), then asserts bit-exact per-frame scores and matching serialized
 * APSNR aggregates.
 *
 * Exits with meson's SKIP code (77) when no CUDA device is available so CI
 * without a GPU reports the test as skipped.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"
#include "libvmaf/picture.h"

#define N_FRAMES 5

static const char *score_keys[] = {
    "psnr_y", "psnr_cb", "psnr_cr",
    "mse_y", "mse_cb", "mse_cr",
    "float_ssim", "float_ssim_l", "float_ssim_c", "float_ssim_s",
};
#define N_KEYS (sizeof(score_keys) / sizeof(score_keys[0]))

static const char *apsnr_keys[] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };
#define N_APSNR (sizeof(apsnr_keys) / sizeof(apsnr_keys[0]))

typedef struct ParityCase {
    const char *name;
    enum VmafPixelFormat pix_fmt;
    unsigned bpc, w, h;
    int scale;
    bool enable_db, clip_db;
    bool reduced_hbd_peak;
    double min_sse;
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
                    r[j] = v;
                    d[j] = vd;
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
                    r[j] = v;
                    d[j] = vd;
                }
                r += ref->stride[p] / 2;
                d += dist->stride[p] / 2;
            }
        }
    }
}

static int read_apsnr(VmafContext *vmaf, double scores[N_APSNR])
{
    char path[128];
    const int len = snprintf(path, sizeof(path),
                             "test_cuda_psnr_ssim_parity_%p.json",
                             (void *)vmaf);
    if (len < 0 || (size_t)len >= sizeof(path))
        return -1;

    if (vmaf_write_output(vmaf, path, VMAF_OUTPUT_FORMAT_JSON))
        return -1;

    FILE *file = fopen(path, "r");
    if (!file) {
        remove(path);
        return -1;
    }

    bool found[N_APSNR] = { false };
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        for (unsigned k = 0; k < N_APSNR; k++) {
            if (!strstr(line, apsnr_keys[k]))
                continue;

            char *value = strchr(line, ':');
            char *end;
            if (!value)
                continue;
            scores[k] = strtod(value + 1, &end);
            found[k] = end != value + 1;
        }
    }

    const int close_err = fclose(file);
    const int remove_err = remove(path);
    if (close_err || remove_err)
        return -1;

    for (unsigned k = 0; k < N_APSNR; k++) {
        if (!found[k])
            return -1;
    }

    return 0;
}

// returns 0 on success, 1 when CUDA is unavailable (caller should skip)
static int run_pass(int use_cuda, const ParityCase *test_case,
                    double scores[N_FRAMES][N_KEYS],
                    double apsnr[N_APSNR], char **fail)
{
    int err = 0;
    *fail = NULL;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_ERROR,
        .n_threads = use_cuda ? 2 : 0, // threads + CUDA: the double-flush path
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) { *fail = "problem during vmaf_init"; return 0; }

    if (use_cuda) {
        VmafCudaState *cu_state;
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            vmaf_close(vmaf);
            return 1; // no CUDA device, skip
        }
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) { *fail = "problem during vmaf_cuda_import_state"; return 0; }
    }

    // APSNR combined with n_threads regression-tests flush idempotency. MSE,
    // reduced peak and min_sse exercise the remaining PSNR scoring options.
    VmafFeatureDictionary *psnr_dict = NULL;
    err = vmaf_feature_dictionary_set(&psnr_dict, "enable_mse", "true");
    err |= vmaf_feature_dictionary_set(&psnr_dict, "enable_apsnr", "true");
    if (test_case->reduced_hbd_peak)
        err |= vmaf_feature_dictionary_set(&psnr_dict, "reduced_hbd_peak", "true");
    if (test_case->min_sse > 0.0) {
        char min_sse[32];
        snprintf(min_sse, sizeof(min_sse), "%.17g", test_case->min_sse);
        err |= vmaf_feature_dictionary_set(&psnr_dict, "min_sse", min_sse);
    }
    if (err) { *fail = "problem configuring psnr options"; return 0; }

    err = vmaf_use_feature(vmaf, use_cuda ? "psnr_cuda" : "psnr", psnr_dict);
    if (err) { *fail = "problem during vmaf_use_feature psnr"; return 0; }

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
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) { *fail = "problem during vmaf_read_pictures"; return 0; }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) { *fail = "problem during vmaf_read_pictures flush"; return 0; }

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            err = vmaf_feature_score_at_index(vmaf, score_keys[k],
                                              &scores[i][k], i);
            if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
        }
    }

    if (read_apsnr(vmaf, apsnr)) {
        *fail = "problem reading APSNR aggregates";
        return 0;
    }

    err = vmaf_close(vmaf);
    if (err) { *fail = "problem during vmaf_close"; return 0; }

    return 0;
}

static char *parity(const ParityCase *test_case)
{
    double cpu[N_FRAMES][N_KEYS], gpu[N_FRAMES][N_KEYS];
    double cpu_apsnr[N_APSNR], gpu_apsnr[N_APSNR];
    char *fail = NULL;

    run_pass(0, test_case, cpu, cpu_apsnr, &fail);
    if (fail) return fail;

    if (run_pass(1, test_case, gpu, gpu_apsnr, &fail)) {
        // meson exitcode protocol: 77 = SKIP, so CI without a GPU reports
        // this as skipped rather than silently passing
        fprintf(stderr, "no CUDA device available, skipping\n");
        exit(77);
    }
    if (fail) return fail;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            if (memcmp(&cpu[i][k], &gpu[i][k], sizeof(cpu[i][k]))) {
                fprintf(stderr, "mismatch %s, frame %u, %s: "
                        "cpu=%a gpu=%a\n", test_case->name, i,
                        score_keys[k], cpu[i][k], gpu[i][k]);
                return "cpu/cuda score mismatch";
            }
        }
    }

    for (unsigned k = 0; k < N_APSNR; k++) {
        if (memcmp(&cpu_apsnr[k], &gpu_apsnr[k], sizeof(cpu_apsnr[k]))) {
            fprintf(stderr, "aggregate mismatch %s, %s: cpu=%a gpu=%a\n",
                    test_case->name, apsnr_keys[k], cpu_apsnr[k],
                    gpu_apsnr[k]);
            return "cpu/cuda APSNR aggregate mismatch";
        }
    }

    return NULL;
}

static char *test_psnr_ssim_cuda_parity_420_8bpc_auto_scale(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 8 bpc 768x432 auto scale",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 8, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_psnr_ssim_cuda_parity_420_10bpc_auto_scale(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 10 bpc 768x432 auto scale",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 10, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_psnr_ssim_cuda_parity_444_12bpc_odd_scale_3(void)
{
    const ParityCase test_case = {
        .name = "yuv444p 12 bpc 65x49 scale 3",
        .pix_fmt = VMAF_PIX_FMT_YUV444P,
        .bpc = 12, .w = 65, .h = 49,
        .scale = 3,
    };
    return parity(&test_case);
}

static char *test_psnr_ssim_cuda_parity_422_16bpc_scale_1_options(void)
{
    const ParityCase test_case = {
        .name = "yuv422p 16 bpc 64x48 scale 1 with options",
        .pix_fmt = VMAF_PIX_FMT_YUV422P,
        .bpc = 16, .w = 64, .h = 48,
        .scale = 1,
        .enable_db = true,
        .clip_db = true,
        .reduced_hbd_peak = true,
        .min_sse = 1.0,
    };
    return parity(&test_case);
}

char *run_tests()
{
    mu_run_test(test_psnr_ssim_cuda_parity_420_8bpc_auto_scale);
    mu_run_test(test_psnr_ssim_cuda_parity_420_10bpc_auto_scale);
    mu_run_test(test_psnr_ssim_cuda_parity_444_12bpc_odd_scale_3);
    mu_run_test(test_psnr_ssim_cuda_parity_422_16bpc_scale_1_options);
    return NULL;
}
