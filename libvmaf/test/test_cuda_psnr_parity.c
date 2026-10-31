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

/* Compare per-frame PSNR/MSE and serialized APSNR for CPU and CUDA.
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
    "psnr_y", "psnr_cb", "psnr_cr",
    "mse_y", "mse_cb", "mse_cr",
};
#define N_KEYS (sizeof(score_keys) / sizeof(score_keys[0]))

static const char *apsnr_keys[] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };
#define N_APSNR (sizeof(apsnr_keys) / sizeof(apsnr_keys[0]))

typedef struct ParityCase {
    const char *name;
    enum VmafPixelFormat pix_fmt;
    unsigned bpc, w, h;
    bool disable_chroma;
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
                    r[j] = index == N_FRAMES - 2 ? 0 : v;
                    d[j] = index == N_FRAMES - 2 ? (int)peak :
                           index == N_FRAMES - 1 ? v : vd;
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
                    r[j] = index == N_FRAMES - 2 ? 0 : v;
                    d[j] = index == N_FRAMES - 2 ? (int)peak :
                           index == N_FRAMES - 1 ? v : vd;
                }
                r += ref->stride[p] / 2;
                d += dist->stride[p] / 2;
            }
        }
    }
}

static int read_apsnr(VmafContext *vmaf, double scores[N_APSNR], unsigned planes)
{
    char path[128];
    const int len = snprintf(path, sizeof(path),
                             "test_cuda_psnr_parity_%p.json",
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
        for (unsigned k = 0; k < planes; k++) {
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

    for (unsigned k = 0; k < planes; k++) {
        if (!found[k])
            return -1;
    }

    return 0;
}

/* Return 1 when CUDA is unavailable; report test failures through fail. */
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

    // Exercise per-frame MSE and the aggregate flush path.
    VmafFeatureDictionary *psnr_dict = NULL;
    err = vmaf_feature_dictionary_set(&psnr_dict, "enable_mse", "true");
    err |= vmaf_feature_dictionary_set(&psnr_dict, "enable_apsnr", "true");
    if (test_case->disable_chroma)
        err |= vmaf_feature_dictionary_set(&psnr_dict, "enable_chroma", "false");
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
            if ((test_case->disable_chroma || test_case->pix_fmt == VMAF_PIX_FMT_YUV400P) && k % 3)
                continue;
            err = vmaf_feature_score_at_index(vmaf, score_keys[k],
                                              &scores[i][k], i);
            if (err) { *fail = "problem during vmaf_feature_score_at_index"; return 0; }
        }
    }

    const unsigned planes = test_case->disable_chroma || test_case->pix_fmt == VMAF_PIX_FMT_YUV400P ? 1 : 3;
    if (read_apsnr(vmaf, apsnr, planes)) {
        *fail = "problem reading APSNR aggregates";
        return 0;
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
    double cpu_apsnr[N_APSNR], gpu_apsnr[N_APSNR];
    char *fail = NULL;

    run_pass(0, test_case, cpu, cpu_apsnr, &fail);
    if (fail) return fail;

    if (run_pass(1, test_case, gpu, gpu_apsnr, &fail)) {
        fprintf(stderr, "no CUDA device available, skipping\n");
        exit(77);
    }
    if (fail) return fail;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        for (unsigned k = 0; k < N_KEYS; k++) {
            if ((test_case->disable_chroma || test_case->pix_fmt == VMAF_PIX_FMT_YUV400P) && k % 3)
                continue;
            if (!isfinite(cpu[i][k]) || !isfinite(gpu[i][k]) ||
                memcmp(&cpu[i][k], &gpu[i][k], sizeof(cpu[i][k]))) {
                fprintf(stderr, "mismatch %s, frame %u, %s: "
                        "cpu=%a gpu=%a\n", test_case->name, i,
                        score_keys[k], cpu[i][k], gpu[i][k]);
                return "cpu/cuda score mismatch";
            }
        }
    }

    const unsigned planes = test_case->disable_chroma || test_case->pix_fmt == VMAF_PIX_FMT_YUV400P ? 1 : 3;
    for (unsigned k = 0; k < planes; k++) {
        const double peak = (1U << test_case->bpc) - 1;
        mu_assert("incorrect maximum-difference MSE",
                  gpu[N_FRAMES - 2][k + 3] == peak * peak);
        mu_assert("identical pictures must have zero MSE",
                  gpu[N_FRAMES - 1][k + 3] == 0.0);
        double cap = 6 * test_case->bpc + 12;
        if (test_case->min_sse > 0.0) {
            const unsigned w = k && test_case->pix_fmt != VMAF_PIX_FMT_YUV444P ?
                               test_case->w / 2 : test_case->w;
            const unsigned h = k && test_case->pix_fmt == VMAF_PIX_FMT_YUV420P ?
                               test_case->h / 2 : test_case->h;
            const double p = test_case->reduced_hbd_peak ?
                             255U << (test_case->bpc - 8) : peak;
            cap = ceil(10.0 * log10(p * p / (test_case->min_sse / ((double)w * h))));
        }
        mu_assert("incorrect identical-picture PSNR cap",
                  gpu[N_FRAMES - 1][k] == cap);
        mu_assert("APSNR must be finite", isfinite(cpu_apsnr[k]) && isfinite(gpu_apsnr[k]));
        if (memcmp(&cpu_apsnr[k], &gpu_apsnr[k], sizeof(cpu_apsnr[k]))) {
            fprintf(stderr, "aggregate mismatch %s, %s: cpu=%a gpu=%a\n",
                    test_case->name, apsnr_keys[k], cpu_apsnr[k],
                    gpu_apsnr[k]);
            return "cpu/cuda APSNR aggregate mismatch";
        }
    }

    return NULL;
}

static char *test_psnr_cuda_parity_420_8bpc(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 8 bpc 768x432",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 8, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_psnr_cuda_parity_420_10bpc(void)
{
    const ParityCase test_case = {
        .name = "yuv420p 10 bpc 768x432",
        .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 10, .w = 768, .h = 432,
    };
    return parity(&test_case);
}

static char *test_psnr_cuda_parity_444_12bpc_odd(void)
{
    const ParityCase test_case = {
        .name = "yuv444p 12 bpc 65x49",
        .pix_fmt = VMAF_PIX_FMT_YUV444P,
        .bpc = 12, .w = 65, .h = 49,
    };
    return parity(&test_case);
}

static char *test_psnr_cuda_parity_422_16bpc_options(void)
{
    const ParityCase test_case = {
        .name = "yuv422p 16 bpc 64x48 with options",
        .pix_fmt = VMAF_PIX_FMT_YUV422P,
        .bpc = 16, .w = 64, .h = 48,
        .reduced_hbd_peak = true,
        .min_sse = 1.0,
    };
    return parity(&test_case);
}

static char *test_psnr_cuda_luma_only(void)
{
    const ParityCase test_case = {
        .name = "luma-only YUV420P", .pix_fmt = VMAF_PIX_FMT_YUV420P,
        .bpc = 10, .w = 64, .h = 48, .disable_chroma = true,
    };
    return parity(&test_case);
}

static char *test_psnr_cuda_gray_tail(void)
{
    const ParityCase test_case = {
        .name = "single-pixel gray", .pix_fmt = VMAF_PIX_FMT_YUV400P,
        .bpc = 8, .w = 1, .h = 1,
    };
    return parity(&test_case);
}

static CudaFunctions driver;
static size_t uploaded_bytes;

static CUresult CUDAAPI count_uploads(const CUDA_MEMCPY2D *copy, CUstream stream)
{
    if (copy->srcMemoryType == CU_MEMORYTYPE_HOST &&
        copy->dstMemoryType == CU_MEMORYTYPE_DEVICE)
        uploaded_bytes += copy->WidthInBytes * copy->Height;
    return driver.cuMemcpy2DAsync(copy, stream);
}

static char *test_psnr_cuda_uploads(void)
{
    for (unsigned psnr = 0; psnr < 2; psnr++) {
        VmafContext *ctx;
        int err = vmaf_init(&ctx, (VmafConfiguration){
            .log_level = VMAF_LOG_LEVEL_ERROR,
        });
        mu_assert("upload-test context init failed", !err);
        VmafCudaState *state = NULL;
        err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){0});
        mu_assert("upload-test CUDA init failed", !err);
        driver = *state->f;
        state->f->cuMemcpy2DAsync = count_uploads;
        uploaded_bytes = 0;
        err = vmaf_cuda_import_state(ctx, state);
        mu_assert("upload-test CUDA import failed", !err);
        err = vmaf_use_feature(ctx, psnr ? "psnr_cuda" : "vif_cuda", NULL);
        mu_assert("upload-test feature registration failed", !err);

        for (unsigned i = 0; i < 2; i++) {
            VmafPicture ref, dist;
            err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 48);
            err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 48);
            mu_assert("upload-test picture allocation failed", !err);
            fill_pictures(&ref, &dist, 8, i);
            err = vmaf_read_pictures(ctx, &ref, &dist, i);
            mu_assert("upload-test extraction failed", !err);
        }
        err = vmaf_read_pictures(ctx, NULL, NULL, 0);
        mu_assert("upload-test flush failed", !err);
        err = vmaf_close(ctx);
        cuda_free_functions(&state->f);
        free(state);
        mu_assert("upload-test close failed", !err);

        /* Two picture pairs: VIF needs luma; PSNR also needs both chroma planes. */
        const size_t expected = 4 * (64 * 48 + (psnr ? 2 * 32 * 24 : 0));
        mu_assert("unexpected host-to-device image transfer size",
                  uploaded_bytes == expected);
    }
    return NULL;
}

#include "test_cuda_score_error.h"

static char *test_psnr_cuda_score_error(void)
{
    return test_cuda_score_error("psnr_cuda", "psnr_y");
}

char *run_tests()
{
    mu_run_test(test_psnr_cuda_parity_420_8bpc);
    mu_run_test(test_psnr_cuda_parity_420_10bpc);
    mu_run_test(test_psnr_cuda_parity_444_12bpc_odd);
    mu_run_test(test_psnr_cuda_parity_422_16bpc_options);
    mu_run_test(test_psnr_cuda_luma_only);
    mu_run_test(test_psnr_cuda_gray_tail);
    mu_run_test(test_psnr_cuda_uploads);
    mu_run_test(test_psnr_cuda_score_error);
    return NULL;
}
