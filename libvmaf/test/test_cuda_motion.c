/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_cuda.h"

#define W 96
#define H 64
#define N_FRAMES 4

/* The mirror at the picture edge only changes the two outermost sample rows
 * and columns, so every frame is noise that differs from its neighbours. */
static uint32_t lcg(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state >> 8;
}

static void fill_picture(VmafPicture *pic, unsigned index)
{
    uint32_t state = 12345u + 7919u * index;
    const unsigned max = (1u << pic->bpc) - 1;
    for (unsigned p = 0; p < 3; p++) {
        for (unsigned y = 0; y < pic->h[p]; y++) {
            for (unsigned x = 0; x < pic->w[p]; x++) {
                const unsigned v = lcg(&state) % (max + 1);
                if (pic->bpc == 8)
                    ((uint8_t *)pic->data[p])[y * pic->stride[p] + x] = v;
                else
                    ((uint16_t *)pic->data[p])[y * (pic->stride[p] / 2) + x] = v;
            }
        }
    }
}

static int motion2_scores(bool cuda, unsigned bpc, double *scores)
{
    VmafConfiguration cfg = { 0 };
    VmafContext *vmaf;
    int err = vmaf_init(&vmaf, cfg);
    if (err) return err;

    if (cuda) {
        VmafCudaState *cu_state;
        VmafCudaConfiguration cuda_cfg = { 0 };
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) return err;
        err = vmaf_cuda_import_state(vmaf, cu_state);
        if (err) return err;
    }

    err = vmaf_use_feature(vmaf, cuda ? "motion_cuda" : "motion", NULL);
    if (err) return err;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, W, H);
        if (err) return err;
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, bpc, W, H);
        if (err) return err;
        fill_picture(&ref, i);
        fill_picture(&dist, i + 100);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        if (err) return err;
    }
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) return err;

    for (unsigned i = 0; i < N_FRAMES; i++) {
        err = vmaf_feature_score_at_index(vmaf, "VMAF_integer_feature_motion2_score",
                                          &scores[i], i);
        if (err) return err;
    }
    return vmaf_close(vmaf);
}

static char *compare_motion2(unsigned bpc)
{
    double cpu[N_FRAMES], gpu[N_FRAMES];
    int err = motion2_scores(false, bpc, cpu);
    mu_assert("problem computing the CPU motion2 scores", !err);
    err = motion2_scores(true, bpc, gpu);
    mu_assert("problem computing the CUDA motion2 scores", !err);

    /* The CUDA kernel still blurs each frame before subtracting, which leaves a
     * rounding residual of about 2e-5 on this input; a wrong edge mirror gives 5e-2. */
    for (unsigned i = 0; i < N_FRAMES; i++) {
        mu_assert("CUDA motion2 differs from the CPU score by more than 1e-3",
                  fabs(cpu[i] - gpu[i]) < 1e-3);
    }
    return NULL;
}

static char *test_cuda_motion_matches_cpu_8bit()
{
    return compare_motion2(8);
}

static char *test_cuda_motion_matches_cpu_10bit()
{
    return compare_motion2(10);
}

char *run_tests()
{
    mu_run_test(test_cuda_motion_matches_cpu_8bit);
    mu_run_test(test_cuda_motion_matches_cpu_10bit);
    return NULL;
}
