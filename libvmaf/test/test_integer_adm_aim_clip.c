/**
 *
 *  Copyright 2026 Lusoris
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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test.h"

#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"

/* AIM is the additive impairment that survives contrast masking, divided by the DLM denominator
 * (the reference's own detail). adm.c clips it to 1: MIN(aim_num / aim_den, 1.0f). integer_adm.c
 * returned aim_num / den, so a flat reference against a picture with visible detail gave an
 * integer AIM above 1 (1.268618 for the noise of +-24 below) and a float AIM of exactly 1. */

typedef enum { DIS_NOISE_24, DIS_NOISE_8, DIS_IDENTICAL } Distortion;

#define FRAME_W 576
#define FRAME_H 324

typedef struct {
    double adm2;
    double aim;
    double adm3;
} Scores;

static void fill_grey(VmafPicture *pic)
{
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *plane = pic->data[p];
        for (unsigned row = 0; row < pic->h[p]; row++)
            memset(plane + row * pic->stride[p], 128, pic->w[p]);
    }
}

/* Luma 128 plus uniform noise in [-amp, amp] from xorshift32. Chroma stays flat. */
static void draw_noise(VmafPicture *pic, unsigned amp)
{
    uint32_t state = 1;

    for (unsigned row = 0; row < pic->h[0]; row++) {
        uint8_t *line = (uint8_t *) pic->data[0] + row * pic->stride[0];
        for (unsigned col = 0; col < pic->w[0]; col++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            line[col] = (uint8_t) (128 + (int) (state % (2 * amp + 1)) - (int) amp);
        }
    }
}

/* A flat grey reference against `dis`. */
static int run(const char *feature, const char *const *keys, Distortion dis, uint64_t cpumask,
               const char *dlm_weight, const char *min_val, Scores *out)
{
    const unsigned w = FRAME_W;
    const unsigned h = FRAME_H;
    VmafConfiguration cfg = {.log_level = VMAF_LOG_LEVEL_NONE, .cpumask = cpumask};
    VmafContext *vmaf = NULL;
    VmafFeatureDictionary *opts = NULL;
    VmafPicture ref, dist;
    int err = vmaf_init(&vmaf, cfg);

    if (err) return err;
    if (dlm_weight) {
        err = vmaf_feature_dictionary_set(&opts, "adm_dlm_weight", dlm_weight);
        err |= vmaf_feature_dictionary_set(&opts, "adm_min_val", min_val);
    }
    err |= vmaf_use_feature(vmaf, feature, opts);
    if (err) {
        vmaf_close(vmaf);
        return err;
    }
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, w, h);
    err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, w, h);
    if (err) {
        vmaf_close(vmaf);
        return err;
    }
    fill_grey(&ref);
    fill_grey(&dist);
    if (dis == DIS_NOISE_24) draw_noise(&dist, 24);
    if (dis == DIS_NOISE_8) draw_noise(&dist, 8);

    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    err |= vmaf_read_pictures(vmaf, NULL, NULL, 0);
    err |= vmaf_feature_score_at_index(vmaf, keys[0], &out->adm2, 0);
    err |= vmaf_feature_score_at_index(vmaf, keys[1], &out->aim, 0);
    err |= vmaf_feature_score_at_index(vmaf, keys[2], &out->adm3, 0);
    vmaf_close(vmaf);
    return err;
}

static const char *const integer_keys[3] = {"VMAF_integer_feature_adm2_score",
                                            "VMAF_integer_feature_aim_score",
                                            "VMAF_integer_feature_adm3_score"};
static const char *const float_keys[3] = {"VMAF_feature_adm2_score", "VMAF_feature_aim_score",
                                          "VMAF_feature_adm3_score"};
static const char *const integer_model_keys[3] = {"integer_adm2_dlmw_0.7_min_0.5",
                                                  "integer_aim_dlmw_0.7_min_0.5",
                                                  "integer_adm3_dlmw_0.7_min_0.5"};
static const char *const float_model_keys[3] = {"adm2_dlmw_0.7_min_0.5", "aim_dlmw_0.7_min_0.5",
                                                "adm3_dlmw_0.7_min_0.5"};

/* The cpumask that leaves every SIMD level on, and the one that forces the scalar path. */
static const uint64_t cpumasks[2] = {0, ~(uint64_t) 0};

/* The float extractor is optional (-Denable_float=false): the comparisons with it are skipped
 * when it is not there. Returns 1 and fills `out`, or 0. */
static int run_float(const char *const *keys, Distortion dis, const char *dlm_weight,
                     const char *min_val, Scores *out)
{
    return run("float_adm", keys, dis, 0, dlm_weight, min_val, out) == 0;
}

/* positive: a flat reference against noise of +-24. The unclipped integer AIM was 1.268618. */
static char *test_integer_aim_is_clipped_at_one()
{
    for (int m = 0; m < 2; m++) {
        Scores s = {0, 0, 0};
        mu_assert("integer adm failed on the noise",
                  run("adm", integer_keys, DIS_NOISE_24, cpumasks[m], NULL, NULL, &s) == 0);
        if (s.aim != 1.0) fprintf(stderr, "\n  integer aim %.17g, expected 1\n", s.aim);
        mu_assert("integer aim on the noise of +-24 is not clipped to 1", s.aim == 1.0);
    }
    return NULL;
}

/* the integer and the float extractor agree on the clipped value, and on adm3 with the default
 * model's weight and floor (0.7 and 0.5): the unclipped AIM took integer adm3 to 0.62, where the
 * float extractor gives 0.7 */
static char *test_integer_matches_float_on_the_noise()
{
    Scores fl = {0, 0, 0}, in = {0, 0, 0};

    if (!run_float(float_keys, DIS_NOISE_24, NULL, NULL, &fl)) return NULL;
    mu_assert("integer adm failed on the noise",
              run("adm", integer_keys, DIS_NOISE_24, 0, NULL, NULL, &in) == 0);
    mu_assert("integer aim differs from float aim", in.aim == fl.aim);

    mu_assert("float adm with the model's weight failed",
              run_float(float_model_keys, DIS_NOISE_24, "0.7", "0.5", &fl));
    mu_assert("integer adm with the model's weight failed",
              run("adm", integer_model_keys, DIS_NOISE_24, 0, "0.7", "0.5", &in) == 0);
    if (fabs(in.adm3 - fl.adm3) >= 1e-4)
        fprintf(stderr, "\n  integer adm3 %.17g, float adm3 %.17g\n", in.adm3, fl.adm3);
    mu_assert("integer adm3 differs from float adm3 with the model's weight and floor",
              fabs(in.adm3 - fl.adm3) < 1e-4);
    return NULL;
}

/* negative: no impairment, AIM 0 */
static char *test_aim_of_identical_pictures_is_zero()
{
    Scores s = {1, 1, 1};

    mu_assert("integer adm failed on identical pictures",
              run("adm", integer_keys, DIS_IDENTICAL, 0, NULL, NULL, &s) == 0);
    mu_assert("integer aim of identical pictures is not 0", s.aim == 0.0);
    return NULL;
}

/* boundary: an AIM below 1 passes the clip unchanged (about 0.4357 for noise of +-8) */
static char *test_aim_below_one_is_unchanged()
{
    Scores s = {0, 0, 0}, fl = {0, 0, 0};

    mu_assert("integer adm failed on the noise",
              run("adm", integer_keys, DIS_NOISE_8, 0, NULL, NULL, &s) == 0);
    mu_assert("integer aim on the noise of +-8 is not below 1", s.aim > 0.4 && s.aim < 0.5);
    if (run_float(float_keys, DIS_NOISE_8, NULL, NULL, &fl))
        mu_assert("integer aim moved away from float aim", fabs(s.aim - fl.aim) < 1e-4);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_integer_aim_is_clipped_at_one);
    mu_run_test(test_integer_matches_float_on_the_noise);
    mu_run_test(test_aim_of_identical_pictures_is_zero);
    mu_run_test(test_aim_below_one_is_unchanged);
    return NULL;
}
