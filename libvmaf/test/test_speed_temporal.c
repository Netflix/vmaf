/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <stdint.h>
#include <stdlib.h>

#include "test.h"
#include "dict.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "feature/feature_name.h"
#include "libvmaf/picture.h"

#define W 256
#define H 256
#define FRAMES 3

static const char *const score_name = "Speed_temporal_feature_speed_temporal_score";

/* xorshift32: successive frames must not be shifted copies of one sequence,
 * or their difference is structured and its covariance singular. */
static uint32_t next(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

static void fill(VmafPicture *pic, uint32_t seed, unsigned amplitude)
{
    uint32_t state = seed * 2654435761u + 1u;
    uint8_t *row = pic->data[0];
    for (unsigned i = 0; i < pic->h[0]; i++) {
        for (unsigned j = 0; j < pic->w[0]; j++)
            row[j] = (uint8_t)(128 + (int)(next(&state) % amplitude) - (int)(amplitude / 2));
        row += pic->stride[0];
    }
}

/* Scores a reference that is strong noise against a distorted video that is
 * weaker noise, and returns the score of frame `index`. */
static int temporal_score(const char *max_val, unsigned index, double *score)
{
    VmafDictionary *opts = NULL;
    if (max_val && vmaf_dictionary_set(&opts, "speed_max_val", max_val, 0))
        return -1;

    VmafFeatureExtractorContext *ctx;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("speed_temporal");
    VmafFeatureCollector *fc;
    int err = !fex;
    err = err || vmaf_feature_extractor_context_create(&ctx, fex, opts);
    err = err || vmaf_feature_collector_init(&fc);
    err = err || vmaf_feature_extractor_context_init(ctx, VMAF_PIX_FMT_YUV420P, 8, W, H);
    if (err)
        return -1;

    for (unsigned i = 0; i < FRAMES && !err; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, W, H);
        err = err || vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, W, H);
        if (err)
            break;
        fill(&ref, 1 + i, 250);
        fill(&dist, 77 + i, 60);
        err = vmaf_feature_extractor_context_extract(ctx, &ref, NULL, &dist, NULL, i, fc);
        vmaf_picture_unref(&ref);
        vmaf_picture_unref(&dist);
    }

    char *name = vmaf_feature_name_from_options(score_name, ctx->fex->options, ctx->fex->priv);
    err = err || !name;
    err = err || vmaf_feature_collector_get_score(fc, name, score, index);

    free(name);
    vmaf_feature_extractor_context_close(ctx);
    vmaf_feature_extractor_context_destroy(ctx);
    vmaf_feature_collector_destroy(fc);
    return err;
}

static char *test_speed_temporal_max_val()
{
    double unclipped, clipped;

    mu_assert("scoring with the default speed_max_val failed",
              !temporal_score(NULL, 1, &unclipped));
    mu_assert("the fixture must score above the bound used below", unclipped > 0.5);

    mu_assert("scoring with speed_max_val=0.5 failed", !temporal_score("0.5", 1, &clipped));
    mu_assert("speed_max_val must clip the score", clipped == 0.5);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_speed_temporal_max_val);
    return NULL;
}
