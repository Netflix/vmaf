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

#include "test.h"
#include "feature/integer_psnr.c"

#define EPS 0.00001

static int almost_equal(double a, double b) {
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

static int get_picture_16b(VmafPicture *pic, int pic_index)
{
    int count = 0;
    uint16_t sample_pic[2][16] = {
        {0, 0, 0, 0, 0, 0},
        {65535, 65535, 65535, 65535, 65535, 65535},
    };

    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV420P, 16, 2, 2);
    if (err) return err;
    for (int c = 0; c < 3; c++) {
        uint16_t *data = (uint16_t *) pic->data[c];
        int stride = pic->stride[c] >> 1;
        for (unsigned i = 0; i < pic->h[c]; i++) {
            for (unsigned j = 0; j < pic->w[c]; j++) {
                data[i * stride + j] = sample_pic[pic_index][count++];
            }
        }
    }
    return 0;
}

static char *test_16b_large_diff()
{
    VmafPicture pic1, pic2;
    int err = 0;
    err |= get_picture_16b(&pic1, 0);
    err |= get_picture_16b(&pic2, 1);
    mu_assert("test_16b_large_diff alloc error", !err);
    VmafFeatureCollector *fc;
    err |= vmaf_feature_collector_init(&fc);
    mu_assert("test_16b_large_diff vmaf_feature_collector_init error", !err);
    PsnrState psnr_state = {
        .enable_chroma = 1,
        .enable_mse = 1,
        .enable_apsnr = 0,
        .peak = 65535,
    };

    err |= psnr_hbd(&pic1, &pic2, 0, fc, &psnr_state);
    mu_assert("failed psnr_hbd", err == 0);

    double psnr_y, psnr_cb, psnr_cr;
    err |= vmaf_feature_collector_get_score(fc, "psnr_y", &psnr_y, 0);
    err |= vmaf_feature_collector_get_score(fc, "psnr_cb", &psnr_cb, 0);
    err |= vmaf_feature_collector_get_score(fc, "psnr_cr", &psnr_cr, 0);
    double mse_y, mse_cb, mse_cr;
    err |= vmaf_feature_collector_get_score(fc, "mse_y", &mse_y, 0);
    err |= vmaf_feature_collector_get_score(fc, "mse_cb", &mse_cb, 0);
    err |= vmaf_feature_collector_get_score(fc, "mse_cr", &mse_cr, 0);
    mu_assert("test_16b_large_diff vmaf_feature_collector_get_score error", !err);

    mu_assert("wrong psnr_y", almost_equal(psnr_y, 0.0));
    mu_assert("wrong psnr_cb", almost_equal(psnr_cb, 0.0));
    mu_assert("wrong psnr_cr", almost_equal(psnr_cr, 0.0));
    mu_assert("wrong mse_y", almost_equal(mse_y, 4294836225.0));
    mu_assert("wrong mse_cb", almost_equal(mse_cb, 4294836225.0));
    mu_assert("wrong mse_cr", almost_equal(mse_cr, 4294836225.0));
    
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&pic1);
    vmaf_picture_unref(&pic2);

    return NULL;
}


static char *test_apsnr_luma_only()
{
    VmafFeatureCollector *fc;
    int err = vmaf_feature_collector_init(&fc);
    mu_assert("collector init error", !err);
    PsnrState state = {
        .enable_chroma = false,
        .enable_apsnr = true,
        .peak = 255,
        .apsnr = { .sse = { 1000000, 0, 0 },
                   .n_pixels = { 1920 * 1080, 0, 0 } },
    };
    VmafFeatureExtractor fex = { .priv = &state };
    err = flush(&fex, fc);
    mu_assert("flush error", err >= 0);

    double y, cb, cr;
    const double expected = MIN(
        10 * (log10(state.peak * state.peak) +
              log10(state.apsnr.n_pixels[0]) - log10(state.apsnr.sse[0])),
        ceil(10 * log10(state.peak * state.peak * state.apsnr.n_pixels[0] * 2)));
    mu_assert("apsnr_y missing", !vmaf_feature_collector_get_aggregate(fc, "apsnr_y", &y));
    mu_assert("incorrect apsnr_y", almost_equal(y, expected));
    mu_assert("apsnr_cb should be absent", vmaf_feature_collector_get_aggregate(fc, "apsnr_cb", &cb) != 0);
    mu_assert("apsnr_cr should be absent", vmaf_feature_collector_get_aggregate(fc, "apsnr_cr", &cr) != 0);
    vmaf_feature_collector_destroy(fc);
    return NULL;
}

static char *test_apsnr_luma_only_16b()
{
    VmafFeatureCollector *fc;
    int err = vmaf_feature_collector_init(&fc);
    mu_assert("collector init error", !err);
    PsnrState state = {
        .enable_chroma = false,
        .enable_apsnr = true,
        .peak = 65535,
        .apsnr = { .sse = { 1, 0, 0 }, .n_pixels = { 1, 0, 0 } },
    };
    VmafFeatureExtractor fex = { .priv = &state };
    err = flush(&fex, fc);
    mu_assert("flush error", err >= 0);
    double y, cb, cr;
    const double expected = MIN(
        10 * (log10(state.peak * state.peak) +
              log10(state.apsnr.n_pixels[0]) - log10(state.apsnr.sse[0])),
        ceil(10 * log10(state.peak * state.peak * state.apsnr.n_pixels[0] * 2)));
    mu_assert("apsnr_y missing", !vmaf_feature_collector_get_aggregate(fc, "apsnr_y", &y));
    mu_assert("incorrect apsnr_y", almost_equal(y, expected));
    mu_assert("apsnr_cb should be absent", vmaf_feature_collector_get_aggregate(fc, "apsnr_cb", &cb) != 0);
    mu_assert("apsnr_cr should be absent", vmaf_feature_collector_get_aggregate(fc, "apsnr_cr", &cr) != 0);
    vmaf_feature_collector_destroy(fc);
    return NULL;
}

static char *test_apsnr_chroma_enabled()
{
    VmafFeatureCollector *fc;
    int err = vmaf_feature_collector_init(&fc);
    mu_assert("collector init error", !err);
    PsnrState state = {
        .enable_chroma = true,
        .enable_apsnr = true,
        .peak = 255,
        .apsnr = { .sse = { 100, 50, 25 }, .n_pixels = { 1000, 250, 250 } },
    };
    VmafFeatureExtractor fex = { .priv = &state };
    err = flush(&fex, fc);
    mu_assert("flush error", err >= 0);
    const char *names[] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };
    for (unsigned i = 0; i < 3; i++) {
        double score;
        mu_assert("aggregate missing", !vmaf_feature_collector_get_aggregate(fc, names[i], &score));
        mu_assert("aggregate not finite", isfinite(score));
    }
    vmaf_feature_collector_destroy(fc);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_16b_large_diff);
    mu_run_test(test_apsnr_luma_only);
    mu_run_test(test_apsnr_luma_only_16b);
    mu_run_test(test_apsnr_chroma_enabled);

    return NULL;
}
