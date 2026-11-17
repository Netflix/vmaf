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


static char *check_apsnr(unsigned bpc, double min_sse, uint64_t y_sse,
                         uint64_t chroma_sse)
{
    const char *names[] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };
    PsnrState s = {
        .enable_chroma = true,
        .enable_apsnr = true,
        .min_sse = min_sse,
        .apsnr = {
            .n_pixels = { 1920 * 1080, 960 * 540, 960 * 540 },
        },
    };
    VmafFeatureExtractor fex = { .priv = &s };
    VmafFeatureCollector *fc;
    int err = init(&fex, VMAF_PIX_FMT_YUV420P, bpc, 1920, 1080);
    mu_assert("init failed", !err);
    s.apsnr.sse[0] = y_sse;
    s.apsnr.sse[1] = s.apsnr.sse[2] = chroma_sse;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("collector init failed", !err);
    mu_assert("flush failed", flush(&fex, fc) == 1);

    for (unsigned i = 0; i < 3; i++) {
        double score;
        err = vmaf_feature_collector_get_aggregate(fc, names[i], &score);
        mu_assert("aggregate missing", !err);
        double expected = s.psnr_max[i];
        if (s.apsnr.sse[i]) {
            double apsnr = 10 * (log10(s.peak * s.peak) +
                                 log10(s.apsnr.n_pixels[i]) -
                                 log10(s.apsnr.sse[i]));
            double max_apsnr = ceil(10 * log10(s.peak * s.peak *
                                               s.apsnr.n_pixels[i] * 2));
            expected = MIN(apsnr, max_apsnr);
        }
        mu_assert("wrong APSNR", almost_equal(score, expected));
    }
    if (min_sse)
        mu_assert("plane caps should differ", s.psnr_max[0] != s.psnr_max[1]);
    else
        mu_assert("wrong default cap", almost_equal(s.psnr_max[0], 6 * bpc + 12));
    vmaf_feature_collector_destroy(fc);
    return NULL;
}

static char *test_apsnr_zero_sse()
{
    const unsigned depths[] = { 8, 10, 12, 16 };
    for (unsigned i = 0; i < sizeof(depths) / sizeof(depths[0]); i++) {
        char *result = check_apsnr(depths[i], 0, 0, 0);
        if (result) return result;
    }
    return NULL;
}

static char *test_apsnr_zero_sse_min_sse()
{
    return check_apsnr(8, 0.5, 0, 0);
}

static char *test_apsnr_mixed_sse()
{
    return check_apsnr(8, 0, 1000000, 0);
}

static char *test_apsnr_one_sse()
{
    return check_apsnr(8, 0, 1, 1);
}

char *run_tests()
{
    mu_run_test(test_16b_large_diff);
    mu_run_test(test_apsnr_zero_sse);
    mu_run_test(test_apsnr_zero_sse_min_sse);
    mu_run_test(test_apsnr_mixed_sse);
    mu_run_test(test_apsnr_one_sse);

    return NULL;
}
