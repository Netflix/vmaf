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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "test.h"
#include "thread_pool.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "predict.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/picture.h"
#include "libvmaf/model.h"

/* ========================================================================
 * Helper utilities
 * ======================================================================== */

#define EPS 0.00001

static int almost_equal(double a, double b) {
    double diff = a > b ? a - b : b - a;
    return diff < EPS;
}

/* ========================================================================
 * Thread pool tests (thundering herd fix, signal vs broadcast)
 * ======================================================================== */

static volatile int tp_counter = 0;
static pthread_mutex_t tp_counter_lock = PTHREAD_MUTEX_INITIALIZER;

static void increment_counter(void *data)
{
    (void)data;
    pthread_mutex_lock(&tp_counter_lock);
    tp_counter++;
    pthread_mutex_unlock(&tp_counter_lock);
}

static char *test_thread_pool_many_jobs()
{
    int err;
    VmafThreadPool *pool;
    const unsigned n_threads = 4;
    const unsigned n_jobs = 1000;

    tp_counter = 0;

    err = vmaf_thread_pool_create(&pool, n_threads);
    mu_assert("problem creating thread pool", !err);

    for (unsigned i = 0; i < n_jobs; i++) {
        err = vmaf_thread_pool_enqueue(pool, increment_counter, NULL, 0);
        mu_assert("problem enqueuing job", !err);
    }

    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem waiting for thread pool", !err);

    mu_assert("not all jobs completed (thundering herd regression)",
              tp_counter == (int)n_jobs);

    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem destroying thread pool", !err);

    return NULL;
}

static void accumulate_value(void *data)
{
    int *val = data;
    pthread_mutex_lock(&tp_counter_lock);
    tp_counter += *val;
    pthread_mutex_unlock(&tp_counter_lock);
}

static char *test_thread_pool_data_passing()
{
    int err;
    VmafThreadPool *pool;

    tp_counter = 0;

    err = vmaf_thread_pool_create(&pool, 2);
    mu_assert("problem creating thread pool", !err);

    for (int i = 1; i <= 100; i++) {
        err = vmaf_thread_pool_enqueue(pool, accumulate_value, &i, sizeof(i));
        mu_assert("problem enqueuing job with data", !err);
    }

    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem waiting for thread pool", !err);

    /* Sum of 1..100 = 5050 */
    mu_assert("data passing through thread pool failed",
              tp_counter == 5050);

    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem destroying thread pool", !err);

    return NULL;
}

/* ========================================================================
 * Feature collector tests (capacity 512, realloc behavior)
 * ======================================================================== */

static char *test_feature_vector_capacity_512()
{
    int err;
    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    /* Append 500 scores to a single feature - should not realloc since
     * initial capacity is now 512 */
    for (unsigned i = 0; i < 500; i++) {
        err = vmaf_feature_collector_append(fc, "test_feature", (double)i, i);
        mu_assert("problem during vmaf_feature_collector_append", !err);
    }

    /* Verify all scores are retrievable */
    double score;
    err = vmaf_feature_collector_get_score(fc, "test_feature", &score, 0);
    mu_assert("could not get score at index 0", !err);
    mu_assert("score at index 0 incorrect", almost_equal(score, 0.0));

    err = vmaf_feature_collector_get_score(fc, "test_feature", &score, 499);
    mu_assert("could not get score at index 499", !err);
    mu_assert("score at index 499 incorrect", almost_equal(score, 499.0));

    /* Now append beyond 512 - should trigger realloc */
    for (unsigned i = 500; i < 600; i++) {
        err = vmaf_feature_collector_append(fc, "test_feature", (double)i, i);
        mu_assert("problem appending beyond initial capacity", !err);
    }

    err = vmaf_feature_collector_get_score(fc, "test_feature", &score, 599);
    mu_assert("could not get score after realloc", !err);
    mu_assert("score after realloc incorrect", almost_equal(score, 599.0));

    vmaf_feature_collector_destroy(fc);
    return NULL;
}

static char *test_feature_collector_multiple_features()
{
    int err;
    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("problem during init", !err);

    /* Add 20 different features - tests collector capacity growth */
    char name[64];
    for (unsigned f = 0; f < 20; f++) {
        snprintf(name, sizeof(name), "feature_%u", f);
        for (unsigned i = 0; i < 10; i++) {
            err = vmaf_feature_collector_append(fc, name, (double)(f * 10 + i), i);
            mu_assert("problem during append", !err);
        }
    }

    /* Verify retrieval from various features */
    double score;
    err = vmaf_feature_collector_get_score(fc, "feature_0", &score, 5);
    mu_assert("get_score failed for feature_0", !err);
    mu_assert("wrong score for feature_0[5]", almost_equal(score, 5.0));

    err = vmaf_feature_collector_get_score(fc, "feature_19", &score, 9);
    mu_assert("get_score failed for feature_19", !err);
    mu_assert("wrong score for feature_19[9]", almost_equal(score, 199.0));

    vmaf_feature_collector_destroy(fc);
    return NULL;
}

/* ========================================================================
 * Predict tests (stack-allocated SVM nodes, cached feature names)
 * ======================================================================== */

static char *test_predict_score_consistency()
{
    int err;

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("problem during vmaf_feature_collector_init", !err);

    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load(&model, &cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    /* Insert known feature scores */
    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(fc, model->feature[i].name, 60., 0);
        mu_assert("problem during feature append", !err);
    }

    /* Predict twice to test caching of name_with_opt */
    double score1 = 0., score2 = 0.;
    err = vmaf_predict_score_at_index(model, fc, 0, &score1, true, false, 0);
    mu_assert("first prediction failed", !err);
    mu_assert("first prediction out of range", score1 > 0. && score1 <= 100.);

    /* Insert scores at index 1 too */
    for (unsigned i = 0; i < model->n_features; i++) {
        err = vmaf_feature_collector_append(fc, model->feature[i].name, 60., 1);
        mu_assert("problem during feature append at index 1", !err);
    }

    err = vmaf_predict_score_at_index(model, fc, 1, &score2, true, false, 0);
    mu_assert("second prediction failed", !err);

    /* Same inputs should give same score */
    mu_assert("cached prediction differs from first",
              almost_equal(score1, score2));

    vmaf_model_destroy(model);
    vmaf_feature_collector_destroy(fc);
    return NULL;
}

/* ========================================================================
 * Feature extractor correctness tests (PSNR, VIF, Motion, ADM)
 * ======================================================================== */

static int alloc_picture_8b(VmafPicture *pic, unsigned w, unsigned h,
                            uint8_t fill_value)
{
    int err = vmaf_picture_alloc(pic, VMAF_PIX_FMT_YUV400P, 8, w, h);
    if (err) return err;
    uint8_t *data = pic->data[0];
    for (unsigned i = 0; i < h; i++) {
        memset(data, fill_value, w);
        data += pic->stride[0];
    }
    return 0;
}

static char *test_psnr_identical_frames()
{
    VmafPicture ref, dist;
    int err = 0;

    err |= alloc_picture_8b(&ref, 64, 64, 128);
    err |= alloc_picture_8b(&dist, 64, 64, 128);
    mu_assert("picture alloc failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("feature collector init failed", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("psnr");
    mu_assert("could not find psnr feature extractor", fex);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("fex context create failed", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &dist, NULL, 0, fc);
    mu_assert("psnr extract failed", !err);

    double psnr_y;
    err = vmaf_feature_collector_get_score(fc, "psnr_y", &psnr_y, 0);
    mu_assert("get psnr_y score failed", !err);

    /* Identical frames should give maximum PSNR */
    mu_assert("identical frames should have max PSNR",
              psnr_y >= 60.0);

    vmaf_feature_extractor_context_close(fex_ctx);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

static char *test_psnr_different_frames()
{
    VmafPicture ref, dist;
    int err = 0;

    err |= alloc_picture_8b(&ref, 64, 64, 0);
    err |= alloc_picture_8b(&dist, 64, 64, 255);
    mu_assert("picture alloc failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("feature collector init failed", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("psnr");
    mu_assert("could not find psnr feature extractor", fex);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("fex context create failed", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &dist, NULL, 0, fc);
    mu_assert("psnr extract failed", !err);

    double psnr_y;
    err = vmaf_feature_collector_get_score(fc, "psnr_y", &psnr_y, 0);
    mu_assert("get psnr_y score failed", !err);

    /* Max difference (0 vs 255) should give very low PSNR */
    mu_assert("max difference should have very low PSNR", psnr_y < 10.0);

    vmaf_feature_extractor_context_close(fex_ctx);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

static char *test_vif_identical_frames()
{
    VmafPicture ref, dist;
    int err = 0;

    err |= alloc_picture_8b(&ref, 64, 64, 128);
    err |= alloc_picture_8b(&dist, 64, 64, 128);
    mu_assert("picture alloc failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("feature collector init failed", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("vif");
    mu_assert("could not find vif feature extractor", fex);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("vif fex context create failed", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &dist, NULL, 0, fc);
    mu_assert("vif extract failed", !err);

    double vif_s0;
    err = vmaf_feature_collector_get_score(fc,
            "VMAF_integer_feature_vif_scale0_score", &vif_s0, 0);
    mu_assert("get vif scale0 score failed", !err);

    /* Identical frames should give VIF = 1.0 */
    mu_assert("identical frames should have VIF ~1.0",
              vif_s0 > 0.99 && vif_s0 < 1.01);

    vmaf_feature_extractor_context_close(fex_ctx);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

static char *test_motion_identical_frames()
{
    VmafPicture ref;
    int err = 0;

    err = alloc_picture_8b(&ref, 64, 64, 100);
    mu_assert("picture alloc failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("feature collector init failed", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("motion");
    mu_assert("could not find motion feature extractor", fex);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("motion fex context create failed", !err);

    /* Feed the same frame twice (index 0 and 1) */
    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &ref, NULL, 0, fc);
    mu_assert("motion extract failed (frame 0)", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &ref, NULL, 1, fc);
    mu_assert("motion extract failed (frame 1)", !err);

    /* Motion of identical frames should be 0 */
    double motion_score;
    err = vmaf_feature_collector_get_score(fc,
            "VMAF_integer_feature_motion2_score", &motion_score, 0);
    mu_assert("get motion2 score failed", !err);
    mu_assert("identical frames should have motion score 0",
              almost_equal(motion_score, 0.0));

    vmaf_feature_extractor_context_close(fex_ctx);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&ref);

    return NULL;
}

static char *test_adm_identical_frames()
{
    VmafPicture ref, dist;
    int err = 0;

    err |= alloc_picture_8b(&ref, 64, 64, 128);
    err |= alloc_picture_8b(&dist, 64, 64, 128);
    mu_assert("picture alloc failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("feature collector init failed", !err);

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("adm");
    mu_assert("could not find adm feature extractor", fex);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    mu_assert("adm fex context create failed", !err);

    err = vmaf_feature_extractor_context_extract(fex_ctx, &ref, NULL,
                                                  &dist, NULL, 0, fc);
    mu_assert("adm extract failed", !err);

    double adm2;
    err = vmaf_feature_collector_get_score(fc,
            "VMAF_integer_feature_adm2_score", &adm2, 0);
    mu_assert("get adm2 score failed", !err);

    /* Identical frames should give ADM2 = 1.0 */
    mu_assert("identical frames should have ADM2 ~1.0",
              adm2 > 0.99 && adm2 < 1.01);

    vmaf_feature_extractor_context_close(fex_ctx);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    vmaf_feature_collector_destroy(fc);
    vmaf_picture_unref(&ref);
    vmaf_picture_unref(&dist);

    return NULL;
}

/* ========================================================================
 * End-to-end VMAF score test
 * ======================================================================== */

static char *test_vmaf_score_range()
{
    int err;
    VmafConfiguration vmaf_cfg = { 0 };
    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("vmaf_init failed", !err);

    VmafModelConfig model_cfg = { .name = "vmaf" };
    VmafModel *model;
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("model load failed", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("use features from model failed", !err);

    /* Create identical ref/dist pictures */
    VmafPicture ref, dist;
    err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("ref alloc failed", !err);
    err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 64, 64);
    mu_assert("dist alloc failed", !err);

    /* Fill with a pattern */
    for (int p = 0; p < 3; p++) {
        uint8_t *rdata = ref.data[p];
        uint8_t *ddata = dist.data[p];
        for (unsigned i = 0; i < ref.h[p]; i++) {
            for (unsigned j = 0; j < ref.w[p]; j++) {
                rdata[j] = (uint8_t)((i + j) & 0xFF);
                ddata[j] = (uint8_t)((i + j) & 0xFF);
            }
            rdata += ref.stride[p];
            ddata += dist.stride[p];
        }
    }

    err = vmaf_read_pictures(vmaf, &ref, &dist, 0);
    mu_assert("read pictures failed", !err);

    /* Flush */
    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("flush failed", !err);

    double vmaf_score;
    err = vmaf_score_at_index(vmaf, model, &vmaf_score, 0);
    mu_assert("score at index failed", !err);

    /* Identical frames should give VMAF near 100 */
    mu_assert("VMAF score for identical frames should be high (> 90)",
              vmaf_score > 90.0);
    mu_assert("VMAF score should not exceed 100", vmaf_score <= 100.0);

    vmaf_model_destroy(model);
    vmaf_close(vmaf);

    return NULL;
}

/* ========================================================================
 * Test runner
 * ======================================================================== */

char *run_tests()
{
    /* Thread pool tests */
    mu_run_test(test_thread_pool_many_jobs);
    mu_run_test(test_thread_pool_data_passing);

    /* Feature collector tests */
    mu_run_test(test_feature_vector_capacity_512);
    mu_run_test(test_feature_collector_multiple_features);

    /* Predict tests */
    mu_run_test(test_predict_score_consistency);

    /* Feature extractor correctness tests */
    mu_run_test(test_psnr_identical_frames);
    mu_run_test(test_psnr_different_frames);
    mu_run_test(test_vif_identical_frames);
    mu_run_test(test_motion_identical_frames);
    mu_run_test(test_adm_identical_frames);

    /* End-to-end test */
    mu_run_test(test_vmaf_score_range);

    return NULL;
}
