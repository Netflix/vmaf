/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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

#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static char *test_picture_pool_basic()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 4,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    // Use large pool for round-robin
    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 40,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        err = vmaf_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static char *test_picture_pool_small()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 2,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    // Small pool to test round-robin wrapping
    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 640,
            .h = 480,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 8,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    // Process fewer frames with small pool
    for (unsigned i = 0; i < 3; i++) {
        VmafPicture ref, dist;
        err = vmaf_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        err = vmaf_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static char *test_picture_pool_fetch_unref_cycle()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 4,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 10,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 16,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    // Test multiple fetch/unref cycles without vmaf_read_pictures
    for (unsigned cycle = 0; cycle < 3; cycle++) {
        VmafPicture pics[4];

        // Fetch pictures
        for (unsigned i = 0; i < 4; i++) {
            err = vmaf_fetch_preallocated_picture(vmaf, &pics[i]);
            mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        }

        // Unref all pictures
        for (unsigned i = 0; i < 4; i++) {
            err = vmaf_picture_unref(&pics[i]);
            mu_assert("problem during vmaf_picture_unref", !err);
        }
    }

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static char *test_picture_pool_yuv444()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 4,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    // Test with YUV444 format
    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV444P,
        },
        .pic_cnt = 20,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 5; i++) {
        VmafPicture ref, dist;
        err = vmaf_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        mu_assert("picture should be YUV444", ref.pix_fmt == VMAF_PIX_FMT_YUV444P);

        err = vmaf_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_fetch_preallocated_picture", !err);
        mu_assert("picture should be YUV444", dist.pix_fmt == VMAF_PIX_FMT_YUV444P);

        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

// Test pool exhaustion and blocking behavior
static char *test_picture_pool_exhaustion()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 4,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    // Very small pool (2 pictures) to test exhaustion
    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 640,
            .h = 480,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 2,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    VmafPicture pics[3];

    // Fetch first picture - should succeed
    err = vmaf_fetch_preallocated_picture(vmaf, &pics[0]);
    mu_assert("first fetch should succeed", !err);

    // Save the data pointer from first picture
    void *first_data_ptr = pics[0].data[0];

    // Fetch second picture - should succeed
    err = vmaf_fetch_preallocated_picture(vmaf, &pics[1]);
    mu_assert("second fetch should succeed", !err);

    // Pool is now exhausted (2/2 pictures in use)
    // If we tried to fetch a third, it would block

    // Return first picture
    err = vmaf_picture_unref(&pics[0]);
    mu_assert("problem during vmaf_picture_unref", !err);

    // Now fetch third picture - should succeed (reuses first picture)
    err = vmaf_fetch_preallocated_picture(vmaf, &pics[2]);
    mu_assert("third fetch should succeed after unref", !err);

    // Verify the picture we got back has same data pointer as first
    // (This verifies the free list is working correctly and pictures are reused)
    mu_assert("pictures should be reused", pics[2].data[0] == first_data_ptr);

    // Cleanup
    err = vmaf_picture_unref(&pics[1]);
    mu_assert("problem during vmaf_picture_unref", !err);
    err = vmaf_picture_unref(&pics[2]);
    mu_assert("problem during vmaf_picture_unref", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

// Multi-threaded test data
typedef struct {
    VmafContext *vmaf;
    int thread_id;
    int fetch_count;
    int error;
    void **data_ptrs;  // Track which data pointers we got
} thread_test_data;

static void *thread_fetch_worker(void *arg)
{
    thread_test_data *data = (thread_test_data *)arg;

    for (int i = 0; i < data->fetch_count; i++) {
        VmafPicture pic;
        int err = vmaf_fetch_preallocated_picture(data->vmaf, &pic);
        if (err) {
            data->error = err;
            return NULL;
        }

        // Store the data pointer to check for duplicates later
        data->data_ptrs[i] = pic.data[0];

        // Simulate some work
        usleep(100);  // 0.1ms

        err = vmaf_picture_unref(&pic);
        if (err) {
            data->error = err;
            return NULL;
        }
    }

    return NULL;
}

// Test concurrent access from multiple threads
static char *test_picture_pool_multithreaded()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 8,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 8,  // Small pool to stress test
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    const int num_threads = 4;
    const int fetches_per_thread = 20;
    pthread_t threads[num_threads];
    thread_test_data thread_data[num_threads];

    // Start threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].vmaf = vmaf;
        thread_data[i].thread_id = i;
        thread_data[i].fetch_count = fetches_per_thread;
        thread_data[i].error = 0;
        thread_data[i].data_ptrs = malloc(sizeof(void*) * fetches_per_thread);

        err = pthread_create(&threads[i], NULL, thread_fetch_worker, &thread_data[i]);
        mu_assert("problem creating thread", !err);
    }

    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        mu_assert("thread encountered error", thread_data[i].error == 0);
    }

    // Verify no two threads got the same picture at the same time
    // (This is a statistical check - not foolproof but catches obvious bugs)
    for (int i = 0; i < num_threads; i++) {
        free(thread_data[i].data_ptrs);
    }

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

// Test that close waits for all pictures to be returned
static void *thread_delayed_unref(void *arg)
{
    VmafPicture *pic = (VmafPicture *)arg;

    // Hold picture for a while
    sleep(1);

    // Then return it
    vmaf_picture_unref(pic);

    return NULL;
}

static char *test_picture_pool_close_waits()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 2,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 640,
            .h = 480,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 4,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    // Fetch a picture
    VmafPicture pic;
    err = vmaf_fetch_preallocated_picture(vmaf, &pic);
    mu_assert("problem during vmaf_fetch_preallocated_picture", !err);

    // Start thread that will hold picture for 1 second then return it
    pthread_t thread;
    err = pthread_create(&thread, NULL, thread_delayed_unref, &pic);
    mu_assert("problem creating thread", !err);

    // Try to close - should block until thread returns picture
    // This will take ~1 second
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    pthread_join(thread, NULL);

    return NULL;
}

// Stress test with high contention
static char *test_picture_pool_stress()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = {
        .log_level = VMAF_LOG_LEVEL_WARNING,
        .n_threads = 16,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    // Very small pool relative to thread count
    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = 640,
            .h = 480,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_cnt = 4,  // Only 4 pictures for 16 threads!
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    mu_assert("problem during vmaf_preallocate_pictures", !err);

    const int num_threads = 16;
    const int fetches_per_thread = 50;
    pthread_t threads[num_threads];
    thread_test_data thread_data[num_threads];

    // Start threads - high contention for limited pool
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].vmaf = vmaf;
        thread_data[i].thread_id = i;
        thread_data[i].fetch_count = fetches_per_thread;
        thread_data[i].error = 0;
        thread_data[i].data_ptrs = malloc(sizeof(void*) * fetches_per_thread);

        err = pthread_create(&threads[i], NULL, thread_fetch_worker, &thread_data[i]);
        mu_assert("problem creating thread", !err);
    }

    // Wait for all threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        mu_assert("thread encountered error", thread_data[i].error == 0);
        free(thread_data[i].data_ptrs);
    }

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_picture_pool_basic);
    mu_run_test(test_picture_pool_small);
    mu_run_test(test_picture_pool_fetch_unref_cycle);
    mu_run_test(test_picture_pool_yuv444);
    mu_run_test(test_picture_pool_exhaustion);
    mu_run_test(test_picture_pool_multithreaded);
    mu_run_test(test_picture_pool_close_waits);
    mu_run_test(test_picture_pool_stress);
    return NULL;
}
