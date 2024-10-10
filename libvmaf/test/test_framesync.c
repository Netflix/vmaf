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
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "framesync.h"
#include "test.h"
#include "thread_pool.h"

#define NUM_TEST_FRAMES 10
#define FRAME_BUF_LEN 1024

typedef struct ThreadData {
    uint8_t *ref;
    uint8_t *dist;
    unsigned index;
    VmafFrameSyncContext *fs_ctx;
    int err;
} ThreadData;

static void my_worker(void *data)
{
    int ctr;
    struct ThreadData *thread_data = data;
    uint8_t *shared_buf;
    uint8_t *dependent_buf;

    //acquire new buffer from frame sync
    vmaf_framesync_acquire_new_buf(thread_data->fs_ctx, (void*)&shared_buf,
                                   FRAME_BUF_LEN, thread_data->index);

    //populate shared buffer with values
    for (ctr = 0; ctr < FRAME_BUF_LEN; ctr++)
        shared_buf[ctr] = thread_data->ref[ctr] + thread_data->dist[ctr] + 2;

    //submit filled buffer back to frame sync
    vmaf_framesync_submit_filled_data(thread_data->fs_ctx, shared_buf,
                                      thread_data->index);

    //sleep to simulate work load
    const int sleep_seconds = 1;
#ifdef _WIN32
    Sleep(1000 * sleep_seconds);
#else
    sleep(sleep_seconds);
#endif

    if (thread_data->index == 0) goto cleanup;

    //retrieve dependent buffer from frame sync
    vmaf_framesync_retrieve_filled_data(thread_data->fs_ctx,
                                        (void*)&dependent_buf,
                                        thread_data->index - 1);

    for (ctr = 0; ctr < FRAME_BUF_LEN; ctr++) {
        if (dependent_buf[ctr] != (thread_data->ref[ctr] + thread_data->dist[ctr])) {
            fprintf(stderr, "verification error in frame index %d\n",
                    thread_data->index);
        }
    }

    //release dependent buffer from frame sync
    vmaf_framesync_release_buf(thread_data->fs_ctx, dependent_buf,
                               thread_data->index - 1);

cleanup:
    free(thread_data->ref);
    free(thread_data->dist);
}

static char *test_framesync_create_process_and_destroy()
{
    int err, frame_index;

    VmafThreadPool *pool;
    VmafFrameSyncContext *fs_ctx;
    unsigned n_threads = 2;

    err = vmaf_thread_pool_create(&pool, n_threads);
    mu_assert("problem during vmaf_thread_pool_init", !err);

    err = vmaf_framesync_init(&fs_ctx);
    mu_assert("problem during vmaf_framesync_init", !err);

    fprintf(stderr, "\n");
    for (frame_index = 0; frame_index < NUM_TEST_FRAMES; frame_index++) {
        uint8_t *pic_a = malloc(FRAME_BUF_LEN);
        uint8_t *pic_b = malloc(FRAME_BUF_LEN);

        fprintf(stderr, "processing frame %d\r", frame_index);

        memset(pic_a, frame_index, FRAME_BUF_LEN);
        memset(pic_b, frame_index, FRAME_BUF_LEN);

        struct ThreadData data = {
            .ref = pic_a,
            .dist = pic_b,
            .index = frame_index,
            .fs_ctx = fs_ctx,
            .err = 0,
        };

        err = vmaf_thread_pool_enqueue(pool, my_worker, &data, sizeof(ThreadData));

        mu_assert("problem during vmaf_thread_pool_enqueue with data", !err);

        //wait once in 2 frames
        if ((frame_index >= 1) && (frame_index & 1)) {
            err = vmaf_thread_pool_wait(pool);
            mu_assert("problem during vmaf_thread_pool_wait", !err);
        }
    }
    fprintf(stderr, "\n");

    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem during vmaf_thread_pool_wait\n", !err);
    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem during vmaf_thread_pool_destroy\n", !err);
    err = vmaf_framesync_destroy(fs_ctx);
    mu_assert("problem during vmaf_framesync_destroy\n", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_framesync_create_process_and_destroy);
    return NULL;
}
