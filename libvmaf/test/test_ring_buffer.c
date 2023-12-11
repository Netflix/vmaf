/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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
#include <string.h>
#include <unistd.h>

#include "test.h"

#include "libvmaf/libvmaf_cuda.h"

#include "cuda/common.h"
#include "cuda/picture_cuda.h"
#include "cuda/ring_buffer.h"
#include "thread_pool.h"

static char *test_ring_buffer()
{
    int err;

    VmafCudaCookie my_cookie = {
        .w = 1920,
        .h = 1080,
        .bpc = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV400P,
        .state = malloc(sizeof(VmafCudaState)),
    };

    VmafCudaConfiguration cu_cfg = { 0 };
    vmaf_cuda_state_init(&my_cookie.state, cu_cfg);

    VmafRingBufferConfig cfg = {
        .pic_cnt = 4,
        .cookie = &my_cookie,
        .alloc_picture_callback = vmaf_cuda_picture_alloc,
        .free_picture_callback = vmaf_cuda_picture_free,
        .synchronize_picture_callback = vmaf_cuda_picture_synchronize,
    };

    VmafRingBuffer *ring_buffer;
    err = vmaf_ring_buffer_init(&ring_buffer, cfg);
    mu_assert("problem during vmaf_picture_pool_init", !err);

    VmafPicture pic_1;
    err = vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_1);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_1.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_1.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_1.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_1.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_2;
    err = vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_2);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_2.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_2.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_2.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_2.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_3;
    err = vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_3);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_3.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_3.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_3.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_3.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_4;
    err = vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_4);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_4.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_4.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_4.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_4.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_5;
    err = vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_5);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_5.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_5.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_5.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_5.pix_fmt == VMAF_PIX_FMT_YUV400P);

    mu_assert("pic_5 should use the same data buffer as pic_1 did.",
              pic_1.data[0] == pic_5.data[0]);

    err = vmaf_ring_buffer_close(ring_buffer);
    mu_assert("problem during vmaf_ring_buffer_close", !err);

    return NULL;
}

/*
typedef struct MyThreadPoolData {
    VmafRingBuffer *ring_buffer;
    unsigned i;
    useconds_t timeout;
    VmafCudaCookie my_cookie;
} MyThreadPoolData;

static void request_picture(void *data)
{
    MyThreadPoolData *my_thread_pool_data = data;
    VmafRingBuffer *ring_buffer = my_thread_pool_data->ring_buffer;

    VmafPicture pic;
    vmaf_picture_alloc(&pic, my_thread_pool_data->my_cookie.pix_fmt,
		       my_thread_pool_data->my_cookie.bpc,
                       my_thread_pool_data->my_cookie.w,
		       my_thread_pool_data->my_cookie.h);

    VmafPicture pic_cuda_ref, pic_cuda_dist;
    //fprintf(stderr, "request: %i\n", my_thread_pool_data->i);
    vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_cuda_ref);
    vmaf_ring_buffer_fetch_next_picture(ring_buffer, &pic_cuda_dist);
    vmaf_cuda_picture_upload_async(&pic_cuda_ref, &pic, 0x1);
    vmaf_cuda_picture_upload_async(&pic_cuda_dist, &pic, 0x1);
    //fprintf(stderr, "usleep=%d: %i\n",
    //        my_thread_pool_data->timeout, my_thread_pool_data->i);
    vmaf_picture_unref(&pic_cuda_ref);
    vmaf_picture_unref(&pic_cuda_dist);
    usleep(my_thread_pool_data->timeout);
    //fprintf(stderr, "unref: %i\n", my_thread_pool_data->i);
}

static char *test_ring_buffer_threaded()
{
    int err;

    VmafCudaCookie my_cookie = {
        .w = 1920,
        .h = 1080,
        .bpc = 8,
        .pix_fmt = VMAF_PIX_FMT_YUV400P,
        .state = malloc(sizeof(VmafCudaState)),
    };

    VmafCudaConfiguration cfg = { 0 };
    vmaf_cuda_state_init(my_cookie.state, cfg);

    VmafRingBufferConfig cfg = {
        .pic_cnt = 4,
        .cookie = &my_cookie,
        .alloc_picture_callback = vmaf_cuda_picture_alloc,
        .free_picture_callback = vmaf_cuda_picture_free,
        .synchronize_picture_callback = vmaf_cuda_picture_synchronize,
    };

    VmafRingBuffer *ring_buffer;
    err = vmaf_ring_buffer_init(&ring_buffer, cfg);
    mu_assert("problem during vmaf_picture_pool_init", !err);

    VmafThreadPool *thread_pool;
    const unsigned n_threads = 4;
    err = vmaf_thread_pool_create(&thread_pool, n_threads);
    mu_assert("problem during vmaf_thread_pool_init", !err);

    const unsigned n = n_threads * 8;
    for (unsigned i = 0; i < n; i++) {
        MyThreadPoolData my_thread_pool_data = {
            .ring_buffer = ring_buffer,
            .i = i,
            .timeout = 100000 - (100000 * ((float)i / n)),
            .my_cookie = my_cookie,
        };
        err = vmaf_thread_pool_enqueue(thread_pool, request_picture,
                                       &my_thread_pool_data,
                                       sizeof(my_thread_pool_data));
        mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    }

    err = vmaf_thread_pool_wait(thread_pool);
    mu_assert("problem during vmaf_thread_pool_wait", !err);
    err = vmaf_thread_pool_destroy(thread_pool);
    mu_assert("problem during vmaf_thread_pool_destroy", !err);

    err = vmaf_ring_buffer_close(ring_buffer);
    mu_assert("problem during vmaf_ring_buffer_close", !err);

    return NULL;
}
*/

char *run_tests()
{
    mu_run_test(test_ring_buffer);
    //mu_run_test(test_ring_buffer_threaded);
    return NULL;
}
