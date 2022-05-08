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
#include <string.h>
#include <unistd.h>

#include "test.h"
#include "picture.h"
#include "libvmaf/picture.h"
#include "ref.h"
#include "thread_pool.h"

static char *test_picture_alloc_ref_and_unref()
{
    int err;

    VmafPicture pic_a, pic_b;
    err = vmaf_picture_alloc(&pic_a, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("pic_a.ref->cnt should be 1", vmaf_ref_load(pic_a.ref) == 1);
    err = vmaf_picture_ref(&pic_b, &pic_a);
    mu_assert("problem during vmaf_picture_ref", !err);
    mu_assert("pic_a.ref->cnt should be 2", vmaf_ref_load(pic_a.ref) == 2);
    mu_assert("pic_b.ref->cnt should be 2", vmaf_ref_load(pic_b.ref) == 2);
    err = vmaf_picture_unref(&pic_a);
    mu_assert("problem during vmaf_picture_unref", !err);
    mu_assert("pic_b.ref->cnt should be 1", vmaf_ref_load(pic_b.ref) == 1);
    err = vmaf_picture_unref(&pic_b);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}

static char *test_picture_data_alignment()
{
    int err;

    VmafPicture pic;
    err = vmaf_picture_alloc(&pic, VMAF_PIX_FMT_YUV420P, 10, 1920+1, 1080);
    mu_assert("problem during vmaf_picture_alloc", !err);
    mu_assert("picture data is not 32-byte alligned",
        !(((uintptr_t) pic.data[0]) % 32) &&
        !(((uintptr_t) pic.data[1]) % 32) &&
        !(((uintptr_t) pic.data[2]) % 32) &&
        !(pic.stride[0] % 32) &&
        !(pic.stride[1] % 32) &&
        !(pic.stride[2] % 32)
    );
    err = vmaf_picture_unref(&pic);
    mu_assert("problem during vmaf_picture_unref", !err);

    return NULL;
}

typedef struct MyCookie {
    size_t data_sz;
    enum VmafPixelFormat pix_fmt;
} MyCookie;

static int my_alloc_picture(VmafPicture *pic, void *cookie)
{
    if (!pic) return -1;

    //fprintf(stderr, "my_alloc_picture\n");
    memset(pic, 0, sizeof(*pic));

    MyCookie *my_cookie = cookie;
    void *data = malloc(my_cookie->data_sz);
    if (!data) return -1;

    pic->data[0] = data;
    pic->data[1] = NULL;
    pic->data[2] = NULL;
    pic->pix_fmt = my_cookie->pix_fmt;

    return 0;
}

static int my_synchronize_picture(VmafPicture *pic, void *cookie)
{
    if (!pic) return -1;

    MyCookie *my_cookie = cookie;
    (void) my_cookie;

    //fprintf(stderr, "my_synchronize_picture\n");

    return 0;
}

static int my_free_picture(VmafPicture *pic, void *cookie)
{
    if (!pic) return -1;
    if (!pic->data[0]) return -1;

    //fprintf(stderr, "my_free_picture\n");

    MyCookie *my_cookie = cookie;
    (void) my_cookie;
    free(pic->data[0]);

    return 0;
}

static char *test_picture_pool()
{
    int err;

    MyCookie my_cookie = {
        .data_sz = 1920*1080,
        .pix_fmt = VMAF_PIX_FMT_YUV400P,
    };

    VmafPicturePoolConfig cfg = {
        .pic_cnt = 4,
        .cookie = &my_cookie,
        .alloc_picture_callback = my_alloc_picture,
        .synchronize_picture_callback = my_synchronize_picture,
        .free_picture_callback = my_free_picture,
    };

    VmafPicturePool *pic_pool;
    err = vmaf_picture_pool_init(&pic_pool, cfg);
    mu_assert("problem during vmaf_picture_pool_init", !err);

    VmafPicture pic_1;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_1);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_1.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_1.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_1.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_1.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_2;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_2);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_2.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_2.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_2.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_2.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_3;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_3);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_3.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_3.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_3.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_3.pix_fmt == VMAF_PIX_FMT_YUV400P);

    VmafPicture pic_4;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_4);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_4.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_4.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_4.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_4.pix_fmt == VMAF_PIX_FMT_YUV400P);

    const void *pic_2_data_buf = pic_2.data[0];
    const void *pic_1_data_buf = pic_1.data[0];
    err |= vmaf_picture_unref(&pic_2);
    err |= vmaf_picture_unref(&pic_1);
    mu_assert("problem during vmaf_picture_unref", !err);

    VmafPicture pic_5;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_5);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_5.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_5.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_5.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_5.pix_fmt == VMAF_PIX_FMT_YUV400P);

    mu_assert("pic_5 should use the same data buffer as pic_1 did. "
              "the underlying queue should be LIFO.",
              pic_1_data_buf == pic_5.data[0]);

    VmafPicture pic_6;
    err = vmaf_picture_pool_request_picture(pic_pool, &pic_6);
    mu_assert("problem during vmaf_picture_pool_request_picture", !err);
    mu_assert("data[0] should have been allocated", pic_6.data[0]);
    mu_assert("data[1] should not have been allocated", !pic_6.data[1]);
    mu_assert("data[2] should not have been allocated", !pic_6.data[2]);
    mu_assert("pix_fmt should be VMAF_PIX_FMT_YUV400P",
              pic_6.pix_fmt == VMAF_PIX_FMT_YUV400P);

    mu_assert("pic_6 should use the same data buffer as pic_2 did. "
              "the underlying queue should be LIFO.",
              pic_2_data_buf == pic_6.data[0]);

    err |= vmaf_picture_unref(&pic_3);
    err |= vmaf_picture_unref(&pic_4);
    err |= vmaf_picture_unref(&pic_5);
    err |= vmaf_picture_unref(&pic_6);
    mu_assert("problem during vmaf_picture_unref", !err);

    err = vmaf_picture_pool_close(pic_pool);
    mu_assert("problem during vmaf_picture_pool_close", !err);

    return NULL;
}

typedef struct MyThreadPoolData {
    VmafPicturePool *pic_pool;
    unsigned i;
    useconds_t timeout;
} MyThreadPoolData;


static void request_picture_and_unref(void *data)
{
    MyThreadPoolData *my_thread_pool_data = data;
    VmafPicturePool *pic_pool = my_thread_pool_data->pic_pool;
    VmafPicture pic;
    //fprintf(stderr, "request: %i\n", my_thread_pool_data->i);
    vmaf_picture_pool_request_picture(pic_pool, &pic);
    //fprintf(stderr, "usleep=%d: %i\n",
    //        my_thread_pool_data->timeout, my_thread_pool_data->i);
    usleep(my_thread_pool_data->timeout);
    //fprintf(stderr, "unref: %i\n", my_thread_pool_data->i);
    vmaf_picture_unref(&pic);
}

static char *test_picture_pool_threaded()
{
    int err;

    MyCookie my_cookie = {
        .data_sz = 1920*1080,
        .pix_fmt = VMAF_PIX_FMT_YUV400P,
    };

    VmafPicturePoolConfig cfg = {
        .pic_cnt = 4,
        .cookie = &my_cookie,
        .alloc_picture_callback = my_alloc_picture,
        .free_picture_callback = my_free_picture,
    };

    VmafPicturePool *pic_pool;
    err = vmaf_picture_pool_init(&pic_pool, cfg);
    mu_assert("problem during vmaf_picture_pool_init", !err);

    VmafThreadPool *thread_pool;
    const unsigned n_threads = 4;
    err = vmaf_thread_pool_create(&thread_pool, n_threads);
    mu_assert("problem during vmaf_thread_pool_init", !err);

    const unsigned n = n_threads * 8;
    for (unsigned i = 0; i < n; i++) {
        MyThreadPoolData my_thread_pool_data = {
            .pic_pool = pic_pool,
            .i = i,
            .timeout = 100000 - (100000 * ((float)i / n)),
        };
        err = vmaf_thread_pool_enqueue(thread_pool, request_picture_and_unref,
                                       &my_thread_pool_data,
                                       sizeof(my_thread_pool_data));
        mu_assert("problem during vmaf_thread_pool_enqueue", !err);
    }

    err = vmaf_thread_pool_wait(thread_pool);
    mu_assert("problem during vmaf_thread_pool_wait", !err);
    err = vmaf_thread_pool_destroy(thread_pool);
    mu_assert("problem during vmaf_thread_pool_destroy", !err);

    err = vmaf_picture_pool_close(pic_pool);
    mu_assert("problem during vmaf_picture_pool_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_picture_alloc_ref_and_unref);
    mu_run_test(test_picture_data_alignment);
    mu_run_test(test_picture_pool);
    mu_run_test(test_picture_pool_threaded);
    return NULL;
}
