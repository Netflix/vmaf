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

#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include "framesync.h"

enum {
    BUF_FREE = 0,
    BUF_ACQUIRED,
    BUF_FILLED,
    BUF_RETRIEVED,
};

typedef struct VmafFrameSyncBuf {
    void *frame_data;
    int buf_status;
    signed long index;
    struct VmafFrameSyncBuf *next;
} VmafFrameSyncBuf;

typedef struct VmafFrameSyncContext {
    VmafFrameSyncBuf *buf_que;
    pthread_mutex_t acquire_lock;
    pthread_mutex_t retrieve_lock;
    pthread_cond_t retrieve;
    unsigned buf_cnt;
} VmafFrameSyncContext;

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx)
{
    VmafFrameSyncContext *const ctx = *fs_ctx = malloc(sizeof(VmafFrameSyncContext));
    if (!ctx) return -ENOMEM;
    memset(ctx, 0, sizeof(VmafFrameSyncContext));
    ctx->buf_cnt = 1;

    pthread_mutex_init(&(ctx->acquire_lock), NULL);
    pthread_mutex_init(&(ctx->retrieve_lock), NULL);
    pthread_cond_init(&(ctx->retrieve), NULL);

    VmafFrameSyncBuf *buf_que = ctx->buf_que = malloc(sizeof(VmafFrameSyncBuf));

    buf_que->frame_data = NULL;
    buf_que->buf_status = BUF_FREE;
    buf_que->index = -1;
    buf_que->next  = NULL;

    return 0;
}

int vmaf_framesync_acquire_new_buf(VmafFrameSyncContext *fs_ctx, void **data,
                                   unsigned data_sz, unsigned index)
{
    VmafFrameSyncBuf *buf_que = fs_ctx->buf_que;
    *data = NULL;

    pthread_mutex_lock(&(fs_ctx->acquire_lock));

    // traverse until a free buffer is found
    for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
        if (buf_que->buf_status == BUF_FREE) {
            buf_que->frame_data = *data = malloc(data_sz);
            if (!buf_que->frame_data)
                return -ENOMEM;
            buf_que->buf_status = BUF_ACQUIRED;
            buf_que->index = index;
            break;
        }
        // move to next node
        if (buf_que->next != NULL)
            buf_que = buf_que->next;
    }

    // create a new node if all nodes are occupied in the list and append to the tail
    if (*data == NULL) {
        VmafFrameSyncBuf *new_buf_node = malloc(sizeof(VmafFrameSyncBuf));
        buf_que->next = new_buf_node;
        new_buf_node->buf_status = BUF_FREE;
        new_buf_node->index = -1;
        new_buf_node->next = NULL;
        fs_ctx->buf_cnt++;

        new_buf_node->frame_data = *data = malloc(data_sz);
        if (!new_buf_node->frame_data)
            return -ENOMEM;
        new_buf_node->buf_status = BUF_ACQUIRED;
        new_buf_node->index = index;
    }

    pthread_mutex_unlock(&(fs_ctx->acquire_lock));

    return 0;
}

int vmaf_framesync_submit_filled_data(VmafFrameSyncContext *fs_ctx, void *data,
                                      unsigned index)
{
    VmafFrameSyncBuf *buf_que = fs_ctx->buf_que;

    pthread_mutex_lock(&(fs_ctx->retrieve_lock));

    // loop until a matchng buffer is found
    for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
        if ((buf_que->index == index) && (buf_que->buf_status == BUF_ACQUIRED)) {
            buf_que->buf_status = BUF_FILLED;
            if (data != buf_que->frame_data)
                return -1;
            break;
        }

        // move to next node
        if (NULL != buf_que->next)
            buf_que = buf_que->next;
    }

    pthread_cond_broadcast(&(fs_ctx->retrieve));
    pthread_mutex_unlock(&(fs_ctx->retrieve_lock));

    return 0;
}

int vmaf_framesync_retrieve_filled_data(VmafFrameSyncContext *fs_ctx,
                                        void **data, unsigned index)
{
    *data = NULL;

    while (*data == NULL) {
        VmafFrameSyncBuf *buf_que = fs_ctx->buf_que;
        pthread_mutex_lock(&(fs_ctx->retrieve_lock));
        // loop until a free buffer is found
        for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
            if ((buf_que->index == index) && (buf_que->buf_status == BUF_FILLED)) {
                buf_que->buf_status = BUF_RETRIEVED;
                *data = buf_que->frame_data;
                break;
            }

            // move to next node
            if (NULL != buf_que->next)
                buf_que = buf_que->next;
        }

        if (*data == NULL)
            pthread_cond_wait(&(fs_ctx->retrieve), &(fs_ctx->retrieve_lock));

        pthread_mutex_unlock(&(fs_ctx->retrieve_lock));
    }

    return 0;
}

int vmaf_framesync_release_buf(VmafFrameSyncContext *fs_ctx, void *data,
                               unsigned index)
{
    VmafFrameSyncBuf *buf_que = fs_ctx->buf_que;

    pthread_mutex_lock(&(fs_ctx->acquire_lock));
    // loop until a matching buffer is found
    for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
        if ((buf_que->index == index) && (buf_que->buf_status == BUF_RETRIEVED)) {
            if (data != buf_que->frame_data)
                return -1;

            free(buf_que->frame_data);
            buf_que->frame_data = NULL;
            buf_que->buf_status = BUF_FREE;
            buf_que->index = -1;
            break;
        }

        // move to next node
        if (NULL != buf_que->next)
            buf_que = buf_que->next;
    }

    pthread_mutex_unlock(&(fs_ctx->acquire_lock));
    return 0;
}

int vmaf_framesync_destroy(VmafFrameSyncContext *fs_ctx)
{
    VmafFrameSyncBuf *buf_que = fs_ctx->buf_que;
    VmafFrameSyncBuf *buf_que_tmp;

    pthread_mutex_destroy(&(fs_ctx->acquire_lock));
    pthread_mutex_destroy(&(fs_ctx->retrieve_lock));
    pthread_cond_destroy(&(fs_ctx->retrieve));

    //check for any data buffers which are not freed
    for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
        if (NULL != buf_que->frame_data) {
            free(buf_que->frame_data);
            buf_que->frame_data = NULL;
        }

        // move to next node
        if (NULL != buf_que->next) {
            buf_que_tmp = buf_que;
            buf_que = buf_que->next;
            free(buf_que_tmp);
        } else {
            free(buf_que);
        }
    }

    free(fs_ctx);

    return 0;
}
