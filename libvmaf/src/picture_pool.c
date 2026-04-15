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

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "picture_pool.h"
#include "libvmaf/picture.h"
#include "mem.h"
#include "picture.h"
#include "ref.h"

/**
 * Extended picture private data that includes pool information.
 * This allows the release callback to return pictures to the pool.
 */
typedef struct PooledPicturePriv {
    VmafPicturePrivate base;
    VmafPicturePool *pool;
    unsigned pic_idx;
} PooledPicturePriv;

/**
 * CPU Picture Pool implementation.
 * Maintains a pool of reusable VmafPicture objects with pre-allocated data.
 * Uses a free list (stack) for O(1) allocation instead of O(n) linear scan.
 */
typedef struct VmafPicturePool {
    VmafPicturePoolConfig cfg;
    pthread_mutex_t lock;
    pthread_cond_t available;

    VmafPicture *pictures;        // Array of pre-allocated pictures

    unsigned *free_list;          // Stack of available picture indices
    unsigned free_list_top;       // Index of top of stack (# of free pictures)
} VmafPicturePool;

/**
 * Release callback invoked when vmaf_picture_unref() brings refcount to 0.
 * Instead of freeing the data, we return the picture to the pool and signal
 * any waiting threads.
 */
static int pooled_picture_release(VmafPicture *pic, void *cookie)
{
    (void) cookie;

    // Extract pool info from priv before it gets freed
    PooledPicturePriv *priv = (PooledPicturePriv *)pic->priv;
    VmafPicturePool *pool = priv->pool;
    unsigned idx = priv->pic_idx;

    // DON'T free pic->data[0] - it belongs to the pool picture and will be reused

    // Return picture to pool (thread-safe) - O(1) push onto free list
    pthread_mutex_lock(&pool->lock);
    pool->free_list[pool->free_list_top++] = idx;
    pthread_cond_signal(&pool->available);  // Wake one waiting thread
    pthread_mutex_unlock(&pool->lock);

    return 0;
}

/* NOLINTNEXTLINE(readability-function-size) */
int vmaf_picture_pool_init(VmafPicturePool **pool,
                           VmafPicturePoolConfig cfg)
{
    if (!pool) return -EINVAL;
    if (!cfg.pic_cnt) return -EINVAL;
    if (!cfg.w || !cfg.h) return -EINVAL;

    int err = 0;

    VmafPicturePool *const p = *pool = malloc(sizeof(*p));
    if (!p) goto fail;
    memset(p, 0, sizeof(*p));
    p->cfg = cfg;

    p->pictures = malloc(sizeof(*p->pictures) * cfg.pic_cnt);
    if (!p->pictures) {
        err = -ENOMEM;
        goto free_pool;
    }
    memset(p->pictures, 0, sizeof(*p->pictures) * cfg.pic_cnt);

    // Allocate free list (stack of available picture indices)
    p->free_list = malloc(sizeof(*p->free_list) * cfg.pic_cnt);
    if (!p->free_list) {
        err = -ENOMEM;
        goto free_pictures;
    }

    err = pthread_mutex_init(&p->lock, NULL);
    if (err) goto free_free_list;

    err = pthread_cond_init(&p->available, NULL);
    if (err) goto free_mutex;

    // Pre-allocate all pictures with their data buffers
    for (unsigned i = 0; i < cfg.pic_cnt; i++) {
        err = vmaf_picture_alloc(&p->pictures[i], cfg.pix_fmt, cfg.bpc,
                                 cfg.w, cfg.h);
        if (err) {
            // Free any pictures we've already allocated
            for (unsigned j = 0; j < i; j++) {
                vmaf_picture_unref(&p->pictures[j]);
            }
            goto free_cond;
        }

        // Clear priv and ref - we'll recreate them on each fetch
        free(p->pictures[i].priv);
        vmaf_ref_close(p->pictures[i].ref);
        p->pictures[i].priv = NULL;
        p->pictures[i].ref = NULL;

        // Push index onto free list (all pictures start available)
        p->free_list[i] = i;
    }
    p->free_list_top = cfg.pic_cnt;  // Stack is full initially

    return 0;

free_cond:
    pthread_cond_destroy(&p->available);
free_mutex:
    pthread_mutex_destroy(&p->lock);
free_free_list:
    free(p->free_list);
free_pictures:
    free(p->pictures);
free_pool:
    free(p);
fail:
    *pool = NULL;
    return err ? err : -ENOMEM;
}

int vmaf_picture_pool_close(VmafPicturePool *pool)
{
    if (!pool) return -EINVAL;

    pthread_mutex_lock(&pool->lock);

    // Wait for all pictures to be returned to the pool
    while (pool->free_list_top < pool->cfg.pic_cnt) {
        pthread_cond_wait(&pool->available, &pool->lock);
    }

    // Free all pictures (including their data buffers)
    for (unsigned i = 0; i < pool->cfg.pic_cnt; i++) {
        // Data pointers are in the picture, just free them directly
        aligned_free(pool->pictures[i].data[0]);
    }

    pthread_mutex_unlock(&pool->lock);
    pthread_cond_destroy(&pool->available);
    pthread_mutex_destroy(&pool->lock);

    free(pool->free_list);
    free(pool->pictures);
    free(pool);
    return 0;
}

int vmaf_picture_pool_fetch(VmafPicturePool *pool, VmafPicture *pic)
{
    if (!pool) return -EINVAL;
    if (!pic) return -EINVAL;

    int err = pthread_mutex_lock(&pool->lock);
    if (err) return err;

    // Wait while no pictures are available (event-driven, no polling)
    while (pool->free_list_top == 0) {
        err = pthread_cond_wait(&pool->available, &pool->lock);
        if (err) {
            pthread_mutex_unlock(&pool->lock);
            return err;
        }
    }

    // Pop picture index from free list - O(1) operation
    unsigned idx = pool->free_list[--pool->free_list_top];

    pthread_mutex_unlock(&pool->lock);

    // Copy the pre-allocated picture (includes all metadata + data pointers)
    *pic = pool->pictures[idx];

    // Set up extended priv with pool information
    PooledPicturePriv *priv = malloc(sizeof(*priv));
    if (!priv) {
        err = -ENOMEM;
        goto return_to_pool;
    }
    memset(priv, 0, sizeof(*priv));
    priv->pool = pool;
    priv->pic_idx = idx;
    pic->priv = (VmafPicturePrivate*)priv;

    // Set custom release callback to return picture to pool
    err = vmaf_picture_set_release_callback(pic, NULL, pooled_picture_release);
    if (err) {
        free(priv);
        goto return_to_pool;
    }

    // Initialize refcount to 1
    err = vmaf_ref_init(&pic->ref);
    if (err) {
        free(priv);
        goto return_to_pool;
    }

    return 0;

return_to_pool:
    // If we failed after popping from free list, return the picture
    pthread_mutex_lock(&pool->lock);
    pool->free_list[pool->free_list_top++] = idx;
    pthread_mutex_unlock(&pool->lock);
    return err;
}
